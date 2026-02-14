import os
import csv
import logging
import numpy as np
import torch
torch.set_num_threads(2)
from torch.utils.data.dataset import random_split
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp
import torch.distributed as dist
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import (roc_auc_score, accuracy_score, f1_score as sk_f1_score,
                             precision_score, recall_score, confusion_matrix)
from collections import defaultdict

from rrl.utils import read_csv, DBEncoder
from rrl.models import RRL

DATA_DIR = './dataset'

# ────────────────────────────────────────────────────────────────────────────
# 数据加载
# ────────────────────────────────────────────────────────────────────────────

def _load_and_split(dataset):
    """加载数据并划分80%训练集和20%测试集 (random_state=42)。"""
    data_path = os.path.join(DATA_DIR, dataset + '.data')
    info_path = os.path.join(DATA_DIR, dataset + '.info')
    X_df, y_df, f_df, label_pos = read_csv(data_path, info_path, shuffle=True)

    db_enc = DBEncoder(f_df, discrete=False)
    db_enc.fit(X_df, y_df)

    X, y = db_enc.transform(X_df, y_df, normalized=True, keep_stat=True)

    train_val_idx, final_test_idx = train_test_split(
        np.arange(len(X)), test_size=0.2, random_state=42, shuffle=True
    )
    return db_enc, X, y, train_val_idx, final_test_idx


def get_data_loader(dataset, world_size, rank, batch_size, k=0, pin_memory=False, save_best=True):
    """在80%训练数据上进行第 k 折交叉验证的数据加载。"""
    db_enc, X, y, train_val_idx, _ = _load_and_split(dataset)

    X_tv = X[train_val_idx]
    y_tv = y[train_val_idx]

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    train_index, test_index = list(kf.split(X_tv))[k]
    X_train = X_tv[train_index]
    y_train = y_tv[train_index]
    X_test = X_tv[test_index]
    y_test = y_tv[test_index]

    train_set = TensorDataset(torch.tensor(X_train.astype(np.float32)), torch.tensor(y_train.astype(np.float32)))
    test_set = TensorDataset(torch.tensor(X_test.astype(np.float32)), torch.tensor(y_test.astype(np.float32)))

    train_len = int(len(train_set) * 0.95)
    train_sub, valid_set = random_split(train_set, [train_len, len(train_set) - train_len])

    if save_best:
        train_set = train_sub

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_set, num_replicas=world_size, rank=rank)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False, pin_memory=pin_memory,
                              sampler=train_sampler)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, pin_memory=pin_memory)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, pin_memory=pin_memory)

    return db_enc, train_loader, valid_loader, test_loader


def get_full_train_loader(dataset, world_size, rank, batch_size, pin_memory=False, save_best=True):
    """使用完整80%训练集训练最终模型的数据加载。"""
    db_enc, X, y, train_val_idx, _ = _load_and_split(dataset)

    X_tv = X[train_val_idx]
    y_tv = y[train_val_idx]

    train_set = TensorDataset(torch.tensor(X_tv.astype(np.float32)), torch.tensor(y_tv.astype(np.float32)))

    train_len = int(len(train_set) * 0.95)
    train_sub, valid_set = random_split(train_set, [train_len, len(train_set) - train_len])

    if save_best:
        train_set = train_sub

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_set, num_replicas=world_size, rank=rank)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False, pin_memory=pin_memory,
                              sampler=train_sampler)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, pin_memory=pin_memory)

    return db_enc, train_loader, valid_loader


def get_final_test_loader(dataset, batch_size):
    """获取20%独立测试集的数据加载器及80%训练集 loader (用于规则打印)。"""
    db_enc, X, y, train_val_idx, final_test_idx = _load_and_split(dataset)

    X_final = X[final_test_idx]
    y_final = y[final_test_idx]
    final_test_set = TensorDataset(torch.tensor(X_final.astype(np.float32)), torch.tensor(y_final.astype(np.float32)))
    test_loader = DataLoader(final_test_set, batch_size=batch_size, shuffle=False)

    X_tv = X[train_val_idx]
    y_tv = y[train_val_idx]
    train_set = TensorDataset(torch.tensor(X_tv.astype(np.float32)), torch.tensor(y_tv.astype(np.float32)))
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False)

    return db_enc, train_loader, test_loader

# ────────────────────────────────────────────────────────────────────────────
# 训练 / 加载
# ────────────────────────────────────────────────────────────────────────────

def train_model(gpu, args):
    rank = args.nr * args.gpus + gpu
    dist.init_process_group(backend='nccl', init_method='env://', world_size=args.world_size, rank=rank)
    torch.manual_seed(42)
    device_id = args.device_ids[gpu]
    torch.cuda.set_device(device_id)

    if gpu == 0:
        writer = SummaryWriter(args.folder_path)
        is_rank0 = True
    else:
        writer = None
        is_rank0 = False

    dataset = args.data_set

    if getattr(args, 'use_full_train', False):
        db_enc, train_loader, valid_loader = get_full_train_loader(
            dataset, args.world_size, rank, args.batch_size, pin_memory=True, save_best=args.save_best)
    else:
        db_enc, train_loader, valid_loader, _ = get_data_loader(
            dataset, args.world_size, rank, args.batch_size,
            k=args.ith_kfold, pin_memory=True, save_best=args.save_best)

    y_fname = db_enc.y_fname
    discrete_flen = db_enc.discrete_flen
    continuous_flen = db_enc.continuous_flen

    rrl = RRL(dim_list=[(discrete_flen, continuous_flen)] + list(map(int, args.structure.split('@'))) + [len(y_fname)],
              device_id=device_id,
              use_not=args.use_not,
              is_rank0=is_rank0,
              log_file=args.log,
              writer=writer,
              save_best=args.save_best,
              estimated_grad=args.estimated_grad,
              use_skip=args.skip,
              save_path=args.model,
              use_nlaf=args.nlaf,
              alpha=args.alpha,
              beta=args.beta,
              gamma=args.gamma,
              temperature=args.temp)

    rrl.train_model(
        data_loader=train_loader,
        valid_loader=valid_loader,
        lr=args.learning_rate,
        epoch=args.epoch,
        lr_decay_rate=args.lr_decay_rate,
        lr_decay_epoch=args.lr_decay_epoch,
        weight_decay=args.weight_decay,
        log_iter=args.log_iter)


def load_model(path, device_id, log_file=None, distributed=True):
    checkpoint = torch.load(path, map_location='cpu')
    saved_args = checkpoint['rrl_args']
    rrl = RRL(
        dim_list=saved_args['dim_list'],
        device_id=device_id,
        is_rank0=True,
        use_not=saved_args['use_not'],
        log_file=log_file,
        distributed=distributed,
        estimated_grad=saved_args['estimated_grad'],
        use_skip=saved_args['use_skip'],
        use_nlaf=saved_args['use_nlaf'],
        alpha=saved_args['alpha'],
        beta=saved_args['beta'],
        gamma=saved_args['gamma'])
    stat_dict = checkpoint['model_state_dict']
    for key in list(stat_dict.keys()):
        stat_dict[key[7:]] = stat_dict.pop(key)
    rrl.net.load_state_dict(checkpoint['model_state_dict'])
    return rrl

# ────────────────────────────────────────────────────────────────────────────
# 预测辅助
# ────────────────────────────────────────────────────────────────────────────

def _predict(rrl, data_loader, device_id):
    """返回 (y_true_onehot, y_pred_prob, y_true_label, y_pred_label)。"""
    y_true_list, y_pred_list = [], []
    with torch.no_grad():
        for X, y_batch in data_loader:
            X = X.cuda(device_id, non_blocking=True)
            y_true_list.append(y_batch)
            y_pred_list.append(rrl.net.forward(X).cpu())
    y_true_oh = torch.cat(y_true_list, dim=0).numpy()
    y_pred_prob = torch.cat(y_pred_list, dim=0).numpy()
    y_true_label = np.argmax(y_true_oh, axis=1)
    y_pred_label = np.argmax(y_pred_prob, axis=1)
    return y_true_oh, y_pred_prob, y_true_label, y_pred_label

# ────────────────────────────────────────────────────────────────────────────
# 95% CI (Bootstrap)
# ────────────────────────────────────────────────────────────────────────────

def _bootstrap_ci(metric_fn, n_bootstrap=2000, seed=42):
    """返回 (point_estimate, ci_lower, ci_upper)。
    metric_fn(indices) 应返回标量指标。"""
    rng = np.random.RandomState(seed)
    n = metric_fn.__code__.co_varnames  # 占位，实际不用
    scores = []
    for _ in range(n_bootstrap):
        idx = rng.choice(metric_fn._n, size=metric_fn._n, replace=True)
        scores.append(metric_fn(idx))
    scores = np.array(scores)
    point = metric_fn(np.arange(metric_fn._n))
    lower = np.percentile(scores, 2.5)
    upper = np.percentile(scores, 97.5)
    return point, lower, upper


def _make_metric_fn(y_true, y_pred, func, **kwargs):
    """包装 sklearn 指标函数以兼容 bootstrap 索引采样。"""
    def fn(indices):
        return func(y_true[indices], y_pred[indices], **kwargs)
    fn._n = len(y_true)
    return fn


def _make_auc_fn(y_true_oh, y_pred_prob):
    """包装 AUC 计算以兼容 bootstrap 索引采样。"""
    if y_true_oh.shape[1] == 2:
        def fn(indices):
            try:
                return roc_auc_score(y_true_oh[indices, 1], y_pred_prob[indices, 1])
            except Exception:
                return np.nan
    else:
        def fn(indices):
            try:
                return roc_auc_score(y_true_oh[indices], y_pred_prob[indices], multi_class='ovr', average='macro')
            except Exception:
                return np.nan
    fn._n = len(y_true_oh)
    return fn

# ────────────────────────────────────────────────────────────────────────────
# 测试 / 评估
# ────────────────────────────────────────────────────────────────────────────

def test_model_cv(args):
    """在交叉验证折的验证集上测试，返回 (accuracy, f1, auc)。"""
    rrl = load_model(args.model, args.device_ids[0], log_file=args.test_res, distributed=False)
    db_enc, train_loader, _, test_loader = get_data_loader(
        args.data_set, 4, 0, args.batch_size, args.ith_kfold, save_best=False)

    accuracy, f1 = rrl.test(test_loader=test_loader, set_name='CV Fold {}'.format(args.ith_kfold))

    y_true_oh, y_pred_prob, _, _ = _predict(rrl, test_loader, args.device_ids[0])
    auc_fn = _make_auc_fn(y_true_oh, y_pred_prob)
    auc = auc_fn(np.arange(auc_fn._n))
    if not np.isnan(auc):
        logging.info('\n\tAUC of RRL Model (fold {}): {}'.format(args.ith_kfold, auc))
    else:
        auc = None

    _print_rules(rrl, db_enc, train_loader, args)
    return accuracy, f1, auc


def test_model_final(args):
    """在20%独立测试集上评估最终模型，返回完整指标字典。"""
    rrl = load_model(args.model, args.device_ids[0], log_file=args.test_res, distributed=False)
    db_enc, train_loader, test_loader = get_final_test_loader(args.data_set, args.batch_size)

    accuracy, f1 = rrl.test(test_loader=test_loader, set_name='Final Test (20% holdout)')

    y_true_oh, y_pred_prob, y_true, y_pred = _predict(rrl, test_loader, args.device_ids[0])

    # ── 基础指标 ──
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)

    # ── Sensitivity / Specificity / PPV / NPV (二分类时直接计算，多分类时取 macro 均值) ──
    cm = confusion_matrix(y_true, y_pred)
    n_classes = cm.shape[0]
    sensitivities, specificities, ppvs, npvs = [], [], [], []
    for c in range(n_classes):
        tp = cm[c, c]
        fn = cm[c, :].sum() - tp
        fp = cm[:, c].sum() - tp
        tn = cm.sum() - tp - fn - fp
        sensitivities.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
        specificities.append(tn / (tn + fp) if (tn + fp) > 0 else 0.0)
        ppvs.append(tp / (tp + fp) if (tp + fp) > 0 else 0.0)
        npvs.append(tn / (tn + fn) if (tn + fn) > 0 else 0.0)
    sensitivity = np.mean(sensitivities)
    specificity = np.mean(specificities)
    ppv = np.mean(ppvs)
    npv = np.mean(npvs)

    # ── AUC ──
    auc_fn = _make_auc_fn(y_true_oh, y_pred_prob)
    auc = auc_fn(np.arange(auc_fn._n))
    auc = auc if not np.isnan(auc) else None

    # ── 95% CI (Bootstrap) ──
    acc_point, acc_lo, acc_hi = _bootstrap_ci(
        _make_metric_fn(y_true, y_pred, accuracy_score))
    auc_point, auc_lo, auc_hi = (None, None, None)
    if auc is not None:
        auc_point, auc_lo, auc_hi = _bootstrap_ci(auc_fn)

    results = {
        'Test_Accuracy': accuracy,
        'Test_Accuracy_95%CI': f'({acc_lo:.4f}, {acc_hi:.4f})',
        'Test_AUC': auc,
        'Test_AUC_95%CI': f'({auc_lo:.4f}, {auc_hi:.4f})' if auc is not None else 'N/A',
        'Test_Precision': precision,
        'Test_Recall': recall,
        'Test_F1-Score': f1,
        'Test_Sensitivity': sensitivity,
        'Test_Specificity': specificity,
        'Test_PPV': ppv,
        'Test_NPV': npv,
    }

    _print_rules(rrl, db_enc, train_loader, args)
    return results


def _print_rules(rrl, db_enc, train_loader, args):
    """打印/保存规则并记录 Log(#Edges)。"""
    if args.print_rule:
        with open(args.rrl_file, 'w') as rrl_file:
            rule2weights = rrl.rule_print(db_enc.X_fname, db_enc.y_fname, train_loader, file=rrl_file,
                                          mean=db_enc.mean, std=db_enc.std)
    else:
        rule2weights = rrl.rule_print(db_enc.X_fname, db_enc.y_fname, train_loader,
                                      mean=db_enc.mean, std=db_enc.std, display=False)

    edge_cnt = 0
    connected_rid = defaultdict(lambda: set())
    ln = len(rrl.net.layer_list) - 1
    for rid, w in rule2weights:
        connected_rid[ln - abs(rid[0])].add(rid[1])
    while ln > 1:
        ln -= 1
        layer = rrl.net.layer_list[ln]
        for r in connected_rid[ln]:
            con_len = len(layer.rule_list[0])
            if r >= con_len:
                opt_id = 1
                r -= con_len
            else:
                opt_id = 0
            rule = layer.rule_list[opt_id][r]
            edge_cnt += len(rule)
            for rid in rule:
                connected_rid[ln - abs(rid[0])].add(rid[1])
    if edge_cnt > 0:
        logging.info('\n\tLog(#Edges) of RRL Model: {}'.format(np.log(edge_cnt)))


def train_main(args):
    os.environ['MASTER_ADDR'] = args.master_address
    os.environ['MASTER_PORT'] = args.master_port
    mp.spawn(train_model, nprocs=args.gpus, args=(args,))


# ════════════════════════════════════════════════════════════════════════════
# 主流程: 4 步
# ════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    from args import rrl_args

    test_file = rrl_args.folder_path  # 保存根目录

    # ──────────────────────────────────────────────────────────────────────
    # 第1步: 数据已在 _load_and_split 中按 80/20 划分 (random_state=42)
    # ──────────────────────────────────────────────────────────────────────
    print("=" * 80)
    print("第1步: 数据集划分为 80% 训练集 + 20% 独立测试集 (random_state=42)")
    print("=" * 80)

    # ──────────────────────────────────────────────────────────────────────
    # 第2步: 在80%训练集上做5折交叉验证
    # ──────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("第2步: 在80%训练集上进行5折交叉验证")
    print("=" * 80)

    cv_results = []
    rrl_args.use_full_train = False

    for k in range(5):
        print(f"\n{'─'*80}")
        print(f"  第 {k+1}/5 折")
        print(f"{'─'*80}")

        rrl_args.ith_kfold = k

        folder_path_ = os.path.join(test_file, str(k))
        if not os.path.exists(folder_path_):
            os.makedirs(folder_path_)

        rrl_args.model = os.path.join(folder_path_, 'model.pth')
        rrl_args.rrl_file = os.path.join(folder_path_, 'rrl.txt')
        rrl_args.test_res = os.path.join(folder_path_, 'test_res.txt')
        rrl_args.log = os.path.join(folder_path_, 'log.txt')
        rrl_args.folder_path = folder_path_

        train_main(rrl_args)
        acc, f1, auc = test_model_cv(rrl_args)
        cv_results.append({'fold': k, 'accuracy': acc, 'f1_score': f1, 'auc': auc})

        print(f"  Fold {k+1}  Accuracy={acc:.4f}  F1={f1:.4f}", end="")
        if auc is not None:
            print(f"  AUC={auc:.4f}")
        else:
            print()

    # 汇总
    print("\n" + "=" * 80)
    print("5折交叉验证汇总")
    print("=" * 80)
    for r in cv_results:
        print(f"  Fold {r['fold']+1}: Accuracy={r['accuracy']:.4f}, F1={r['f1_score']:.4f}", end="")
        if r['auc'] is not None:
            print(f", AUC={r['auc']:.4f}")
        else:
            print()

    accs = [r['accuracy'] for r in cv_results]
    f1s  = [r['f1_score'] for r in cv_results]
    aucs = [r['auc'] for r in cv_results if r['auc'] is not None]

    print(f"\n  CV_Accuracy : {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    print(f"  CV_F1       : {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
    if aucs:
        print(f"  CV_AUC      : {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")

    # ──────────────────────────────────────────────────────────────────────
    # 第3步: 用完整80%训练集训练最终模型
    # ──────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("第3步: 在完整80%训练集上训练最终模型")
    print("=" * 80)

    final_folder = os.path.join(test_file, 'final')
    if not os.path.exists(final_folder):
        os.makedirs(final_folder)

    rrl_args.use_full_train = True
    rrl_args.model = os.path.join(final_folder, 'model.pth')
    rrl_args.rrl_file = os.path.join(final_folder, 'rrl.txt')
    rrl_args.test_res = os.path.join(final_folder, 'test_res.txt')
    rrl_args.log = os.path.join(final_folder, 'log.txt')
    rrl_args.folder_path = final_folder

    train_main(rrl_args)

    # ──────────────────────────────────────────────────────────────────────
    # 第4步: 在20%独立测试集上评估最终模型
    # ──────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("第4步: 在20%独立测试集上评估最终模型")
    print("=" * 80)

    final_results = test_model_final(rrl_args)

    print("\n最终独立测试集结果:")
    for key, val in final_results.items():
        if isinstance(val, float):
            print(f"  {key:30s}: {val:.4f}")
        else:
            print(f"  {key:30s}: {val}")

    # 保存到 CSV
    csv_path = os.path.join(final_folder, 'final_test_results.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=final_results.keys())
        writer.writeheader()
        writer.writerow(final_results)
    print(f"\n结果已保存至: {csv_path}")
    print("=" * 80)
