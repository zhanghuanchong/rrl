# ============================================================================
# experiment12.py
# ============================================================================
# 实验流程概述：
#
# 第一阶段：数据集划分 (Hold-out Split)
#   将原始数据按 80% : 20% 的比例进行分层采样 (Stratified Split)。
#   80% (训练池)：用于后续的交叉验证和最终模型训练。
#   20% (独立测试集/Hold-out Set)：完全隔离，在交叉验证过程中绝对不可见，
#                                   仅用于最后评估最终模型的泛化性能。
#
# 第二阶段：5折交叉验证 (5-Fold CV)
#   仅在 80% 训练池 内部进行 5折 Stratified K-Fold 划分。
#   每一折训练一个模型，并在该折的验证集上评估。
#   目的是调整超参数、验证模型稳定性，并计算 CV 平均指标（Accuracy, F1, AUC）。
#
# 第三阶段：在测试集上验证模型性能并评估
#   使用5折交叉检验后得到的模型在 20% 独立测试集上进行最终评估。
#   输出的结果被视为论文或报告中的"主结果"，代表模型在未见数据上的真实表现。
# ============================================================================

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
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (roc_auc_score, accuracy_score, f1_score as sk_f1_score,
                             precision_score, recall_score, confusion_matrix,
                             classification_report)
from scipy.special import softmax as sp_softmax
from collections import defaultdict

from rrl.utils import read_csv, DBEncoder
from rrl.models import RRL

DATA_DIR = './dataset'


# ════════════════════════════════════════════════════════════════════════════
# 第一阶段：数据加载与 80/20 分层 Hold-out 划分
# ════════════════════════════════════════════════════════════════════════════

def _load_and_encode(dataset):
    """加载数据集并进行编码，返回编码器和编码后的特征/标签数组。"""
    data_path = os.path.join(DATA_DIR, dataset + '.data')
    info_path = os.path.join(DATA_DIR, dataset + '.info')
    X_df, y_df, f_df, label_pos = read_csv(data_path, info_path, shuffle=True)

    db_enc = DBEncoder(f_df, discrete=False)
    db_enc.fit(X_df, y_df)

    X, y = db_enc.transform(X_df, y_df, normalized=True, keep_stat=True)
    # y_labels: 一维整型类标签，用于分层采样
    y_labels = np.argmax(y, axis=1) if y.ndim == 2 else y.astype(int)
    return db_enc, X, y, y_labels


def get_holdout_split(dataset, test_size=0.2, random_state=42):
    """第一阶段 — 分层 80/20 Hold-out 划分。

    返回值
    ------
    db_enc       : 数据编码器
    X_pool       : 80% 训练池特征
    y_pool       : 80% 训练池标签
    y_pool_labels: 80% 训练池的一维整型类标签（用于分层）
    X_test       : 20% 独立测试集特征
    y_test       : 20% 独立测试集标签
    y_test_labels: 20% 独立测试集的一维整型类标签
    """
    db_enc, X, y, y_labels = _load_and_encode(dataset)

    pool_idx, test_idx = train_test_split(
        np.arange(len(X)), test_size=test_size,
        stratify=y_labels, random_state=random_state, shuffle=True)

    return (db_enc,
            X[pool_idx], y[pool_idx], y_labels[pool_idx],
            X[test_idx], y[test_idx], y_labels[test_idx])


# ════════════════════════════════════════════════════════════════════════════
# 数据加载器构建
# ════════════════════════════════════════════════════════════════════════════

def get_data_loader(dataset, world_size, rank, batch_size, k=0,
                    pin_memory=False, save_best=True):
    """在 80% 训练池上进行第 k 折交叉验证的数据加载。

    返回值
    ------
    db_enc       : 数据编码器
    train_loader : 训练集 DataLoader（若 save_best=True 则去除 5% 作为早停验证集）
    valid_loader : 早停验证集 DataLoader
    fold_val_loader : 当前折的验证集 DataLoader（用于评估交叉验证性能）
    test_loader  : 20% 独立测试集 DataLoader
    """
    (db_enc,
     X_pool, y_pool, y_pool_labels,
     X_test, y_test, _) = get_holdout_split(dataset)

    # 在训练池内部做 5 折分层划分
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    train_idx, val_idx = list(skf.split(X_pool, y_pool_labels))[k]

    X_train = X_pool[train_idx]
    y_train = y_pool[train_idx]
    X_val = X_pool[val_idx]
    y_val = y_pool[val_idx]

    train_set = TensorDataset(torch.tensor(X_train.astype(np.float32)),
                              torch.tensor(y_train.astype(np.float32)))
    fold_val_set = TensorDataset(torch.tensor(X_val.astype(np.float32)),
                                 torch.tensor(y_val.astype(np.float32)))
    test_set = TensorDataset(torch.tensor(X_test.astype(np.float32)),
                             torch.tensor(y_test.astype(np.float32)))

    # 从训练集中划出 5% 作为早停验证集（用于 save_best 模型选择）
    train_len = int(len(train_set) * 0.95)
    train_sub, es_valid_set = random_split(train_set, [train_len, len(train_set) - train_len])

    if save_best:
        # 使用去除早停验证集后的训练子集进行训练
        train_set = train_sub

    # 分布式采样器
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_set, num_replicas=world_size, rank=rank)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False,
                              pin_memory=pin_memory, sampler=train_sampler)
    valid_loader = DataLoader(es_valid_set, batch_size=batch_size, shuffle=False,
                              pin_memory=pin_memory)
    fold_val_loader = DataLoader(fold_val_set, batch_size=batch_size, shuffle=False,
                                 pin_memory=pin_memory)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                             pin_memory=pin_memory)

    return db_enc, train_loader, valid_loader, fold_val_loader, test_loader


def get_final_test_loader(dataset, batch_size):
    """获取 20% 独立测试集的 DataLoader，以及 80% 训练池 DataLoader（用于规则打印）。"""
    (db_enc,
     X_pool, y_pool, _,
     X_test, y_test, _) = get_holdout_split(dataset)

    test_set = TensorDataset(torch.tensor(X_test.astype(np.float32)),
                             torch.tensor(y_test.astype(np.float32)))
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    pool_set = TensorDataset(torch.tensor(X_pool.astype(np.float32)),
                             torch.tensor(y_pool.astype(np.float32)))
    pool_loader = DataLoader(pool_set, batch_size=batch_size, shuffle=False)

    return db_enc, pool_loader, test_loader


# ════════════════════════════════════════════════════════════════════════════
# 第二阶段：训练（分布式）
# ════════════════════════════════════════════════════════════════════════════

def train_model(gpu, args):
    """单个 GPU 上训练一个 CV 折的模型（由 mp.spawn 调用）。"""
    rank = args.nr * args.gpus + gpu
    dist.init_process_group(backend='nccl', init_method='env://',
                            world_size=args.world_size, rank=rank)
    torch.manual_seed(42)
    device_id = args.device_ids[gpu]
    torch.cuda.set_device(device_id)

    is_rank0 = (gpu == 0)
    writer = SummaryWriter(args.folder_path) if is_rank0 else None

    dataset = args.data_set
    db_enc, train_loader, valid_loader, _, _ = get_data_loader(
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


def train_main(args):
    """启动分布式训练。"""
    os.environ['MASTER_ADDR'] = args.master_address
    os.environ['MASTER_PORT'] = args.master_port
    mp.spawn(train_model, nprocs=args.gpus, args=(args,))


# ════════════════════════════════════════════════════════════════════════════
# 模型加载
# ════════════════════════════════════════════════════════════════════════════

def load_model(path, device_id, log_file=None, distributed=True):
    """从检查点文件加载 RRL 模型。"""
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
        # 移除分布式训练的 'module.' 前缀
        stat_dict[key[7:]] = stat_dict.pop(key)
    rrl.net.load_state_dict(checkpoint['model_state_dict'])
    return rrl


# ════════════════════════════════════════════════════════════════════════════
# 预测与评估辅助函数
# ════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def _predict(rrl, data_loader, device_id):
    """对 DataLoader 中的数据进行前向推理。

    返回值
    ------
    y_true_oh   : one-hot 格式的真实标签 (numpy)
    y_pred_raw  : 模型原始输出 (logits, numpy)
    y_true_label: 一维整型真实类标签
    y_pred_label: 一维整型预测类标签
    """
    y_true_list, y_pred_list = [], []
    for X, y_batch in data_loader:
        X = X.cuda(device_id, non_blocking=True)
        y_true_list.append(y_batch)
        y_pred_list.append(rrl.net.forward(X).cpu())
    y_true_oh = torch.cat(y_true_list, dim=0).numpy()
    y_pred_raw = torch.cat(y_pred_list, dim=0).numpy()
    y_true_label = np.argmax(y_true_oh, axis=1)
    y_pred_label = np.argmax(y_pred_raw, axis=1)
    return y_true_oh, y_pred_raw, y_true_label, y_pred_label


def _compute_auc(y_true_oh, y_pred_raw):
    """计算 AUC，支持二分类和多分类。"""
    n_classes = y_pred_raw.shape[1]
    try:
        y_prob = sp_softmax(y_pred_raw, axis=1)
        if n_classes == 2:
            # 二分类：使用正类概率
            auc = roc_auc_score(np.argmax(y_true_oh, axis=1), y_prob[:, 1])
        else:
            # 多分类：OVR 策略，macro 平均
            auc = roc_auc_score(y_true_oh, y_prob, multi_class='ovr', average='macro')
    except ValueError:
        auc = float('nan')
    return auc


# ────────────────────────────────────────────────────────────────────────────
# Bootstrap 95% 置信区间
# ────────────────────────────────────────────────────────────────────────────

def _bootstrap_ci(metric_fn, n_bootstrap=2000, seed=42):
    """使用 Bootstrap 方法计算 95% 置信区间。

    参数
    ----
    metric_fn : 可调用对象，接受索引数组，返回标量指标。
                需要有 ._n 属性表示样本总数。

    返回值
    ------
    (point_estimate, ci_lower, ci_upper)
    """
    rng = np.random.RandomState(seed)
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
    """将 sklearn 指标函数包装为兼容 bootstrap 索引采样的可调用对象。"""
    def fn(indices):
        return func(y_true[indices], y_pred[indices], **kwargs)
    fn._n = len(y_true)
    return fn


def _make_auc_fn(y_true_oh, y_pred_raw):
    """将 AUC 计算包装为兼容 bootstrap 索引采样的可调用对象。"""
    y_prob = sp_softmax(y_pred_raw, axis=1)
    if y_true_oh.shape[1] == 2:
        def fn(indices):
            try:
                return roc_auc_score(y_true_oh[indices, 1], y_prob[indices, 1])
            except Exception:
                return np.nan
    else:
        def fn(indices):
            try:
                return roc_auc_score(y_true_oh[indices], y_prob[indices],
                                     multi_class='ovr', average='macro')
            except Exception:
                return np.nan
    fn._n = len(y_true_oh)
    return fn


# ════════════════════════════════════════════════════════════════════════════
# 规则打印与边数统计
# ════════════════════════════════════════════════════════════════════════════

def _print_rules(rrl, db_enc, train_loader, args):
    """打印/保存规则并记录 Log(#Edges)。"""
    if args.print_rule:
        with open(args.rrl_file, 'w') as rrl_file:
            rule2weights = rrl.rule_print(db_enc.X_fname, db_enc.y_fname, train_loader,
                                          file=rrl_file, mean=db_enc.mean, std=db_enc.std)
    else:
        rule2weights = rrl.rule_print(db_enc.X_fname, db_enc.y_fname, train_loader,
                                      mean=db_enc.mean, std=db_enc.std, display=False)

    # 统计边数
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


# ════════════════════════════════════════════════════════════════════════════
# 第二阶段：交叉验证评估
# ════════════════════════════════════════════════════════════════════════════

def evaluate_cv_fold(args, fold_k):
    """在第 k 折的验证集上评估模型，返回 (accuracy, f1, auc)。"""
    rrl = load_model(args.model, args.device_ids[0], log_file=args.test_res, distributed=False)

    # 重新构造第 k 折的验证集
    (db_enc,
     X_pool, y_pool, y_pool_labels,
     _, _, _) = get_holdout_split(args.data_set)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    _, val_idx = list(skf.split(X_pool, y_pool_labels))[fold_k]

    val_set = TensorDataset(torch.tensor(X_pool[val_idx].astype(np.float32)),
                            torch.tensor(y_pool[val_idx].astype(np.float32)))
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

    # 调用模型自带的 test 方法获取准确率和 F1
    accuracy, f1 = rrl.test(test_loader=val_loader,
                            set_name='CV Fold {}'.format(fold_k))

    # 计算 AUC
    y_true_oh, y_pred_raw, _, _ = _predict(rrl, val_loader, args.device_ids[0])
    auc = _compute_auc(y_true_oh, y_pred_raw)
    if not np.isnan(auc):
        logging.info('\tAUC of RRL Model (fold {}): {:.4f}'.format(fold_k, auc))
    else:
        auc = None

    # 打印规则
    pool_set = TensorDataset(torch.tensor(X_pool.astype(np.float32)),
                             torch.tensor(y_pool.astype(np.float32)))
    pool_loader = DataLoader(pool_set, batch_size=args.batch_size, shuffle=False)
    _print_rules(rrl, db_enc, pool_loader, args)

    return accuracy, f1, auc


# ════════════════════════════════════════════════════════════════════════════
# 第三阶段：在 20% 独立测试集上进行最终评估
# ════════════════════════════════════════════════════════════════════════════

def evaluate_on_holdout(args, model_path=None):
    """在 20% 独立测试集上评估指定模型，返回完整指标字典。

    这是论文或报告中的"主结果"，代表模型在未见数据上的真实表现。
    """
    if model_path is None:
        model_path = args.model

    rrl = load_model(model_path, args.device_ids[0], log_file=args.test_res, distributed=False)
    db_enc, pool_loader, test_loader = get_final_test_loader(args.data_set, args.batch_size)

    # 使用模型自带的 test 方法获取基础指标
    accuracy, f1 = rrl.test(test_loader=test_loader, set_name='Hold-out Test (20%)')

    # 详细预测结果
    y_true_oh, y_pred_raw, y_true_label, y_pred_label = _predict(rrl, test_loader, args.device_ids[0])

    # ── 基础指标 ──
    precision = precision_score(y_true_label, y_pred_label, average='macro', zero_division=0)
    recall_val = recall_score(y_true_label, y_pred_label, average='macro', zero_division=0)

    # ── Sensitivity / Specificity / PPV / NPV ──
    # 二分类时直接计算，多分类时取 macro 均值
    cm = confusion_matrix(y_true_label, y_pred_label)
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
    auc = _compute_auc(y_true_oh, y_pred_raw)
    auc = auc if not np.isnan(auc) else None

    # ── 95% Bootstrap 置信区间 ──
    acc_point, acc_lo, acc_hi = _bootstrap_ci(
        _make_metric_fn(y_true_label, y_pred_label, accuracy_score))
    if auc is not None:
        auc_fn = _make_auc_fn(y_true_oh, y_pred_raw)
        auc_point, auc_lo, auc_hi = _bootstrap_ci(auc_fn)
    else:
        auc_point, auc_lo, auc_hi = None, None, None

    results = {
        'Test_Accuracy': accuracy,
        'Test_Accuracy_95%CI': '({:.4f}, {:.4f})'.format(acc_lo, acc_hi),
        'Test_AUC': auc,
        'Test_AUC_95%CI': '({:.4f}, {:.4f})'.format(auc_lo, auc_hi) if auc is not None else 'N/A',
        'Test_Precision': precision,
        'Test_Recall': recall_val,
        'Test_F1-Score': f1,
        'Test_Sensitivity': sensitivity,
        'Test_Specificity': specificity,
        'Test_PPV': ppv,
        'Test_NPV': npv,
    }

    # 打印规则
    _print_rules(rrl, db_enc, pool_loader, args)

    return results


# ════════════════════════════════════════════════════════════════════════════
# 主流程
# ════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    from args import rrl_args

    # 设置根日志，使 CV 汇总信息可在控制台看到
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - [%(levelname)s] - %(message)s')

    # 保存根目录
    root_folder = rrl_args.folder_path

    # ──────────────────────────────────────────────────────────────────────
    # 第一阶段：数据集划分
    #   数据已在 get_holdout_split 中按 80/20 分层划分 (random_state=42)
    # ──────────────────────────────────────────────────────────────────────
    print("=" * 80)
    print("第一阶段：数据集划分为 80% 训练池 + 20% 独立测试集 (Stratified, random_state=42)")
    print("=" * 80)

    # ──────────────────────────────────────────────────────────────────────
    # 第二阶段：在 80% 训练池上进行 5 折分层交叉验证
    # ──────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("第二阶段：在 80% 训练池上进行 5 折分层交叉验证 (Stratified 5-Fold CV)")
    print("=" * 80)

    cv_results = []  # 保存每一折的 (accuracy, f1, auc, model_path)

    for k in range(5):
        print("\n" + "─" * 80)
        print("  第 {}/5 折".format(k + 1))
        print("─" * 80)

        # 为每一折创建独立的输出目录
        fold_folder = os.path.join(root_folder, str(k))
        if not os.path.exists(fold_folder):
            os.makedirs(fold_folder)

        # 更新参数指向当前折的路径
        rrl_args.ith_kfold = k
        rrl_args.model = os.path.join(fold_folder, 'model.pth')
        rrl_args.rrl_file = os.path.join(fold_folder, 'rrl.txt')
        rrl_args.test_res = os.path.join(fold_folder, 'test_res.txt')
        rrl_args.log = os.path.join(fold_folder, 'log.txt')
        rrl_args.folder_path = fold_folder

        # 训练当前折的模型
        train_main(rrl_args)

        # 在当前折的验证集上评估
        acc, f1, auc = evaluate_cv_fold(rrl_args, k)
        fold_model_path = rrl_args.model
        cv_results.append({
            'fold': k,
            'accuracy': acc,
            'f1_score': f1,
            'auc': auc,
            'model_path': fold_model_path
        })

        print("  Fold {} — Accuracy={:.4f}  F1={:.4f}".format(k + 1, acc, f1), end="")
        if auc is not None:
            print("  AUC={:.4f}".format(auc))
        else:
            print()

    # ── 交叉验证汇总 ──
    print("\n" + "=" * 80)
    print("5 折交叉验证汇总")
    print("=" * 80)
    for r in cv_results:
        print("  Fold {}: Accuracy={:.4f}, F1={:.4f}".format(r['fold'] + 1, r['accuracy'], r['f1_score']), end="")
        if r['auc'] is not None:
            print(", AUC={:.4f}".format(r['auc']))
        else:
            print()

    accs = [r['accuracy'] for r in cv_results]
    f1s  = [r['f1_score'] for r in cv_results]
    aucs = [r['auc'] for r in cv_results if r['auc'] is not None]

    print("\n  CV_Accuracy : {:.4f} ± {:.4f}".format(np.mean(accs), np.std(accs)))
    print("  CV_F1       : {:.4f} ± {:.4f}".format(np.mean(f1s), np.std(f1s)))
    if aucs:
        print("  CV_AUC      : {:.4f} ± {:.4f}".format(np.mean(aucs), np.std(aucs)))

    # 选择验证集 F1 最高的折作为最优模型
    best_fold_idx = int(np.argmax(f1s))
    best_model_path = cv_results[best_fold_idx]['model_path']
    print("\n  最优折: 第 {} 折 (F1={:.4f})".format(best_fold_idx + 1, f1s[best_fold_idx]))
    print("  最优模型路径: {}".format(best_model_path))

    # ──────────────────────────────────────────────────────────────────────
    # 第三阶段：在 20% 独立测试集上评估最优模型
    # ──────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("第三阶段：在 20% 独立测试集上评估最优模型（最终泛化性能）")
    print("=" * 80)

    # 设置最终评估的输出路径
    final_folder = os.path.join(root_folder, 'final_holdout')
    if not os.path.exists(final_folder):
        os.makedirs(final_folder)

    rrl_args.test_res = os.path.join(final_folder, 'test_res.txt')
    rrl_args.rrl_file = os.path.join(final_folder, 'rrl.txt')
    rrl_args.folder_path = final_folder

    final_results = evaluate_on_holdout(rrl_args, model_path=best_model_path)

    print("\n最终独立测试集结果（论文/报告主结果）:")
    for key, val in final_results.items():
        if isinstance(val, float):
            print("  {:30s}: {:.4f}".format(key, val))
        else:
            print("  {:30s}: {}".format(key, val))

    # 保存结果到 CSV
    csv_path = os.path.join(final_folder, 'final_test_results.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=final_results.keys())
        writer.writeheader()
        writer.writerow(final_results)
    print("\n结果已保存至: {}".format(csv_path))

    # 同时保存 CV 汇总到 CSV
    cv_csv_path = os.path.join(root_folder, 'cv_summary.csv')
    with open(cv_csv_path, 'w', newline='') as f:
        fieldnames = ['fold', 'accuracy', 'f1_score', 'auc', 'model_path']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in cv_results:
            writer.writerow(r)
        # 写入均值行
        writer.writerow({
            'fold': 'mean±std',
            'accuracy': '{:.4f}±{:.4f}'.format(np.mean(accs), np.std(accs)),
            'f1_score': '{:.4f}±{:.4f}'.format(np.mean(f1s), np.std(f1s)),
            'auc': '{:.4f}±{:.4f}'.format(np.mean(aucs), np.std(aucs)) if aucs else 'N/A',
            'model_path': ''
        })
    print("CV 汇总已保存至: {}".format(cv_csv_path))
    print("=" * 80)

