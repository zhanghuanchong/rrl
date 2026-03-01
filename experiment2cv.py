# ============================================================================
# experiment13.py
# ============================================================================
# 实验流程概述：
#
# 第一阶段：5折交叉验证网格搜索调优超参数
#   针对整个数据集进行 5 折分层交叉验证网格搜索。
#   根据 args.py 文件生成 param_grid，random_state=42。
#   每一折训练一个模型，在该折验证集上评估：
#     Accuracy、F1、AUC(含95%Bootstrap 置信区间)、Precision、Recall、
#     Sensitivity、Specificity、PPV、NPV。
#   汇总 5 折结果(均值±标准差)，输出最优超参数、交叉验证最佳得分，获取最优模型。
#
# 第二阶段：在提前准备的新 data 文件(即测试集)上进行最终评估
#   输出完整指标: Accuracy、AUC(含95%Bootstrap 置信区间)、Precision、Recall、
#   F1、Sensitivity、Specificity、PPV、NPV。结果保存为 CSV。
# ============================================================================

import os
import csv
import logging
import itertools
import copy
import numpy as np
import torch
torch.set_num_threads(2)
from torch.utils.data.dataset import random_split
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp
import torch.distributed as dist
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (roc_auc_score, accuracy_score, f1_score as sk_f1_score,
                             precision_score, recall_score, confusion_matrix)
from scipy.special import softmax as sp_softmax
from collections import defaultdict

from rrl.utils import read_csv, DBEncoder
from rrl.models import RRL

DATA_DIR = './dataset'

# ════════════════════════════════════════════════════════════════════════════
# 超参数网格（参考 search.py 中的 param_grid）
# ════════════════════════════════════════════════════════════════════════════

param_grid = {
    "learning_rate": [0.0003,0.0001],
    "lr_decay_epoch": [50,60],
    "weight_decay": [5e-3,1e-2],
    "batch_size": [32],
    "epoch": [250],
    "structure": [
        "2@256",
        "5@32",
        "10@32",
        "10@32@32"
    ],
    "temp": [0.01],  # 选最好的
    "use_not": [False],
    "skip": [True],
    # 这三个参数只试那三种组合，现在这样是8种
    "alpha": [0.9],
    "beta": [3],
    "gamma": [3],
    "nlaf": [True],
    "weighted":[True]
}


# ════════════════════════════════════════════════════════════════════════════
# 数据加载与编码
# ════════════════════════════════════════════════════════════════════════════

def _load_and_encode(dataset):
    """加载数据集并编码，返回编码器和编码后的特征/标签数组。"""
    data_path = os.path.join(DATA_DIR, dataset + '.data')
    info_path = os.path.join(DATA_DIR, dataset + '.info')
    X_df, y_df, f_df, label_pos = read_csv(data_path, info_path, shuffle=True)

    db_enc = DBEncoder(f_df, discrete=False)
    db_enc.fit(X_df, y_df)

    X, y = db_enc.transform(X_df, y_df, normalized=True, keep_stat=True)
    y_labels = np.argmax(y, axis=1) if y.ndim == 2 else y.astype(int)
    return db_enc, X, y, y_labels


# ════════════════════════════════════════════════════════════════════════════
# 数据加载器构建
# ════════════════════════════════════════════════════════════════════════════

def get_cv_data_loader(dataset, world_size, rank, batch_size, k=0,
                       pin_memory=False, save_best=True, random_state=42):
    """在整个数据集上进行第 k 折交叉验证的数据加载。"""
    db_enc, X, y, y_labels = _load_and_encode(dataset)

    skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=random_state)
    train_idx, val_idx = list(skf.split(X, y_labels))[k]

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    train_set = TensorDataset(torch.tensor(X_train.astype(np.float32)),
                              torch.tensor(y_train.astype(np.float32)))
    val_set = TensorDataset(torch.tensor(X_val.astype(np.float32)),
                            torch.tensor(y_val.astype(np.float32)))

    # 从训练集中划出 5% 作为早停验证集
    train_len = int(len(train_set) * 0.95)
    train_sub, es_valid_set = random_split(train_set, [train_len, len(train_set) - train_len])

    if save_best:
        train_set = train_sub

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_set, num_replicas=world_size, rank=rank)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False,
                              pin_memory=pin_memory, sampler=train_sampler)
    es_valid_loader = DataLoader(es_valid_set, batch_size=batch_size, shuffle=False,
                                pin_memory=pin_memory)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                            pin_memory=pin_memory)

    return db_enc, train_loader, es_valid_loader, val_loader


def get_test_data_loader(test_dataset, db_enc_ref, batch_size):
    """加载独立测试集（新的 data 文件），使用训练集的编码器进行变换。"""
    data_path = os.path.join(DATA_DIR, test_dataset + '.data')
    info_path = os.path.join(DATA_DIR, test_dataset + '.info')
    X_df, y_df, f_df, label_pos = read_csv(data_path, info_path, shuffle=False)

    X, y = db_enc_ref.transform(X_df, y_df, normalized=True, keep_stat=False)

    test_set = TensorDataset(torch.tensor(X.astype(np.float32)),
                             torch.tensor(y.astype(np.float32)))
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    # 同时返回全量训练集 loader（用于规则打印）
    return test_loader


# ════════════════════════════════════════════════════════════════════════════
# 训练（分布式）
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
    db_enc, train_loader, es_valid_loader, _ = get_cv_data_loader(
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
        valid_loader=es_valid_loader,
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
        stat_dict[key[7:]] = stat_dict.pop(key)
    rrl.net.load_state_dict(checkpoint['model_state_dict'])
    return rrl


# ════════════════════════════════════════════════════════════════════════════
# 预测与评估辅助函数
# ════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def _predict(rrl, data_loader, device_id):
    """对 DataLoader 中的数据进行前向推理。"""
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
            auc = roc_auc_score(np.argmax(y_true_oh, axis=1), y_prob[:, 1])
        else:
            auc = roc_auc_score(y_true_oh, y_prob, multi_class='ovr', average='macro')
    except ValueError:
        auc = float('nan')
    return auc


# ────────────────────────────────────────────────────────────────────────────
# Bootstrap 95% 置信区间
# ────────────────────────────────────────────────────────────────────────────

def _bootstrap_ci(metric_fn, n_bootstrap=2000, seed=42):
    """使用 Bootstrap 方法计算 95% 置信区间。"""
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


def compute_full_metrics(y_true_oh, y_pred_raw, y_true_label, y_pred_label):
    """计算完整指标集合，包括 Bootstrap 95% CI。

    返回字典包含:
    Accuracy, F1, AUC, AUC_95%CI, Precision, Recall,
    Sensitivity, Specificity, PPV, NPV
    """
    accuracy = accuracy_score(y_true_label, y_pred_label)
    f1 = sk_f1_score(y_true_label, y_pred_label, average='macro', zero_division=0)
    precision = precision_score(y_true_label, y_pred_label, average='macro', zero_division=0)
    recall_val = recall_score(y_true_label, y_pred_label, average='macro', zero_division=0)

    # Sensitivity / Specificity / PPV / NPV
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

    # AUC
    auc = _compute_auc(y_true_oh, y_pred_raw)
    auc = auc if not np.isnan(auc) else None

    # 95% Bootstrap CI for AUC
    if auc is not None:
        auc_fn = _make_auc_fn(y_true_oh, y_pred_raw)
        _, auc_lo, auc_hi = _bootstrap_ci(auc_fn)
        auc_ci = '({:.4f}, {:.4f})'.format(auc_lo, auc_hi)
    else:
        auc_ci = 'N/A'

    return {
        'Accuracy': accuracy,
        'F1': f1,
        'AUC': auc,
        'AUC_95%CI': auc_ci,
        'Precision': precision,
        'Recall': recall_val,
        'Sensitivity': sensitivity,
        'Specificity': specificity,
        'PPV': ppv,
        'NPV': npv,
    }


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
# 网格搜索工具
# ════════════════════════════════════════════════════════════════════════════

def _generate_param_combinations(grid):
    """将 param_grid 字典展开为参数组合列表。"""
    keys = list(grid.keys())
    values = list(grid.values())
    combos = []
    for vals in itertools.product(*values):
        combos.append(dict(zip(keys, vals)))
    return combos


def _apply_params_to_args(args, params):
    """将一组超参数应用到 args 对象上。"""
    for key, val in params.items():
        setattr(args, key, val)


# ════════════════════════════════════════════════════════════════════════════
# 第一阶段：10折交叉验证网格搜索
# ════════════════════════════════════════════════════════════════════════════

def grid_search_cv(args, grid=None):
    """在整个数据集上进行 10折交叉验证网格搜索。

    返回
    ------
    best_params     : 最优超参数字典
    best_cv_score   : 最优交叉验证 F1 均值
    best_model_path : 最优模型路径
    all_results     : 所有参数组合的结果列表
    """
    if grid is None:
        grid = param_grid

    param_combos = _generate_param_combinations(grid)
    total_combos = len(param_combos)
    logging.info('网格搜索: 共 {} 组超参数组合'.format(total_combos))

    root_folder = args.folder_path
    all_results = []
    best_cv_score = -1.0
    best_params = None
    best_model_path = None

    for combo_idx, params in enumerate(param_combos):
        logging.info('\n' + '=' * 80)
        logging.info('超参数组合 {}/{}: {}'.format(combo_idx + 1, total_combos, params))
        logging.info('=' * 80)

        # 将参数应用到 args
        args_copy = copy.deepcopy(args)
        _apply_params_to_args(args_copy, params)

        # 为当前组合创建独立目录
        combo_name = 'combo_{}'.format(combo_idx)
        combo_folder = os.path.join(root_folder, combo_name)
        if not os.path.exists(combo_folder):
            os.makedirs(combo_folder)

        # 重新计算 batch_size（按 GPU 数均分）
        args_copy.batch_size = int(args_copy.batch_size / args_copy.gpus)

        fold_results = []  # 每折的完整指标
        fold_model_paths = []

        for k in range(2):
            logging.info('\n' + '─' * 60)
            logging.info('  组合 {}, 第 {}/2 折'.format(combo_idx + 1, k + 1))
            logging.info('─' * 60)

            fold_folder = os.path.join(combo_folder, 'fold_{}'.format(k))
            if not os.path.exists(fold_folder):
                os.makedirs(fold_folder)

            args_copy.ith_kfold = k
            args_copy.model = os.path.join(fold_folder, 'model.pth')
            args_copy.rrl_file = os.path.join(fold_folder, 'rrl.txt')
            args_copy.test_res = os.path.join(fold_folder, 'test_res.txt')
            args_copy.log = os.path.join(fold_folder, 'log.txt')
            args_copy.folder_path = fold_folder

            # 训练当前折
            train_main(args_copy)

            # 在当前折验证集上评估
            rrl = load_model(args_copy.model, args_copy.device_ids[0],
                             log_file=args_copy.test_res, distributed=False)

            db_enc, X, y, y_labels = _load_and_encode(args_copy.data_set)
            skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
            _, val_idx = list(skf.split(X, y_labels))[k]

            val_set = TensorDataset(torch.tensor(X[val_idx].astype(np.float32)),
                                    torch.tensor(y[val_idx].astype(np.float32)))
            val_loader = DataLoader(val_set, batch_size=args_copy.batch_size, shuffle=False)

            y_true_oh, y_pred_raw, y_true_label, y_pred_label = _predict(
                rrl, val_loader, args_copy.device_ids[0])
            fold_metric = compute_full_metrics(y_true_oh, y_pred_raw, y_true_label, y_pred_label)

            logging.info('  Fold {} 指标: Acc={:.4f}, F1={:.4f}, AUC={}'.format(
                k + 1, fold_metric['Accuracy'], fold_metric['F1'],
                '{:.4f}'.format(fold_metric['AUC']) if fold_metric['AUC'] is not None else 'N/A'))

            fold_results.append(fold_metric)
            fold_model_paths.append(args_copy.model)

        # 汇总 10 折结果
        metric_names = ['Accuracy', 'F1', 'AUC', 'Precision', 'Recall',
                        'Sensitivity', 'Specificity', 'PPV', 'NPV']
        cv_summary = {}
        for m in metric_names:
            vals = [fr[m] for fr in fold_results if fr[m] is not None]
            if vals:
                cv_summary[m + '_mean'] = np.mean(vals)
                cv_summary[m + '_std'] = np.std(vals)
            else:
                cv_summary[m + '_mean'] = None
                cv_summary[m + '_std'] = None

        logging.info('\n  组合 {} — 2折 CV 汇总 (均值±标准差):'.format(combo_idx + 1))
        for m in metric_names:
            mean_val = cv_summary[m + '_mean']
            std_val = cv_summary[m + '_std']
            if mean_val is not None:
                logging.info('    {:15s}: {:.4f} ± {:.4f}'.format(m, mean_val, std_val))

        cv_f1_mean = cv_summary.get('F1_mean', 0.0) or 0.0

        # 选当前组合中 F1 最高的折作为该组合的代表模型
        f1_scores = [fr['F1'] for fr in fold_results]
        best_fold_idx = int(np.argmax(f1_scores))

        combo_result = {
            'combo_idx': combo_idx,
            'params': params,
            'cv_summary': cv_summary,
            'cv_f1_mean': cv_f1_mean,
            'fold_results': fold_results,
            'best_fold_model': fold_model_paths[best_fold_idx],
        }
        all_results.append(combo_result)

        # 更新全局最优
        if cv_f1_mean > best_cv_score:
            best_cv_score = cv_f1_mean
            best_params = params
            best_model_path = fold_model_paths[best_fold_idx]

        logging.info('  当前最优 CV F1 均值: {:.4f}'.format(best_cv_score))

    # 恢复 folder_path
    args.folder_path = root_folder

    logging.info('\n' + '=' * 80)
    logging.info('网格搜索完成!')
    logging.info('最优超参数: {}'.format(best_params))
    logging.info('最优交叉验证 F1 均值: {:.4f}'.format(best_cv_score))
    logging.info('最优模型路径: {}'.format(best_model_path))
    logging.info('=' * 80)

    return best_params, best_cv_score, best_model_path, all_results


# ════════════════════════════════════════════════════════════════════════════
# 第二阶段：在新的测试集上进行最终评估
# ════════════════════════════════════════════════════════════════════════════

def evaluate_on_test_set(args, model_path, test_dataset):
    """在提前准备的新 data 文件（测试集）上评估最优模型。

    参数
    ----
    args         : 命令行参数
    model_path   : 最优模型检查点路径
    test_dataset : 测试集名称（对应 dataset/ 下的 .data 和 .info 文件）

    返回
    ------
    results : 完整指标字典
    """
    rrl = load_model(model_path, args.device_ids[0], log_file=args.test_res, distributed=False)

    # 使用训练数据的编码器
    db_enc, _, _, _ = _load_and_encode(args.data_set)
    test_loader = get_test_data_loader(test_dataset, db_enc, args.batch_size)

    # 在测试集上预测
    y_true_oh, y_pred_raw, y_true_label, y_pred_label = _predict(
        rrl, test_loader, args.device_ids[0])

    results = compute_full_metrics(y_true_oh, y_pred_raw, y_true_label, y_pred_label)

    # 为结果 key 加上 Test_ 前缀
    test_results = {}
    for k, v in results.items():
        test_results['Test_' + k] = v

    logging.info('\n' + '=' * 80)
    logging.info('第二阶段: 在测试集 ({}) 上的最终评估结果'.format(test_dataset))
    logging.info('=' * 80)
    for k, v in test_results.items():
        if isinstance(v, float):
            logging.info('  {:30s}: {:.4f}'.format(k, v))
        else:
            logging.info('  {:30s}: {}'.format(k, v))

    return test_results


# ════════════════════════════════════════════════════════════════════════════
# 主流程
# ════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='RRL 2-Fold CV Grid Search + Test Set Evaluation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--data_set', type=str, default='tic-tac-toe',
                        help='训练数据集名称（dataset/ 目录下的文件前缀）')
    parser.add_argument('-td', '--test_data_set', type=str, default=None,
                        help='测试数据集名称（dataset/ 目录下的文件前缀）。若不指定则跳过第二阶段。')
    parser.add_argument('-i', '--device_ids', type=str, default='0',
                        help='GPU 设备 ID，多个用 @ 分隔，如 0@1')
    parser.add_argument('-nr', '--nr', default=0, type=int, help='节点排名')
    parser.add_argument('-e', '--epoch', type=int, default=41, help='训练轮次')
    parser.add_argument('-bs', '--batch_size', type=int, default=64, help='批大小')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.01, help='初始学习率')
    parser.add_argument('-lrdr', '--lr_decay_rate', type=float, default=0.75, help='学习率衰减率')
    parser.add_argument('-lrde', '--lr_decay_epoch', type=int, default=10, help='学习率衰减轮次')
    parser.add_argument('-wd', '--weight_decay', type=float, default=0.0, help='权重衰减(L2正则)')
    parser.add_argument('-ki', '--ith_kfold', type=int, default=0, help='第 i 折交叉验证, 0 <= ki < 5')
    parser.add_argument('-rc', '--round_count', type=int, default=0, help='实验轮次计数')
    parser.add_argument('-ma', '--master_address', type=str, default='127.0.0.1',
                        help='主节点地址')
    parser.add_argument('-mp', '--master_port', type=str, default='0',
                        help='主节点端口')
    parser.add_argument('-li', '--log_iter', type=int, default=500,
                        help='每隔多少迭代记录一次日志')
    parser.add_argument('--nlaf', action='store_true', help='使用 NLAF 激活函数')
    parser.add_argument('--alpha', type=float, default=0.999, help='NLAF alpha')
    parser.add_argument('--beta', type=int, default=8, help='NLAF beta')
    parser.add_argument('--gamma', type=int, default=1, help='NLAF gamma')
    parser.add_argument('--temp', type=float, default=1.0, help='温度系数')
    parser.add_argument('--use_not', action='store_true', help='使用 NOT 算子')
    parser.add_argument('--save_best', action='store_true',
                        help='在验证集上保存最优模型')
    parser.add_argument('--skip', action='store_true', help='使用跳连接')
    parser.add_argument('--estimated_grad', action='store_true', help='使用估计梯度')
    parser.add_argument('--weighted', action='store_true', help='使用加权损失')
    parser.add_argument('--print_rule', action='store_true',
                        help='打印规则')
    parser.add_argument('-s', '--structure', type=str, default='5@64',
                        help='二值化层和逻辑层结构，如 10@64, 10@64@32@16')
    parser.add_argument('-o', '--output_dir', type=str, default='log_folder/grid_search',
                        help='网格搜索输出目录')

    exp_args = parser.parse_args()

    # 设置设备
    exp_args.device_ids = list(map(int, exp_args.device_ids.strip().split('@')))
    exp_args.gpus = len(exp_args.device_ids)
    exp_args.nodes = 1
    exp_args.world_size = exp_args.gpus * exp_args.nodes

    # 设置输出目录
    exp_args.folder_path = os.path.join(exp_args.output_dir, exp_args.data_set)
    if not os.path.exists(exp_args.folder_path):
        os.makedirs(exp_args.folder_path)
    exp_args.model = os.path.join(exp_args.folder_path, 'model.pth')
    exp_args.rrl_file = os.path.join(exp_args.folder_path, 'rrl.txt')
    exp_args.test_res = os.path.join(exp_args.folder_path, 'test_res.txt')
    exp_args.log = os.path.join(exp_args.folder_path, 'log.txt')
    exp_args.round_count = 0

    # 设置日志
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - [%(levelname)s] - %(message)s')

    # ──────────────────────────────────────────────────────────────────────
    # 第一阶段：10折交叉验证网格搜索
    # ──────────────────────────────────────────────────────────────────────
    print('=' * 80)
    print('第一阶段: 10折交叉验证网格搜索 (random_state=42)')
    print('=' * 80)

    best_params, best_cv_score, best_model_path, all_results = grid_search_cv(exp_args)

    print('\n最优超参数: {}'.format(best_params))
    print('交叉验证最佳 F1 均值: {:.4f}'.format(best_cv_score))
    print('最优模型路径: {}'.format(best_model_path))

    # 保存 CV 汇总结果到 CSV
    cv_csv_path = os.path.join(exp_args.folder_path, 'grid_search_cv_results.csv')
    metric_names = ['Accuracy', 'F1', 'AUC', 'Precision', 'Recall',
                    'Sensitivity', 'Specificity', 'PPV', 'NPV']
    fieldnames = ['combo_idx', 'cv_f1_mean'] + list(param_grid.keys())
    for m in metric_names:
        fieldnames += [m + '_mean', m + '_std']

    with open(cv_csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_results:
            row = {'combo_idx': r['combo_idx'], 'cv_f1_mean': r['cv_f1_mean']}
            row.update(r['params'])
            row.update(r['cv_summary'])
            writer.writerow(row)
    print('\nCV 网格搜索结果已保存至: {}'.format(cv_csv_path))

    # ──────────────────────────────────────────────────────────────────────
    # 第二阶段：在新测试集上最终评估
    # ──────────────────────────────────────────────────────────────────────
    if exp_args.test_data_set is not None:
        print('\n' + '=' * 80)
        print('第二阶段: 在测试集 ({}) 上进行最终评估'.format(exp_args.test_data_set))
        print('=' * 80)

        test_results = evaluate_on_test_set(exp_args, best_model_path, exp_args.test_data_set)

        print('\n最终测试集结果:')
        for key, val in test_results.items():
            if isinstance(val, float):
                print('  {:30s}: {:.4f}'.format(key, val))
            else:
                print('  {:30s}: {}'.format(key, val))

        # 保存测试结果到 CSV
        test_csv_path = os.path.join(exp_args.folder_path, 'final_test_results.csv')
        with open(test_csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=test_results.keys())
            writer.writeheader()
            writer.writerow(test_results)
        print('\n测试结果已保存至: {}'.format(test_csv_path))
    else:
        print('\n未指定测试数据集 (-td)，跳过第二阶段。')

    print('=' * 80)
