# 第一阶段：数据集划分 (Hold-out Split)
# 将原始数据按 80% : 20% 的比例进行分层采样 (Stratified Split)。
# 80% (训练池)：用于后续的交叉验证和最终模型训练。
# 20% (独立测试集/Hold-out Set)：完全隔离，在交叉验证过程中绝对不可见，仅用于最后评估最终模型的泛化性能。
# 第二阶段：5折交叉验证 (5-Fold CV)
# 仅在 80% 训练池 内部进行 5折 Stratified K-Fold 划分。
# 每一折训练一个模型，并在该折的验证集上评估。
# 目的是调整超参数、验证模型稳定性，并计算 CV 平均指标（Accuracy, F1, AUC）。
# 第三阶段：在用测试集验证模型性能并评估
# 使用5折交叉检验后得到的最优RRL模型在 20% 独立测试集 上进行最终评估。
# 输出的结果被视为论文或报告中的“主结果”，代表模型在未见数据上的真实表现。

import os
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
from sklearn import metrics
from collections import defaultdict

from rrl.utils import read_csv, DBEncoder
from rrl.models import RRL

DATA_DIR = './dataset'


# ==============================================================================
# Phase 1: Hold-out Split — 80% training pool / 20% independent test set
# Phase 2: 5-Fold Stratified CV on 80% training pool
# Phase 3: Evaluate best CV model on 20% hold-out test set
# ==============================================================================


def _load_and_encode(dataset):
    """Load dataset and return encoded arrays together with the encoder."""
    data_path = os.path.join(DATA_DIR, dataset + '.data')
    info_path = os.path.join(DATA_DIR, dataset + '.info')
    X_df, y_df, f_df, label_pos = read_csv(data_path, info_path, shuffle=True)

    db_enc = DBEncoder(f_df, discrete=False)
    db_enc.fit(X_df, y_df)

    X, y = db_enc.transform(X_df, y_df, normalized=True, keep_stat=True)
    # y_labels: 1-D integer class labels for stratification
    y_labels = np.argmax(y, axis=1) if y.ndim == 2 else y.astype(int)
    return db_enc, X, y, y_labels


def _make_loaders(X_train, y_train, X_val, y_val, batch_size,
                  world_size=1, rank=0, pin_memory=False, distributed=True, save_best=True):
    """Build DataLoaders from numpy arrays."""
    train_set = TensorDataset(torch.tensor(X_train.astype(np.float32)),
                              torch.tensor(y_train.astype(np.float32)))
    val_set = TensorDataset(torch.tensor(X_val.astype(np.float32)),
                            torch.tensor(y_val.astype(np.float32)))

    if save_best:
        # carve out 5 % of the training fold as an early-stop validation split
        train_len = int(len(train_set) * 0.95)
        train_set, es_val_set = random_split(train_set, [train_len, len(train_set) - train_len])
    else:
        es_val_set = val_set  # fall back to fold validation set

    if distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(
            train_set, num_replicas=world_size, rank=rank)
    else:
        sampler = None

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=(sampler is None),
                              pin_memory=pin_memory, sampler=sampler)
    es_val_loader = DataLoader(es_val_set, batch_size=batch_size, shuffle=False, pin_memory=pin_memory)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, pin_memory=pin_memory)
    return train_loader, es_val_loader, val_loader


def get_holdout_split(dataset, test_size=0.2, random_state=42):
    """Phase 1 — Stratified 80/20 hold-out split.

    Returns
    -------
    db_enc, X_pool, y_pool, y_pool_labels, X_test, y_test, y_test_labels
    """
    db_enc, X, y, y_labels = _load_and_encode(dataset)

    pool_idx, test_idx = train_test_split(
        np.arange(len(X)), test_size=test_size,
        stratify=y_labels, random_state=random_state, shuffle=True)

    return (db_enc,
            X[pool_idx], y[pool_idx], y_labels[pool_idx],
            X[test_idx], y[test_idx], y_labels[test_idx])


# ---------- legacy wrapper kept for backward-compat (single fold) ----------
def get_data_loader(dataset, world_size, rank, batch_size, k=0,
                    pin_memory=False, save_best=True):
    """Legacy interface — wraps the new hold-out + CV logic."""
    (db_enc, X_pool, y_pool, y_pool_labels,
     X_test, y_test, _) = get_holdout_split(dataset)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    train_idx, val_idx = list(skf.split(X_pool, y_pool_labels))[k]

    train_loader, es_val_loader, val_loader = _make_loaders(
        X_pool[train_idx], y_pool[train_idx],
        X_pool[val_idx], y_pool[val_idx],
        batch_size, world_size, rank, pin_memory,
        distributed=True, save_best=save_best)

    test_set = TensorDataset(torch.tensor(X_test.astype(np.float32)),
                             torch.tensor(y_test.astype(np.float32)))
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                             pin_memory=pin_memory)

    return db_enc, train_loader, es_val_loader, val_loader, test_loader


# =====================  Phase 2: 5-Fold Stratified CV  =======================

def train_model(gpu, args):
    """Train a single model for one CV fold (called by mp.spawn)."""
    rank = args.nr * args.gpus + gpu
    dist.init_process_group(backend='nccl', init_method='env://',
                            world_size=args.world_size, rank=rank)
    torch.manual_seed(42)
    device_id = args.device_ids[gpu]
    torch.cuda.set_device(device_id)

    is_rank0 = (gpu == 0)
    writer = SummaryWriter(args.folder_path) if is_rank0 else None

    dataset = args.data_set
    db_enc, train_loader, es_val_loader, _, _ = get_data_loader(
        dataset, args.world_size, rank, args.batch_size,
        k=args.ith_kfold, pin_memory=True, save_best=args.save_best)

    X_fname = db_enc.X_fname
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
        valid_loader=es_val_loader,
        lr=args.learning_rate,
        epoch=args.epoch,
        lr_decay_rate=args.lr_decay_rate,
        lr_decay_epoch=args.lr_decay_epoch,
        weight_decay=args.weight_decay,
        log_iter=args.log_iter)


def train_fold(args, fold_k):
    """Train model for a single fold and return the saved model path."""
    original_kfold = args.ith_kfold

    # Ensure 'model' and 'log' attributes exist (they are set in args.py but
    # may be absent if the arg-parsing script on this machine differs).
    if not hasattr(args, 'model'):
        args.model = os.path.join(args.folder_path, 'model.pth')
    if not hasattr(args, 'log'):
        args.log = os.path.join(args.folder_path, 'log.txt')

    original_model = args.model
    original_log = args.log

    fold_model_path = original_model.replace('.pth', f'_fold{fold_k}.pth')
    fold_log_path = original_log.replace('.txt', f'_fold{fold_k}.txt')

    args.ith_kfold = fold_k
    args.model = fold_model_path
    args.log = fold_log_path

    os.environ['MASTER_ADDR'] = args.master_address
    os.environ['MASTER_PORT'] = args.master_port
    mp.spawn(train_model, nprocs=args.gpus, args=(args,))

    # restore
    args.ith_kfold = original_kfold
    args.model = original_model
    args.log = original_log

    return fold_model_path


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
        # remove 'module.' prefix
        stat_dict[key[7:]] = stat_dict.pop(key)
    rrl.net.load_state_dict(checkpoint['model_state_dict'])
    return rrl


@torch.no_grad()
def _evaluate_loader(rrl, loader, device_id, set_name='Validation'):
    """Evaluate model on a DataLoader and return (accuracy, f1_macro, auc)."""
    y_list, y_pred_list = [], []
    for X, y in loader:
        X = X.cuda(device_id, non_blocking=True)
        output = rrl.net.forward(X)
        y_list.append(y)
        y_pred_list.append(output)

    y_true = torch.cat(y_list, dim=0).cpu().numpy().astype(int)
    y_pred = torch.cat(y_pred_list).cpu().numpy()

    y_true_arg = np.argmax(y_true, axis=1)
    y_pred_arg = np.argmax(y_pred, axis=1)

    accuracy = metrics.accuracy_score(y_true_arg, y_pred_arg)
    f1 = metrics.f1_score(y_true_arg, y_pred_arg, average='macro')

    # AUC — handle binary and multi-class
    n_classes = y_pred.shape[1]
    try:
        if n_classes == 2:
            # use probability of positive class
            from scipy.special import softmax as sp_softmax
            y_prob = sp_softmax(y_pred, axis=1)[:, 1]
            auc = metrics.roc_auc_score(y_true_arg, y_prob)
        else:
            from scipy.special import softmax as sp_softmax
            y_prob = sp_softmax(y_pred, axis=1)
            auc = metrics.roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
    except ValueError:
        auc = float('nan')

    logging.info('-' * 60)
    logging.info('On {} Set:\n\tAccuracy : {:.4f}\n\tF1 (macro): {:.4f}\n\tAUC      : {:.4f}'.format(
        set_name, accuracy, f1, auc))
    logging.info('Confusion Matrix:\n{}'.format(metrics.confusion_matrix(y_true_arg, y_pred_arg)))
    logging.info('Classification Report:\n{}'.format(metrics.classification_report(y_true_arg, y_pred_arg)))
    logging.info('-' * 60)

    return accuracy, f1, auc


def cross_validate(args):
    """Phase 2 — Run 5-fold Stratified CV and return per-fold metrics + best fold."""
    n_folds = 5
    fold_metrics = []  # list of (accuracy, f1, auc, model_path)

    for k in range(n_folds):
        logging.info('=' * 70)
        logging.info('  Fold {}/{}'.format(k + 1, n_folds))
        logging.info('=' * 70)

        fold_model_path = train_fold(args, k)

        # evaluate this fold's model on its own validation split
        rrl = load_model(fold_model_path, args.device_ids[0], distributed=False)

        (db_enc, X_pool, y_pool, y_pool_labels,
         _, _, _) = get_holdout_split(args.data_set)

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        _, val_idx = list(skf.split(X_pool, y_pool_labels))[k]

        val_set = TensorDataset(torch.tensor(X_pool[val_idx].astype(np.float32)),
                                torch.tensor(y_pool[val_idx].astype(np.float32)))
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

        acc, f1, auc = _evaluate_loader(rrl, val_loader, args.device_ids[0],
                                        set_name=f'CV-Fold-{k+1} Validation')
        fold_metrics.append((acc, f1, auc, fold_model_path))

    # summarise CV results
    accs = [m[0] for m in fold_metrics]
    f1s  = [m[1] for m in fold_metrics]
    aucs = [m[2] for m in fold_metrics]

    logging.info('=' * 70)
    logging.info('5-Fold CV Summary (mean ± std):')
    logging.info('  Accuracy : {:.4f} ± {:.4f}'.format(np.mean(accs), np.std(accs)))
    logging.info('  F1 (macro): {:.4f} ± {:.4f}'.format(np.mean(f1s), np.std(f1s)))
    logging.info('  AUC      : {:.4f} ± {:.4f}'.format(np.nanmean(aucs), np.nanstd(aucs)))
    logging.info('=' * 70)

    # select the fold with the highest validation F1 as the best model
    best_fold = int(np.argmax(f1s))
    best_model_path = fold_metrics[best_fold][3]
    logging.info('Best fold: {} (F1={:.4f})  ->  {}'.format(
        best_fold + 1, f1s[best_fold], best_model_path))

    return best_model_path, fold_metrics


# =====================  Phase 3: Final evaluation on hold-out set  ============

def test_model(args, model_path=None):
    """Evaluate a trained model on the 20 % independent hold-out test set."""
    # Ensure optional attributes exist
    if not hasattr(args, 'model'):
        args.model = os.path.join(args.folder_path, 'model.pth')
    if not hasattr(args, 'test_res'):
        args.test_res = os.path.join(args.folder_path, 'test_res.txt')
    if not hasattr(args, 'rrl_file'):
        args.rrl_file = os.path.join(args.folder_path, 'rrl.txt')
    if not hasattr(args, 'print_rule'):
        args.print_rule = False

    if model_path is None:
        model_path = args.model

    rrl = load_model(model_path, args.device_ids[0], log_file=args.test_res, distributed=False)

    (db_enc, X_pool, y_pool, _,
     X_test, y_test, _) = get_holdout_split(args.data_set)

    test_set = TensorDataset(torch.tensor(X_test.astype(np.float32)),
                             torch.tensor(y_test.astype(np.float32)))
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    logging.info('=' * 70)
    logging.info('Phase 3: Final Evaluation on 20%% Independent Hold-out Test Set')
    logging.info('  Model: {}'.format(model_path))
    logging.info('=' * 70)

    acc, f1, auc = _evaluate_loader(rrl, test_loader, args.device_ids[0], set_name='Hold-out Test')

    # --- rule printing & edge counting (same as original) ---
    pool_set = TensorDataset(torch.tensor(X_pool.astype(np.float32)),
                             torch.tensor(y_pool.astype(np.float32)))
    pool_loader = DataLoader(pool_set, batch_size=args.batch_size, shuffle=False)

    if args.print_rule:
        with open(args.rrl_file, 'w') as rrl_file:
            rule2weights = rrl.rule_print(db_enc.X_fname, db_enc.y_fname, pool_loader,
                                          file=rrl_file, mean=db_enc.mean, std=db_enc.std)
    else:
        rule2weights = rrl.rule_print(db_enc.X_fname, db_enc.y_fname, pool_loader,
                                      mean=db_enc.mean, std=db_enc.std, display=False)

    metric = 'Log(#Edges)'
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
        logging.info('\n\t{} of RRL  Model: {}'.format(metric, np.log(edge_cnt)))

    return acc, f1, auc


def train_main(args):
    """Legacy single-fold training entry point."""
    os.environ['MASTER_ADDR'] = args.master_address
    os.environ['MASTER_PORT'] = args.master_port
    mp.spawn(train_model, nprocs=args.gpus, args=(args,))


if __name__ == '__main__':
    from args import rrl_args

    # Set up root logging so that CV summary is visible on stdout
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')

    # Phase 1 + 2: Hold-out split then 5-fold Stratified CV
    best_model_path, fold_metrics = cross_validate(rrl_args)

    # Phase 3: Evaluate the best model on the 20% independent test set
    acc, f1, auc = test_model(rrl_args, model_path=best_model_path)
    logging.info('Final Hold-out Test Results  ->  Accuracy: {:.4f}  F1: {:.4f}  AUC: {:.4f}'.format(acc, f1, auc))
