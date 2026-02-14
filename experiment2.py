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
from sklearn.model_selection import KFold, train_test_split
from collections import defaultdict

from rrl.utils import read_csv, DBEncoder
from rrl.models import RRL

DATA_DIR = './dataset'


def get_data_loader(dataset, world_size, rank, batch_size, k=0, pin_memory=False, save_best=True, use_final_test=False):
    data_path = os.path.join(DATA_DIR, dataset + '.data')
    info_path = os.path.join(DATA_DIR, dataset + '.info')
    X_df, y_df, f_df, label_pos = read_csv(data_path, info_path, shuffle=True)

    db_enc = DBEncoder(f_df, discrete=False)
    db_enc.fit(X_df, y_df)

    X, y = db_enc.transform(X_df, y_df, normalized=True, keep_stat=True)

    if use_final_test:
        # 用于最终测试集评估，不做5折划分
        X_train = X
        y_train = y
        X_test = X
        y_test = y
    else:
        # 先划分80%训练集和20%测试集
        train_val_index, final_test_index = train_test_split(
            range(len(X)), test_size=0.2, random_state=42, shuffle=True
        )

        # 在80%的数据上进行5折交叉验证
        X_train_val = X[train_val_index]
        y_train_val = y[train_val_index]

        kf = KFold(n_splits=5, shuffle=True, random_state=0)
        train_index, test_index = list(kf.split(X_train_val))[k]
        X_train = X_train_val[train_index]
        y_train = y_train_val[train_index]
        X_test = X_train_val[test_index]
        y_test = y_train_val[test_index]

    train_set = TensorDataset(torch.tensor(X_train.astype(np.float32)), torch.tensor(y_train.astype(np.float32)))
    test_set = TensorDataset(torch.tensor(X_test.astype(np.float32)), torch.tensor(y_test.astype(np.float32)))

    train_len = int(len(train_set) * 0.95)
    train_sub, valid_set = random_split(train_set, [train_len, len(train_set) - train_len])

    if save_best:  # use validation set for model selections.
        train_set = train_sub

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_set, num_replicas=world_size, rank=rank)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False, pin_memory=pin_memory,
                              sampler=train_sampler)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, pin_memory=pin_memory)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, pin_memory=pin_memory)

    return db_enc, train_loader, valid_loader, test_loader


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
    db_enc, train_loader, valid_loader, _ = get_data_loader(dataset, args.world_size, rank, args.batch_size,
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
        # remove 'module.' prefix
        stat_dict[key[7:]] = stat_dict.pop(key)
    rrl.net.load_state_dict(checkpoint['model_state_dict'])
    return rrl


def test_model(args, use_final_test=False):
    from sklearn.metrics import roc_auc_score

    rrl = load_model(args.model, args.device_ids[0], log_file=args.test_res, distributed=False)
    dataset = args.data_set

    if use_final_test:
        # 加载20%的最终测试集
        data_path = os.path.join(DATA_DIR, dataset + '.data')
        info_path = os.path.join(DATA_DIR, dataset + '.info')
        X_df, y_df, f_df, label_pos = read_csv(data_path, info_path, shuffle=True)

        db_enc = DBEncoder(f_df, discrete=False)
        db_enc.fit(X_df, y_df)

        X, y = db_enc.transform(X_df, y_df, normalized=True, keep_stat=True)

        # 划分出20%的最终测试集
        _, final_test_index = train_test_split(
            range(len(X)), test_size=0.2, random_state=42, shuffle=True
        )

        X_final_test = X[final_test_index]
        y_final_test = y[final_test_index]

        final_test_set = TensorDataset(
            torch.tensor(X_final_test.astype(np.float32)),
            torch.tensor(y_final_test.astype(np.float32))
        )
        test_loader = DataLoader(final_test_set, batch_size=args.batch_size, shuffle=False, pin_memory=False)

        # 获取train_loader用于规则打印
        train_val_index, _ = train_test_split(
            range(len(X)), test_size=0.2, random_state=42, shuffle=True
        )
        X_train_val = X[train_val_index]
        y_train_val = y[train_val_index]
        train_set = TensorDataset(
            torch.tensor(X_train_val.astype(np.float32)),
            torch.tensor(y_train_val.astype(np.float32))
        )
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False, pin_memory=False)
    else:
        db_enc, train_loader, _, test_loader = get_data_loader(dataset, 4, 0, args.batch_size, args.ith_kfold,
                                                               save_best=False)

    # 测试模型并获取预测结果
    accuracy, f1_score = rrl.test(test_loader=test_loader, set_name='Final Test' if use_final_test else 'Test')

    # 计算AUC
    y_true_list = []
    y_pred_list = []
    for X, y in test_loader:
        X = X.cuda(args.device_ids[0], non_blocking=True)
        y_true_list.append(y)
        output = rrl.net.forward(X)
        y_pred_list.append(output.cpu().detach())

    y_true = torch.cat(y_true_list, dim=0).numpy()
    y_pred = torch.cat(y_pred_list, dim=0).numpy()

    # 计算AUC (支持多分类)
    try:
        if y_true.shape[1] == 2:  # 二分类
            auc_score = roc_auc_score(y_true[:, 1], y_pred[:, 1])
        else:  # 多分类
            auc_score = roc_auc_score(y_true, y_pred, multi_class='ovr', average='macro')
        logging.info('\n\tAUC of RRL Model: {}'.format(auc_score))
    except Exception as e:
        logging.warning('\n\tFailed to compute AUC: {}'.format(str(e)))
        auc_score = None

    if args.print_rule:
        with open(args.rrl_file, 'w') as rrl_file:
            rule2weights = rrl.rule_print(db_enc.X_fname, db_enc.y_fname, train_loader, file=rrl_file, mean=db_enc.mean,
                                          std=db_enc.std)
    else:
        rule2weights = rrl.rule_print(db_enc.X_fname, db_enc.y_fname, train_loader, mean=db_enc.mean, std=db_enc.std,
                                      display=False)

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
    logging.info('\n\t{} of RRL  Model: {}'.format(metric, np.log(edge_cnt)))

    return accuracy, f1_score, auc_score


def train_main(args):
    os.environ['MASTER_ADDR'] = args.master_address
    os.environ['MASTER_PORT'] = args.master_port
    mp.spawn(train_model, nprocs=args.gpus, args=(args,))


if __name__ == '__main__':
    from args import rrl_args

    # 存储5折交叉验证的结果
    cv_results = []

    print("=" * 80)
    print("开始5折交叉验证 (在80%的训练数据上)")
    print("=" * 80)

    # 5折交叉验证
    for k in range(5):
        print(f"\n{'='*80}")
        print(f"第 {k+1}/5 折交叉验证")
        print(f"{'='*80}")

        # 更新参数
        rrl_args.ith_kfold = k

        # 创建每折的文件夹
        test_file = rrl_args.folder_path
        folder_path_ = os.path.join(test_file, str(k))
        if not os.path.exists(folder_path_):
            os.makedirs(folder_path_)

        # 更新文件路径
        rrl_args.model = os.path.join(folder_path_, 'model.pth')
        rrl_args.rrl_file = os.path.join(folder_path_, 'rrl.txt')
        rrl_args.test_res = os.path.join(folder_path_, 'test_res.txt')
        rrl_args.log = os.path.join(folder_path_, 'log.txt')

        # 训练模型
        train_main(rrl_args)

        # 测试模型（在验证集上）
        accuracy, f1_score, auc_score = test_model(rrl_args, use_final_test=False)
        cv_results.append({
            'fold': k,
            'accuracy': accuracy,
            'f1_score': f1_score,
            'auc': auc_score
        })

        print(f"\n第 {k+1} 折结果:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  F1 Score: {f1_score:.4f}")
        if auc_score is not None:
            print(f"  AUC: {auc_score:.4f}")

    # 打印5折交叉验证结果汇总
    print("\n" + "=" * 80)
    print("5折交叉验证结果汇总")
    print("=" * 80)

    for result in cv_results:
        print(f"第 {result['fold']+1} 折: Accuracy={result['accuracy']:.4f}, F1={result['f1_score']:.4f}", end="")
        if result['auc'] is not None:
            print(f", AUC={result['auc']:.4f}")
        else:
            print()

    # 计算平均值
    avg_accuracy = np.mean([r['accuracy'] for r in cv_results])
    avg_f1 = np.mean([r['f1_score'] for r in cv_results])
    aucs = [r['auc'] for r in cv_results if r['auc'] is not None]
    avg_auc = np.mean(aucs) if aucs else None

    print("\n平均结果:")
    print(f"  Average Accuracy: {avg_accuracy:.4f} ± {np.std([r['accuracy'] for r in cv_results]):.4f}")
    print(f"  Average F1 Score: {avg_f1:.4f} ± {np.std([r['f1_score'] for r in cv_results]):.4f}")
    if avg_auc is not None:
        print(f"  Average AUC: {avg_auc:.4f} ± {np.std(aucs):.4f}")

    # 使用最后一折的模型在20%的独立测试集上进行最终评估
    print("\n" + "=" * 80)
    print("在20%独立测试集上进行最终评估")
    print("=" * 80)

    # 使用最后一折的模型
    final_accuracy, final_f1, final_auc = test_model(rrl_args, use_final_test=True)

    print("\n最终测试集结果:")
    print(f"  Accuracy: {final_accuracy:.4f}")
    print(f"  F1 Score: {final_f1:.4f}")
    if final_auc is not None:
        print(f"  AUC: {final_auc:.4f}")
    print("=" * 80)
