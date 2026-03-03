#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_test_only.py
================
仅在测试集上进行评估的独立脚本。
跳过第一阶段的网格搜索训练，直接加载已训练好的模型，在指定测试集上评估。

用法示例:

  # 方式一：指定包含 combo_*/fold_*/model.pth 的网格搜索根目录，自动寻找最优组合
  python run_test_only.py \
      -d FREET_MultiP13Train \
      -td FREET_MultiP13test \
      --search_dir log_folder/grid_search/FREET_MultiP13Train

  # 方式二：直接指定模型路径（逗号分隔，用于 5 折集成）
  python run_test_only.py \
      -d FREET_MultiP13Train \
      -td FREET_MultiP13test \
      --model_paths "log_folder/grid_search/.../fold_0/model.pth,log_folder/grid_search/.../fold_1/model.pth,..."

  # 方式三：只指定单个模型
  python run_test_only.py \
      -d FREET_MultiP13Train \
      -td FREET_MultiP13test \
      --model_paths "path/to/model.pth"
"""

import os
import csv
import glob
import copy
import shutil
import logging
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from experiment2cv import (
    _load_and_encode,
    load_model,
    get_test_data_loader,
    _predict,
    _print_rules,
    compute_full_metrics,
)

DATA_DIR = './dataset'


def find_best_combo_models(search_dir):
    """在网格搜索输出目录中自动查找所有 combo，返回每个 combo 的模型路径列表。

    如果目录中存在 grid_search_cv_results.csv，则根据 cv_f1_mean 找到最优 combo；
    否则返回第一个包含完整 5 折模型的 combo。
    """
    csv_path = os.path.join(search_dir, 'grid_search_cv_results.csv')

    # 尝试从 CSV 中读取最优 combo
    best_combo_idx = None
    if os.path.exists(csv_path):
        logging.info('找到 CV 结果文件: {}'.format(csv_path))
        best_f1 = -1.0
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                f1_val = float(row.get('cv_f1_mean', 0) or 0)
                combo_idx = int(row.get('combo_idx', -1))
                if f1_val > best_f1:
                    best_f1 = f1_val
                    best_combo_idx = combo_idx
        if best_combo_idx is not None:
            logging.info('根据 CSV，最优 combo_idx={}, CV F1 均值={:.4f}'.format(best_combo_idx, best_f1))

    # 查找 combo 目录
    combo_dirs = sorted(glob.glob(os.path.join(search_dir, 'combo_*')))
    if not combo_dirs:
        raise FileNotFoundError('在 {} 中未找到 combo_* 目录'.format(search_dir))

    if best_combo_idx is not None:
        target_dir = os.path.join(search_dir, 'combo_{}'.format(best_combo_idx))
        if os.path.isdir(target_dir):
            combo_dirs = [target_dir]
        else:
            logging.warning('combo_{} 目录不存在，将扫描所有 combo'.format(best_combo_idx))

    # 在目标 combo 目录中查找所有 fold 模型
    for combo_dir in combo_dirs:
        fold_dirs = sorted(glob.glob(os.path.join(combo_dir, 'fold_*')))
        model_paths = []
        for fd in fold_dirs:
            mp = os.path.join(fd, 'model.pth')
            if os.path.exists(mp):
                model_paths.append(mp)
        if model_paths:
            logging.info('在 {} 中找到 {} 个折模型'.format(combo_dir, len(model_paths)))
            return model_paths

    raise FileNotFoundError('未在任何 combo 中找到 model.pth 文件')


def evaluate_on_test_set_standalone(args, model_paths, test_dataset):
    """独立版本的测试集评估，包含 handle_unknown 修复。

    与 experiment2cv.evaluate_on_test_set 类似，但增加了对未知类别的处理。
    同时保存 model.pth、log.txt、rrl.txt、test_res.txt 到输出目录。
    """
    # 使用训练数据的编码器
    db_enc, X_all, y_all, y_labels_all = _load_and_encode(args.data_set)

    # ═══ 关键修复：处理测试集中出现训练集未见过的类别 ═══
    if hasattr(db_enc, 'feature_enc') and db_enc.feature_enc is not None:
        db_enc.feature_enc.set_params(handle_unknown='ignore')
        logging.info('已设置 feature_enc.handle_unknown = "ignore"')

    # 如果 label_enc 也是 OneHotEncoder，同样处理
    if hasattr(db_enc, 'label_enc') and hasattr(db_enc.label_enc, 'handle_unknown'):
        db_enc.label_enc.set_params(handle_unknown='ignore')
        logging.info('已设置 label_enc.handle_unknown = "ignore"')

    test_loader = get_test_data_loader(test_dataset, db_enc, args.batch_size)

    # 构建训练集 DataLoader（用于规则打印时检测死节点）
    train_set_for_rules = TensorDataset(
        torch.tensor(X_all.astype(np.float32)),
        torch.tensor(y_all.astype(np.float32)))
    train_loader_for_rules = DataLoader(
        train_set_for_rules, batch_size=args.batch_size, shuffle=False)

    # test_res.txt 路径
    test_res_path = os.path.join(args.output_dir, 'test_res.txt')

    # 对每个折模型进行推理，收集原始输出
    all_y_pred_raw = []
    y_true_oh = None
    for i, mp in enumerate(model_paths):
        logging.info('加载模型 [{}/{}]: {}'.format(i + 1, len(model_paths), mp))
        rrl = load_model(mp, args.device_ids[0], log_file=test_res_path, distributed=False)
        y_true_oh_i, y_pred_raw_i, _, _ = _predict(rrl, test_loader, args.device_ids[0])
        all_y_pred_raw.append(y_pred_raw_i)
        if y_true_oh is None:
            y_true_oh = y_true_oh_i

        # ── 保存 model.pth 副本到输出目录 ──
        if len(model_paths) == 1:
            dest_model = os.path.join(args.output_dir, 'model.pth')
        else:
            dest_model = os.path.join(args.output_dir, 'model_fold{}.pth'.format(i))
        shutil.copy2(mp, dest_model)
        logging.info('模型已复制至: {}'.format(dest_model))

        # ── 保存 rrl.txt 规则文件 ──
        if len(model_paths) == 1:
            rrl_file_path = os.path.join(args.output_dir, 'rrl.txt')
        else:
            rrl_file_path = os.path.join(args.output_dir, 'rrl_fold{}.txt'.format(i))
        args_for_rules = argparse.Namespace(
            print_rule=True,
            rrl_file=rrl_file_path,
        )
        _print_rules(rrl, db_enc, train_loader_for_rules, args_for_rules)
        logging.info('规则已保存至: {}'.format(rrl_file_path))

    logging.info('test_res 已保存至: {}'.format(test_res_path))

    # 对所有折模型的原始输出取平均
    y_pred_raw = np.mean(all_y_pred_raw, axis=0)
    y_true_label = np.argmax(y_true_oh, axis=1)
    y_pred_label = np.argmax(y_pred_raw, axis=1)

    results = compute_full_metrics(y_true_oh, y_pred_raw, y_true_label, y_pred_label)

    # 为结果 key 加上 Test_ 前缀
    test_results = {}
    for k, v in results.items():
        test_results['Test_' + k] = v

    return test_results


def main():
    parser = argparse.ArgumentParser(
        description='仅在测试集上评估已训练好的 RRL 模型（跳过训练阶段）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)

    parser.add_argument('-d', '--data_set', type=str, required=True,
                        help='训练数据集名称（用于重建编码器，对应 dataset/ 下的文件前缀）')
    parser.add_argument('-td', '--test_data_set', type=str, required=True,
                        help='测试数据集名称（对应 dataset/ 下的 .data 和 .info 文件前缀）')

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--model_paths', type=str, default=None,
                       help='模型路径，多个用逗号分隔（用于 5 折集成）')
    group.add_argument('--search_dir', type=str, default=None,
                       help='网格搜索输出目录（自动查找最优 combo 下的模型）')

    parser.add_argument('-i', '--device_ids', type=str, default='0',
                        help='GPU 设备 ID，多个用 @ 分隔（如 0@1）。CPU 模式请设为 0')
    parser.add_argument('-bs', '--batch_size', type=int, default=32,
                        help='推理时的批大小')
    parser.add_argument('-o', '--output_dir', type=str, default=None,
                        help='结果输出目录（默认为 search_dir 或当前目录）')

    args = parser.parse_args()

    # 设置设备
    device_ids_str = str(args.device_ids).strip()
    args.device_ids = list(map(int, device_ids_str.split('@')))
    args.gpus = len(args.device_ids)

    # 设置日志
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - [%(levelname)s] - %(message)s')

    # 解析模型路径
    if args.model_paths is not None:
        model_paths = [p.strip() for p in args.model_paths.split(',') if p.strip()]
    else:
        model_paths = find_best_combo_models(args.search_dir)

    # 验证模型文件存在
    for mp in model_paths:
        if not os.path.exists(mp):
            raise FileNotFoundError('模型文件不存在: {}'.format(mp))

    # 输出目录
    if args.output_dir is None:
        if args.search_dir:
            args.output_dir = args.search_dir
        else:
            args.output_dir = os.path.dirname(model_paths[0]) if model_paths else '.'
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # ── 添加文件日志 handler，将日志同时写入 log.txt ──
    log_file_path = os.path.join(args.output_dir, 'log.txt')
    file_handler = logging.FileHandler(log_file_path, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - [%(levelname)s] - %(message)s'))
    logging.getLogger().addHandler(file_handler)
    logging.info('日志将同时保存至: {}'.format(log_file_path))

    # 为 evaluate 函数需要的属性补充默认值
    args.test_res = os.path.join(args.output_dir, 'test_res.txt')

    print('=' * 80)
    print('独立测试集评估')
    print('=' * 80)
    print('训练数据集 (编码器来源): {}'.format(args.data_set))
    print('测试数据集:              {}'.format(args.test_data_set))
    print('模型数量:                {} 个 (集成)'.format(len(model_paths)))
    for i, mp in enumerate(model_paths):
        print('  模型 {}: {}'.format(i, mp))
    print('设备 ID:                 {}'.format(args.device_ids))
    print('批大小:                  {}'.format(args.batch_size))
    print('输出目录:                {}'.format(args.output_dir))
    print('=' * 80)

    # 执行评估
    test_results = evaluate_on_test_set_standalone(args, model_paths, args.test_data_set)

    # 打印结果
    print('\n' + '=' * 80)
    print('测试集 ({}) 最终评估结果'.format(args.test_data_set))
    print('=' * 80)
    for key, val in test_results.items():
        if isinstance(val, float):
            print('  {:30s}: {:.4f}'.format(key, val))
        else:
            print('  {:30s}: {}'.format(key, val))

    # 保存结果到 CSV
    test_csv_path = os.path.join(args.output_dir, 'final_test_results.csv')
    with open(test_csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=test_results.keys())
        writer.writeheader()
        writer.writerow(test_results)
    print('\n测试结果已保存至: {}'.format(test_csv_path))
    print('\n输出文件汇总:')
    print('  日志文件:       {}'.format(log_file_path))
    print('  测试结果 CSV:   {}'.format(test_csv_path))
    print('  test_res.txt:   {}'.format(os.path.join(args.output_dir, 'test_res.txt')))
    for i in range(len(model_paths)):
        if len(model_paths) == 1:
            print('  model.pth:      {}'.format(os.path.join(args.output_dir, 'model.pth')))
            print('  rrl.txt:        {}'.format(os.path.join(args.output_dir, 'rrl.txt')))
        else:
            print('  model_fold{}.pth: {}'.format(i, os.path.join(args.output_dir, 'model_fold{}.pth'.format(i))))
            print('  rrl_fold{}.txt:   {}'.format(i, os.path.join(args.output_dir, 'rrl_fold{}.txt'.format(i))))
    print('=' * 80)
if __name__ == '__main__':
    main()



