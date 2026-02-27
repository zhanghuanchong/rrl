# 第一阶段:针对数据集进行RRL模型的5折交叉验证调优超参数，用5折交叉验证进行网格搜索，根据args.py文件生成param_grid, random state=42
# 每一折训练一个模型，在该折验证集上评估 Accuracy、F1、AUC(含95%Bootstrap 置信区间)、Precision、Recall、Sensitivity、Specificity、PPV、NPV。
# 汇总5折结果(均值土标准差)，输出最优超参数，交叉验证最佳得分，获取最优模型。
# 第二个阶段:在提前准备的新的data文件(即测试集)上，进行最终评估。
# 输出完整指标:Accuracy、AUC(含95%Bootstrap 置信区间)、Precision、Recall、F1、Sensitivity、Specificity、PPV、NPV.结果保存为 CSV，可直接用于论文/报告。
# 适配1800样本×12列医疗数据的RRL模型网格搜索参数配置
param_grid = {
    # ===================== 基础训练参数 =====================
    # 学习率：小样本适配，论文推荐5e-3~5e-5，选中间区间
    "learning_rate": [0.002, 0.0005, 0.0002],
    # 学习率衰减率：论文固定0.75，保留1个备选验证
    "lr_decay_rate": [0.75, 0.8],
    # 学习率衰减轮次：小样本训练轮次少，适配400epoch的1/4/1/5
    "lr_decay_epoch": [80, 100],
    # L2正则（weight_decay）：医疗数据需规则稀疏性，选1e-3~1e-5
    "weight_decay": [1e-3, 1e-4, 1e-5],
    # 批大小：按GPU数均分后，单GPU取16/32（小样本避免batch过小）
    "batch_size": [64, 128],  # 若用1个GPU，实际为64/128；2个GPU则为32/64
    # 训练轮次：论文推荐400epoch，小样本可适当减少
    "epoch": [300, 400],

    # ===================== 模型结构参数 =====================
    # 二值化层+逻辑层结构：适配12列特征，分箱数(k)选5/10，逻辑层节点从64递减
    "structure": [
        "5@64",       # 二值化层k=5，1层逻辑层64节点（最简结构，优先验证）
        "5@64@32",    # 二值化层k=5，2层逻辑层64→32
        "10@64",      # 二值化层k=10，1层逻辑层64节点
        "10@64@32"    # 二值化层k=10，2层逻辑层64→32
    ],
    # 温度系数：控制规则稀疏性，医疗场景优先稀疏规则
    "temp": [1.0, 0.1, 0.01],
    # NOT算子：增强能力但增加复杂度，医疗数据可选开启/关闭
    "use_not": [False, True],
    # 跳连接：逻辑层>2时生效，配合structure参数验证
    "skip": [False, True],

    # ===================== NLAF激活函数参数 =====================
    # 论文推荐的3组核心组合，直接复用
    "alpha": [0.999, 0.9],
    "beta": [3, 8],
    "gamma": [1, 3],
    # 是否使用NLAF：减少GPU内存，适配小样本训练
    "nlaf": [True, False],

    # ===================== 其他优化参数 =====================
    # 估计梯度：加速训练，小样本可选
    "estimated_grad": [False, True],
    # 加权损失：医疗数据多类别不平衡，优先开启
    "weighted": [True, False]
}

# ------------------------------
# 补充：若用Optuna等自动调参工具，可将参数范围改为如下格式（更高效）
# ------------------------------
def suggest_params(trial):
    # 基础训练参数
    lr = trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True)
    lr_decay_rate = trial.suggest_float("lr_decay_rate", 0.7, 0.8, step=0.05)
    lr_decay_epoch = trial.suggest_int("lr_decay_epoch", 50, 100, step=10)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [64, 128])
    epoch = trial.suggest_categorical("epoch", [300, 400])

    # 模型结构参数
    structure = trial.suggest_categorical("structure", ["5@64", "5@64@32", "10@64", "10@64@32", "2@256@256"])
    temp = trial.suggest_categorical("temp", [0.01, 0.1, 1.0])
    use_not = trial.suggest_categorical("use_not", [False, True])
    skip = trial.suggest_categorical("skip", [False, True])

    # NLAF参数
    alpha = trial.suggest_categorical("alpha", [0.999, 0.9])
    beta = trial.suggest_categorical("beta", [3, 8])
    gamma = trial.suggest_categorical("gamma", [1, 3])
    nlaf = trial.suggest_categorical("nlaf", [True, False])

    # 其他优化参数
    estimated_grad = trial.suggest_categorical("estimated_grad", [False, True])
    weighted = trial.suggest_categorical("weighted", [True, False])

    return {
        "learning_rate": lr, "lr_decay_rate": lr_decay_rate, "lr_decay_epoch": lr_decay_epoch,
        "weight_decay": weight_decay, "batch_size": batch_size, "epoch": epoch,
        "structure": structure, "temp": temp, "use_not": use_not, "skip": skip,
        "alpha": alpha, "beta": beta, "gamma": gamma, "nlaf": nlaf,
        "estimated_grad": estimated_grad, "weighted": weighted
    }
