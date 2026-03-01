# -- coding: utf-8 --
""" Reproductive Medicine Outcome Prediction Model (Binary/Multi-class) - Simplified Version
Features:
- Removed feature engineering code
- Removed RandomForest SHAP analysis
- Updated metrics to 5-fold cross validation mean values
"""
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from typing import Tuple, Dict, List
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, LabelBinarizer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.metrics import (
    roc_auc_score, roc_curve, f1_score, classification_report, confusion_matrix,
    accuracy_score, precision_score, recall_score, matthews_corrcoef, precision_recall_curve, auc
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier, VotingClassifier
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils.multiclass import type_of_target
import xgboost as xgb
from itertools import cycle
from scipy.stats import bootstrap
from scipy.stats import norm
import shap
import matplotlib.pyplot as plt
import os
import shap
import warnings
warnings.filterwarnings('ignore')


# ---------------------- 全局配置 ----------------------
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.figsize'] = (12, 8)    # 设置默认画布大小
RANDOM_SEED = 42
TEST_SIZE = 0.2
N_JOBS = -1
K_FOLD = 5  # 5折交叉验证
ID_COLUMN = "ID"


def safe_bootstrap_metric(y_true, y_score, metric_func, n_bootstraps=1000):
    """
    终极版：完全隔离sklearn内置Bootstrap,强制保证样本数一致
    """
    # 步骤1：强制标准化数据类型和长度（三重校验）
    # 转为numpy一维数组
    y_true = np.array(y_true).ravel()
    y_score = np.array(y_score).ravel()
    # 过滤缺失值（根源性问题：可能有nan导致样本数统计错误）
    mask = ~(np.isnan(y_true) | np.isnan(y_score))
    y_true = y_true[mask]
    y_score = y_score[mask]
    # 最终强制对齐长度
    min_len = min(len(y_true), len(y_score))
    y_true = y_true[:min_len]
    y_score = y_score[:min_len]
    
    # 步骤2：提前判断是否可计算AUC
    if metric_func == roc_auc_score:
        if len(np.unique(y_true)) < 2:
            return 0.0, (0.0, 0.0)
    
    # 步骤3：自定义Bootstrap（完全不使用sklearn的resample,避免内置逻辑报错）
    metrics = []
    rng = np.random.RandomState(RANDOM_SEED)
    n_samples = len(y_true)
    
    for _ in range(n_bootstraps):
        # 手动生成重采样索引（确保长度绝对一致）
        idx = rng.randint(0, n_samples, size=n_samples)
        y_t = y_true[idx]
        y_s = y_score[idx]
        
        # 跳过单类别
        if metric_func == roc_auc_score and len(np.unique(y_t)) < 2:
            continue
        
        # 强制捕获所有异常,避免中断
        try:
            score = metric_func(y_t, y_s)
            metrics.append(score)
        except:
            continue
    
    # 步骤4：返回结果（无有效值时退回到普通计算）
    if len(metrics) == 0:
        try:
            score = metric_func(y_true, y_score)
            return score, (score, score)
        except:
            return 0.0, (0.0, 0.0)
    
    mean_score = np.mean(metrics)
    ci = np.percentile(metrics, [2.5, 97.5])
    return mean_score, ci
# ---------------------- 辅助函数：计算扩展评估指标 ----------------------
def calculate_auc_se(auc_val, n):
    """计算AUC标准误"""
    auc_val = np.clip(auc_val, 0.0, 1.0)
    if n < 2:
        return 0.0
    numerator = auc_val * (1 - auc_val) + (n - 1) * (auc_val**2 - auc_val)
    denominator = n * (n - 1)
    ratio = numerator / denominator
    ratio = max(ratio, 1e-10)
    return np.sqrt(ratio)

def calculate_extended_metrics(y_true, y_pred, y_pred_proba, task_type):
    """完善的指标计算函数（修复多分类Bootstrap样本数不一致问题）"""
    # ========== 核心修复1：统一数据格式,根除索引/维度问题 ==========
    # 强制转为numpy一维/二维数组,丢弃pandas索引（根源性问题）
    y_true = np.array(y_true).ravel()  # 标签转为一维数组
    y_pred = np.array(y_pred).ravel()  # 预测标签转为一维数组
    y_pred_proba = np.array(y_pred_proba)  # 预测概率保留原维度（多分类为N×k）
    
    # ========== 核心修复2：强制样本数完全对齐 ==========
    # 1. 确定最小样本数（覆盖所有输入）
    min_samples = len(y_true)
    if len(y_pred) < min_samples:
        min_samples = len(y_pred)
    # 多分类时y_pred_proba是二维数组,取第一维长度
    if len(y_pred_proba.shape) >= 1 and y_pred_proba.shape[0] < min_samples:
        min_samples = y_pred_proba.shape[0]
    
    # 2. 截断所有输入到最小样本数
    if min_samples < len(y_true):
        print(f"Warning: Sample size mismatch! Truncating to {min_samples} samples")
        y_true = y_true[:min_samples]
        y_pred = y_pred[:min_samples]
        y_pred_proba = y_pred_proba[:min_samples]
    
    # ========== 二分类概率格式兼容 ==========
    if task_type == 'binary':
        # 确保二分类概率是N×2的格式
        if len(y_pred_proba.shape) == 1:
            y_pred_proba = np.column_stack([1 - y_pred_proba, y_pred_proba])
        elif len(y_pred_proba.shape) == 2 and y_pred_proba.shape[1] == 1:
            y_pred_proba = np.column_stack([1 - y_pred_proba, y_pred_proba])

    metrics = {}

    # ========== 基础分类指标 ==========
    metrics['Accuracy'] = round(accuracy_score(y_true, y_pred), 4)

    # 精确率
    try:
        if task_type == 'multiclass':
            metrics['Precision'] = round(precision_score(y_true, y_pred, average='weighted'), 4)
        else:
            metrics['Precision'] = round(precision_score(y_true, y_pred, zero_division=0), 4)
    except:
        metrics['Precision'] = 0.0

    # 召回率
    try:
        if task_type == 'multiclass':
            metrics['Recall'] = round(recall_score(y_true, y_pred, average='weighted'), 4)
        else:
            metrics['Recall'] = round(recall_score(y_true, y_pred, zero_division=0), 4)
    except:
        metrics['Recall'] = 0.0

    # F1分数
    try:
        if task_type == 'multiclass':
            metrics['F1-Score'] = round(f1_score(y_true, y_pred, average='weighted'), 4)
        else:
            metrics['F1-Score'] = round(f1_score(y_true, y_pred, zero_division=0), 4)
    except:
        metrics['F1-Score'] = 0.0

    # 二分类特有指标
    if task_type == 'binary':
        try:
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            metrics['Sensitivity'] = round(tp / (tp + fn) if (tp + fn) > 0 else 0.0, 4)
            metrics['Specificity'] = round(tn / (tn + fp) if (tn + fp) > 0 else 0.0, 4)
            metrics['PPV'] = round(tp / (tp + fp) if (tp + fp) > 0 else 0.0, 4)
            metrics['NPV'] = round(tn / (tn + fn) if (tn + fn) > 0 else 0.0, 4)
        except:
            metrics['Sensitivity'] = 0.0
            metrics['Specificity'] = 0.0
            metrics['PPV'] = 0.0
            metrics['NPV'] = 0.0
    else:
        metrics['Sensitivity'] = 0.0
        metrics['Specificity'] = 0.0
        metrics['PPV'] = 0.0
        metrics['NPV'] = 0.0

    # ========== AUC及置信区间（核心修复3：替换sklearn bootstrap为自定义函数） ==========
    try:
        if task_type == 'multiclass':
            auc_val = roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='weighted')
        else:
            auc_val = roc_auc_score(y_true, y_pred_proba[:, 1])
        metrics['AUC'] = round(auc_val, 4)
    except Exception as e:
        print(f"AUC calculation failed: {e}")
        auc_val = 0.0
        metrics['AUC'] = 0.0

    # Bootstrap计算AUC的95%CI（完全使用自定义safe_bootstrap_metric）
    try:
        # 定义多分类/二分类的AUC计算函数
        if task_type == 'multiclass':
            def multiclass_auc_func(yt, yp):
                return roc_auc_score(yt, yp, multi_class='ovr', average='weighted')
            # 调用自定义Bootstrap函数
            auc_mean, auc_ci = safe_bootstrap_metric(
                y_true, y_pred_proba, multiclass_auc_func
            )
        else:
            # 二分类取正类概率
            y_pred_pos_proba = y_pred_proba[:, 1]
            auc_mean, auc_ci = safe_bootstrap_metric(
                y_true, y_pred_pos_proba, roc_auc_score
            )
        metrics['AUC_95%CI'] = (round(auc_ci[0], 4), round(auc_ci[1], 4))
    except Exception as e:
        print(f"Bootstrap AUC failed: {e}, falling back to normal approximation")
        n = len(y_true)
        se_auc = calculate_auc_se(auc_val, n)
        ci_lower = max(0.0, auc_val - 1.96 * se_auc)
        ci_upper = min(1.0, auc_val + 1.96 * se_auc)
        metrics['AUC_95%CI'] = (round(ci_lower, 4), round(ci_upper, 4))

    # 准确率的95%CI
    try:
        n = len(y_true)
        acc = metrics['Accuracy']
        se_acc = np.sqrt((acc * (1 - acc)) / n)
        acc_ci_lower = max(0.0, acc - 1.96 * se_acc)
        acc_ci_upper = min(1.0, acc + 1.96 * se_acc)
        metrics['Accuracy_95%CI'] = (round(acc_ci_lower, 4), round(acc_ci_upper, 4))
    except:
        metrics['Accuracy_95%CI'] = (0.0, 0.0)

    return metrics
# ---------------------- 计算模型边数 ----------------------
def calculate_model_edges(model, model_name):
    """计算模型的边数"""
    if model_name == "DecisionTree":
        if hasattr(model, "tree_"):
            return model.tree_.node_count - 1
        return 1
    elif model_name in ["RandomForest", "ExtraTrees", "GBM", "AdaBoost", "XGBoost"]:
        total_edges = 0
        if hasattr(model, "estimators_"):
            for estimator in model.estimators_:
                if hasattr(estimator, "tree_"):
                    total_edges += estimator.tree_.node_count - 1
        elif hasattr(model, "get_booster"):
            booster = model.get_booster()
            num_trees = booster.num_boosted_rounds()
            max_depth = model.get_params().get("max_depth", 3)
            single_tree_edges = (2 ** (max_depth + 1) - 1) - 1
            total_edges = num_trees * single_tree_edges
        return total_edges if total_edges > 0 else 1
    else:
        return 1

# ---------------------- 多分类混淆矩阵可视化 ----------------------
def plot_multiclass_confusion_matrix(y_true, y_pred, label_encoder, model_name, task_type, normalize=True):
    """多分类混淆矩阵可视化"""
    print(f"Plotting confusion matrix for {model_name}...")
    class_names = label_encoder.classes_
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = np.round(cm, 4)
        vmin, vmax, fmt = 0.0, 1.0, '.2f'
        cbar_label = 'Normalized Ratio'
    else:
        vmin, vmax, fmt = 0, np.max(cm), 'd'
        cbar_label = 'Sample Count'

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap='Blues',
        vmin=vmin,
        vmax=vmax,
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': cbar_label},
        linewidths=0.5
    )
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title(f'Confusion Matrix - {model_name} - {task_type}\n(Normalized: {normalize})', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    save_name = f'confusion_matrix_{model_name.lower().replace(" ", "_")}_{task_type}.png'
    plt.savefig(save_name, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Confusion matrix saved as {save_name}")

# ---------------------- 数据预处理函数 ----------------------
def data_preprocessing_opt(df: pd.DataFrame, label_col: str) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str], pd.Series, LabelEncoder]:
    """数据预处理函数"""
    print("="*50)
    print("Starting data preprocessing...")

    if ID_COLUMN not in df.columns:
        raise ValueError(f"ID column '{ID_COLUMN}' not found in dataset!")
    id_series = df[ID_COLUMN].copy()

    # 分离特征和标签
    if label_col not in df.columns:
        raise ValueError(f"Label column {label_col} not found in dataset!")

    X = df.drop(columns=[label_col, ID_COLUMN])
    y = df[label_col].copy()
    feature_names = X.columns.tolist()
    print(f"Raw data shape: Features {X.shape}, Labels {y.shape}, ID count {len(id_series)}")

    # 编码标签
    le = LabelEncoder()
    y_processed = le.fit_transform(y)
    print(f"Label classes: {le.classes_}, Encoded labels: {np.unique(y_processed)}")

    # 分离数值和分类特征
    num_features = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_features = X.select_dtypes(include=[object, 'category']).columns.tolist()
    print(f"Numeric features: {len(num_features)}, Categorical features: {len(cat_features)}")

    # 处理数值特征
    if num_features:
        knn_imputer = KNNImputer(n_neighbors=5, weights='distance')
        X_num_imputed = knn_imputer.fit_transform(X[num_features])
        X_num_imputed = pd.DataFrame(X_num_imputed, columns=num_features, index=X.index)
        
        # 异常值处理
        for col in num_features:
            mean = X_num_imputed[col].mean()
            std = X_num_imputed[col].std()
            upper_limit = mean + 3 * std
            lower_limit = mean - 3 * std
            X_num_imputed[col] = np.where(
                X_num_imputed[col] > upper_limit, upper_limit,
                np.where(X_num_imputed[col] < lower_limit, lower_limit, X_num_imputed[col])
            )
    else:
        X_num_imputed = pd.DataFrame(index=X.index)

    # 处理分类特征
    if cat_features:
        cat_imputer = SimpleImputer(strategy='most_frequent', fill_value='Unknown')
        X_cat_imputed = cat_imputer.fit_transform(X[cat_features])
        X_cat_imputed = pd.DataFrame(X_cat_imputed, columns=cat_features, index=X.index)
        
        # 编码分类特征
        for col in cat_features:
            le_cat = LabelEncoder()
            X_cat_imputed[col] = le_cat.fit_transform(X_cat_imputed[col].astype(str))
    else:
        X_cat_imputed = pd.DataFrame(index=X.index)

    # 合并特征
    X_processed = pd.concat([X_num_imputed, X_cat_imputed], axis=1)
    feature_names = X_processed.columns.tolist()

    # 最终数据验证
    X_processed = X_processed.dropna()
    y_processed = y_processed[X_processed.index]
    id_series = id_series[X_processed.index]
    print(f"Processed data shape: Features {X_processed.shape}, Labels {y_processed.shape}, ID count {len(id_series)}")
    print("Data preprocessing completed!")
    print("="*50)

    return X_processed, y_processed, feature_names, num_features, id_series, le

# ---------------------- 数值特征直方图 ----------------------
def plot_numeric_feature_histogram(X: pd.DataFrame, num_features: List[str], task_type: str):
    """绘制数值特征直方图"""
    print("="*50)
    print("Plotting numeric feature frequency distribution histogram...")

    if not num_features:
        print("No numeric features found, skipping histogram plot!")
        return

    top_num_features = num_features[:13]
    n_cols = 2
    n_rows = (len(top_num_features) + 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    axes = axes.flatten()

    for i, feature in enumerate(top_num_features):
        ax = axes[i]
        ax.hist(X[feature], bins=30, alpha=0.7, color='skyblue', edgecolor='black', density=True)
        mu, sigma = norm.fit(X[feature])
        x = np.linspace(X[feature].min(), X[feature].max(), 100)
        ax.plot(x, norm.pdf(x, mu, sigma), 'r--', linewidth=2, label=f'Normal fit (μ={mu:.2f}, σ={sigma:.2f})')
        
        ax.set_title(f'Frequency Distribution of {feature}', fontsize=12)
        ax.set_xlabel(feature, fontsize=10)
        ax.set_ylabel('Density', fontsize=10)
        ax.legend()
        ax.grid(alpha=0.3)

    for i in range(len(top_num_features), len(axes)):
        fig.delaxes(axes[i])

    plt.suptitle(f'Frequency Distribution Histogram of Numeric Features - {task_type}', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'numeric_feature_histogram_{task_type}.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("Numeric feature histogram plot completed!")
    print("="*50)

# ---------------------- Spearman相关性分析 ----------------------
def plot_spearman_correlation(X: pd.DataFrame, task_type: str):
    """绘制Spearman相关性热力图"""
    print("="*50)
    print("Plotting Spearman correlation analysis heatmap...")

    corr = X.corr(method='spearman')

    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr,
        mask=mask,
        cmap='coolwarm',
        annot=True,
        fmt='.2f',
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        cbar_kws={'label': 'Spearman Correlation Coefficient'}
    )
    plt.title(f'Spearman Correlation Analysis of Prediction Model Dataset - {task_type}', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'spearman_correlation_{task_type}.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("Spearman correlation plot completed!")
    print("="*50)

# ---------------------- 模型复杂度与AUC关系图 ----------------------
def plot_model_complexity_vs_auc(performance_list: List[Dict], task_type: str):
    """绘制模型复杂度与AUC关系图"""
    print("="*50)
    print("Plotting model complexity (log(#edges)) vs AUC performance...")

    model_names = [p["Model Name"] for p in performance_list]
    auc_scores = [p["CV_AUC"] for p in performance_list]  # 使用交叉验证AUC
    log_edges = [p["Log_Model_Edges"] for p in performance_list]

    if "ensemble_results" in performance_list[0]:
        ensemble_y_pred, ensemble_y_proba = performance_list[0]["ensemble_results"]
        y_test = performance_list[0]["y_test"]
        if task_type == "multiclass":
            ensemble_auc = roc_auc_score(y_test, ensemble_y_proba, multi_class="ovr", average="weighted")
        else:
            ensemble_auc = roc_auc_score(y_test, ensemble_y_proba[:, 1])
        
        core_model_log_edges = [
            p["Log_Model_Edges"] for p in performance_list
            if p["Model Name"] in ["RandomForest", "ExtraTrees", "GBM", "XGBoost"]
        ]
        ensemble_log_edges = np.mean(core_model_log_edges) if core_model_log_edges else 0

        model_names.append("Ensemble")
        auc_scores.append(round(ensemble_auc, 4))
        log_edges.append(ensemble_log_edges)

    complexity = []
    colors = []
    for le in log_edges:
        if le < 1:
            complexity.append("Low")
            colors.append("green")
        elif le < 2:
            complexity.append("Medium")
            colors.append("orange")
        else:
            complexity.append("High")
            colors.append("red")

    fig, ax = plt.subplots(figsize=(12, 8))
    scatter = ax.scatter(log_edges, auc_scores, c=colors, s=200, alpha=0.7, edgecolors="black")

    for i, txt in enumerate(model_names):
        ax.annotate(
            txt, 
            (log_edges[i], auc_scores[i]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7)
        )

    if len(log_edges) > 1 and len(auc_scores) > 1:
        z = np.polyfit(log_edges, auc_scores, 1)
        p = np.poly1d(z)
        ax.plot(log_edges, p(log_edges), "b--", alpha=0.5, label=f"Trend line (slope={z[0]:.4f})")

    ax.set_xlabel("log(#edges) - Model Complexity", fontsize=12)
    ax.set_ylabel("AUC Score - Model Performance", fontsize=12)
    ax.set_title(f"Relationship between Model Complexity (log(#edges)) and AUC Performance - {task_type}", fontsize=14)
    ax.grid(alpha=0.3)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="green", label="Low Complexity (log(#edges) < 1)"),
        Patch(facecolor="orange", label="Medium Complexity (1 ≤ log(#edges) < 2)"),
        Patch(facecolor="red", label="High Complexity (log(#edges) ≥ 2)")
    ]
    ax.legend(handles=legend_elements, loc="lower right")

    plt.tight_layout()
    plt.savefig(f"model_complexity_log_edges_vs_auc_{task_type}.png", dpi=300, bbox_inches="tight")
    plt.show()

    print("Model complexity (log(#edges)) vs AUC plot completed!")
    print("="*50)

# ---------------------- ROC和PR曲线绘制 ----------------------
def plot_model_roc_pr_curve(y_true, y_pred_proba, model_name, task_type):
    """绘制ROC和PR曲线"""
    print(f"="*50)
    print(f"Plotting ROC & PR curves for {model_name}...")

    y_true = np.array(y_true)
    y_pred_proba = np.array(y_pred_proba)

    plt.figure(figsize=(14, 6))

    # ROC曲线
    plt.subplot(1, 2, 1)
    if task_type == "binary":
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba[:, 1])
        roc_auc = roc_auc_score(y_true, y_pred_proba[:, 1])
        plt.plot(fpr, tpr, lw=2, label=f"{model_name}\nAUC = {roc_auc:.4f}")
    else:
        lb = LabelBinarizer()
        y_true_binarized = lb.fit_transform(y_true)
        fpr, tpr, _ = roc_curve(y_true_binarized.ravel(), y_pred_proba.ravel())
        roc_auc = roc_auc_score(y_true_binarized, y_pred_proba, multi_class='ovr', average='micro')
        plt.plot(fpr, tpr, lw=2, label=f"{model_name}\nMicro-AUC = {roc_auc:.4f}")
    
    plt.plot([0, 1], [0, 1], lw=2, linestyle='--', color='gray', label="Random Guess")
    plt.xlabel("1 - Specificity (FPR)", fontsize=11)
    plt.ylabel("Sensitivity (TPR)", fontsize=11)
    plt.title(f"ROC Curve - {model_name}", fontsize=12)
    plt.legend(
                loc='lower right',
                fontsize=8,
                ncol=2,
                handlelength=1.5,
                borderaxespad=0.3,
                columnspacing=0.8,
                handletextpad=0.5
    )
    plt.grid(alpha=0.3)

    # PR曲线
    plt.subplot(1, 2, 2)
    if task_type == "binary":
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba[:, 1])
        auprc = auc(recall, precision)
        plt.plot(recall, precision, lw=2, label=f"{model_name}\nAUPRC = {auprc:.4f}")
    else:
        lb = LabelBinarizer()
        y_true_binarized = lb.fit_transform(y_true)
        precision, recall, _ = precision_recall_curve(y_true_binarized.ravel(), y_pred_proba.ravel())
        auprc = auc(recall, precision)
        plt.plot(recall, precision, lw=2, label=f"{model_name}\nMicro-AUPRC = {auprc:.4f}")
    
    pos_ratio = np.sum(y_true) / len(y_true)
    plt.plot([0, 1], [pos_ratio, pos_ratio], lw=2, linestyle='--', color='gray', label="Random Guess")
    plt.xlabel("Recall", fontsize=11)
    plt.ylabel("Precision", fontsize=11)
    plt.title(f"PR Curve - {model_name}", fontsize=12)
    plt.legend(
                loc='lower left',
                fontsize=8,
                ncol=2,
                handlelength=1.5,
                borderaxespad=0.3,
                columnspacing=0.8,
                handletextpad=0.5
    )
    plt.grid(alpha=0.3)

    plt.tight_layout()
    save_path = f"{model_name}_roc_pr_curve_{task_type}.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"ROC & PR curves saved to: {save_path}")
    print(f"="*50)

# ---------------------- 特征重要性可视化 ----------------------
def extract_and_visualize_feature_importance(models: Dict, X: pd.DataFrame, feature_names: List[str], task_type: str):
    """特征重要性可视化"""
    print("="*50)
    print("Plotting feature importance ranking...")

    importance_supported_models = ['RandomForest', 'ExtraTrees', 'GBM', 'AdaBoost', 'XGBoost']
    available_models = [name for name in models.keys() if name in importance_supported_models]

    if not available_models:
        print("No models support feature importance extraction, skipping!")
        return

    plt.figure(figsize=(12, 8))
    target_model = models['RandomForest']
    if hasattr(target_model, 'feature_importances_'):
        importances = target_model.feature_importances_
        indices = np.argsort(importances)[::-1]
        top_n = min(20, len(feature_names))
        top_indices = indices[:top_n]
        
        plt.bar(range(top_n), importances[top_indices], align='center', color='skyblue', alpha=0.8)
        plt.xticks(range(top_n), [feature_names[i] for i in top_indices], rotation=45, ha='right')
        plt.xlabel('Feature Names', fontsize=12)
        plt.ylabel('Feature Importance', fontsize=12)
        plt.title(f'Model Feature Importance Ranking (Top {top_n}) - {task_type}', fontsize=14)
        plt.tight_layout()
        plt.savefig(f'feature_importance_{task_type}.png', dpi=300, bbox_inches='tight')
        plt.show()

    print("Feature importance plot completed!")
    print("="*50)

# ---------------------- 合并ROC和PR曲线 ----------------------
def plot_combined_roc_pr_curve(model_perf_data, task_type):
    """合并所有模型的ROC和PR曲线"""
    print("="*50)
    print("Plotting combined ROC & PR curves for all models...")

    color_cycle = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive'])

    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(16, 6))

    # ROC曲线
    ax_roc.set_title('ROC Curves (All Models)', fontsize=14)
    ax_roc.set_xlabel('1 - Specificity (FPR)', fontsize=12)
    ax_roc.set_ylabel('Sensitivity (TPR)', fontsize=12)
    ax_roc.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Guess')
    
    for model_data, color in zip(model_perf_data, color_cycle):
        model_name = model_data['model_name']
        fpr = model_data['fpr']
        tpr = model_data['tpr']
        roc_auc = model_data['roc_auc']
        ax_roc.plot(fpr, tpr, color=color, lw=2, label=f'{model_name}\nAUC = {roc_auc:.4f}')
    
    ax_roc.legend(
                loc='lower right',
                fontsize=8,
                ncol=1,
                handlelength=1.5,
                borderaxespad=0.3,
                columnspacing=0.8,
                handletextpad=0.5
    )
    ax_roc.grid(alpha=0.3)

    # PR曲线
    ax_pr.set_title('PR Curves (All Models)', fontsize=14)
    ax_pr.set_xlabel('Recall', fontsize=12)
    ax_pr.set_ylabel('Precision', fontsize=12)
    y_true = model_perf_data[0]['y_true']
    pos_ratio = np.sum(y_true) / len(y_true)
    ax_pr.plot([0, 1], [pos_ratio, pos_ratio], 'k--', lw=2, label='Random Guess')
    
    color_cycle = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive'])
    for model_data, color in zip(model_perf_data, color_cycle):
        model_name = model_data['model_name']
        recall = model_data['recall']
        precision = model_data['precision']
        auprc = model_data['auprc']
        ax_pr.plot(recall, precision, color=color, lw=2, label=f'{model_name}\nAUPRC = {auprc:.4f}')
    
    ax_pr.legend(
                loc='lower right',
                fontsize=8,
                ncol=2,
                handlelength=1.5,
                borderaxespad=0.3,
                columnspacing=0.8,
                handletextpad=0.5
    )
    ax_pr.grid(alpha=0.3)

    plt.tight_layout()
    save_path = f'combined_roc_pr_curves_{task_type}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    print(f"Combined ROC & PR curves saved to: {save_path}")
    print("="*50)

def evaluate_model_with_cv(model, X, y, X_test, y_test, model_name, task_type):
    """使用5折交叉验证评估模型并计算均值 + 测试集指标（修复样本数不一致+scaler未定义问题）"""
    print(f"\nEvaluating {model_name} with {K_FOLD}-fold cross validation...")
    
    # 确保X是DataFrame,y是一维数组
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    y = np.array(y).ravel()  # 确保y是一维,避免维度错误
    
    cv = StratifiedKFold(n_splits=K_FOLD, shuffle=True, random_state=RANDOM_SEED)
    metrics_list = []
    all_y_pred_proba = []
    all_y_true = []
    
    # 存储每折的指标
    fold_metrics = {
        'Accuracy': [], 'Precision': [], 'Recall': [], 'F1-Score': [],
        'Sensitivity': [], 'Specificity': [], 'PPV': [], 'NPV': [], 'AUC': [],
        'AUC_95%CI': [], 'Accuracy_95%CI': []
    }
    
    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        print(f"\n===== Fold {fold+1}/{K_FOLD} =====")
        
        # 关键：通过索引分割,确保样本数严格一致
        X_train, X_val = X.iloc[train_idx].copy(), X.iloc[val_idx].copy()
        y_train, y_val = y[train_idx].copy(), y[val_idx].copy()
        
        # 打印分割后样本数,校验一致性
        print(f"训练集：X={X_train.shape[0]}样本, y={len(y_train)}样本")
        print(f"验证集：X={X_val.shape[0]}样本, y={len(y_val)}样本")
        
        # 标准化（修复scaler未定义问题：先初始化,再判断是否有数值列）
        # 步骤1：筛选数值列
        numeric_cols = X_train.select_dtypes(include=[np.number]).columns
        # 步骤2：初始化标准化器（无论是否有数值列,先定义scaler）
        scaler = StandardScaler()
        # 步骤3：复制数据,避免修改原数据
        X_train_scaled = X_train.copy()
        X_val_scaled = X_val.copy()
        # 步骤4：仅对数值列做标准化
        if len(numeric_cols) > 0:
            X_train_scaled[numeric_cols] = scaler.fit_transform(X_train_scaled[numeric_cols])
            X_val_scaled[numeric_cols] = scaler.transform(X_val_scaled[numeric_cols])
        else:
            print("警告：无数值特征,跳过标准化")
        
        # 训练模型
        model.fit(X_train_scaled, y_train)
        
        # 预测（确保预测概率是一维数组,核心修复）
        y_pred = model.predict(X_val_scaled)
        try:
            # 多分类时返回完整概率矩阵,二分类取正类概率
            y_pred_proba = model.predict_proba(X_val_scaled)
            if task_type == 'binary':
                y_pred_proba = y_pred_proba[:, 1]
        except:
            # 若模型无predict_proba（如SVM）,用predict替代
            y_pred_proba = y_pred
            print(f"{model_name} 无predict_proba方法,使用预测标签替代概率")
        
        # 强制转为一维数组,避免维度错误导致样本数异常
        y_pred_proba = np.array(y_pred_proba).ravel()
        y_val = np.array(y_val).ravel()
        y_pred = np.array(y_pred).ravel()
        
        # 打印预测结果样本数,定位问题
        print(f"验证集标签数：{len(y_val)}, 预测标签数：{len(y_pred)}, 预测概率数：{len(y_pred_proba)}")
        
        # 计算指标（包含Bootstrap）
        metrics = calculate_extended_metrics(y_val, y_pred, y_pred_proba, task_type)
        
        # 保存每折的指标
        for key in fold_metrics.keys():
            if key in metrics:
                fold_metrics[key].append(metrics[key])
        
        # 收集所有预测结果（对齐样本数后）
        min_len = min(len(y_val), len(y_pred_proba))
        all_y_true.append(y_val[:min_len])
        all_y_pred_proba.append(y_pred_proba[:min_len])
    
    # 计算CV均值
    mean_metrics = {}
    for key in fold_metrics.keys():
        if key in ['AUC_95%CI', 'Accuracy_95%CI']:
            # 计算CI的均值
            if fold_metrics[key]:
                lows = [x[0] for x in fold_metrics[key]]
                highs = [x[1] for x in fold_metrics[key]]
                mean_metrics[key] = (round(np.mean(lows), 4), round(np.mean(highs), 4))
            else:
                mean_metrics[key] = (0.0, 0.0)
        else:
            mean_metrics[key] = round(np.mean(fold_metrics[key]), 4) if fold_metrics[key] else 0.0
    
    # ========== 新增：计算测试集指标 ==========
    print(f"\nCalculating {model_name} test set metrics...")
    # 对测试集做预测
    y_test_pred = model.predict(X_test)
    try:
        y_test_pred_proba = model.predict_proba(X_test)
        # 二分类时需要调整概率格式（适配calculate_extended_metrics）
        if task_type == 'binary':
            y_test_pred_proba_2d = np.column_stack([1 - y_test_pred_proba[:, 1], y_test_pred_proba[:, 1]])
        else:
            y_test_pred_proba_2d = y_test_pred_proba
    except:
        y_test_pred_proba = y_test_pred
        y_test_pred_proba_2d = y_test_pred
        print(f"{model_name} 无predict_proba方法,使用预测标签替代概率（测试集）")
    
    # 计算测试集指标
    test_metrics = calculate_extended_metrics(y_test, y_test_pred, y_test_pred_proba_2d, task_type)
    
    # 计算模型复杂度
    model_edges = calculate_model_edges(model, model_name)
    log_model_edges = np.log10(model_edges) if model_edges > 0 else 0.0
    
    # 准备返回结果（包含CV和Test两类指标）
    performance = {
        # CV指标（原有）
        "Model Name": model_name,
        "CV_Accuracy": mean_metrics['Accuracy'],
        "CV_Accuracy_95%CI": mean_metrics['Accuracy_95%CI'],
        "CV_AUC": mean_metrics['AUC'],
        "CV_AUC_95%CI": mean_metrics['AUC_95%CI'],
        "CV_Precision": mean_metrics['Precision'],
        "CV_Recall": mean_metrics['Recall'],
        "CV_F1-Score": mean_metrics['F1-Score'],
        "CV_Sensitivity": mean_metrics['Sensitivity'],
        "CV_Specificity": mean_metrics['Specificity'],
        "CV_PPV": mean_metrics['PPV'],
        "CV_NPV": mean_metrics['NPV'],
        # Test指标（新增）
        "Test_Accuracy": test_metrics['Accuracy'],
        "Test_Accuracy_95%CI": test_metrics['Accuracy_95%CI'],
        "Test_AUC": test_metrics['AUC'],
        "Test_AUC_95%CI": test_metrics['AUC_95%CI'],
        "Test_Precision": test_metrics['Precision'],
        "Test_Recall": test_metrics['Recall'],
        "Test_F1-Score": test_metrics['F1-Score'],
        "Test_Sensitivity": test_metrics['Sensitivity'],
        "Test_Specificity": test_metrics['Specificity'],
        "Test_PPV": test_metrics['PPV'],
        "Test_NPV": test_metrics['NPV'],
        # 模型复杂度
        "Model_Edges": model_edges,
        "Log_Model_Edges": log_model_edges
    }
    
    # 打印汇总结果
    print(f"\n{model_name} 评估结果汇总：")
    print(f"CV均值 - Accuracy: {mean_metrics['Accuracy']}, AUC: {mean_metrics['AUC']}")
    print(f"测试集 - Accuracy: {test_metrics['Accuracy']}, AUC: {test_metrics['AUC']}")
    
    return performance, all_y_true, all_y_pred_proba
# ---------------------- 任务类型判断 ----------------------
def auto_judge_task_type(y: pd.Series) -> str:
    """自动判断任务类型"""
    target_type = type_of_target(y)
    unique_classes = len(np.unique(y))

    if target_type == 'binary' or unique_classes == 2:
        return 'binary'
    elif target_type == 'multiclass' and unique_classes > 2:
        return 'multiclass'
    else:
        raise ValueError(f"Unsupported task type: {target_type}, Number of classes: {unique_classes}")

# ---------------------- 定义模型和参数网格 ----------------------
def define_models_and_param_grids(task_type: str) -> Tuple[Dict, Dict]:
    """定义模型和参数网格"""
    models = {
        'LogisticRegression': LogisticRegression(random_state=RANDOM_SEED),
        'LASSO_LogisticRegression': LogisticRegression(penalty='l1', solver='saga', random_state=RANDOM_SEED, n_jobs=N_JOBS),
        'SVM': SVC(probability=True, random_state=RANDOM_SEED),
        'DecisionTree': DecisionTreeClassifier(random_state=RANDOM_SEED),
        'RandomForest': RandomForestClassifier(random_state=RANDOM_SEED, n_jobs=N_JOBS),
        'KNN': KNeighborsClassifier(n_jobs=N_JOBS),
        'ExtraTrees': ExtraTreesClassifier(random_state=RANDOM_SEED, n_jobs=N_JOBS),
        'GBM': GradientBoostingClassifier(random_state=RANDOM_SEED),
        'AdaBoost': AdaBoostClassifier(random_state=RANDOM_SEED),
        'XGBoost': xgb.XGBClassifier(
            random_state=RANDOM_SEED,
            n_jobs=N_JOBS,
            eval_metric='mlogloss' if task_type == 'multiclass' else 'logloss',
            objective='multi:softprob' if task_type == 'multiclass' else 'binary:logistic'
        )
    }

    param_grids = {
        'LogisticRegression': {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'solver': ['lbfgs', 'liblinear'] if task_type == 'binary' else ['lbfgs'],
            'max_iter': [1000, 2000],
        },
        'LASSO_LogisticRegression': {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'max_iter': [1000, 3000],
        },
        'SVM': {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
            'kernel': ['linear', 'rbf', 'poly']
        },
        'DecisionTree': {
            'max_depth': [3, 5, 7, 10, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        },
        'RandomForest': {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7, None],
            'min_samples_split': [2, 5, 10],
            'bootstrap': [True, False]
        },
        'KNN': {
            'n_neighbors': [3, 5, 7, 9, 11],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan']
        },
        'ExtraTrees': {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7, None],
            'min_samples_split': [2, 5, 10],
            'bootstrap': [True, False]
        },
        'GBM': {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'subsample': [0.8, 0.9, 1.0]
        },
        'AdaBoost': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.5, 1.0],
            'estimator': [DecisionTreeClassifier(max_depth=1), DecisionTreeClassifier(max_depth=3)]
        },
        'XGBoost': {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0],
        }
    }

    if task_type == 'multiclass':
        models['SVM'].decision_function_shape = 'ovr'

    return models, param_grids

# ---------------------- 模型训练和优化 ----------------------
def train_and_optimize_models(models: Dict, param_grids: Dict, X_train: pd.DataFrame, y_train: pd.Series, task_type: str) -> Dict:
    """模型训练和参数优化"""
    print("="*50)
    print("Starting model training and hyperparameter optimization...")
    trained_models = {}

    y_train = np.array(y_train)
    n_classes = len(np.unique(y_train))
    label_min, label_max = y_train.min(), y_train.max()
    print(f"Detected training info: n_classes={n_classes}, label range=[{label_min}, {label_max}]")

    if task_type == 'multiclass' and not (label_min == 0 and label_max == n_classes - 1):
        raise ValueError(f"XGBoost多分类要求标签为0~{n_classes-1}的连续整数")

    for model_name, model in models.items():
        print(f"\nTraining model: {model_name}")
        start_time = time.time()

        if task_type == 'multiclass' and model_name == 'XGBoost':
            model = xgb.XGBClassifier(
                random_state=RANDOM_SEED,
                n_jobs=N_JOBS,
                eval_metric='mlogloss',
                objective='multi:softprob',
                num_class=n_classes,
                n_estimators=100, learning_rate=0.1, max_depth=3
            )
            model.set_params(num_class=n_classes)

        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grids[model_name],
            cv=StratifiedKFold(n_splits=K_FOLD, shuffle=True, random_state=RANDOM_SEED),
            scoring='roc_auc_ovr_weighted' if task_type == 'multiclass' else 'roc_auc',
            n_jobs=N_JOBS,
            verbose=0
        )
        
        grid_search.fit(X_train, y_train)
        
        train_time = time.time() - start_time
        best_model = grid_search.best_estimator_
        best_model.grid_search_time = train_time
        trained_models[model_name] = best_model
        
        print(f"Model {model_name} training completed!")
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Cross-validation best score: {round(grid_search.best_score_, 4)}")
        print(f"Training time: {round(train_time, 4)} seconds")

    print("\nAll models training completed!")
    print("="*50)
    return trained_models

# ---------------------- 模型集成 ----------------------
def ensemble_model_fusion(trained_models: Dict, X_test: pd.DataFrame, task_type: str) -> Tuple[np.ndarray, np.ndarray]:
    """模型集成融合"""
    print("="*50)
    print("Starting model ensemble fusion (soft voting)...")

    core_models = ['RandomForest', 'ExtraTrees', 'GBM', 'AdaBoost', 'XGBoost']
    available_core_models = [name for name in trained_models.keys() if name in core_models]

    if len(available_core_models) < 2:
        raise ValueError("Insufficient core models for ensemble fusion!")

    y_proba_list = []
    for model_name in available_core_models:
        model = trained_models[model_name]
        y_proba = model.predict_proba(X_test)
        y_proba_list.append(y_proba)
    y_proba_array = np.array(y_proba_list)
    ensemble_y_proba = np.mean(y_proba_array, axis=0)
    ensemble_y_pred = np.argmax(ensemble_y_proba, axis=1)

    print(f"Ensemble fusion completed! Used core models: {available_core_models}")
    print("="*50)

    return ensemble_y_pred, ensemble_y_proba

# ---------------------- 基础结果可视化 ----------------------
def visualize_basic_results(performance_list, ensemble_results, task_type, label_encoder):
    """基础结果可视化"""
    print("="*50)
    print("Visualizing basic model performance results...")

    model_names = [p['Model Name'] for p in performance_list]
    auc_scores = [p.get('CV_AUC', 0) for p in performance_list]  # 使用交叉验证AUC
    f1_scores = [p.get('CV_F1-Score', 0) for p in performance_list]  # 使用交叉验证F1
    train_times = [0] * len(model_names)  # 简化处理

    ensemble_y_pred, ensemble_y_proba = ensemble_results
    y_true = performance_list[0].get('y_test', None)
    
    if y_true is not None:
        if task_type == "binary":
            ensemble_f1 = round(f1_score(y_true, ensemble_y_pred, average='binary'), 4)
            ensemble_auc = round(roc_auc_score(y_true, ensemble_y_proba[:, 1]), 4)
        else:
            ensemble_f1 = round(f1_score(y_true, ensemble_y_pred, average='weighted'), 4)
            ensemble_auc = round(roc_auc_score(y_true, ensemble_y_proba, multi_class='ovr', average='weighted'), 4)
        
        model_names.append("Ensemble (Soft Voting)")
        auc_scores.append(ensemble_auc)
        f1_scores.append(ensemble_f1)
        train_times.append(0)

    # AUC对比图
    plt.figure(figsize=(12, 6))
    bars = plt.bar(model_names, auc_scores, color=['skyblue', 'lightgreen', 'lightcoral', 'gold', 'mediumpurple'])
    plt.title(f'Model CV AUC Comparison ({task_type})', fontsize=14)
    plt.ylabel('CV AUC Score', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    
    for bar, score in zip(bars, auc_scores):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{score:.4f}', ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f'model_auc_comparison_{task_type}.png', dpi=300, bbox_inches='tight')
    plt.close()

    # F1对比图
    plt.figure(figsize=(12, 6))
    bars = plt.bar(model_names, f1_scores, color=['skyblue', 'lightgreen', 'lightcoral', 'gold', 'mediumpurple'])
    plt.title(f'Model CV F1-Score Comparison ({task_type})', fontsize=14)
    plt.ylabel('CV F1-Score', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    
    for bar, score in zip(bars, f1_scores):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{score:.4f}', ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f'model_f1_comparison_{task_type}.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("Basic performance visualization completed!")
    print("="*50)

# ---------------------- 主函数 ----------------------
def main(csv_path: str, label_col: str):
    """主函数"""
    try:
        # 1. 加载数据
        print("="*50)
        print("Loading CSV dataset...")
        df = pd.read_csv(csv_path, encoding='utf-8')
        print(f"CSV file loaded successfully! Data shape: {df.shape}")
        print(f"ID column '{ID_COLUMN}' exists: {ID_COLUMN in df.columns}")
        print("="*50)

        # 2. 数据预处理
        X_processed, y_processed, feature_names, num_features, id_series, label_encoder = data_preprocessing_opt(df, label_col)
        
        # 3. 自动判断任务类型
        task_type = auto_judge_task_type(y_processed)
        print(f"Auto-detected task type: {task_type}")
        
        # 4. 直接使用预处理后的特征（删除特征工程）
        X_eng = X_processed.copy()
        final_feature_names = feature_names.copy()
        id_eng = id_series.copy()
        
        # 5. 数据拆分
        X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
            X_eng, y_processed, id_eng,
            test_size=TEST_SIZE, stratify=y_processed, random_state=RANDOM_SEED
        )
        
        y_train = y_train.values.ravel() if hasattr(y_train, 'values') else np.ravel(y_train)
        y_test = y_test.values.ravel() if hasattr(y_test, 'values') else np.ravel(y_test)
        
        print(f"Data split completed:")
        print(f"  Train set: Features {X_train.shape}, Labels {y_train.shape}, ID {len(id_train)}")
        print(f"  Test set: Features {X_test.shape}, Labels {y_test.shape}, ID {len(id_test)}")
        
        # 6. 标准化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=final_feature_names, index=X_train.index)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=final_feature_names, index=X_test.index)
        
        # 7. 定义模型和参数网格
        models, param_grids = define_models_and_param_grids(task_type)
        
        # 8. 训练模型
        trained_models = train_and_optimize_models(models, param_grids, X_train_scaled, y_train, task_type)
        
        # 9. 使用5折交叉验证评估模型
        performance_list = []
        for model_name, model in trained_models.items():
            # 使用整个训练集进行5折交叉验证,同时传入测试集计算测试集指标
            performance, _, _ = evaluate_model_with_cv(model, X_train_scaled, y_train, X_test_scaled, y_test, model_name, task_type)
            performance['y_test'] = y_test  # 保存测试集标签用于后续绘图
            performance_list.append(performance)
        
        # 10. 集成模型
        ensemble_y_pred, ensemble_y_proba = ensemble_model_fusion(trained_models, X_test_scaled, task_type)
        ensemble_results = (ensemble_y_pred, ensemble_y_proba)
        
        # 计算集成模型的交叉验证指标
        ensemble_metrics = calculate_extended_metrics(y_test, ensemble_y_pred, ensemble_y_proba, task_type)
        
        # 11. 收集ROC/PR数据
        print("Starting to collect ROC & PR data for all models...")
        model_perf_data = []
        y_true = y_test
        
        # 基础模型
        for model_name, model in trained_models.items():
            y_pred_proba = model.predict_proba(X_test_scaled)
            if task_type == "binary":
                fpr, tpr, _ = roc_curve(y_true, y_pred_proba[:, 1])
                roc_auc = roc_auc_score(y_true, y_pred_proba[:, 1])
                precision, recall, _ = precision_recall_curve(y_true, y_pred_proba[:, 1])
                auprc = auc(recall, precision)
            else:
                lb = LabelBinarizer()
                y_true_binarized = lb.fit_transform(y_true)
                fpr, tpr, _ = roc_curve(y_true_binarized.ravel(), y_pred_proba.ravel())
                roc_auc = roc_auc_score(y_true_binarized, y_pred_proba, multi_class='ovr', average='micro')
                precision, recall, _ = precision_recall_curve(y_true_binarized.ravel(), y_pred_proba.ravel())
                auprc = auc(recall, precision)
            
            model_perf_data.append({
                'model_name': model_name,
                'fpr': fpr,
                'tpr': tpr,
                'roc_auc': roc_auc,
                'precision': precision,
                'recall': recall,
                'auprc': auprc,
                'y_true': y_true
            })
        
        # 集成模型
        if task_type == "binary":
            ens_fpr, ens_tpr, _ = roc_curve(y_true, ensemble_y_proba[:, 1])
            ens_roc_auc = roc_auc_score(y_true, ensemble_y_proba[:, 1])
            ens_precision, ens_recall, _ = precision_recall_curve(y_true, ensemble_y_proba[:, 1])
            ens_auprc = auc(ens_recall, ens_precision)
        else:
            y_true_binarized = lb.transform(y_true)
            ens_fpr, ens_tpr, _ = roc_curve(y_true_binarized.ravel(), ensemble_y_proba.ravel())
            ens_roc_auc = roc_auc_score(y_true_binarized, ensemble_y_proba, multi_class='ovr', average='micro')
            ens_precision, ens_recall, _ = precision_recall_curve(y_true_binarized.ravel(), ensemble_y_proba.ravel())
            ens_auprc = auc(ens_recall, ens_precision)
        
        model_perf_data.append({
            'model_name': 'Ensemble (Soft Voting)',
            'fpr': ens_fpr,
            'tpr': ens_tpr,
            'roc_auc': ens_roc_auc,
            'precision': ens_precision,
            'recall': ens_recall,
            'auprc': ens_auprc,
            'y_true': y_true
        })
        
        # 绘制合并ROC/PR曲线
        plot_combined_roc_pr_curve(model_perf_data, task_type)
        
        # 12. 基础结果可视化
        visualize_basic_results(performance_list, ensemble_results, task_type, label_encoder)
        
        # 13. 特征重要性可视化
        extract_and_visualize_feature_importance(trained_models, X_train_scaled, final_feature_names, task_type)
        
        # 14. 数值特征直方图
        plot_numeric_feature_histogram(X_processed, num_features, task_type)
        
        # 15. Spearman相关性分析
        plot_spearman_correlation(X_processed, task_type)
        
        # 16. 模型复杂度与AUC关系图
        plot_model_complexity_vs_auc(performance_list, task_type)

        # ========== 核心修改1：定义要分析的目标模型列表 ==========
        # 你可以按需修改这个列表，比如只保留 ['LogisticRegression', 'RandomForest']
        target_models = ['LogisticRegression', 'SVM']
        
        # shap可视化
        shap_dict = {}
        # ========== 修复1：遍历目标模型（而非所有模型） ==========
        for model_name, model in trained_models.items():
            # 只处理目标模型列表中的模型
            if model_name not in target_models:
                print(f"ℹ️  跳过非目标模型：{model_name}")
                continue
            
            # 跳过不支持SHAP的模型（如SVM线性核、KNN）
            if model_name in ['KNN']:
                print(f"⚠️  {model_name} 暂不支持SHAP解释,跳过")
                continue
            
            print(f"\n📊 正在处理 {model_name} 的SHAP解释...")
            # ========== 修复2：适配不同模型的SHAP解释器 ==========
            background = X_train_scaled[:100]  # 背景数据取前100个样本（平衡速度和准确性）
            try:
                # 1. 初始化对应类型的SHAP解释器
                if 'LogisticRegression' in model_name:
                    # 线性模型使用LinearExplainer
                    explainer = shap.LinearExplainer(model, background, feature_perturbation="interventional")
                elif model_name in ['RandomForest', 'ExtraTrees', 'XGBoost', 'GBM', 'AdaBoost', 'DecisionTree']:
                    # 树模型使用TreeExplainer（效率最高）
                    explainer = shap.TreeExplainer(model, background)
                else:
                    # 其他模型使用通用解释器（如神经网络、LightGBM等）
                    explainer = shap.Explainer(model, background)
            
                # 2. 计算核心SHAP值（测试集）
                shap_values = explainer(X_test_scaled)
                # 分类任务中,SHAP值可能是二维（样本×特征）,需提取目标类别（如第1类）
                if len(shap_values.shape) == 3:
                    shap_values = shap_values[:, :, 1]  # 取正类的SHAP值（根据你的标签调整）
                
                # 3. 尝试计算交互值（部分模型不支持,失败则跳过）
                shap_interaction_values = None
                try:
                    shap_interaction_values = explainer.shap_interaction_values(X_test_scaled)
                    print(f"✅ {model_name} 成功计算SHAP交互值")
                except Exception as e:
                    print(f"⚠️ {model_name} 不支持计算SHAP交互值: {str(e)[:50]}")
                
                # 4. 将结果存入字典,便于后续复用
                shap_dict[model_name] = {
                    "explainer": explainer,
                    "shap_values": shap_values,
                    "shap_interaction_values": shap_interaction_values,
                    "feature_names": X_test_scaled.columns if hasattr(X_test_scaled, 'columns') else [f"特征{i}" for i in range(X_test_scaled.shape[1])]
                }
                
                # 5. 生成并保存核心可视化图表
                feature_names = shap_dict[model_name]["feature_names"]
                
                # 全局绘图设置
                plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
                plt.rcParams['axes.unicode_minus'] = False
                shap.initjs()
                
                # ==============================================
                # 图1：SHAP特征重要性柱状图
                # ==============================================
                plt.figure(figsize=(12, 8))
                shap.summary_plot(shap_values, X_test_scaled, plot_type="bar", show=False)
                plt.title(f"{model_name} -Clinical- SHAP Feature Importance Bar Plot", fontsize=16, pad=20)
                plt.xlabel("|SHAP value| (average impact)", fontsize=12)
                plt.tight_layout()
                plt.savefig(f"{model_name}_Clinical-shap_bar_plot.png", dpi=300, bbox_inches='tight')
                plt.show()
                
                # ==============================================
                # 图2：TOP5特征SHAP散点图
                # ==============================================
                # 修复：feature_cols未定义的问题，改用shap_dict中的feature_names
                feature_cols = shap_dict[model_name]["feature_names"]
                feat_imp = np.abs(shap_values.values).mean(axis=0)
                feat_imp_df = pd.DataFrame({"feature": feature_cols, "importance": feat_imp}).sort_values(by="importance", ascending=False)
                top_n_features = feat_imp_df["feature"].head(min(5, len(feature_cols))).tolist()
                print(f"\n生成 TOP{len(top_n_features)} 特征的 SHAP 散点图：{top_n_features}")
                
                for idx, feat in enumerate(top_n_features, 1):
                    plt.figure(figsize=(12, 8))
                    shap.dependence_plot(feat, shap_values.values, X_test_scaled, show=False, alpha=0.6, title=None)
                    plt.title(f"{model_name} -Clinical- SHAP Scatter Plot (TOP{idx} Feature: {feat})", fontsize=16, pad=20)
                    plt.tight_layout()
                    plt.savefig(f"{model_name}_Clinical-shap_scatter_TOP{idx}_{feat}.png", dpi=300, bbox_inches='tight')
                    plt.show()
                
                # ==============================================
                # 图3：单样本SHAP力图
                # ==============================================
                sample_idx = 0  # 修复：定义样本索引（可改为你想分析的样本，如10、20）
                plt.figure(figsize=(20, 7))  # 保留画布大小
                shap.force_plot(
                    explainer.expected_value,
                    shap_values[sample_idx].values,
                    X_test_scaled.iloc[sample_idx],
                    matplotlib=True,
                    show=False
                )
                
                # ========== 核心调整：整体轴线下移（关键修改） ==========
                # 1. 调整subplots_adjust：增大bottom值（底部留白更多）,适度调高top值
                plt.subplots_adjust(top=0.8, bottom=0.2, left=0.05, right=0.95)
                
                # 2. 精准下移轴线（可选：进一步精细化控制）
                ax = plt.gca()
                current_pos = ax.get_position()
                ax.set_position([current_pos.x0, 0.5, current_pos.width, current_pos.height])
                
                # 3. 保留原有f(x)文本上移逻辑
                for text in ax.texts:
                    text_content = text.get_text()
                    if 'base value' in text_content or 'f(x)' in text_content:
                        current_y = text.get_position()[1]
                        text.set_y(current_y +0.01)  # 按需微调偏移量
                        text.set_fontsize(10)  # 默认字体大小约10,改为8（可调6-9）
                
                # 调整标题：适配新的轴线位置
                plt.title(
                    f"{model_name} -Clinical- Single Sample SHAP Force Plot (Sample Index: {sample_idx})",
                    fontsize=16,
                    pad=30,
                    y=1.4  # 从1.2微调为1.15,适配下移的轴线
                )
                
                plt.tight_layout(rect=[0.05, 0.2, 0.95, 0.9])
                plt.savefig(
                    f"{model_name}_Clinical-shap_force_plot.png", 
                    dpi=300, 
                    bbox_inches='tight',
                    pad_inches=0.8
                )
                plt.show()
                
                # ==============================================
                # 图6：SHAP Summary Scatter Plot
                # ==============================================
                plt.figure(figsize=(12, 10))
                shap.summary_plot(shap_values, X_test_scaled, show=False, alpha=0.7)
                plt.title(f"{model_name} -Clinical- SHAP Summary Scatter Plot", fontsize=16, pad=30)
                plt.tight_layout()
                plt.savefig(f"{model_name}_Clinical_shap_summary_scatter_plot.png", dpi=300, bbox_inches='tight')
                plt.show()
                
                # ---------------------- 结果保存 ----------------------
                shap_imp = pd.DataFrame({"feature": feature_cols, "shap_abs_mean": np.abs(shap_values.values).mean(axis=0)}).sort_values(by="shap_abs_mean", ascending=False)
                shap_imp.to_csv(f"{model_name}_Clinical_shap_feature_importance.csv", index=False, encoding="utf-8-sig")
                
                force_plot_html = shap.force_plot(explainer.expected_value, shap_values[sample_idx].values, X_test_scaled.iloc[sample_idx], show=False)
                shap.save_html(f"{model_name}_Clinical_shap_force_plot.html", force_plot_html)
                
                print(f"\n所有结果已保存：")
                print(f"1. 特征重要性图：{model_name}_shap_bar_plot.png")
                print(f"2. TOP5特征散点图：{model_name}_shap_scatter_TOP*.png")
                print(f"3. 单样本力图：{model_name}_shap_force_plot.png / .html")
                print(f"6. SHAP汇总散点图：{model_name}_shap_summary_scatter_plot.png")
                print(f"8. 特征重要性CSV：{model_name}_shap_feature_importance.csv")
        
            # ========== 内层try对应的except：捕获单个模型的SHAP异常 ==========
            except Exception as shap_e:
                print(f"❌ 处理{model_name} SHAP时出错: {shap_e}")
                continue  # 跳过当前模型，继续处理下一个
       #     continue  # 跳过当前模型，继续处理下一个 
        
        # 17. 生成性能汇总报告
        # 17. 生成性能汇总报告
        print("="*50)
        print("Generating model performance summary report...")
        
        summary_data = []
        for p in performance_list:
            summary_row = {
                'Model Name': p['Model Name'],
                # CV指标（原有）
                'CV_Accuracy': p['CV_Accuracy'],
                'CV_Accuracy_95%CI': f"{p['CV_Accuracy_95%CI'][0]}-{p['CV_Accuracy_95%CI'][1]}",
                'CV_AUC': p['CV_AUC'],
                'CV_AUC_95%CI': f"{p['CV_AUC_95%CI'][0]}-{p['CV_AUC_95%CI'][1]}",
                'CV_Precision': p['CV_Precision'],
                'CV_Recall': p['CV_Recall'],
                'CV_F1-Score': p['CV_F1-Score'],
                'CV_Sensitivity': p['CV_Sensitivity'],
                'CV_Specificity': p['CV_Specificity'],
                'CV_PPV': p['CV_PPV'],
                'CV_NPV': p['CV_NPV'],
                # Test指标（新增）
                'Test_Accuracy': p['Test_Accuracy'],
                'Test_Accuracy_95%CI': f"{p['Test_Accuracy_95%CI'][0]}-{p['Test_Accuracy_95%CI'][1]}",
                'Test_AUC': p['Test_AUC'],
                'Test_AUC_95%CI': f"{p['Test_AUC_95%CI'][0]}-{p['Test_AUC_95%CI'][1]}",
                'Test_Precision': p['Test_Precision'],
                'Test_Recall': p['Test_Recall'],
                'Test_F1-Score': p['Test_F1-Score'],
                'Test_Sensitivity': p['Test_Sensitivity'],
                'Test_Specificity': p['Test_Specificity'],
                'Test_PPV': p['Test_PPV'],
                'Test_NPV': p['Test_NPV']
            }
            summary_data.append(summary_row)
        
        # 添加集成模型（同时包含CV和Test指标）
        ensemble_row = {
            'Model Name': 'Ensemble',
            # CV指标（集成模型无CV,用测试集值填充或留空）
            'CV_Accuracy': '-',
            'CV_Accuracy_95%CI': '-',
            'CV_AUC': '-',
            'CV_AUC_95%CI': '-',
            'CV_Precision': '-',
            'CV_Recall': '-',
            'CV_F1-Score': '-',
            'CV_Sensitivity': '-',
            'CV_Specificity': '-',
            'CV_PPV': '-',
            'CV_NPV': '-',
            # Test指标（集成模型）
            'Test_Accuracy': ensemble_metrics['Accuracy'],
            'Test_Accuracy_95%CI': f"{ensemble_metrics.get('Accuracy_95%CI', (0.0, 0.0))[0]}-{ensemble_metrics.get('Accuracy_95%CI', (0.0, 0.0))[1]}",
            'Test_AUC': ensemble_metrics.get('AUC', 0.0),
            'Test_AUC_95%CI': f"{ensemble_metrics.get('AUC_95%CI', (0.0, 0.0))[0]}-{ensemble_metrics.get('AUC_95%CI', (0.0, 0.0))[1]}",
            'Test_Precision': ensemble_metrics['Precision'],
            'Test_Recall': ensemble_metrics['Recall'],
            'Test_F1-Score': ensemble_metrics['F1-Score'],
            'Test_Sensitivity': ensemble_metrics.get('Sensitivity', 0.0),
            'Test_Specificity': ensemble_metrics.get('Specificity', 0.0),
            'Test_PPV': ensemble_metrics.get('PPV', 0.0),
            'Test_NPV': ensemble_metrics.get('NPV', 0.0)
        }
        summary_data.append(ensemble_row)
        
        # 创建并保存汇总数据
        performance_df = pd.DataFrame(summary_data)
        print("\nModel Performance Summary (CV Mean + Test Set):")
        print(performance_df.round(4))
        
        # 保存为CSV文件
        performance_df.to_csv(f'FREET_MultiP13_model_performance_summary_{task_type}.csv', index=False, encoding='utf-8-sig')
        print(f"\nPerformance summary saved to: FREET_MultiP13_model_performance_summary_{task_type}.csv")

        # 保存测试集预测结果
        print("\nSaving test set ID + prediction results...")
        id_test_vals = id_test.values if hasattr(id_test, 'values') else id_test
        y_test_vals = y_test if hasattr(y_test, 'values') else y_test
        
        pred_result_df = pd.DataFrame({
            ID_COLUMN: id_test_vals,
            'True_Label': label_encoder.inverse_transform(y_test_vals)
        })
        
        for model_name, model in trained_models.items():
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)
            pred_result_df[f'{model_name}_Pred_Label'] = label_encoder.inverse_transform(y_pred)
            if task_type == 'binary':
                pred_result_df[f'{model_name}_Pred_Prob'] = y_pred_proba[:, 1]
            else:
                pred_result_df[f'{model_name}_Pred_Prob'] = np.max(y_pred_proba, axis=1)
        
        pred_result_df['Ensemble_Pred_Label'] = label_encoder.inverse_transform(ensemble_y_pred)
        if task_type == 'binary':
            pred_result_df['Ensemble_Pred_Prob'] = ensemble_y_proba[:, 1]
        else:
            pred_result_df['Ensemble_Pred_Prob'] = np.max(ensemble_y_proba, axis=1)
        
        pred_result_df.to_csv(f'FREET_test_set_id_predictions_{task_type}.csv', index=False, encoding='utf-8-sig')
        print(f"Test set ID + predictions saved to: FREET_MultiP13_test_set_id_predictions_{task_type}.csv")

        print("="*50)
        print("All processes completed successfully!")
        print(f"\nKey output files:")
        print(f"  1. Performance summary: FREET_MultiP13_model_performance_summary_{task_type}.csv")
        print(f"  2. Test set ID + predictions: FREET_MultiP13_test_set_id_predictions_{task_type}.csv")
        
    except Exception as e:
        print(f"Program error: {e}")
        raise

# ---------------------- 运行主函数 ----------------------
if __name__ == "__main__":
    # 替换为你的数据集路径和标签列
    CSV_FILE_PATH = "/data/ruth/rrl2/dataset/FREET_MultiP13.csv"
    LABEL_COLUMN = "OP"
    
    # 运行主函数
    main(CSV_FILE_PATH, LABEL_COLUMN)