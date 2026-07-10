import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
import warnings
import joblib
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
warnings.filterwarnings('ignore')

# --------------------------
# 1. 配置参数
# --------------------------
TRAIN_PATH = r"D:\项目\模型预测\标准化数据集\train_data_归一化后.csv"
TEST_PATH = r"D:\项目\模型预测\标准化数据集\test_data_归一化后.csv"
LABEL_COL = "总价(万)"
ID_COL = "小区名称"

OUTPUT_DIR = "result"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print(f"✅ 已创建输出文件夹：{OUTPUT_DIR}")

# --------------------------
# 核心：按你的要求重新定义特征列
# --------------------------
# 1. 无大小依据的纯类别特征（全部声明为LightGBM类别特征）
CATEGORICAL_COLS = ["装修类型", "朝向", "楼层",'户型']
# 2. 有数值意义的离散特征（无需声明类别）
DISCRETE_NUM_COLS = ["区域","建造年份", "总楼层", "室", "厅", "卫", "房龄",'房间总数']
# 3. 连续数值特征
CONTINUOUS_NUM_COLS = ["面积(㎡)",'卫室比', '厅室比', '功能区密度']
# 4. 所有参与训练的特征列
FEATURE_COLS = CATEGORICAL_COLS + DISCRETE_NUM_COLS + CONTINUOUS_NUM_COLS

# LightGBM参数
LGB_PARAMS = {
    "objective": "poisson",
    "metric": "mse",
    "boosting_type": "gbdt",
    "max_depth": 6,
    "num_leaves": 27,
    "min_child_samples": 20,
    "learning_rate": 0.01,
    "random_state": 42,
    "verbose": -1,
     "subsample":0.8,
    "colsample_bytree":0.8
}

RELATIVE_ERROR_THRESHOLD = 25.0
THRESHOLD_LABEL = f"相对误差≤{RELATIVE_ERROR_THRESHOLD}%"

# --------------------------
# 全局配置
# --------------------------
plt.rcParams["font.family"] = ["SimHei",]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 100

# --------------------------
# 核心工具函数
# --------------------------
def plot_feature_importance(model, feature_cols, save_filename="feature_importance_final.png"):
    save_path = os.path.join(OUTPUT_DIR, save_filename)
    importance = model.feature_importance(importance_type='gain')
    importance_norm = (importance / importance.sum()) * 100
    feat_imp = pd.DataFrame({
        '特征名称': feature_cols,
        '特征重要性（Gain）': importance,
        '归一化占比(%)': importance_norm.round(2)
    }).sort_values('特征重要性（Gain）', ascending=False)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    # 按特征类型设置颜色：纯类别（紫）、离散数值（蓝）、连续数值（红）
    colors = []
    for col in feat_imp['特征名称'][::-1]:
        if col in CATEGORICAL_COLS:
            colors.append('#9B59B6')  # 纯类别特征（紫色）
        elif col in DISCRETE_NUM_COLS:
            colors.append('#3498DB')  # 离散数值特征（蓝色）
        elif col in CONTINUOUS_NUM_COLS:
            colors.append('#E74C3C')  # 连续数值特征（红色）
    bars = ax.barh(feat_imp['特征名称'][::-1], feat_imp['特征重要性（Gain）'][::-1], color=colors)
    
    # 标注数值+占比
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ratio = feat_imp['归一化占比(%)'].iloc[-(i+1)]
        ax.text(width + 100, bar.get_y() + bar.get_height()/2, 
                f'{width:.0f} ({ratio}%)', ha='left', va='center', fontsize=9)
    
    ax.set_xlabel('特征重要性（Gain）', fontsize=12)
    ax.set_ylabel('特征名称', fontsize=12)
    ax.set_title('LightGBM回归模型 - 特征重要性排序', fontsize=14, fontweight='bold')
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#9B59B6', label='纯类别特征'),
        Patch(facecolor='#3498DB', label='离散数值特征'),
        Patch(facecolor='#E74C3C', label='连续数值特征')
    ]
    ax.legend(handles=legend_elements, loc='lower right')
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 特征重要性图已保存至：{save_path}")
    return feat_imp

def check_necessary_cols(df, df_name, necessary_cols):
    missing_cols = [col for col in necessary_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"❌ {df_name} 缺失必要列：{missing_cols}")
    return df

# --------------------------
# 数据读取与预处理
# --------------------------
train_df = pd.read_csv(TRAIN_PATH, encoding='utf-8-sig')
test_df = pd.read_csv(TEST_PATH, encoding='utf-8-sig')

print(f"📊 训练集形状：{train_df.shape}")
print(f"📊 测试集形状：{test_df.shape}")

# 检查必要列
train_df = check_necessary_cols(train_df, "训练集", [ID_COL, LABEL_COL] + FEATURE_COLS)
test_df = check_necessary_cols(test_df, "测试集", [ID_COL, LABEL_COL] + FEATURE_COLS)

# 删除目标列缺失样本
train_df = train_df.dropna(subset=[LABEL_COL])
test_df = test_df.dropna(subset=[LABEL_COL])
print(f"🧹 仅删除目标列缺失样本后，训练集样本数：{train_df.shape[0]}")
print(f"🧹 仅删除目标列缺失样本后，测试集样本数：{test_df.shape[0]}")

# 提取特征和目标
X_train, y_train = train_df[FEATURE_COLS], train_df[LABEL_COL]
X_test, y_test = test_df[FEATURE_COLS], test_df[LABEL_COL]

# --------------------------
# 训练LightGBM模型（声明所有纯类别特征）
# --------------------------
lgb_train = lgb.Dataset(X_train, label=y_train, categorical_feature=CATEGORICAL_COLS)
lgb_test = lgb.Dataset(X_test, label=y_test, reference=lgb_train, categorical_feature=CATEGORICAL_COLS)

print("\n🚀 开始训练LightGBM回归模型（总价为原始值）...")
print(f"🔧 无大小依据的纯类别特征：{CATEGORICAL_COLS}")
print(f"🔧 有数值意义的离散特征：{DISCRETE_NUM_COLS}")
print(f"🔧 连续数值特征：{CONTINUOUS_NUM_COLS}")

model = lgb.train(
    LGB_PARAMS,
    lgb_train,
    valid_sets=[lgb_train, lgb_test],
    num_boost_round=10000,
    callbacks=[
        lgb.early_stopping(stopping_rounds=100, verbose=True),
        lgb.log_evaluation(period=200)
    ]
)
print("✅ 模型训练完成！")

# --------------------------
# 模型预测与评估
# --------------------------
y_train_pred = model.predict(X_train, num_iteration=model.best_iteration)
y_test_pred = model.predict(X_test, num_iteration=model.best_iteration)

# 训练集评估
train_mae = mean_absolute_error(y_train, y_train_pred)
train_mse = mean_squared_error(y_train, y_train_pred)
train_rmse = np.sqrt(train_mse)
train_r2 = r2_score(y_train, y_train_pred)

print("\n" + "="*60)
print("【训练集评估结果】")
print(f"平均绝对误差（MAE）：{train_mae:.2f} 万")
print(f"均方误差（MSE）：{train_mse:.2f} 万²")
print(f"均方根误差（RMSE）：{train_rmse:.2f} 万")
print(f"决定系数（R²）：{train_r2:.4f}")
print("="*60)

# 测试集评估
test_mae = mean_absolute_error(y_test, y_test_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
test_rmse = np.sqrt(test_mse)
test_r2 = r2_score(y_test, y_test_pred)

print("\n" + "="*60)
print("【测试集评估结果】")
print(f"平均绝对误差（MAE）：{test_mae:.2f} 万")
print(f"均方误差（MSE）：{test_mse:.2f} 万²")
print(f"均方根误差（RMSE）：{test_rmse:.2f} 万")
print(f"决定系数（R²）：{test_r2:.4f}")
print("="*60)

# --------------------------
# 生成特征重要性与预测结果
# --------------------------
feat_imp_df = plot_feature_importance(model, FEATURE_COLS)
feat_imp_path = os.path.join(OUTPUT_DIR, "feature_importance_final.csv")
feat_imp_df.to_csv(feat_imp_path, index=False, encoding="utf-8-sig")
print(f"✅ 特征重要性表已保存至：{feat_imp_path}")
print("\n【特征重要性（含归一化占比）】")
print(feat_imp_df[['特征名称', '特征重要性（Gain）', '归一化占比(%)']].head(20))

# 预测结果表
result_df = test_df[[ID_COL]].copy()
result_df["真实总价(万)"] = y_test.round(2)
result_df["预测总价(万)"] = y_test_pred.round(2)
result_df["绝对误差(万)"] = abs(result_df["预测总价(万)"] - result_df["真实总价(万)"]).round(2)

def calculate_relative_error(row):
    true_val = row["真实总价(万)"]
    abs_error = row["绝对误差(万)"]
    if true_val == 0:
        return np.inf
    return (abs_error / true_val) * 100

result_df["相对误差(%)"] = result_df.apply(calculate_relative_error, axis=1).round(2)
result_df[f"是否正确（{THRESHOLD_LABEL}）"] = result_df["相对误差(%)"].apply(
    lambda x: "是" if x <= RELATIVE_ERROR_THRESHOLD else "否"
)

# 统计正确率
valid_mask = result_df["真实总价(万)"] != 0
valid_count = valid_mask.sum()
correct_count = (result_df.loc[valid_mask, f"是否正确（{THRESHOLD_LABEL}）"] == "是").sum()

if valid_count > 0:
    correct_rate = correct_count / valid_count * 100
    print(f"\n📊 测试集预测正确率（{THRESHOLD_LABEL}）：{correct_rate:.2f}%（{correct_count}/{valid_count}）")
    zero_true_count = (~valid_mask).sum()
    if zero_true_count > 0:
        print(f"⚠️  注意：测试集中有{zero_true_count}个样本真实总价为0，已排除在正确率统计外")
else:
    print("\n❌ 测试集中无有效样本（真实总价均为0），无法计算正确率")
    correct_rate = 0

result_path = os.path.join(OUTPUT_DIR, "prediction_results_final.csv")
result_df.to_csv(result_path, index=False, encoding="utf-8-sig")
print(f"✅ 预测结果对比表已保存至：{result_path}")
print("\n【预测结果前10条示例】")
print(result_df[["小区名称", "真实总价(万)", "预测总价(万)", "绝对误差(万)", "相对误差(%)", f"是否正确（{THRESHOLD_LABEL}）"]].head(10))

# 保存模型
model_path = os.path.join(OUTPUT_DIR, "house_price_prediction_model_final.pkl")
joblib.dump(model, model_path)
print(f"\n✅ 模型已保存至：{model_path}")

print("\n🎉 全部流程完成！所有输出文件均已保存至当前目录的result文件夹中")