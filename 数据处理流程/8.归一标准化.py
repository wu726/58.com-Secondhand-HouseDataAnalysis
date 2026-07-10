import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler  # 替换为MinMaxScaler
import joblib
import os

# ===================== 1. 定义路径和变量 =====================
# 文件路径
train_path = r"D:\项目\模型预测\映射数据集\train_data_编码后.csv"
test_path = r"D:\项目\模型预测\映射数据集\test_data_编码后.csv"
# 归一化器保存路径
scaler_save_path = r"D:\项目\模型预测\标准化数据集\minmax_scaler_params.pkl"
# 处理后文件保存路径
train_processed_path = r"D:\项目\模型预测\标准化数据集\train_data_归一化后.csv"
test_processed_path = r"D:\项目\模型预测\标准化数据集\test_data_归一化后.csv"

# 变量分类
id_col = "小区名称"  # 标识列
category_cols = [ '区域','装修类型','户型','朝向','楼层','建造年份','总价(万)','总楼层','室','厅', '卫','房龄','房间总数'
    ]  # 类别变量/离散数值变量（不归一化）
numeric_cols = [
    '面积(㎡)','卫室比', '厅室比', '功能区密度'
]  # 连续       数值变量（需要归一化）

# ===================== 2. 工具函数：自动创建文件夹 =====================
def create_dir_if_not_exists(file_path):
    """
    检查文件路径对应的文件夹是否存在，不存在则创建
    :param file_path: 文件的完整路径
    """
    # 提取文件所在的文件夹路径
    dir_path = os.path.dirname(file_path)
    if not os.path.exists(dir_path):
        # 创建文件夹（包括多级目录）
        os.makedirs(dir_path)
        print(f"文件夹 {dir_path} 不存在，已自动创建")
    else:
        print(f"文件夹 {dir_path} 已存在，无需创建")

# ===================== 3. 读取数据 =====================
def load_data(file_path):
    """读取CSV文件，处理缺失值"""
    try:
        df = pd.read_csv(file_path, encoding='utf-8-sig')
        # 检查必要列是否存在
        missing_cols = [col for col in [id_col] + category_cols + numeric_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"文件 {file_path} 缺失列：{missing_cols}")
        # 填充数值列缺失值（用0，也可改为训练集最小值，避免影响归一化范围）
        df[numeric_cols] = df[numeric_cols].fillna(0)
        print(f"成功读取 {file_path}，数据形状：{df.shape}")
        return df
    except FileNotFoundError:
        print(f"错误：找不到文件 {file_path}")
        return None
    except Exception as e:
        print(f"读取数据出错：{e}")
        return None

# ===================== 4. 主流程执行 =====================
# 1. 提前创建所有保存路径的文件夹
create_dir_if_not_exists(scaler_save_path)
create_dir_if_not_exists(train_processed_path)
create_dir_if_not_exists(test_processed_path)

# 2. 读取训练集和测试集
train_df = load_data(train_path)
test_df = load_data(test_path)

if train_df is None or test_df is None:
    exit()

# 3. 训练集归一化（MinMaxScaler，缩放到0-1）
# 分离非数值列和数值列
train_non_numeric = train_df[[id_col] + category_cols]  # 标识列+类别列
train_numeric = train_df[numeric_cols]  # 数值列

# 初始化归一化器（仅用训练集拟合）
minmax_scaler = MinMaxScaler(feature_range=(0, 1))  # 可调整范围，如(0,10)
train_numeric_scaled = minmax_scaler.fit_transform(train_numeric)

# 转换为DataFrame
train_numeric_scaled_df = pd.DataFrame(
    train_numeric_scaled,
    columns=numeric_cols,
    index=train_df.index  # 保持索引一致，方便合并
)

# 保存归一化器参数（供测试集使用）
joblib.dump(minmax_scaler, scaler_save_path)
print(f"归一化器参数已保存至：{scaler_save_path}")

# 4. 测试集归一化（使用训练集参数）
# 加载保存的归一化器
scaler_loaded = joblib.load(scaler_save_path)

# 分离测试集非数值列和数值列
test_non_numeric = test_df[[id_col] + category_cols]
test_numeric = test_df[numeric_cols]

# 用训练集参数归一化测试集（仅transform，不fit）
test_numeric_scaled = scaler_loaded.transform(test_numeric)
test_numeric_scaled_df = pd.DataFrame(
    test_numeric_scaled,
    columns=numeric_cols,
    index=test_df.index
)

# 5. 合并数据并保存
# 合并训练集
train_processed = pd.concat([train_non_numeric, train_numeric_scaled_df], axis=1)
# 合并测试集
test_processed = pd.concat([test_non_numeric, test_numeric_scaled_df], axis=1)

# 保存处理后的数据
train_processed.to_csv(train_processed_path, index=False, encoding='utf-8-sig')
test_processed.to_csv(test_processed_path, index=False, encoding='utf-8-sig')

print(f"训练集归一化后已保存至：{train_processed_path}")
print(f"测试集归一化后已保存至：{test_processed_path}")

# 6. 输出归一化参数（方便核对）
print("\n===== 归一化参数（训练集） =====")
for col, min_val, max_val in zip(numeric_cols, minmax_scaler.data_min_, minmax_scaler.data_max_):
    print(f"{col} - 最小值：{min_val:.4f}，最大值：{max_val:.4f}")
    print(f"{col} - 归一化公式：x_scaled = (x - {min_val:.4f}) / ({max_val:.4f} - {min_val:.4f})")