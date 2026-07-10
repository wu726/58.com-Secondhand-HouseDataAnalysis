import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
import os

# 1. 配置文件路径
input_path = r"D:\项目\processed_data\武汉市二手房数据_清洗后.csv"
output_dir = r"D:\项目\模型预测\数据集"
train_output_path = os.path.join(output_dir, "train_data.csv")
test_output_path = os.path.join(output_dir, "test_data.csv")

# 2. 自动创建输出文件夹
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"输出文件夹 {output_dir} 不存在，已自动创建")

# 定义特征计算函数（剔除采光效果，仅保留房龄+户型延伸特征）
def calculate_features(df_input):
    """
    计算房龄+户型延伸特征
    :param df_input: 输入DataFrame（训练集/测试集）
    :return: 含新增特征的DataFrame
    """
    df = df_input.copy()
    
    # 1. 预处理基础字段
    # 建造年份预处理（房龄计算）
    df['建造年份'] = pd.to_numeric(df['建造年份'], errors='coerce')
    # 室/厅/卫/面积预处理（确保数值类型）
    for col in ['室', '厅', '卫', '面积(㎡)']:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)  # 空值填充0，避免计算错误

    # 2. 计算房龄特征
    base_year = 2026
    df['房龄'] = base_year - df['建造年份']
    
    # 3. 新增户型延伸特征
    # 卫室比：卫数/室数（反映舒适度，分母为0时填0）
    df['卫室比'] = np.where(df['室'] != 0, df['卫'] / df['室'], 0)
    # 厅室比：厅数/室数（反映公共空间，分母为0时填0）
    df['厅室比'] = np.where(df['室'] != 0, df['厅'] / df['室'], 0)
    # 房间总数：室+厅+卫（反映户型复杂度）
    df['房间总数'] = df['室'] + df['厅'] + df['卫']
    # 功能区密度：房间总数/面积(㎡)（反映紧凑度，面积为0时填0）
    df['功能区密度'] = np.where(df['面积(㎡)'] != 0, df['房间总数'] / df['面积(㎡)'], 0)

    # 4. 清理异常值（避免无穷大/负数）
    for col in ['卫室比', '厅室比', '功能区密度']:
        df[col] = df[col].replace([np.inf, -np.inf], 0)  # 替换无穷大
        df[col] = df[col].apply(lambda x: max(x, 0))  # 确保非负

    # 5. 打印特征统计信息（验证计算结果）
    valid_age_count = df['房龄'].notna().sum()
    invalid_age_count = df['房龄'].isna().sum()
    print(f"\n房龄计算完成：")
    print(f"有效房龄记录数：{valid_age_count}")
    print(f"无效/未知建造年份记录数：{invalid_age_count}")
    if valid_age_count > 0:
        print(f"房龄范围：{df['房龄'].min():.0f} - {df['房龄'].max():.0f} 年")
    
    print(f"\n户型延伸特征统计：")
    print(f"卫室比均值：{df['卫室比'].mean():.2f}")
    print(f"厅室比均值：{df['厅室比'].mean():.2f}")
    print(f"房间总数均值：{df['房间总数'].mean():.1f}")
    print(f"功能区密度均值：{df['功能区密度'].mean():.3f}")

    # 6. 剔除所有价格类新增特征
    drop_cols = ['小区平均价', '区域均价(万/㎡)', '装修溢价率', '朝向溢价率', '楼层溢价率']
    df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors='ignore')

    return df

try:
    # 3. 读取去重后的基础数据
    df = pd.read_csv(input_path, encoding='utf-8-sig')
    print(f"✅ 成功读取数据文件，总数据行数：{len(df)}")

    # 删除单价列
    target_col = '单价(元/㎡)'
    if target_col in df.columns:
        df = df.drop(columns=[target_col])
        print(f"✅ 已删除 '{target_col}' 列")
    else:
        print(f"⚠️  提示：数据中未找到 '{target_col}' 列，无需删除")
    
    # 检查分层字段是否存在
    stratify_cols = ['区域', '朝向', '楼层']
    missing_cols = [col for col in stratify_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"数据中缺失分层所需字段：{missing_cols}")

    # 4. 构建分层标识
    df['stratify_key'] = df[stratify_cols].apply(lambda x: '_'.join(x.astype(str)), axis=1)
    
    # 检查分层标识的样本量
    stratify_count = df['stratify_key'].value_counts()
    small_groups = stratify_count[stratify_count < 2].index.tolist()
    if small_groups:
        print(f"⚠️  警告：以下分层组合样本量小于2，可能影响抽样效果：")
        print(f"   {small_groups}")
        df = df[~df['stratify_key'].isin(small_groups)]
        print(f"   移除后剩余数据行数：{len(df)}")

    # 5. 分层抽样（7:3划分）
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    for train_idx, test_idx in sss.split(df, df['stratify_key']):
        train_df_base = df.iloc[train_idx].drop(columns=['stratify_key'])
        test_df_base = df.iloc[test_idx].drop(columns=['stratify_key'])

    # 6. 分别计算训练集/测试集的特征（剔除采光效果）
    print("\n========== 计算训练集特征 ==========")
    train_df = calculate_features(train_df_base)
    print("\n========== 计算测试集特征 ==========")
    test_df = calculate_features(test_df_base)

    # 7. 定义最终输出字段顺序（剔除采光效果字段）
    output_columns = [
        '小区名称', '区域','户型', '装修类型', '面积(㎡)', '朝向', '楼层', '建造年份', '总价(万)',
        '总楼层', '室', '厅', '卫', '房龄',  # 原有字段
        '卫室比', '厅室比', '房间总数', '功能区密度'  # 新增字段（剔除采光效果）
    ]
    
    # 确保字段存在并按顺序保存
    for col in output_columns:
        if col not in train_df.columns:
            train_df[col] = pd.NA
        if col not in test_df.columns:
            test_df[col] = pd.NA

    train_df_final = train_df[output_columns]
    test_df_final = test_df[output_columns]

    # 8. 保存训练集和测试集
    train_df_final.to_csv(train_output_path, index=False, encoding='utf-8-sig')
    test_df_final.to_csv(test_output_path, index=False, encoding='utf-8-sig')

    # 9. 打印划分结果（更新新增特征说明）
    print(f"\n📊 数据集划分+特征计算完成：")
    print(f"   训练集行数：{len(train_df_final)}（占比 {len(train_df_final)/(len(train_df_final)+len(test_df_final)):.2%}）")
    print(f"   测试集行数：{len(test_df_final)}（占比 {len(test_df_final)/(len(train_df_final)+len(test_df_final)):.2%}）")
    print(f"   训练集保存路径：{train_output_path}")
    print(f"   测试集保存路径：{test_output_path}")
    print(f"   最终新增特征：房龄、卫室比、厅室比、房间总数、功能区密度")

    # 验证分层字段分布一致性
    print("\n📈 分层字段分布验证（前5个组合）：")
    train_dist = train_df_base[stratify_cols].apply(lambda x: '_'.join(x.astype(str)), axis=1)
    test_dist = test_df_base[stratify_cols].apply(lambda x: '_'.join(x.astype(str)), axis=1)
    compare_df = pd.DataFrame({
        '训练集占比': train_dist.value_counts(normalize=True).head(5),
        '测试集占比': test_dist.value_counts(normalize=True).head(5)
    })
    print(compare_df)

except FileNotFoundError:
    print(f"❌ 错误：未找到文件 {input_path}，请检查文件路径是否正确")
except ValueError as e:
    print(f"❌ 数据验证错误：{e}")
except Exception as e:
    print(f"❌ 处理过程中出现错误：{str(e)}")
    import traceback
    traceback.print_exc()