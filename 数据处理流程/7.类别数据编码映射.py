import pandas as pd
import os
import json

# -------------------------- 1. 路径定义 --------------------------
train_path = r"D:\项目\模型预测\数据集\train_data.csv"
test_path = r"D:\项目\模型预测\数据集\test_data.csv"
mapping_save_dir = r"D:\项目\模型预测\映射数据集"
pure_mapping_path = os.path.join(mapping_save_dir, "类别变量编码映射表.json")
encoded_train_path = os.path.join(mapping_save_dir, "train_data_编码后.csv")
encoded_test_path = os.path.join(mapping_save_dir, "test_data_编码后.csv")

os.makedirs(mapping_save_dir, exist_ok=True)
print(f"📁 输出目录：{mapping_save_dir}")

# -------------------------- 2. 通用函数 --------------------------
def read_csv_file(file_path):
    try:
        df = pd.read_csv(file_path, encoding='utf-8-sig')
        df = df.dropna(how='all')
        print(f"✅ 成功读取：{file_path}（utf-8-sig）")
        return df
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, encoding='gbk')
        df = df.dropna(how='all')
        print(f"✅ 成功读取：{file_path}（gbk）")
        return df
    except FileNotFoundError as e:
        print(f"❌ 错误：{file_path} 不存在，{e}")
        return None

def save_pure_mapping(mapping_dict, save_path):
    if not mapping_dict:
        print("⚠️  映射表为空，跳过保存")
        return
    try:
        with open(save_path, 'w', encoding='utf-8-sig') as f:
            json.dump(mapping_dict, f, ensure_ascii=False, indent=4)
        print(f"✅ 纯映射表已按指定格式保存至：{save_path}")
    except Exception as e:
        print(f"❌ 保存映射表失败：{e}")

def load_pure_mapping(load_path):
    try:
        with open(load_path, 'r', encoding='utf-8-sig') as f:
            data = json.load(f)
        print(f"✅ 纯映射表已加载：{load_path}")
        return data
    except FileNotFoundError:
        print(f"❌ 映射表不存在：{load_path}")
        return {}
    except json.JSONDecodeError:
        print(f"❌ 映射表格式错误：{load_path}")
        return {}

# -------------------------- 3. 核心：区域编码（价格越高，编码值越大） --------------------------
def encode_region_by_price_median(train_df, region_col="区域", label_col="总价(万)"):
    """
    最终版区域编码规则：
    1. 计算每个区域的总价中位数，按降序排序
    2. 计算相邻区域价格差，按价格差大小分配间隔（差越大，间隔越大）
    3. 倒序累加编码值（从最低价区域开始，让最高价区域编码值最大）
    """
    # 过滤无效数据（总价≤0/空值、区域空值）
    valid_df = train_df[(train_df[label_col] > 0) & (train_df[label_col].notna()) & (train_df[region_col].notna())]
    if len(valid_df) == 0:
        print("⚠️  无有效区域/总价数据，区域按原始排序编码")
        train_valid_region = train_df[region_col].dropna().unique()
        unique_regions = sorted(train_valid_region)
        return {val: idx + 1 for idx, val in enumerate(unique_regions)}
    
    # 步骤1：计算各区域总价中位数并降序排序
    region_median = valid_df.groupby(region_col)[label_col].median().reset_index()
    region_median.columns = [region_col, "区域总价中位数"]
    region_median_sorted = region_median.sort_values("区域总价中位数", ascending=False).reset_index(drop=True)
    print("\n=== 区域总价中位数排序（降序）===")
    for _, row in region_median_sorted.iterrows():
        print(f"{row[region_col]} | 中位数：{row['区域总价中位数']:.2f}万")
    
    # 步骤2：计算相邻区域价格差（当前区域 - 下一个区域）
    region_median_sorted["价格差"] = region_median_sorted["区域总价中位数"].diff(-1).fillna(0)
    # 计算价格差分位数，划分间隔阈值
    price_diff_quantiles = region_median_sorted["价格差"].quantile([0.33, 0.66]).tolist()
    q33, q66 = price_diff_quantiles
    print(f"\n=== 区域价格差分位数（用于动态间隔）===")
    print(f"33分位：{q33:.2f}万 | 66分位：{q66:.2f}万")
    
    # 步骤3：按价格差分配间隔（差越大，间隔越大）
    def get_encode_gap(price_diff):
        if price_diff >= q66:  # 大价格差
            return 3
        elif price_diff >= q33:  # 中价格差
            return 2
        else:  # 小价格差
            return 1
    region_median_sorted["编码间隔"] = region_median_sorted["价格差"].apply(get_encode_gap)
    
    # 步骤4：倒序累加编码值（核心修改：让高价区编码值最大）
    region_median_sorted["编码值"] = 0
    current_code = 1
    # 反转数据，从最低价区域（最后一行）开始累加
    reversed_df = region_median_sorted.iloc[::-1].reset_index(drop=True)
    for idx, row in reversed_df.iterrows():
        reversed_df.loc[idx, "编码值"] = current_code
        current_code += row["编码间隔"]
    # 再反转回来，恢复原排序（高价区在前，编码值最大）
    region_median_sorted["编码值"] = reversed_df["编码值"].iloc[::-1].values
    
    # 输出最终编码结果
    print("\n=== 最终区域编码结果（价格越高，编码值越大）===")
    for _, row in region_median_sorted.iterrows():
        print(f"{row[region_col]} | 编码值：{row['编码值']} | 间隔：{row['编码间隔']} | 与下一区价格差：{row['价格差']:.2f}万")
    
    return dict(zip(region_median_sorted[region_col], region_median_sorted["编码值"]))

# -------------------------- 4. 读取数据 --------------------------
train_df = read_csv_file(train_path)
test_df = read_csv_file(test_path)
if train_df is None or test_df is None:
    exit()
print("\n⚠️  编码仅替换原列，原始数据不会修改")

# -------------------------- 5. 训练集编码 --------------------------
pure_mapping = {}

# 1. 朝向编码（固定映射）
if '朝向' in train_df.columns:
    pure_mapping["朝向"] = {
        "东": 1, "南": 2, "西": 3, "北": 4, "东南": 5,
        "东北": 6, "西南": 7, "西北": 8, "东西": 9, "南北": 10
    }
    train_df['朝向'] = train_df['朝向'].map(pure_mapping["朝向"])
    print("✅ 训练集'朝向'已编码（固定映射）")

# 2. 户型编码
if '户型' in train_df.columns:
    train_valid_house = train_df['户型'].dropna().unique()
    unique_house_types = sorted(train_valid_house)
    house_mapping = {val: idx + 1 for idx, val in enumerate(unique_house_types)}
    house_mapping["未知户型"] = 0
    pure_mapping["户型"] = house_mapping
    train_df['户型'] = train_df['户型'].map({k:v for k,v in house_mapping.items() if k != "未知户型"})
    print(f"✅ 训练集'户型'已编码：{len(unique_house_types)}个已知户型，未知户型编码为0")

# 3. 区域编码（价格越高，编码值越大）
if '区域' in train_df.columns:
    if '总价(万)' in train_df.columns:
        region_mapping = encode_region_by_price_median(train_df)
    else:
        train_valid_region = train_df['区域'].dropna().unique()
        unique_regions = sorted(train_valid_region)
        region_mapping = {val: idx + 1 for idx, val in enumerate(unique_regions)}
        print("⚠️  训练集无'总价(万)'列，区域按原始排序编码")
    pure_mapping["区域"] = region_mapping
    train_df['区域'] = train_df['区域'].map(region_mapping)
    print(f"✅ 训练集'区域'已按总价中位数编码完成（价格越高，编码值越大）")

# 4. 装修类型编码
if '装修类型' in train_df.columns:
    train_valid_decoration = train_df['装修类型'].dropna().unique()
    unique_decorations = sorted(train_valid_decoration)
    decoration_mapping = {val: idx + 1 for idx, val in enumerate(unique_decorations)}
    pure_mapping["装修类型"] = decoration_mapping
    train_df['装修类型'] = train_df['装修类型'].map(decoration_mapping)
    print(f"✅ 训练集'装修类型'已编码：{len(unique_decorations)}个已知类型")

# 5. 楼层编码
if '楼层' in train_df.columns:
    train_valid_floor = train_df['楼层'].dropna().unique()
    unique_floors = sorted(train_valid_floor)
    floor_mapping = {val: idx + 1 for idx, val in enumerate(unique_floors)}
    pure_mapping["楼层"] = floor_mapping
    train_df['楼层'] = train_df['楼层'].map(floor_mapping)
    print(f"✅ 训练集'楼层'已编码：{len(unique_floors)}个已知楼层")

# -------------------------- 6. 保存纯映射表 --------------------------
save_pure_mapping(pure_mapping, pure_mapping_path)

# -------------------------- 7. 测试集编码 --------------------------
loaded_mapping = load_pure_mapping(pure_mapping_path)
if not loaded_mapping:
    print("❌ 映射表加载失败，跳过测试集编码")
else:
    print("\n=== 测试集编码（仅复用训练集映射）===")
    for col_name, mapping in loaded_mapping.items():
        if col_name in test_df.columns:
            if col_name == "户型":
                test_df[col_name] = test_df[col_name].map({k:v for k,v in mapping.items() if k != "未知户型"}).fillna(0)
                unknown_count = (test_df[col_name] == 0).sum()
            else:
                test_df[col_name] = test_df[col_name].map(mapping)
                unknown_count = test_df[col_name].isna().sum()
            print(f"✅ 测试集'{col_name}'：复用训练集映射，未知值数量={unknown_count}")

# -------------------------- 8. 保存编码后数据 --------------------------
try:
    train_df.to_csv(encoded_train_path, index=False, encoding='utf-8-sig')
    test_df.to_csv(encoded_test_path, index=False, encoding='utf-8-sig')
    print(f"\n✅ 编码后训练集：{encoded_train_path}")
    print(f"✅ 编码后测试集：{encoded_test_path}")
except Exception as e:
    print(f"❌ 保存编码后数据失败：{e}")

# -------------------------- 9. 验证映射表格式 --------------------------
print("\n=== 验证映射表格式 ===")
final_mapping = load_pure_mapping(pure_mapping_path)
if final_mapping:
    print("1. 朝向映射（与指定格式一致）：")
    print(final_mapping.get("朝向", "无"))
    print("\n2. 户型映射（前5条+未知户型）：")
    house_map = final_mapping.get("户型", {})
    house_items = list(house_map.items())[:5] + [("未知户型", house_map.get("未知户型", 0))]
    print(dict(house_items))
    print("\n3. 区域映射（价格越高，编码值越大）：")
    print(final_mapping.get("区域", "无"))
    print("\n4. 装修类型映射：")
    print(final_mapping.get("装修类型", "无"))
    print("\n5. 楼层映射：")
    print(final_mapping.get("楼层", "无"))
else:
    print("❌ 映射表验证失败：无有效映射数据")