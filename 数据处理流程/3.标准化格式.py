import pandas as pd
import re
import os
import random

# 定义文件路径配置
# 1. 常规处理的文件列表（江岸/江汉/江夏/武昌/洪山）
normal_files = [
    r"D:\项目\data\洪山区二手房数据.csv",
    r"D:\项目\data\江岸区二手房数据.csv",
    r"D:\项目\data\江汉区二手房数据.csv",
    r"D:\项目\data\江夏区二手房数据.csv",
    r"D:\项目\data\武昌区二手房数据.csv"
]

# 2. 特殊处理的文件（东湖高新区）
special_file = r"D:\项目\统一格式\东湖高新区二手房数据_格式统一.csv"

# 3. 输出文件夹
output_dir = r"D:\项目\标准化格式"

# 创建输出文件夹（不存在则创建）
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"输出文件夹 {output_dir} 不存在，已自动创建")

# 通用函数1：拆分楼层字段（提取楼层类型+总楼层）
def split_floor_info(floor_text):
    """
    处理楼层字段：
    - 楼层类型：仅保留中/低/高（纯"共X层"填充为低）
    - 总楼层：提取括号内或"共X层"中的数字
    """
    if pd.isna(floor_text):
        return "", ""
    
    # 提取总楼层（匹配 "共(\d+)层" 中的数字）
    floor_pattern = re.compile(r'共(\d+)层')
    floor_match = floor_pattern.search(str(floor_text))
    total_floor = floor_match.group(1) if floor_match else ""
    
    # 提取楼层类型（中/低/高）
    if re.search(r'中层', str(floor_text)):
        floor_type = "中"
    elif re.search(r'低层', str(floor_text)):
        floor_type = "低"
    elif re.search(r'高层', str(floor_text)):
        floor_type = "高"
    elif floor_match:  # 仅"共X层"格式，无楼层类型
        floor_type = "低"
    else:
        floor_type = ""
    
    return floor_type, total_floor

# --------------------------
# 通用函数2：常规户型拆分（提取室、厅、卫）
# --------------------------
def process_normal_house_type(df):
    """处理常规文件的户型字段，拆分室、厅、卫为数字列"""
    # 正则提取 室、厅、卫 的数字（兼容 X室X厅 / X室X厅X卫 格式）
    df[['室', '厅', '卫']] = (
        df['户型']
        .str.extract(r'(\d+)室(\d+)厅(?:(\d+)卫)?')  # 卫是可选分组
        .fillna(1)  # 无卫的字段填充1
        .astype(int)  # 转为整数类型
    )
    
    return df

# --------------------------
# 通用函数3：东湖高新区特殊户型处理（卫字段随机填充1/2）
# --------------------------
def process_special_house_type(df):
    """处理东湖高新区文件：拆分室、厅，卫字段随机填充1或2，并重构户型字段"""
    # 1. 先提取室、厅（保留原始户型字段，不删除）
    df[['室', '厅']] = (
        df['户型']
        .str.extract(r'(\d+)室(\d+)厅')
        .astype(int)
    )
    # 2. 卫字段随机填充1或2
    df['卫'] = [random.choice([1, 2]) for _ in range(len(df))]
    # 3. 基于室、厅、卫字段反向重构完善的户型字段
    df['户型'] = df['室'].astype(str) + '室' + df['厅'].astype(str) + '厅' + df['卫'].astype(str) + '卫'
    
    return df

# --------------------------
# 批量处理常规文件（洪山/江岸/江汉/江夏/武昌）
# --------------------------
# 最终保存的字段顺序
final_columns = [
    '小区名称', '区域', '户型','装修类型', '面积(㎡)', '朝向', 
    '楼层', '建造年份', '总价(万)', '单价(元/㎡)', '总楼层', '室', '厅', '卫'
]

for file_path in normal_files:
    try:
        # 1. 读取文件（utf-8-sig编码）
        df = pd.read_csv(file_path, encoding='utf-8-sig')
        
        # 2. 处理户型字段（拆分原户型）
        df = process_normal_house_type(df)
        
        # 3. 处理楼层字段（拆分楼层类型+总楼层）
        df['楼层'], df['总楼层'] = zip(*df['楼层'].apply(split_floor_info))
        
        # 4. 删除核心卖点字段
        df = df.drop(columns=['核心卖点'], errors='ignore')
        
        # 5. 重命名装修字段（确保字段名统一为"装修类型"）
        df = df.rename(columns={'装修情况': '装修类型'}, errors='ignore')
        
        # 6. 按指定顺序筛选字段（缺失字段填充空）
        for col in final_columns:
            if col not in df.columns:
                df[col] = ""
        df_processed = df[final_columns]
        
        # 7. 生成输出文件名并保存
        file_name = os.path.basename(file_path).replace('.csv', '_标准格式.csv')
        output_path = os.path.join(output_dir, file_name)
        df_processed.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"✅ 常规文件处理完成：{output_path}")
    
    except Exception as e:
        print(f"❌ 处理文件 {file_path} 失败：{str(e)}")

# 处理东湖高新区特殊文件
try:
    # 1. 读取文件
    df_special = pd.read_csv(special_file, encoding='utf-8-sig')
    
    # 2. 仅处理户型字段
    df_special = process_special_house_type(df_special)
    
    # 3. 仅重命名装修字段
    df_special = df_special.rename(columns={'装修情况': '装修类型'}, errors='ignore')
    
    # 4. 按指定顺序筛选字段（缺失字段填充空，楼层/总楼层保留原始值）
    for col in final_columns:
        if col not in df_special.columns:
            df_special[col] = ""
    df_special_processed = df_special[final_columns]
    
    # 5. 自定义文件名并保存
    special_output_path = os.path.join(output_dir, "东湖高新区二手房数据_标准格式.csv")
    df_special_processed.to_csv(special_output_path, index=False, encoding='utf-8-sig')
    print(f"✅ 特殊文件处理完成：{special_output_path}")

except Exception as e:
    print(f"❌ 处理特殊文件 {special_file} 失败：{str(e)}")

print("\n所有文件处理完成！")