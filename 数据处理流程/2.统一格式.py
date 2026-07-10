import pandas as pd
import re
import os  # 新增：导入os模块用于文件夹操作

# 读取CSV文件
file_path = r"D:\项目\data\东湖高新区二手房数据.csv"
df = pd.read_csv(file_path, encoding='utf-8')  # 编码报错可换为 encoding='gbk'

# 1. 处理户型字段，拆分出真正的户型和总楼层
def extract_house_info(text):
    """
    从混合的户型文本中提取 户型（如3室2厅）和总楼层（如32）
    """
    if pd.isna(text):
        return "", ""
    
    # 提取总楼层：匹配 (共XX层) 中的数字
    floor_pattern = re.compile(r'共(\d+)层')
    floor_match = floor_pattern.search(text)
    total_floor = floor_match.group(1) if floor_match else ""
    
    # 提取户型：匹配 X室X厅 格式
    house_pattern = re.compile(r'(\d+室\d+厅)')
    house_match = house_pattern.search(text)
    house_type = house_match.group(1) if house_match else ""
    
    return house_type, total_floor

# 应用函数拆分字段
df['户型_clean'], df['总楼层'] = zip(*df['户型'].apply(extract_house_info))

# 2. 删除原始的户型字段
df = df.drop(columns=['户型'])

# 3. 重命名字段并调整顺序
# 字段映射：原字段名 -> 新字段名
field_mapping = {
    '小区名称': '小区名称',
    '区域': '区域',
    '装修情况': '装修类型',
    '户型_clean': '户型',  
    '面积(㎡)': '面积(㎡)',
    '朝向': '朝向',
    '楼层': '楼层',
    '建造年份': '建造年份',
    '总价(万)': '总价(万)',
    '单价(元/㎡)': '单价(元/㎡)',
    '总楼层': '总楼层'
}

# 按指定顺序筛选并重命名字段
new_columns = [
    '小区名称', '区域', '装修类型', '户型', '面积(㎡)', 
    '朝向', '楼层', '建造年份', '总价(万)', '单价(元/㎡)', '总楼层'
]
df_new = df.rename(columns=field_mapping)[new_columns]

# 4. 保存处理后的数据
output_path = r"D:\作业\统一格式\东湖高新区二手房数据_格式统一.csv"
# 提取保存路径的文件夹部分
output_dir = os.path.dirname(output_path)

# 检查文件夹是否存在，不存在则创建（exist_ok=True 避免文件夹已存在时报错）
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"文件夹 {output_dir} 不存在，已自动创建")

# 保存文件
df_new.to_csv(output_path, index=False, encoding='utf-8-sig')
print(f"数据已成功保存至：{output_path}")

# 打印前5行预览结果
print("\n处理后的数据预览：")
print(df_new.head())