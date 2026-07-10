import pandas as pd
import os

# 1. 定义需要合并的文件列表
input_files = [
    r"D:\项目\标准化格式\东湖高新区二手房数据_标准格式.csv",
    r"D:\项目\标准化格式\洪山区二手房数据_标准格式.csv",
    r"D:\项目\标准化格式\江岸区二手房数据_标准格式.csv",
    r"D:\项目\标准化格式\江汉区二手房数据_标准格式.csv",
    r"D:\项目\标准化格式\江夏区二手房数据_标准格式.csv",
    r"D:\项目\标准化格式\武昌区二手房数据_标准格式.csv"
]

# 2. 定义输出路径和文件名
output_dir = r"D:\项目\processed_data"
output_file = r"武汉市二手房数据.csv"
output_path = os.path.join(output_dir, output_file)

# 3. 自动创建输出文件夹（不存在则创建）
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"输出文件夹 {output_dir} 不存在，已自动创建")                                                                                                                                                                                                                                                                                                                                                            

# 4. 批量读取并合并文件
df_list = []  # 存储每个文件的DataFrame
total_rows = 0  # 统计总数据行数

for file_path in input_files:
    try:
        # 读取文件
        df = pd.read_csv(file_path, encoding='utf-8-sig')
        district_name = os.path.basename(file_path).split('二手房数据')[0]  
        df_rows = len(df)
        total_rows += df_rows
        
        df_list.append(df)
        print(f"✅ 成功读取：{os.path.basename(file_path)} | 数据行数：{df_rows}")
    
    except Exception as e:
        print(f"❌ 读取文件 {file_path} 失败：{str(e)}")

# 5. 合并所有DataFrame
if df_list:
    # 按列合并（确保字段顺序一致，缺失字段填充空）
    merged_df = pd.concat(df_list, ignore_index=True, sort=False)
    
    # 保存合并后的文件
    merged_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    print(f"\n📊 合并完成！")
    print(f"   合并文件数量：{len(df_list)}")
    print(f"   总数据行数：{total_rows}")
    print(f"   合并后文件保存路径：{output_path}")
else:
    print("\n❌ 无有效文件可合并！")