import pandas as pd
import os
import re

# 定义根目录（所有区域文件夹的上级目录）
root_dir = r"D:\项目"

# 1. 定义需要处理的5个目标区域（文件夹名+输出文件名对应）
# 格式：{"文件夹名": "输出文件名前缀"}
target_areas = {
    "洪山": "洪山区",
    "江岸": "江岸区",
    "江汉": "江汉区",
    "江夏": "江夏区",
    "武昌": "武昌区"
}

# 2. 固定4类装修类型（仅处理这4类，避免匹配其他无关文件）
target_decorations = ["豪华装修", "简单装修", "精装修", "毛坯"]

# 遍历每个目标区域，单独处理并保存
for area_folder, area_name in target_areas.items():
    # 拼接区域文件夹完整路径
    area_dir = os.path.join(root_dir, area_folder)
    
    # 检查文件夹是否存在
    if not os.path.isdir(area_dir):
        print(f"⚠️ 警告：未找到区域文件夹 {area_dir}，跳过该区域！")
        print("-" * 50)
        continue
    
    # 筛选当前区域下，属于4类装修类型的CSV文件
    csv_files = []
    for f in os.listdir(area_dir):
        # 仅匹配：包含4类装修类型 + 二手房数据 + CSV后缀的文件
        if f.endswith(".csv") and "二手房数据" in f:
            if any(deco in f for deco in target_decorations):
                csv_files.append(f)
    
    if not csv_files:
        print(f"⚠️ 警告：{area_folder} 文件夹内未找到4类装修类型的CSV文件，跳过！")
        print("-" * 50)
        continue
    
    # 临时列表存储当前区域的所有装修类型数据
    area_data = []
    
    # 遍历当前区域的目标CSV文件
    for csv_file in csv_files:
        file_path = os.path.join(area_dir, csv_file)
        
        # 从文件名提取装修类型（区域名已通过字典确定，避免文件名错误）
        pattern = r"[^_]+区_([^_]+装修|毛坯)_二手房数据\.csv"
        match = re.search(pattern, csv_file)
        
        if not match:
            print(f"⚠️ 警告：{area_name} - 文件 {csv_file} 命名格式不符，跳过！")
            continue
        
        decoration_type = match.group(1)  # 提取装修类型（如豪华装修/毛坯）
        
        # 校验提取的装修类型是否在4类范围内
        if decoration_type not in target_decorations:
            print(f"⚠️ 警告：{area_name} - {csv_file} 提取的装修类型[{decoration_type}]不在4类范围内，跳过！")
            continue
        
        try:
            # 读取CSV文件（兼容utf-8-sig/GBK编码，解决中文乱码）
            try:
                df = pd.read_csv(file_path, encoding="utf-8-sig")
            except:
                df = pd.read_csv(file_path, encoding="gbk")
            
            # 添加标准化的区域和装修类型字段
            df["区域"] = area_name
            df["装修类型"] = decoration_type
            
            # 添加到当前区域数据列表
            area_data.append(df)
            print(f"✅ {area_name} - 成功处理：{decoration_type}，数据量：{len(df)}条")
        
        except FileNotFoundError:
            print(f"❌ {area_name} - 错误：文件 {file_path} 不存在！")
        except Exception as e:
            print(f"❌ {area_name} - 处理文件 {csv_file} 出错：{str(e)}，跳过！")
    
    # 合并当前区域的所有装修类型数据并保存
    if area_data:
        area_merged_df = pd.concat(area_data, ignore_index=True)
        
        # 定义当前区域的输出文件路径（如D:\爬虫\洪山区二手房数据.csv）
        output_path = os.path.join(root_dir, f"{area_name}二手房数据.csv")
        
        # 保存当前区域的合并数据
        area_merged_df.to_csv(output_path, index=False, encoding="utf-8-sig")
        
        # 输出当前区域汇总信息
        print(f"\n📌 {area_name} 合并完成！")
        print(f"📊 数据总量：{len(area_merged_df)} 条")
        print(f"📁 保存路径：{output_path}")
        print(f"🔍 装修类型分布：")
        deco_count = area_merged_df["装修类型"].value_counts()
        for deco, count in deco_count.items():
            print(f"   - {deco}：{count} 条")
    else:
        print(f"⚠️ {area_name} 无有效4类装修数据，未生成文件！")
    
    print("-" * 50)

print("🎉 所有区域处理完成！")