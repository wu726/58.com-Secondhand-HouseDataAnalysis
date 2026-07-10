import pandas as pd
import os

file_path = r"D:\项目\processed_data\武汉市二手房数据去重.csv"

# 读取数据
df = pd.read_csv(file_path, encoding='utf-8-sig')
# 统计户型类型及数量
type_count = df['户型'].value_counts().reset_index()
type_count.columns = ['户型类型', '数量']  # 重命名列
# 保存统计结果
output_name = os.path.basename(file_path).replace('.csv', '_户型统计.csv')
output_path = os.path.join(os.path.dirname(file_path), output_name)
type_count.to_csv(output_path, index=False, encoding='utf-8-sig')
print(f"✅ 统计完成：{output_path}")
