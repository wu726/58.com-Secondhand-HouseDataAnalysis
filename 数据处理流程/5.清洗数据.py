import pandas as pd
import os
import traceback

# ===================== 配置项 =====================
# 输入输出文件路径
input_path = r"D:\项目\processed_data\武汉市二手房数据.csv"
output_path = r"D:\项目\processed_data\武汉市二手房数据_清洗后.csv"

#基础字段列表
output_columns = [
    '小区名称', '区域','户型','装修类型', '面积(㎡)', '朝向', '楼层', '建造年份', '总价(万)',
    '单价(元/㎡)', '总楼层', '室', '厅', '卫'
]

# ===================== 数据清洗函数 =====================
def clean_data(df):
    #第一步清洗：剔除包含"未知"或空值的行
    
    print("\n开始剔除缺失值/未知值")
    original_rows = len(df)
    
    # 1. 替换空字符串/空格为NaN，方便统一处理
    df = df.replace(['', ' ', '  '], pd.NA)
    
    # 2. 剔除包含"未知"的行（所有字段）
    df = df[~df.astype(str).apply(lambda x: x.str.contains('未知', na=False)).any(axis=1)]
    
    # 3. 剔除包含空值（NaN）的行（所有字段）
    df = df.dropna(how='any')
    
    cleaned_rows = len(df)
    deleted_rows = original_rows - cleaned_rows
    print(f"原始行数：{original_rows} | 剔除缺失值后行数：{cleaned_rows} | 剔除行数：{deleted_rows}")
    
    return df

def deduplicate_data(df):
    """
    第二步去重：删除完全重复的行，仅保留第一次出现的行
    """
    print("\n===== 开始数据去重 =====")
    # 重置索引，生成连续行号
    df = df.reset_index(drop=True)
    # 添加原始行号（方便标注重复行，从1开始）
    df['原始行号'] = df.index + 1
    original_cleaned_rows = len(df)
    print(f"去重前行数（已剔除缺失值）：{original_cleaned_rows}")

    # 标记重复行（排除原始行号字段）
    df['是否重复行'] = df.duplicated(subset=df.columns.drop(['原始行号']), keep='first')
    # 提取重复行信息
    duplicate_row_numbers = df[df['是否重复行']]['原始行号'].tolist()
    
    # 输出重复行详情
    if duplicate_row_numbers:
        print(f"检测到重复行的原始行号：{duplicate_row_numbers}")
        print(f"重复行数量：{len(duplicate_row_numbers)}")
        print("\n重复行具体内容（前10条）：")
        print(df[df['是否重复行']][['原始行号'] + list(df.columns.drop(['原始行号', '是否重复行']))].head(10))
    else:
        print("未检测到任何完全重复的行")
    
    # 执行去重
    df = df[~df['是否重复行']].drop(columns=['是否重复行'])
    deduplicated_rows = len(df)
    deleted_dup_rows = original_cleaned_rows - deduplicated_rows
    print(f"去重后剩余行数：{deduplicated_rows} | 删除的重复行数：{deleted_dup_rows}")
    
    return df

# ===================== 主处理逻辑 =====================
def main():
    try:
        # 1. 读取原始数据
        print(f"\n===== 开始读取数据 =====")
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"文件不存在：{input_path}")
        
        df = pd.read_csv(input_path, encoding='utf-8-sig')
        print(f"成功读取文件：{input_path}")
        print(f"原始数据总行数：{len(df)}")

        # 2. 第一步：剔除缺失值/未知值
        df_cleaned = clean_data(df)
        
        # 3. 第二步：数据去重
        df_deduplicated = deduplicate_data(df_cleaned)
        
        # 4. 整理输出字段并保存
        print("\n===== 开始保存最终数据 =====")
        # 确保所有输出字段存在（缺失字段填充为空）
        for col in output_columns:
            if col not in df_deduplicated.columns:
                df_deduplicated[col] = pd.NA
        
        # 删除辅助字段，按指定顺序保存
        df_final = df_deduplicated.drop(columns=['原始行号'], errors='ignore')[output_columns]
        df_final.to_csv(output_path, index=False, encoding='utf-8-sig')
        
        # 输出最终结果
        print(f"\n===== 数据处理完成！=====")
        print(f"最终保存行数：{len(df_final)}")
        print(f"文件保存路径：{output_path}")
        print(f"最终输出字段列表：{output_columns}")

    except FileNotFoundError as e:
        print(f"\n❌ 错误：{str(e)}，请检查文件路径是否正确")
    except ValueError as ve:
        print(f"\n❌ 参数错误：{str(ve)}")
    except Exception as e:
        print(f"\n❌ 处理过程中出现错误：{str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    main()                                                   