import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import re  # 新增：用于处理特殊字符
warnings.filterwarnings('ignore')

# --------------------------
# 1. 配置全局参数
# --------------------------
# 文件路径
FILE_PATH = r"D:\项目\processed_data\武汉市二手房数据去重.csv"
# 设置中文字体，解决图表中文显示问题
plt.rcParams["font.family"] = ["SimHei",]
plt.rcParams["axes.unicode_minus"] = False
# 异常值报告保存路径（可选）
REPORT_PATH = "数值特征异常值检测报告.csv"

# --------------------------
# 新增函数：处理文件名中的特殊字符
# --------------------------
def clean_filename(filename):
    """
    清理文件名中的特殊字符（Windows不允许：\/:*?"<>|）
    :param filename: 原始文件名
    :return: 清理后的安全文件名
    """
    # 替换特殊字符为下划线
    special_chars = r'[\\/:*?"<>|]'
    clean_name = re.sub(special_chars, '_', filename)
    # 移除连续的下划线，避免冗余
    clean_name = re.sub(r'_+', '_', clean_name)
    # 移除首尾下划线
    clean_name = clean_name.strip('_')
    return clean_name

# --------------------------
# 2. 核心函数：异常值检测
# --------------------------
def detect_numerical_outliers(df):
    """
    检测数值特征的异常值
    方法1：3σ原则（适用于正态分布特征）
    方法2：四分位数法（IQR，适用于非正态分布特征，更通用）
    """
    # 步骤1：筛选数值特征（排除字符串、日期等非数值列）
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    print(f"🔍 检测到数值特征：{numerical_cols}")
    print(f"📊 总计 {len(numerical_cols)} 个数值特征需要检测异常值\n")
    
    # 步骤2：初始化异常值报告
    outlier_report = []
    
    # 步骤3：逐个特征检测异常值
    for col in numerical_cols:
        # 跳过全为空的列
        if df[col].isnull().sum() == len(df):
            print(f"⚠️ 特征 {col} 全为空，跳过检测")
            continue
        
        # 基础统计信息
        data = df[col].dropna()  # 剔除缺失值后检测
        mean_val = data.mean()
        std_val = data.std()
        q1 = data.quantile(0.25)  # 下四分位数
        q3 = data.quantile(0.75)  # 上四分位数
        iqr = q3 - q1             # 四分位距
        min_normal = q1 - 1.5 * iqr  # 正常范围下限
        max_normal = q3 + 1.5 * iqr  # 正常范围上限
        
        # 方法1：3σ原则检测异常值
        sigma_outliers = data[(data < mean_val - 3 * std_val) | (data > mean_val + 3 * std_val)]
        # 方法2：IQR法检测异常值（更推荐）
        iqr_outliers = data[(data < min_normal) | (data > max_normal)]
        
        # 统计异常值信息
        outlier_info = {
            "特征名称": col,
            "样本总数": len(data),
            "缺失值数": df[col].isnull().sum(),
            "均值": round(mean_val, 2),
            "标准差": round(std_val, 2),
            "下四分位数(Q1)": round(q1, 2),
            "上四分位数(Q3)": round(q3, 2),
            "IQR正常范围": f"[{round(min_normal, 2)}, {round(max_normal, 2)}]",
            "3σ异常值数量": len(sigma_outliers),
            "3σ异常值占比(%)": round(len(sigma_outliers)/len(data)*100, 2),
            "IQR异常值数量": len(iqr_outliers),
            "IQR异常值占比(%)": round(len(iqr_outliers)/len(data)*100, 2),
            "IQR异常值示例": iqr_outliers.head(5).tolist()  # 前5个异常值示例
        }
        outlier_report.append(outlier_info)
        
        # 打印该特征的检测结果
        print(f"📈 特征 {col} 检测结果：")
        print(f"   - IQR法异常值数量：{len(iqr_outliers)} (占比 {round(len(iqr_outliers)/len(data)*100, 2)}%)")
        print(f"   - 3σ法异常值数量：{len(sigma_outliers)} (占比 {round(len(sigma_outliers)/len(data)*100, 2)}%)")
        print(f"   - 正常范围（IQR）：{round(min_normal, 2)} ~ {round(max_normal, 2)}\n")
        
        # 可视化：箱线图（直观展示异常值）
        plt.figure(figsize=(8, 4))
        plt.boxplot(data, patch_artist=True, boxprops={'facecolor': '#3498DB'})
        plt.title(f"{col} 数值分布（箱线图）- 异常值检测", fontsize=12, fontweight='bold')
        plt.ylabel("数值")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # 关键修复：清理文件名中的特殊字符
        clean_col_name = clean_filename(col)
        save_path = f"{clean_col_name}_异常值箱线图.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   - 箱线图已保存为：{save_path}")
    
    # 步骤4：生成异常值报告DataFrame
    report_df = pd.DataFrame(outlier_report)
    return report_df

# --------------------------
# 3. 主流程执行
# --------------------------
if __name__ == "__main__":
    # 读取数据
    print("📥 正在读取数据...")
    try:
        df = pd.read_csv(FILE_PATH, encoding='utf-8-sig')  # 优先用utf-8-sig，兼容中文
    except UnicodeDecodeError:
        df = pd.read_csv(FILE_PATH, encoding='gbk')  # 备用编码（中文Windows常见）
    print(f"✅ 数据读取完成，数据形状：{df.shape}\n")
    
    # 检测异常值
    print("🔍 开始检测数值特征异常值...\n")
    outlier_report = detect_numerical_outliers(df)
    
    # 保存异常值报告
    if not outlier_report.empty:
        outlier_report.to_csv(REPORT_PATH, index=False, encoding='utf-8-sig')
        print(f"\n✅ 异常值检测报告已保存至：{REPORT_PATH}")
    else:
        print("\n⚠️ 无有效数值特征，未生成报告")
    
    print("\n🎉 异常值检测完成！")                                                                    