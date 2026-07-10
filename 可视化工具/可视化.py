import pandas as pd
import matplotlib.pyplot as plt
import warnings
import os
warnings.filterwarnings('ignore')

# 解决中文显示
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 微软雅黑（Windows自带）
plt.rcParams['axes.unicode_minus'] = False            # 负号正常显示

# 文件夹与数据读取
save_dir = r"D:\项目\processed_data\数据可视化"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# 读取数据
df = pd.read_csv(r"D:\项目\processed_data\武汉市二手房数据_清洗后.csv", encoding='utf-8-sig')
area_price = df.groupby('区域').agg({
    '单价(元/㎡)':'mean',
    '总价(万)':'mean'
}).round(2).sort_values('单价(元/㎡)', ascending=False)

# -------------------------- 1. 各行政区平均单价（独立图） --------------------------
fig, ax = plt.subplots(figsize=(12,7))
x = area_price.index
bars = ax.bar(x, area_price['单价(元/㎡)'], color='#4287f5', alpha=0.7)
ax.twinx().plot(x, area_price['单价(元/㎡)'], color='red', marker='o', linewidth=2)
ax.set_title('武汉市各行政区二手房平均单价', fontsize=14, fontweight='bold')
ax.set_xlabel('行政区')
ax.set_ylabel('单价（元/㎡）')
ax.tick_params(axis='x', rotation=45)
# 数值标签
for bar in bars:
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+100, f'{bar.get_height():.0f}', ha='center')
plt.tight_layout()
plt.savefig(os.path.join(save_dir, '各行政区平均单价.png'), dpi=300, bbox_inches='tight')
plt.show()

# -------------------------- 2. 各行政区平均总价（独立图） --------------------------
fig, ax = plt.subplots(figsize=(12,7))
bars = ax.bar(x, area_price['总价(万)'], color='#f5a642', alpha=0.7)
ax.twinx().plot(x, area_price['总价(万)'], color='green', marker='s', linewidth=2)
ax.set_title('武汉市各行政区二手房平均总价', fontsize=14, fontweight='bold')
ax.set_xlabel('行政区')
ax.set_ylabel('总价（万）')
ax.tick_params(axis='x', rotation=45)
# 数值标签
for bar in bars:
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+2, f'{bar.get_height():.1f}', ha='center')
plt.tight_layout()
plt.savefig(os.path.join(save_dir, '各行政区平均总价.png'), dpi=300, bbox_inches='tight')
plt.show()

# -------------------------- 3. 房源特征饼图（修正autopct_params错误） --------------------------
features = {
    '户型':'户型分布占比',
    '装修类型':'装修类型分布占比',
    '楼层':'楼层分布占比',
    '朝向':'朝向分布占比'
}
colors = ['#ff6b6b','#4ecdc4','#45b7d1','#96ceb4','#ffeaa7']

for feature, title in features.items():
    # 统计数量并合并小类别
    count = df[feature].value_counts()
    threshold = count.sum() * 0.01  # 1%阈值
    small_cats = count[count < threshold]
    main_cats = count[count >= threshold]
    if not small_cats.empty:
        main_cats['其他'] = small_cats.sum()
    
    # 绘制饼图（删除错误的autopct_params参数）
    fig, ax = plt.subplots(figsize=(10,8))
    wedges, texts, autotexts = ax.pie(
        main_cats.values,
        labels=main_cats.index,
        autopct='%1.1f%%',  # 仅保留autopct参数显示百分比
        colors=colors[:len(main_cats)],
        startangle=90,
        wedgeprops={'edgecolor':'white', 'linewidth':2}  # 扇区白色边框
    )
    
    # 单独设置百分比文字样式
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(11)
        autotext.set_fontweight('bold')
    
    ax.set_title(f'武汉市二手房{title}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{title}.png'), dpi=300, bbox_inches='tight')
    plt.show()

# -------------------------- 4. 建筑年代与单价（散点图） --------------------------
build_df = df.dropna(subset=['建造年份','单价(元/㎡)'])
fig, ax = plt.subplots(figsize=(14,8))
ax.scatter(build_df['建造年份'], build_df['单价(元/㎡)'], c='#219ebc', alpha=0.6, s=40)
ax.set_title('建筑年代与二手房单价关系（散点图）', fontsize=14, fontweight='bold')
ax.set_xlabel('建造年份')
ax.set_ylabel('单价（元/㎡）')
ax.grid(alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig(os.path.join(save_dir, '建筑年代与单价_散点图.png'), dpi=300, bbox_inches='tight')
plt.show()

