import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
import pandas as pd

# 重新整理数据
models = ["VSVIO", "NASVIO", "ULVIO", "OURS"]
conditions = ["Visual degradation", "Inertial degradation", "All degradation"]
sequences = ["Seq 05", "Seq 07", "Seq 10"]
errors = [
    [4.24, 3.62, 8.14, 38.78, 55.84, 52.98, 25.51, 37.79, 31.48],
    [2.92, 5.01, 5.74, 31.69, 38.75, 23.02, 15.02, 19.54, 12.10],
    [10.93, 9.37, 13.57, 20.24, 21.36, 19.19, 10.74, 15.59, 8.94],
    [5.32, 4.28, 5.87, 5.18, 4.73, 6.83, 5.74, 5.20, 6.39],
]

# errors = [
#     [0.1136 ,0.0793 ,0.1285 ,0.0690 ,0.0702 ,0.0937 ,0.0955 ,0.1074 ,0.1285],
#     [0.0791 ,0.1178 ,0.0803 ,0.0545 ,0.0713 ,0.0671 ,0.0664 ,0.0913 ,0.0803],
#     [0.1695 ,0.2015 ,0.1755 ,0.0629 ,0.0695 ,0.0701 ,0.1484 ,0.1376 ,0.1755],
#     [0.0606 ,0.0535 ,0.0734 ,0.0517 ,0.0433 ,0.0691 ,0.0565 ,0.0458 ,0.0734 ],
# ]

# 创建 DataFrame
data = []
for model, error_list in zip(models, errors):
    for i, condition in enumerate(conditions):
        for j, sequence in enumerate(sequences):
            data.append([model, condition, sequence, error_list[i * 3 + j]])

df_user = pd.DataFrame(data, columns=["Model", "Condition", "Sequence", "Error"])

# # 绘制箱线图
# plt.figure(figsize=(12, 6))
# sns.boxplot(x='Condition', y='Error', hue='Model', data=df_user, palette='Set2')


# 方案 2: 采用 Violin Plot（小提琴图）
plt.figure(figsize=(8, 6))
sns.violinplot(x='Condition', y='Error', hue='Model', data=df_user, palette='Set2', split=True)



plt.grid(True)
plt.grid(False, axis='x')  
# 2. 在X轴项与项之间添加灰色竖线
ax = plt.gca()  # 获取当前坐标轴
ax.set_xticks(range(len(conditions)))  # 设定刻度
ax.set_xticklabels(conditions)  # 设定刻度标签
for i in range(1, len(conditions)):  # 在X轴项与项之间画线
    plt.axvline(i - 0.5, color='gray', linestyle='--', linewidth=0.8)
# 添加图例和标签
plt.legend(title='Models', loc='upper left')
plt.tight_layout()
plt.savefig('analysis.png', dpi=300)