import numpy as np
import matplotlib.pyplot as plt

# 示例数据
categories = ['FL', "PFL", "BPFL", "LR-BPFL"]  # 每簇的类别
values = [1.22, 1.22, 2.44, 1.27]  # 第一簇数据

# 创建柱状图
plt.figure(figsize=(10, 6))
x = np.arange(len(categories))

# 通过减少宽度 (width) 来缩小柱子之间的间距
plt.bar(x, values, width=0.5, color=['#2E86C1', "#EF767A", "#456990", "#48C0AA"], edgecolor='black')

plt.xticks(x, categories)

# 添加标题和标签
plt.ylabel('Model size (MB)', fontsize=30)

plt.xticks(size=30)
plt.yticks(size=30)

# 设置字体为 Times New Roman
plt.rcParams["font.family"] = "Times New Roman"

# 显示网格
plt.grid(axis='y', linestyle='--', alpha=0.7)

# 显示图表
plt.show()