import numpy as np
import matplotlib.pyplot as plt

# 示例数据
categories = ['CIFAR-10(2/10 labels)', 'CIFAR-10(5/10 labels)']  # 每簇的类别
group1_values = [2.54, 3.02]  # 第一簇数据
group2_values = [5.22, 6.98]
group3_values = [19.36, 23.53]
group4_values = [20.86, 25.29] # 第二簇数据

# 设置X轴的位置
x = np.array([0.4, 0.7])  # 为每个类别创建X轴位置
width = 0.04  # 每个柱子的宽度

# 创建柱状图
plt.figure(figsize=(8, 6))
plt.bar(x - 1.5 * width, group1_values, width, label='Ours', color='#2E86C1', edgecolor='black')
plt.bar(x - 0.5 * width, group2_values, width, label='pFedBayes', color='#EF767A',edgecolor='black')
plt.bar(x + 0.5 * width, group3_values, width, label='pFedMe', color='#456990', edgecolor='black')  # 第一簇
plt.bar(x + 1.5 * width, group4_values, width, label='FedAvg', color='#48C0AA', edgecolor='black')  # 第二簇

# 添加标题和标签
plt.xlabel('Dataset', fontsize=14)
plt.ylabel('ECE(%)', fontsize=14)

# 为X轴添加类别标签
plt.xticks(x, categories)

# 显示图例
plt.legend()

# 显示网格
plt.grid(axis='y', linestyle='--', alpha=0.7)

# 显示图表
plt.show()