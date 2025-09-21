import torch
import numpy as np
import matplotlib.pyplot as plt


x = [0, 0.1, 0.25, 0.5, 0.75, 1.0]
# ACC
# y1 = [64.59, 64.45, 63.99, 64.48, 63.53, 64.51]
# y2 = [82.92, 78.08, 56.14, 49.56, 40.86, 40.76]
# y3 = [86.48, 76.91, 58.31, 44.35, 39.25, 36.05]
# y4 = [88.20, 87.24, 75.18, 67.05, 60.88, 60.24]

# Avg ECE
# y1 = [21.65, 21.52, 21.99, 21.57, 22.04, 21.47]
y2 = [10.17, 10.35, 21.05, 24.45, 32.82, 29.67]
y3 = [6.08, 6.64, 6.54, 5.24, 5.56, 5.65]
y4 = [0.98, 1.54, 1.88, 2.01, 2.24, 2.09]

# Worst-client ECE
# y2 = [23.04, 22.59, 36.46, 33.68, 34.86, 36.64]
# y3 = [27.25, 30.31, 25.34, 23.34, 18.98, 19.97]
# y4 = [5.56, 6.84, 8.80, 7.98, 8.84, 8.11]


plt.figure(0, figsize=(8, 10))

# plt.plot(x, y1, marker='o', markersize=8, label='FedAvg', linewidth=2, color="#2E86C1")
plt.plot(x, y2, marker='o', markersize=8, label='per-FedAvg', linewidth=2, color="#EF767A")
plt.plot(x, y3, marker='o', markersize=8, label='pFedBayes', linewidth=2, color="#456990")
plt.plot(x, y4, marker='o', markersize=8, label='LR-BPFL', linewidth=2, color="#48C0AA")

plt.xlabel(r"$\alpha$", size=20)
plt.ylabel("Expected Calibration Error (%)", size=20)
# plt.ylabel("Accuracy(%)", size=25)

plt.xticks(size=20)
plt.yticks(size=20)

plt.rcParams["font.family"] = "Times New Roman"
plt.legend(fontsize=25, loc='center right')

plt.show()
