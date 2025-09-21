import torch
import numpy as np
import matplotlib.pyplot as plt

# 设置字体为 Times New Roman
plt.rcParams["font.family"] = "Times New Roman"

n_bins = 15  # Reliability diagram
bins = torch.linspace(0, 1, n_bins + 1)
width = 1.0 / n_bins
bin_centers = np.linspace(0, 1.0 - width, n_bins) + width / 2

bin1 = np.array([np.nan, np.nan, np.nan, 0, 0.1379, 0.2291, 0.2678, 0.3044, 0.3259, 0.3397, 0.3633, 0.3673,  0.3962, 0.4113, 0.6328])
bin2 = np.array([np.nan, np.nan, np.nan, np.nan, np.nan, 0, 0.45, 0.4821, 0.5173, 0.5326, 0.5219, 0.5318,  0.5213, 0.5582, 0.6888])
bin3 = np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 0, 0.5264, 0.5510, 0.5428, 0.5719, 0.6384,  0.6755, 0.7033, 0.8173])
bin4 = np.array([np.nan, np.nan, 0, 0.2733, 0.3429, 0.4462, 0.4984, 0.5358, 0.5815, 0.6509, 0.7115, 0.7913,  0.8456, 0.9134, 0.9686])

label1 = "FL (FedAvg)                                                 "
label2 = "PFL (per-FedAvg)                                            "
label3 = "BPFL (pFedBayes)                                              "
label4 = "LR-BPFL (Ours)                                               "

plt.figure(0, figsize=(8, 8))
plt.plot(bin_centers, bin1, marker='o', markersize=8, label=label1, linewidth=2, color="#2E86C1")
plt.plot(bin_centers, bin2, marker='o', markersize=8, label=label2, linewidth=2, color="#EF767A")
plt.plot(bin_centers, bin3, marker='o', markersize=8, label=label3, linewidth=2, color="#456990")
plt.plot(bin_centers, bin4, marker='o', markersize=8, label=label4, linewidth=2, color="#48C0AA")


plt.plot([0, 1], [0, 1], '--', color='grey')

plt.ylabel("Accuracy", size=20)
plt.xlabel("Confidence", size=20)
plt.xlim(0.15, 1)
plt.ylim(0, 1)

plt.xticks(size=20)
plt.yticks(size=20)

# 调整图例的左侧对齐
plt.legend(fontsize=15, loc='upper left')

plt.show()
