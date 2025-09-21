import torch
import numpy as np
import matplotlib.pyplot as plt


# x = [31.66, 23.52, 32.04, 30.27, 39.49, 42.56]
# y = [53.96, 34.79, 32.15, 28.64, 29.49, 3.75]

x = [53.80, 54.84, 54.58, 52.27, 62.25, 64.61]
y = [38.21, 31.64, 23.06, 22.95, 8.17, 3.62]

labels = ["FedAvg", "per-FedAvg", "MetaVD", "pFedGP", "pFedBayes", "Ours"]
colors = ["#2E86C1", "#EF767A", "#966b80", "#f6df05", "#48C0AA", "#456990"]
markers = ['o', 'o', 'o', 'o', 'o', 'o']

fig, ax = plt.subplots(figsize=(7, 10))
ax.set_facecolor('#f0f0f0')

for i in range(len(x)):
    ax.scatter(x[i], y[i], color=colors[i], marker=markers[i], s=350, label=labels[i] if labels[i] not in plt.gca().get_legend_handles_labels()[1] else "")

ax.axhline(y=y[-1], color="#456990", linestyle='--', linewidth=2)
ax.axvline(x=x[-1], color="#456990", linestyle='--', linewidth=2)

ax.set_xlabel("Accuracy (%)", size=25)
ax.set_ylabel("Expected Calibration Error (%)", size=25)

ax.tick_params(axis='x', labelsize=20)
ax.tick_params(axis='y', labelsize=20)

plt.ylim(0, 55)

plt.text(0.5, 1.02, "Original CIFAR-10 dataset", fontname='Microsoft YaHei',horizontalalignment='center', fontsize=30, transform=ax.transAxes)
# ax.legend(fontsize=25, loc='center left',  bbox_to_anchor=(0, 0.28))

plt.xticks(np.arange(51, 66, 3))

ax.grid(True, which='major', axis='both', linestyle='--', linewidth=0.5, alpha=0.5)

plt.show()
