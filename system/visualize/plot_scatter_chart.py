
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 字体设置，参考plot_cifar_fr_comparison.py
preferred_fonts = ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"]
available_fonts = {f.name for f in font_manager.fontManager.ttflist}
font_name = next((f for f in preferred_fonts if f in available_fonts), "DejaVu Serif")
plt.rcParams.update({
    "font.family": font_name,
    "mathtext.fontset": "custom",
    "mathtext.rm": font_name,
    "mathtext.it": f"{font_name}:italic",
    "mathtext.bf": f"{font_name}:bold",
    "font.size": 20,
    "axes.titlesize": 30,
    "axes.labelsize": 25,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
})


x = [31.66, 23.52, 32.04, 30.27, 39.49, 42.56, 45.12]
y = [53.96, 34.79, 32.15, 28.64, 29.49, 3.75, 2.89]

# x = [53.80, 54.84, 54.58, 52.27, 62.25, 64.61, 67.14]
# y = [38.21, 31.64, 23.06, 22.95, 8.17, 3.62, 2.25]

labels = ["FedAvg", "per-FedAvg", "MetaVD", "pFedGP", "pFedBayes", "LR-BPFL", "FT-BPFL"]
colors = ["#2E86C1", "#EF767A", "#966b80", "#f6df05", "#48C0AA", "#456990", "#FF5733"]
markers = ['o', 'o', 'o', 'o', 'o', 'o', 'o']

fig, ax = plt.subplots(figsize=(7.5, 6))

for i in range(len(x)):
    ax.scatter(x[i], y[i], color=colors[i], marker=markers[i], s=350, label=labels[i] if labels[i] not in plt.gca().get_legend_handles_labels()[1] else "")

ax.axhline(y=y[-1], color="#FF5733", linestyle='--', linewidth=2)
ax.axvline(x=x[-1], color="#FF5733", linestyle='--', linewidth=2)

ax.set_xlabel("Accuracy (%)", size=25)
ax.set_ylabel("Expected Calibration Error (%)", size=25)

ax.tick_params(axis='x', labelsize=20)
ax.tick_params(axis='y', labelsize=20)

plt.ylim(0, 55)

plt.text(0.5, 1.02, "Noisy CIFAR-10 2/10", fontname=font_name, horizontalalignment='center', fontsize=25, transform=ax.transAxes)
# 加粗FT-BPFL图例
handles, legend_labels = ax.get_legend_handles_labels()
from matplotlib.legend import Legend
legend = ax.legend(handles, legend_labels, fontsize=16, loc='lower left', frameon=True)
for text, label in zip(legend.get_texts(), legend_labels):
    if label == "FT-BPFL":
        text.set_weight('bold')

plt.xticks(np.arange(20, 50, 5))

ax.grid(True, which='major', axis='both', linestyle='--', linewidth=0.5, alpha=0.5)

plt.savefig("scatter_chart_noise.pdf", dpi=300, bbox_inches="tight", pad_inches=0.02)
plt.show()
print("Saved plot to: scatter_chart.pdf")
