import csv
import math
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

# Paths
BASE_DIR = Path(__file__).resolve().parent
CSV_CIFAR10_2M = BASE_DIR / "summary_Cifar10-pat-2M.csv"
CSV_CIFAR10_5M = BASE_DIR / "summary_Cifar10-pat-5M.csv"
CSV_CIFAR100_5M = BASE_DIR / "summary_Cifar100-pat-5M.csv"
OUT_PATH = BASE_DIR / "fr_comparison_plot.pdf"

# Plot settings
preferred_fonts = ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"]
available_fonts = {f.name for f in font_manager.fontManager.ttflist}
font_name = next((f for f in preferred_fonts if f in available_fonts), "DejaVu Serif")
plt.rcParams.update({
    "font.family": font_name,
    "mathtext.fontset": "custom",
    "mathtext.rm": font_name,
    "mathtext.it": f"{font_name}:italic",
    "mathtext.bf": f"{font_name}:bold",
    "font.size": 14,
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
})

fig = plt.figure(figsize=(5, 4))
gs = fig.add_gridspec(3, 1, height_ratios=[1, 1, 1], hspace=0.18)
ax_top = fig.add_subplot(gs[0, 0])
ax_mid = fig.add_subplot(gs[1, 0], sharex=ax_top)
ax_bot = fig.add_subplot(gs[2, 0], sharex=ax_top)
axes = [ax_top, ax_mid, ax_bot]
# 断轴y范围可根据数据调整
ylims = [(87, 90), (73, 76), (66, 70)]

def load_summary(csv_path: Path):
    sl_vals, mean_vals, var_vals = [], [], []
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sl_vals.append(float(row["sl"]))
            mean_vals.append(float(row["mean"]))
            var_vals.append(float(row["variance"]))
    return sl_vals, mean_vals, var_vals

# 数据集配置
datasets = [
    ("CIFAR-10 2/10", CSV_CIFAR10_2M, "#1f77b4"),
    ("CIFAR-10 5/10", CSV_CIFAR10_5M, "#ff7f0e"),
    ("CIFAR-100 5/100", CSV_CIFAR100_5M, "#2ca02c"),
]

all_x = []
for label, csv_path, color in datasets:
    sl_vals, mean_vals, var_vals = load_summary(csv_path)
    std_vals = [math.sqrt(v) for v in var_vals]
    all_x.extend(sl_vals)
    for ax, (y0, y1) in zip(axes, ylims):
        ax.plot(sl_vals, mean_vals, linewidth=2, color=color, label=label)
        ax.fill_between(
            sl_vals,
            [m - s for m, s in zip(mean_vals, std_vals)],
            [m + s for m, s in zip(mean_vals, std_vals)],
            color=color,
            alpha=0.12,
        )
        ax.set_ylim(y0, y1)
        ax.grid(True, linestyle="--", alpha=0.3)

# X轴只在底部显示
x_min = math.floor(min(all_x) / 0.1) * 0.1
x_max = math.ceil(max(all_x) / 0.1) * 0.1
ax_bot.set_xticks(np.arange(x_min, x_max + 0.001, 0.1))
ax_bot.set_xlabel(r"$\lambda\times10^{-3}$")

ax_bot.set_xticklabels([f"{x:.1f}" for x in np.arange(x_min, x_max + 0.001, 0.1)])
ax_mid.set_ylabel(r"Test accuracy (%)")
ax_top.tick_params(labelbottom=False, bottom=False)
ax_mid.tick_params(labelbottom=False, bottom=False)
ax_top.spines["bottom"].set_visible(False)
ax_mid.spines["top"].set_visible(False)
ax_mid.spines["bottom"].set_visible(False)
ax_bot.spines["top"].set_visible(False)

def _add_diag(ax, where="bottom", d=0.015, lw=1.2):
    kwargs = dict(transform=ax.transAxes, color="black", clip_on=False, linewidth=lw)
    if where == "bottom":
        ax.plot((-d, +d), (-d, +d), **kwargs)
        ax.plot((1 - d, 1 + d), (-d, +d), **kwargs)
    else:
        ax.plot((-d, +d), (1 - d, 1 + d), **kwargs)
        ax.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

_add_diag(ax_top, "bottom")
_add_diag(ax_mid, "top")
_add_diag(ax_mid, "bottom")
_add_diag(ax_bot, "top")

# 图例放顶部
handles, labels = ax_top.get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    frameon=False,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.0),
    ncol=3,
    columnspacing=0.8,
    fontsize=12,
    bbox_transform=fig.transFigure,
)
fig.subplots_adjust(left=0.18, right=0.98, bottom=0.18, top=0.90)
fig.savefig(OUT_PATH, dpi=300, bbox_inches="tight", pad_inches=0.02)
print(f"Saved plot to: {OUT_PATH}")
