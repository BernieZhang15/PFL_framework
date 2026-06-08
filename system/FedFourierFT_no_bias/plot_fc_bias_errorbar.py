import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

# Paths
BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "fc_bias.csv"
OUT_PATH = BASE_DIR / "fc_bias_errorbar.png"

# Plot settings (match plot_cifar10_fr.py)
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

# Expected order and labels
x_labels = ["Low", "Random", "High"]
sample_key_map = {
    "low": "Low",
    "rand": "Random",
    "random": "Random",
    "high": "High",
}

# Dataset colors and legend labels
series = [
    ("CIFAR-10 2/10", "#1f77b4"),
    ("CIFAR-10 5/10", "#ff7f0e"),
    ("CIFAR-100 5/100", "#2ca02c"),
]

# Load data
records = []
with CSV_PATH.open(newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        dataset = row["dataset"].strip()
        sample_raw = row["sample"].strip()
        sample = sample_key_map.get(sample_raw, sample_raw)
        acc_mean = float(row["acc_mean"]) * 1.0
        acc_std = float(row["acc_std"]) * 1.0
        records.append((dataset, sample, acc_mean, acc_std))

# Organize by dataset
by_dataset = {name: {} for name, _ in series}
for dataset, sample, acc_mean, acc_std in records:
    if dataset in by_dataset:
        by_dataset[dataset][sample] = (acc_mean, acc_std)

fig = plt.figure(figsize=(5, 4))
gs = fig.add_gridspec(3, 1, height_ratios=[1, 1, 1], hspace=0.18)
ax_top = fig.add_subplot(gs[0, 0])
ax_mid = fig.add_subplot(gs[1, 0], sharex=ax_top)
ax_bot = fig.add_subplot(gs[2, 0], sharex=ax_top)

axes = [ax_top, ax_mid, ax_bot]
ylims = [(85, 90), (72, 77), (64, 69)]


# Align points on the same x positions
x = np.arange(len(x_labels))
offsets = np.zeros(len(series))

for (dataset, color), dx in zip(series, offsets):
    means = []
    stds = []
    for label in x_labels:
        mean, std = by_dataset[dataset][label]
        means.append(mean)
        stds.append(std)

    for ax in axes:
        ax.errorbar(
            x + dx,
            means,
            yerr=stds,
            fmt="o-",
            markersize=5,
            linewidth=1.5,
            color=color,
            ecolor=color,
            elinewidth=1.5,
            capsize=3,
            label=dataset,
        )

for ax, (y0, y1) in zip(axes, ylims):
    ax.set_ylim(y0, y1)
    ax.grid(True, linestyle="--", alpha=0.3)

# Set y-ticks to match the three ranges
ax_bot.set_yticks([64, 66, 68])
ax_mid.set_yticks([72, 74, 76])
ax_top.set_yticks([85, 87, 89])

# X axis only on bottom subplot
ax_bot.set_xticks(x)
ax_bot.set_xticklabels(x_labels)
ax_bot.set_xlabel("Frequency selection")

# Y label on middle subplot
ax_mid.set_ylabel("Test accuracy(%)")

# Hide x tick labels on upper axes
ax_top.tick_params(labelbottom=False, bottom=False)
ax_mid.tick_params(labelbottom=False, bottom=False)

# Remove horizontal spines between broken axes
ax_top.spines["bottom"].set_visible(False)
ax_mid.spines["top"].set_visible(False)
ax_mid.spines["bottom"].set_visible(False)
ax_bot.spines["top"].set_visible(False)

# Add diagonal break marks
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

# Legend on top
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
