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
CSV_CIFAR100_5M = BASE_DIR.parent / "FedFourierFT_sweep_cifar100_5M" / "summary_Cifar100-pat-5M.csv"
OUT_PATH = BASE_DIR / "fr_comparison_plot.pdf"

# Model-specific frequency settings (from FTFedAvgCNN)
BASE_FREQ1, BASE_FREQ2, BASE_FREQ3 = 1024, 512, 256


def fr_to_params(fr: float) -> int:
    """Estimate trainable FourierFT parameters for a given fr.

    FourierFTLinear uses spectrum_mu and spectrum_rho, each of size n_frequency.
    n_frequency scales with fr.
    """
    n1 = max(1, int(BASE_FREQ1 * fr))
    n2 = max(1, int(BASE_FREQ2 * fr))
    n3 = max(1, int(BASE_FREQ3 * fr))
    return 2 * (n1 + n2 + n3)


def load_summary(csv_path: Path):
    fr_vals, mean_vals, std_vals = [], [], []
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fr_vals.append(float(row["fr"]))
            mean_vals.append(float(row["mean"]))
            std_vals.append(float(row["std"]))
    return fr_vals, mean_vals, std_vals


datasets = [
    ("CIFAR-10 2/10", CSV_CIFAR10_2M, "#1f77b4"),
    ("CIFAR-10 5/10", CSV_CIFAR10_5M, "#ff7f0e"),
    ("CIFAR-100 5/100", CSV_CIFAR100_5M, "#2ca02c"),
]

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

fig, ax = plt.subplots(figsize=(5, 4))

all_x = []
for label, csv_path, color in datasets:
    fr_vals, mean_vals, std_vals = load_summary(csv_path)
    x_params_k = [fr for fr in fr_vals]
    mean_pct = [m * 100.0 for m in mean_vals]
    std_pct = [s * 100.0 for s in std_vals]
    all_x.extend(x_params_k)

    ax.plot(x_params_k, mean_pct, linewidth=2, color=color, label=label)
    ax.fill_between(
        x_params_k,
        [m - s for m, s in zip(mean_pct, std_pct)],
        [m + s for m, s in zip(mean_pct, std_pct)],
        color=color,
        alpha=0.12,
    )

ax.set_xlabel(r"# parameters $\times 10^{3}$")
ax.set_ylabel(r"Test accuracy(%)")
# ax.set_title("FR Sweep Comparison")

# X ticks every 0.4
x_min = math.floor(min(all_x) / 0.4) * 0.4
x_max = math.ceil(max(all_x) / 0.4) * 0.4
ax.set_xticks(np.arange(x_min, x_max + 0.001, 0.4))

# Format ticks with LaTeX-like mathtext
ax.xaxis.set_major_formatter(lambda x, pos: rf"${x:g}$")
ax.yaxis.set_major_formatter(lambda y, pos: rf"${y:g}$")

ax.grid(True, linestyle="--", alpha=0.3)
# Legend
ax.legend(frameon=False)
# Fix margins to keep consistent output size
fig.subplots_adjust(left=0.18, right=0.98, bottom=0.18, top=0.90)

fig.savefig(OUT_PATH, dpi=300)
print(f"Saved plot to: {OUT_PATH}")
