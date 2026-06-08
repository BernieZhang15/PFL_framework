import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

# ── Paths ──────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "test_acc_summary.csv"

# ── Font settings (same style as plot_fc_bias_errorbar.py) ─────────────
preferred_fonts = ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"]
available_fonts = {f.name for f in font_manager.fontManager.ttflist}
font_name = next((f for f in preferred_fonts if f in available_fonts), "DejaVu Serif")

plt.rcParams.update({
    "font.family": font_name,
    "mathtext.fontset": "custom",
    "mathtext.rm": font_name,
    "mathtext.it": f"{font_name}:italic",
    "mathtext.bf": f"{font_name}:bold",
    "font.size": 26,
    "axes.titlesize": 28,
    "axes.labelsize": 28,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
})

# ── Load CSV ───────────────────────────────────────────────────────────
rounds = []
data = {}

with CSV_PATH.open(newline="") as f:
    reader = csv.DictReader(f)
    cols = reader.fieldnames
    for col in cols[1:]:
        data[col] = []
    for row in reader:
        rounds.append(int(row["Round"]))
        for col in cols[1:]:
            val = row[col]
            data[col].append(float(val) * 100 if val else None)  # convert to %

rounds = np.array(rounds)

# ── Define 3 figures: one per dataset ──────────────────────────────────
datasets = [
    {
        "title": "CIFAR-10  2/10",
        "out": BASE_DIR / "acc_cifar10_pat2M.pdf",
        "cols": [
            ("FedFourierFT_Cifar10-pat-2M", "FT-BPFL", "#FF5733", "-"),
            ("FedMetaBayes(LR)_Cifar10-pat-2M", "LR-BPFL", "#456990", "--"),
        ],
    },
    {
        "title": "CIFAR-10  5/10",
        "out": BASE_DIR / "acc_cifar10_pat5M.pdf",
        "cols": [
            ("FedFourierFT_Cifar10-pat-5M", "FT-BPFL", "#FF5733", "-"),
            ("FedMetaBayes(LR)_Cifar10-pat-5M", "LR-BPFL", "#456990", "--"),
        ],
    },
    {
        "title": "CIFAR-100  5/100",
        "out": BASE_DIR / "acc_cifar100_pat5M.pdf",
        "cols": [
            ("FedFourierFT_Cifar100-pat-5M", "FT-BPFL", "#FF5733", "-"),
            ("FedMetaBayes(LR)_Cifar100-pat-5M", "LR-BPFL", "#456990", "--"),
        ],
    },
]

# ── Plot each figure ──────────────────────────────────────────────────
for ds in datasets:
    fig, ax = plt.subplots(figsize=(4, 6))

    for col_key, label, color, ls in ds["cols"]:
        acc = np.array(data[col_key])
        ax.plot(
            rounds,
            acc,
            linestyle=ls,
            linewidth=3,
            color=color,
            label=label,
        )

    ax.set_xlabel("Communication round")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title(ds["title"])
    ax.set_xlim(0, 1000)
    ax.grid(True, linestyle="--", alpha=0.3)

    ax.legend(
        frameon=False,
        loc="lower right",
        fontsize=16,
    )

    fig.subplots_adjust(left=0.15, right=0.97, bottom=0.15, top=0.92)
    fig.savefig(ds["out"], dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"Saved: {ds['out']}")

print("All done.")
