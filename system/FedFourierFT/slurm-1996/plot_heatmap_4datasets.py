import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager, cm
from pathlib import Path

# ── Reproducible random sampling ──────────────────────────────────────
RNG_SEED = 3
TOTAL_CLIENTS = 50

# ── Paths ──────────────────────────────────────────────────────────────
BASE_DATA = Path("/users/3147883g/sharedscratch/PFL_framework/dataset")
OUT_PATH = Path(__file__).resolve().parent / "heatmap_4datasets.pdf"

# ── Font settings ─────────────────────────────────────────────────────
preferred_fonts = ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"]
available_fonts = {f.name for f in font_manager.fontManager.ttflist}
font_name = next((f for f in preferred_fonts if f in available_fonts), "DejaVu Serif")

plt.rcParams.update({
    "font.family": font_name,
    "mathtext.fontset": "custom",
    "mathtext.rm": font_name,
    "mathtext.it": f"{font_name}:italic",
    "mathtext.bf": f"{font_name}:bold",
    "font.size": 16,
    "axes.titlesize": 22,
    "axes.labelsize": 18,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
})

# ── Dataset configs ───────────────────────────────────────────────────
datasets = [
    ("Cifar10-pat-2S", "CIFAR-10  2/10 Small"),
    ("Cifar10-pat-2M", "CIFAR-10  2/10 Medium"),
    ("Cifar10-pat-5S", "CIFAR-10  5/10 Small"),
    ("Cifar10-pat-5M", "CIFAR-10  5/10 Medium"),
]
num_clients = 10
num_classes = 10

# ── Load all heatmaps (randomly sample 10 clients per dataset) ────────
rng = np.random.RandomState(RNG_SEED)
heatmaps = []
sampled_ids_list = []          # keep track of which clients were picked
for ds_name, _ in datasets:
    chosen = sorted(rng.choice(TOTAL_CLIENTS, num_clients, replace=False))
    sampled_ids_list.append(chosen)
    hm = np.zeros((num_clients, num_classes), dtype=int)
    for row, cid in enumerate(chosen):
        fpath = BASE_DATA / ds_name / "train" / f"{cid}.npz"
        obj = np.load(fpath, allow_pickle=True)["data"].item()
        labels = obj["y"]
        for cls in range(num_classes):
            hm[row, cls] = int(np.sum(labels == cls))
    heatmaps.append(hm)

# ── Global vmax for unified bubble scale ──────────────────────────────
global_vmax = max(hm.max() for hm in heatmaps)
D_MIN, D_MAX = 6, 22
cmap = cm.get_cmap("tab10")

# ── Plot 2×2 subplots ────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
axes = axes.flatten()

for idx, (ax, (ds_name, title), heatmap) in enumerate(zip(axes, datasets, heatmaps)):
    chosen = sampled_ids_list[idx]

    for i in range(num_clients):
        for j in range(num_classes):
            val = int(heatmap[i, j])
            if val <= 0:
                continue
            diam = D_MIN + (val / global_vmax) * (D_MAX - D_MIN)
            s = diam ** 2
            ax.scatter(j, i, s=s, c=[cmap(i % 10)],
                       marker="o", linewidths=0.3, edgecolors="white", alpha=0.9)

    ax.set_xlim(-0.5, num_classes - 0.5)
    ax.set_ylim(-0.5, num_clients - 0.5)
    ax.set_xticks(range(num_classes))
    ax.set_xticklabels(range(num_classes))
    ax.set_yticks(range(num_clients))
    ax.set_yticklabels(chosen)
    ax.invert_yaxis()

    ax.set_title(title)
    ax.set_axisbelow(True)
    ax.grid(which="major", axis="both", color="#EAEAEA", linewidth=0.6, linestyle="-")
    for spine in ["top", "right", "left", "bottom"]:
        ax.spines[spine].set_visible(False)

    # Only add axis labels on edges
    if idx >= 2:
        ax.set_xlabel("Class")
    if idx % 2 == 0:
        ax.set_ylabel("Client")

fig.subplots_adjust(hspace=0.30, wspace=0.10)
fig.savefig(OUT_PATH, dpi=300, bbox_inches="tight", pad_inches=0.05)
plt.close(fig)
print(f"Saved: {OUT_PATH}")
