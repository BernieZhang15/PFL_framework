import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager, cm
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────
DATA_DIR = Path("/users/3147883g/sharedscratch/PFL_framework/dataset/Cifar10-pat-5M/train")
OUT_PATH = (Path.cwd() / "heatmap_cifar10_5M.pdf")

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
    "font.size": 18,
    "axes.titlesize": 28,
    "axes.labelsize": 28,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
})

# ── Load data ─────────────────────────────────────────────────────────
num_clients = 10
num_classes = 10
heatmap = np.zeros((num_clients, num_classes), dtype=int)

for cid in range(num_clients):
    fpath = DATA_DIR / f"{cid}.npz"
    obj = np.load(fpath, allow_pickle=True)["data"].item()
    labels = obj["y"]
    for cls in range(num_classes):
        heatmap[cid, cls] = int(np.sum(labels == cls))

# ── Plot bubble heatmap (dot diameter ∝ count) ────────────────────────
assert isinstance(heatmap, np.ndarray) and heatmap.ndim == 2, "heatmap 尚未构造为二维 ndarray"

fig_w = max(8, 0.6 * num_classes + 2)
fig_h = max(4, 0.55 * num_clients + 1.2)
fig, ax = plt.subplots(figsize=(fig_w, fig_h))

xs, ys, sizes, cols = [], [], [], []
vmax = int(heatmap.max()) if heatmap.size else 0

# 最小/最大“直径”（注意：scatter 的 s 是面积 points^2）
D_MIN, D_MAX = 2, 24
cmap = cm.get_cmap("tab20")

for i in range(num_clients):
    for j in range(num_classes):
        val = int(heatmap[i, j])
        if val <= 0:
            continue
        if vmax > 0:
            diam = D_MIN + (val / vmax) * (D_MAX - D_MIN)
        else:
            diam = D_MIN
        s = diam ** 2
        xs.append(j)
        ys.append(i)
        sizes.append(s)
        cols.append(cmap(i % cmap.N))  # 每一行（client）颜色一致

sc = ax.scatter(
    xs, ys, s=sizes, c=cols,
    marker="o", linewidths=0.3, edgecolors="white", alpha=0.95
)

ax.set_xlim(-0.5, num_classes - 0.5)
ax.set_ylim(-0.5, num_clients - 0.5)
ax.set_xticks(range(num_classes))
ax.set_xticklabels(range(num_classes))

# ========= 你要的两处改动 =========
ax.set_yticks(range(num_clients))                 # 纵轴刻度：0..9
ax.set_yticklabels(range(num_clients))            # 纵轴显示：0..9
ax.set_ylabel("Client")                           # 纵轴标签
# =================================

# 让 0 在最上方（保留你之前的视觉习惯；若想让 0 在最下方，请把下一行删掉）
ax.invert_yaxis()

# 网格和外观
ax.set_axisbelow(True)
ax.grid(which="major", axis="x", color="#EAEAEA", linewidth=0.8, linestyle="-")
ax.grid(which="major", axis="y", color="#F1F1F1", linewidth=0.9, linestyle="-")
for spine in ["top", "right", "left", "bottom"]:
    ax.spines[spine].set_visible(False)

ax.set_xlabel("Class")
ax.set_title("CIFAR-10  5/10 Medium")

# 尺寸图例（显示直径代表的样本数）
if vmax > 0:
    anchor_vals = np.unique(
        np.clip(np.round(np.linspace(max(1, vmax*0.25), vmax, 4)), 1, None).astype(int)
    )
    handles = []
    for v in anchor_vals:
        d = D_MIN + (v / vmax) * (D_MAX - D_MIN)
        s = d ** 2
        h = ax.scatter([], [], s=s, color="#999999", alpha=0.6,
                       edgecolors="white", linewidths=0.3)
        handles.append(h)

plt.margins(x=0.02, y=0.05)
fig.savefig(OUT_PATH, dpi=300, bbox_inches="tight", pad_inches=0.02)
plt.close(fig)
print(f"Saved: {OUT_PATH}")