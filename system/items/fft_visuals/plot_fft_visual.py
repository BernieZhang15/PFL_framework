import argparse
import os
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import PowerNorm

# font settings (preferred serif fonts)
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


def save_image(arr, out_path, title, cmap="viridis", vmin=None, vmax=None, dpi=300, norm=None):
    plt.figure(figsize=(5, 4))
    if norm is not None and (vmin is not None or vmax is not None):
        if getattr(norm, "vmin", None) is None:
            norm.vmin = vmin
        if getattr(norm, "vmax", None) is None:
            norm.vmax = vmax
        plt.imshow(arr, cmap=cmap, aspect="auto", norm=norm)
    else:
        plt.imshow(arr, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax, norm=norm)
    plt.colorbar()
    plt.title(title)
    # plt.tight_layout()
    plt.savefig(out_path, dpi=dpi)
    plt.close()


def load_npy(path):
    return np.load(path)


def main():
    parser = argparse.ArgumentParser(description="Plot FourierFT spectra visualizations from saved .npy files")
    parser.add_argument("--input_dir", type=str, default=".", help="Directory containing saved .npy files")
    parser.add_argument("--files", type=str, nargs="*", default=None,
                        help="Explicit list of .npy files to plot (e.g., 6 files)")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save plots (default: input_dir)")
    parser.add_argument("--dpi", type=int, default=300, help="Output image DPI")
    parser.add_argument("--delta_clip", type=float, nargs=2, default=[5, 99],
                        help="Percentile clipping for delta visuals")
    parser.add_argument("--delta_vmin", type=float, default=None, help="Manual vmin for delta spectrum")
    parser.add_argument("--delta_vmax", type=float, default=None, help="Manual vmax for delta spectrum")
    parser.add_argument("--delta_gamma", type=float, default=1,
                        help="PowerNorm gamma for delta visuals (gamma<1 increases contrast for small values)")
    parser.add_argument("--delta_dilate", type=int, default=2,
                        help="Dilation radius for fc2_delta to make points thicker (0=no dilation)")
    parser.add_argument("--delta_dilate_fc1", type=int, default=4,
                        help="Dilation radius for fc1_delta (larger matrix, needs bigger radius)")
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir or input_dir
    os.makedirs(output_dir, exist_ok=True)

    # collect npy files
    npy_files = []
    if args.files:
        npy_files = [os.path.abspath(p) for p in args.files]
    else:
        npy_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.npy')]

    if not npy_files:
        print("No .npy files found in input_dir")
        return

    # find specific files for fc1, fc2 and fc3, base and delta
    mapping = {
        'fc1_base': None,
        'fc1_delta': None,
        'fc2_base': None,
        'fc2_delta': None,
        'fc3_base': None,
        'fc3_delta': None,
    }

    for p in npy_files:
        n = os.path.basename(p).lower()
        if 'fc1' in n and ('base' in n or 'base_fft' in n or '_base' in n):
            mapping['fc1_base'] = p
        if 'fc1' in n and ('delta' in n or 'delta_spectrum' in n or '_delta' in n):
            mapping['fc1_delta'] = p
        if 'fc2' in n and ('base' in n or 'base_fft' in n or '_base' in n):
            mapping['fc2_base'] = p
        if 'fc2' in n and ('delta' in n or 'delta_spectrum' in n or '_delta' in n):
            mapping['fc2_delta'] = p
        if 'fc3' in n and ('base' in n or 'base_fft' in n or '_base' in n):
            mapping['fc3_base'] = p
        if 'fc3' in n and ('delta' in n or 'delta_spectrum' in n or '_delta' in n):
            mapping['fc3_delta'] = p

    missing = [k for k, v in mapping.items() if v is None]
    if missing:
        raise ValueError(f"Missing required files for: {missing}. Available files: {npy_files}")

    # load arrays
    fc1_base = np.log1p(np.abs(load_npy(mapping['fc1_base'])))
    fc2_base = np.log1p(np.abs(load_npy(mapping['fc2_base'])))
    fc3_base = np.log1p(np.abs(load_npy(mapping['fc3_base'])))
    fc1_delta = np.log1p(np.abs(load_npy(mapping['fc1_delta'])))
    fc2_delta = np.log1p(np.abs(load_npy(mapping['fc2_delta'])))
    fc3_delta = np.log1p(np.abs(load_npy(mapping['fc3_delta'])))

    # determine vmin/vmax per row (fixed range 0–5)
    base_vmin = 0.0
    base_vmax = 5.0

    # delta clipping using percentiles
    clip_lo, clip_hi = args.delta_clip
    nonzero = np.concatenate([fc1_delta[fc1_delta > 0].ravel(), fc2_delta[fc2_delta > 0].ravel(), fc3_delta[fc3_delta > 0].ravel()])
    if nonzero.size > 0:
        d_vmin = np.percentile(nonzero, clip_lo)
        d_vmax = np.percentile(nonzero, clip_hi)
    else:
        d_vmin = min(fc2_delta.min(), fc3_delta.min())
        d_vmax = max(fc2_delta.max(), fc3_delta.max())
    if args.delta_vmin is not None:
        d_vmin = args.delta_vmin
    if args.delta_vmax is not None:
        d_vmax = args.delta_vmax

    def dilate_max(arr, radius=1):
        if radius <= 0:
            return arr
        out = arr.copy()
        h, w = arr.shape[:2]
        for dy in range(-radius, radius+1):
            for dx in range(-radius, radius+1):
                shifted = np.zeros_like(arr)
                ys_src = slice(max(0, -dy), min(h, h-dy))
                xs_src = slice(max(0, -dx), min(w, w-dx))
                ys_dst = slice(max(0, dy), min(h, h+dy))
                xs_dst = slice(max(0, dx), min(w, w+dx))
                shifted[ys_dst, xs_dst] = arr[ys_src, xs_src]
                out = np.maximum(out, shifted)
        return out

    # optionally dilate fc1_delta and fc2_delta to make points thicker
    if args.delta_dilate_fc1 and args.delta_dilate_fc1 > 0:
        fc1_delta = dilate_max(fc1_delta, radius=args.delta_dilate_fc1)
    if args.delta_dilate and args.delta_dilate > 0:
        fc2_delta = dilate_max(fc2_delta, radius=args.delta_dilate)

    # create 2x4 gridspec: 2 rows, 4 cols (last col narrow for colorbars)
    fig = plt.figure(figsize=(15, 8))
    # reduce vertical spacing between the two rows
    # set horizontal spacing equal to vertical spacing so left/right gap matches up/down gap
    hspace = 0.06
    wspace = hspace
    gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, 0.08], height_ratios=[1, 1], wspace=wspace, hspace=hspace)

    ax00 = fig.add_subplot(gs[0, 0])
    ax01 = fig.add_subplot(gs[0, 1])
    ax02 = fig.add_subplot(gs[0, 2])
    cax0 = fig.add_subplot(gs[0, 3])

    ax10 = fig.add_subplot(gs[1, 0])
    ax11 = fig.add_subplot(gs[1, 1])
    ax12 = fig.add_subplot(gs[1, 2])
    cax1 = fig.add_subplot(gs[1, 3])

    # Row 0: base weights
    im00 = ax00.imshow(fc1_base, aspect='auto', cmap='viridis', vmin=base_vmin, vmax=base_vmax)
    ax00.set_title('Layer 1', fontsize=plt.rcParams['axes.labelsize'], pad=14)
    ax00.set_xticks([])
    ax00.set_yticks([])
    for s in ax00.spines.values():
        s.set_visible(False)
    ax00.set_ylabel(r'Shared model  $\mathbf{W}$', rotation=90, labelpad=12)

    im01 = ax01.imshow(fc2_base, aspect='auto', cmap='viridis', vmin=base_vmin, vmax=base_vmax)
    ax01.set_title('Layer 2', fontsize=plt.rcParams['axes.labelsize'], pad=14)
    ax01.set_xticks([])
    ax01.set_yticks([])
    for s in ax01.spines.values():
        s.set_visible(False)

    im02 = ax02.imshow(fc3_base, aspect='auto', cmap='viridis', vmin=base_vmin, vmax=base_vmax)
    ax02.set_title('Layer 3', fontsize=plt.rcParams['axes.labelsize'], pad=14)
    ax02.set_xticks([])
    ax02.set_yticks([])
    for s in ax02.spines.values():
        s.set_visible(False)

    # Row 1: delta corrections
    delta_norm = PowerNorm(gamma=args.delta_gamma, vmin=d_vmin, vmax=d_vmax)
    im10 = ax10.imshow(fc1_delta, aspect='auto', cmap='magma', norm=delta_norm, interpolation='nearest')
    ax10.set_title('')
    ax10.set_xticks([])
    ax10.set_yticks([])
    for s in ax10.spines.values():
        s.set_visible(False)
    ax10.set_ylabel(r'Correction  $\Delta \mathbf{W}_k$', rotation=90, labelpad=12)

    im11 = ax11.imshow(fc2_delta, aspect='auto', cmap='magma', norm=delta_norm, interpolation='nearest')
    ax11.set_title('')
    ax11.set_xticks([])
    ax11.set_yticks([])
    for s in ax11.spines.values():
        s.set_visible(False)

    im12 = ax12.imshow(fc3_delta, aspect='auto', cmap='magma', norm=delta_norm, interpolation='nearest')
    ax12.set_title('')
    ax12.set_xticks([])
    ax12.set_yticks([])
    for s in ax12.spines.values():
        s.set_visible(False)

    # colorbars per row
    fig.colorbar(im02, cax=cax0)
    fig.colorbar(im12, cax=cax1)

    out_path = os.path.join(output_dir, 'fc1_fc2_fc3_composite.pdf')
    # plt.tight_layout()
    plt.savefig(out_path, dpi=args.dpi, bbox_inches='tight')
    plt.close(fig)

    print(f"Saved composite plot to: {out_path}")


if __name__ == "__main__":
    main()
