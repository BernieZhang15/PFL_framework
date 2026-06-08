import numpy as np
import matplotlib.pyplot as plt

def p_of_r(r, fc=10.0, sigma=3.0, eps=1e-6):
    r2 = r**2
    return np.exp(-((r2 - fc**2)**2) / ((r2 + eps) * sigma**2))

# --- 1D plot: p(r) ---
fc = 10.0
sigma = 3.0
eps = 1e-6

r = np.linspace(0, 30, 2000)
p = p_of_r(r, fc=fc, sigma=sigma, eps=eps)

plt.figure(figsize=(6, 3))
plt.plot(r, p)
plt.xlabel("r")
plt.ylabel("p(r)")
plt.title(f"p(r) with f_c={fc}, sigma={sigma}, eps={eps}")
plt.ylim(-0.05, 1.05)
plt.grid(True, alpha=0.3)
plt.savefig("p_of_r_1d.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
plt.close()
print("Saved: p_of_r_1d.png")

# --- 2D plot: p(u,v) with r(u,v)=sqrt(u^2+v^2) ---
U = np.linspace(-30, 30, 600)
V = np.linspace(-30, 30, 600)
uu, vv = np.meshgrid(U, V)
rr = np.sqrt(uu**2 + vv**2)
pp = p_of_r(rr, fc=fc, sigma=sigma, eps=eps)

plt.figure(figsize=(5, 4))
plt.imshow(pp, extent=[U.min(), U.max(), V.min(), V.max()], origin="lower")
plt.colorbar(label="p(u,v)")
plt.xlabel("u")
plt.ylabel("v")
plt.title("p(u,v) heatmap (r = sqrt(u^2+v^2))")
plt.savefig("p_of_r_2d.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
plt.close()
print("Saved: p_of_r_2d.png")