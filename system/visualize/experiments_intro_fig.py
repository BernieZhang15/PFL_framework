import torch
import numpy as np
import matplotlib.pyplot as plt


n_bins = 15

# Reliability diagram
bins = torch.linspace(0, 1, n_bins + 1)
width = 1.0 / n_bins
bin_centers = np.linspace(0, 1.0 - width, n_bins) + width / 2

bin1 = [0, 0, 0, 1.85168040e-05, 5.36987316e-04, 4.44403296e-03, 9.61022128e-03, 2.54235719e-02, 3.40894362e-02, 3.54781965e-02, 3.92926581e-02, 4.34033886e-02, 5.35505972e-02, 7.74743079e-02, 6.16813258e-01]
bin2 = [0, 0, 0, 0, 0, 1.85168040e-05, 3.70336080e-05, 8.79548190e-03, 1.81094343e-02, 1.95722618e-02, 2.18868623e-02, 2.44421813e-02, 3.07193778e-02, 4.94583835e-02, 6.08462179e-01]
bin3 = [0, 0, 0, 0, 0, 0, 3.70336080e-05, 3.78113138e-02, 2.92009999e-02, 2.48310342e-02, 3.61633182e-02, 9.32506249e-02, 2.91084159e-02, 3.08304787e-02, 4.23738543e-01]
bin4 = [0, 0, 1.85109770e-05, 1.11065862e-04, 5.73840287e-04, 2.20280626e-03, 6.96012736e-03, 3.43748843e-02, 4.53889156e-02, 5.11828514e-02, 5.65325238e-02, 6.71578246e-02, 8.43175003e-02, 1.19821554e-01, 5.31357595e-01]

plt.figure(0, figsize=(7, 10))

plt.plot(bin_centers, bin1, marker='o', markersize=8, label='FedAvg ECE(%): 34.25', linewidth=2, color="#2E86C1")
plt.plot(bin_centers, bin2, marker='o', markersize=8, label='per-FedAvg ECE(%): 12.70', linewidth=2, color="#EF767A")
plt.plot(bin_centers, bin3, marker='o', markersize=8, label='pFedBayes ECE(%): 6.26', linewidth=2, color="#456990")
plt.plot(bin_centers, bin4, marker='o', markersize=8, label='pFedBLoRA ECE(%): 1.74', linewidth=2, color="#48C0AA")

plt.ylabel("% of Sample", size=25)
plt.xlabel("Confidence", size=25)
plt.xlim(0, 1)
plt.ylim(0, 1)

plt.xticks(size=15)
plt.yticks(size=15)

plt.legend(fontsize=20, loc='lower right')

plt.show()
