import torch
import numpy as np
import matplotlib.pyplot as plt

preferred_fonts = ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"]
plt.rcParams["font.family"] = preferred_fonts

def make_model_diagrams(outputs, labels, ece, mce, algorithm="FedAvg", n_bins=15):
    """
    outputs - a torch tensor (size n x num_classes) with the outputs from the final linear layer
    - Already softmaxed
    labels - a torch tensor (size n) with the labels
    """
    confidences, predictions = outputs.max(1)
    accuracies = torch.eq(predictions, labels)

    # Reliability diagram
    bins = torch.linspace(0, 1, n_bins + 1)
    width = 1.0 / n_bins
    bin_centers = np.linspace(0, 1.0 - width, n_bins) + width / 2
    bin_indices = [confidences.ge(bin_lower) * confidences.lt(bin_upper) for bin_lower, bin_upper in
                   zip(bins[:-1], bins[1:])]

    bin_corrects = np.array([torch.mean(accuracies[bin_index].float()) for bin_index in bin_indices])
    bin_scores = np.array([torch.mean(confidences[bin_index].float()) for bin_index in bin_indices])

    plt.figure(0, figsize=(8, 8))
    confs = plt.bar(bin_centers, bin_corrects, width=width, alpha=0.7, ec='black')
    gaps = plt.bar(bin_centers, (bin_scores - bin_corrects), bottom=bin_corrects, color=[1, 0.7, 0.7], alpha=0.5,
                   width=width, hatch='//', edgecolor='r')
    plt.plot([0, 1], [0, 1], '--', color='grey')
    plt.legend(
        [confs, gaps],
        ['Outputs', 'Gap'],
        loc=(0.035, 0.6),
        fontsize=30,
        labelspacing=0.2,
        handletextpad=0.4,
        borderpad=0.3,
    )

    # Clean up
    bbox_props = dict(boxstyle="round", fc="#F2F2F2", ec="#808080", lw=1.8)
    plt.text(0.24, 0.9, "ECE: {:.4f} \nMCE: {:.4f}".format(ece, mce), ha="center", va="center", size=30, weight='bold', bbox=bbox_props)

    # plt.title("{}".format(algorithm), size=35, weight="bold")
    plt.title("{}".format(algorithm), size=35)

    
    plt.ylabel("Accuracy", size=40)
    plt.xlabel("Confidence", size=40)
    plt.xlim(0, 1)
    plt.ylim(0, 1)

    plt.xticks(size=20)
    plt.yticks(size=20)

    plt.tight_layout()
    safe_algorithm = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(algorithm))
    plt.savefig(f"Reliability_diagram_{safe_algorithm}.pdf", bbox_inches="tight")
    # plt.show()
