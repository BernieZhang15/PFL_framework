import torch
import numpy as np
import matplotlib.pyplot as plt


def make_model_diagrams(outputs, labels, ece, mce, n_bins=15, algorithm=None):
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
    confs = plt.bar(bin_centers, bin_corrects, width=width, color='darkgreen', alpha=0.7, ec='black')

    plt.plot([0, 1], [0, 1], '--', color='grey')

    # Clean up
    bbox_props = dict(boxstyle="round", fc="white", ec="grey", lw=2)
    plt.text(0.2, 0.9, "ECE: {:.4f} \n MCE: {:.4f}".format(ece, mce), ha="center", va="center", size=20,  bbox=bbox_props)

    # plt.title("Reliability Diagram for {}".format(algorithm), size=20)
    plt.ylabel("Accuracy", size=20)
    plt.xlabel("Confidence", size=20)
    plt.xlim(0, 1)
    plt.ylim(0, 1)

    # plt.savefig("Reliability_diagram.jpg")
    plt.show()
