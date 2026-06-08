import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve


def plot_calibration_curve(y_pred, y_test):
    prob_true, prob_pred = calibration_curve(y_test, y_pred, n_bins=15)
    plt.figure(figsize=(8, 6))
    plt.plot(prob_pred, prob_true, marker='o', label="Calibration Curve")
    plt.plot([0, 1], [0, 1], linestyle='--', label="Ideal Calibration (Perfect)")
    plt.xlabel("Predicted Probability")
    plt.ylabel("True Probability")
    plt.title("Calibration Curve")
    plt.legend()
    plt.grid(True)
    plt.show()
