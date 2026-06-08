import os
import sys
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from torch.utils.data import DataLoader
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from flcore.trainmodel.be_models import *
from utils.data_utils import read_client_data
from torch.utils.tensorboard import SummaryWriter
from plot_reliability_diagram import make_model_diagrams
from torchmetrics.functional.classification import multiclass_calibration_error

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

algorithm = "FedAvg"
dataset = "Cifar10-pat-2M"
model_path = os.path.join('./../models', dataset, algorithm + "_server_573_0.01" + ".pt")
model = torch.load(model_path)
model = model.to(device)

def evaluate(dataloader):

    model.eval()

    eval_cor = 0
    eval_num = 0
    y_prob = []
    y_true = []

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)

            outputs  = model(x)

            outputs = F.softmax(outputs, dim=1)

            eval_cor += (torch.sum(torch.argmax(outputs, dim=1) == y)).item()
            eval_num += y.shape[0]

            y_prob.append(outputs.detach().cpu())
            y_true.append(y.cpu())

        y_prob = torch.cat(y_prob, axis=0)
        y_true = torch.cat(y_true, axis=0)

        test_ece = multiclass_calibration_error(y_prob, y_true, num_classes=10, n_bins=15, norm="l1")
        test_mce = multiclass_calibration_error(y_prob, y_true, num_classes=10, n_bins=15, norm="max")

        return eval_cor / eval_num, test_ece, test_mce, y_prob, y_true


c_start = 6
c_end = 7
y_prob = []
y_true = []

for c in range(c_start, c_end):

    print("Start evaluating client {}".format(c))

    test_data = read_client_data(dataset, c, is_train=False)
    test_loader = DataLoader(test_data, batch_size=128, shuffle=False)

    stats = evaluate(test_loader)

    y_prob.append(stats[3])
    y_true.append(stats[4])

y_prob = torch.cat(y_prob, axis=0)
y_true = torch.cat(y_true, axis=0)

ece = multiclass_calibration_error(y_prob, y_true, num_classes=10, n_bins=15, norm="l1")
mce = multiclass_calibration_error(y_prob, y_true, num_classes=10, n_bins=15, norm="max")
make_model_diagrams(y_prob, y_true, ece, mce, algorithm="pFedME")








