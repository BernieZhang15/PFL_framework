import os
import torch
import random
import numpy as np
from system.flcore.trainmodel.pFBModel import *
from torch.utils.data import DataLoader
from system.utils.data_utils import read_client_data
from torchmetrics.functional.classification import multiclass_calibration_error


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

algorithm = "FedAvg"
dataset = "Cifar10-pat-2M"
model_path = os.path.join("..", 'models', dataset, algorithm + "_server_2" + ".pt")
global_model = torch.load(model_path)
global_model = global_model.to(device)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def evaluate(dataloader):

    global_model.eval()

    eval_cor = 0
    eval_num = 0
    y_prob = []
    y_true = []

    with torch.no_grad():
        for x, y in dataloader:

            x, y = x.to(device), y.to(device)

            outputs = global_model(x)

            eval_cor += (torch.sum(torch.argmax(outputs, dim=1) == y)).item()
            eval_num += y.shape[0]

            y_prob.append(outputs.detach().cpu())
            y_true.append(y.cpu())

        y_prob = torch.cat(y_prob, axis=0)
        y_true = torch.cat(y_true, axis=0)

        test_ece = multiclass_calibration_error(y_prob, y_true, num_classes=10, n_bins=15, norm="l1")
        test_mce = multiclass_calibration_error(y_prob, y_true, num_classes=10, n_bins=15, norm="max")

    return eval_cor / eval_num, test_ece, test_mce, y_prob, y_true

test_accs = [[] for _ in range(50)]
test_eces = [[] for _ in range(50)]
test_mces = [[] for _ in range(50)]

c_start = 0
c_end = 50

for c in range(c_start, c_end):

    test_data = read_client_data(dataset, c, is_train=False)
    test_loader = DataLoader(test_data, batch_size=len(test_data), shuffle=False)

    print("Start evaluating client {}".format(c))

    for _, s in enumerate([531890, 957355, 96236, 773865, 209954, 463854, 642602, 688091, 690056, 246082]):
        set_seed(s)

        stats = evaluate(test_loader)

        test_accs[c].append(stats[0])
        test_eces[c].append(stats[1])
        test_mces[c].append(stats[2])

print(test_accs)
print(test_eces)
print(test_mces)





