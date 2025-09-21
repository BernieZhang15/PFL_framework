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
dataset = "Cifar10-pat-2S"
train_one_step = False
model_path = os.path.join("", 'models', "Cifar10-pat-2S", algorithm + "_server_2" + ".pt")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def get_next_batch(dataloader):
    iter_dataloader = iter(dataloader)
    try:
        x_input, label = next(iter_dataloader)
    except StopIteration:
        iter_trainloader = iter(dataloader)
        x_input, label = next(iter_trainloader)
    return x_input, label

def evaluate(network, dataloader):

    network.eval()

    eval_cor = 0
    eval_num = 0
    y_prob = []
    y_true = []

    with torch.no_grad():
        for x, y in dataloader:

            x, y = x.to(device), y.to(device)

            outputs = network(x)

            eval_cor += (torch.sum(torch.argmax(outputs, dim=1) == y)).item()
            eval_num += y.shape[0]

            y_prob.append(outputs.detach().cpu())
            y_true.append(y.cpu())

        y_prob = torch.cat(y_prob, axis=0)
        y_true = torch.cat(y_true, axis=0)

        ece = multiclass_calibration_error(y_prob, y_true, num_classes=10, n_bins=15, norm="l1")
        mce = multiclass_calibration_error(y_prob, y_true, num_classes=10, n_bins=15, norm="max")

        print(ece, mce)

    return eval_cor, eval_num, y_prob, y_true

test_cor = []
test_num = []
test_probs = []
test_trues = []

c_start = 0
c_end = 50

set_seed(55)

for c in range(c_start, c_end):

    train_data = read_client_data(dataset, c, is_train=True)
    train_loader = DataLoader(train_data, batch_size=32, shuffle=False)

    network = torch.load(model_path)
    network = network.to(device)

    if train_one_step:
        optimizer = torch.optim.SGD(network.parameters(), lr=0.005, momentum=0.9, weight_decay=5e-4)

        network.train()

        x, y = get_next_batch(train_loader)
        x, y = x.to(device), y.to(device)

        pred = network(x)
        optimizer.zero_grad()
        loss = F.cross_entropy(pred, y)
        loss.backward()
        optimizer.step()

    test_data = read_client_data(dataset, c, is_train=False)
    test_loader = DataLoader(test_data, batch_size=len(test_data), shuffle=False)

    print("Start evaluating client {}".format(c))

    stats = evaluate(network, test_loader)

    test_cor.append(stats[0])
    test_num.append(stats[1])
    test_probs.append(stats[2])
    test_trues.append(stats[3])

test_trues = torch.cat(test_trues, axis=0)
test_probs = torch.cat(test_probs, axis=0)

test_acc = sum(test_cor) / sum(test_num)
test_ece = multiclass_calibration_error(test_probs, test_trues, num_classes=10, n_bins=15, norm="l1")
test_mce = multiclass_calibration_error(test_probs, test_trues, num_classes=10, n_bins=15, norm="max")

print(test_acc, test_ece, test_mce)











