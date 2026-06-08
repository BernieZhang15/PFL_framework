import os
import copy
import random
import numpy as np
import matplotlib.pyplot as plt
from flcore.trainmodel.pFBModel import *
from torch.utils.data import DataLoader
from utils.data_utils import read_client_data
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.functional.classification import multiclass_calibration_error
from plot_reliability_diagram import make_model_diagrams


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

algorithm = "pFedBayes"
dataset = "Cifar10-pat-2"
model_path = os.path.join('models', dataset, algorithm + "_server_187" + ".pt")
global_model = torch.load(model_path)
global_model = global_model.to(device)
global_model.transform_rhos(global_model.rhos)

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

def fine_tuning(trainloader, train_nums=None, update_step=None, learning_rate=None):

    Round = 5

    network = pBNN(device=device, output_dim=10).to(device)

    optimizer = torch.optim.SGD(network.parameters(), lr=learning_rate)

    for e in range(update_step):

        network.train()

        x, y = get_next_batch(trainloader)
        x, y = x.to(device), y.to(device)
        y_onehot = F.one_hot(y, num_classes=10).to(device)

        for r in range(Round):
            epsilons = network.sample_epsilons(network.layer_param_shapes)
            layer_params = network.transform_gaussian_samples(network.mus, network.rhos, epsilons)

            outputs = network.net(x, layer_params)

            p_loss = network.combined_loss_personalized(outputs, y_onehot, network.mus, network.sigmas,
                copy.deepcopy(global_model.mus), [t.clone().detach() for t in global_model.sigmas], train_nums)

            optimizer.zero_grad()
            p_loss.backward()
            optimizer.step()

    return network

def evaluate(network, dataloader, ens_num=4):

    network.eval()

    eval_cor = 0
    eval_num = 0
    y_prob = []
    y_true = []

    with torch.no_grad():
        for x, y in dataloader:

            x, y = x.to(device), y.to(device)

            outputs = []
            for _ in range(ens_num):

                epsilons = network.sample_epsilons(network.layer_param_shapes)
                layer_params = network.transform_gaussian_samples(network.mus, network.rhos, epsilons)

                # forward-propagate the batch
                output = network.net(x, layer_params)
                outputs.append(output)

            outputs = torch.stack(outputs, dim=0)
            outputs = F.softmax(outputs, dim=2)
            outputs = torch.mean(outputs, dim=0)

            eval_cor += (torch.sum(torch.argmax(outputs, dim=1) == y)).item()
            eval_num += y.shape[0]

            y_prob.append(outputs.detach().cpu())
            y_true.append(y.cpu())

        y_prob = torch.cat(y_prob, axis=0)
        y_true = torch.cat(y_true, axis=0)

        test_ece = multiclass_calibration_error(y_prob, y_true, num_classes=10, n_bins=15, norm="l1")
        test_mce = multiclass_calibration_error(y_prob, y_true, num_classes=10, n_bins=15, norm="max")

    return eval_cor / eval_num, test_ece, test_mce, y_prob, y_true

ood_test_accs = [[] for _ in range(10)]
ood_test_eces = [[] for _ in range(10)]
ood_test_mces = [[] for _ in range(10)]

id_test_accs = [[] for _ in range(40)]
id_test_eces = [[] for _ in range(40)]
id_test_mces = [[] for _ in range(40)]

c_start = 0
c_end = 40

for c in range(c_start, c_end):

    test_data = read_client_data(dataset, c, is_train=False)
    test_loader = DataLoader(test_data, batch_size=len(test_data), shuffle=False)

    print("Start evaluating client {}".format(c))

    if c >= 40:
        train_data = read_client_data(dataset, c, is_train=True)
        train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

        for _, s in enumerate([890, 355, 236, 865, 954, 854, 602, 91, 56, 82]):
            set_seed(s)

            local_model = fine_tuning(train_loader, train_nums=len(train_data), update_step=40, learning_rate=0.001)
            stats = evaluate(local_model, test_loader)

            ood_test_accs[c - 40].append(stats[0])
            ood_test_eces[c - 40].append(stats[1])
            ood_test_mces[c - 40].append(stats[2])
    else:
        model_path = os.path.join('models', dataset, "pFedBayes_clients_187", "pFedBayes_client_{}".format(c) + ".pt")
        local_model = torch.load(model_path)
        local_model = local_model.to(device)

        for _, s in enumerate([531890, 957355, 96236, 773865, 209954, 463854, 642602, 688091, 690056, 246082]):
            set_seed(s)

            stats = evaluate(local_model, test_loader)
            id_test_accs[c].append(stats[0])
            id_test_eces[c].append(stats[1])
            id_test_mces[c].append(stats[2])

if c_start < 40:
    id_test_accs = np.array(id_test_accs)
    id_test_eces = np.array(id_test_eces)
    id_test_mces = np.array(id_test_mces)

    id_mean_acc = np.mean(id_test_accs, axis=1)
    id_mean_ece = np.mean(id_test_eces, axis=1)
    id_mean_mce = np.mean(id_test_mces, axis=1)

    print(id_mean_acc)
    print(id_mean_ece)
    print(id_mean_mce)

#
# if c_end > 40:
#     ood_test_accs = np.array(ood_test_accs)
#     ood_test_eces = np.array(ood_test_eces)
#     ood_test_mces = np.array(ood_test_mces)
#
#     ood_test_acc = np.mean(ood_test_accs, axis=1)
#     ood_test_ece = np.mean(ood_test_eces, axis=1)
#     ood_test_mce = np.mean(ood_test_mces, axis=1)
#
#     print(ood_test_acc)
#     print(ood_test_ece)
#     print(ood_test_mce)



