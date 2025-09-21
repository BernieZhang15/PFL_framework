import os
import random
import numpy as np
from flcore.trainmodel.pFBModel import *
from torch.utils.data import DataLoader
from utils.data_utils import read_client_data
from torchmetrics.functional.classification import multiclass_calibration_error
from visualize.plot_reliability_diagram import make_model_diagrams


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

algorithm = "pFedBayes"
dataset = "Cifar10-pat-2S"

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


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

    return eval_cor / eval_num, y_prob, y_true

c_start = 7
c_end = 8

y_prob = []
y_true = []


for c in range(c_start, c_end):

    set_seed(8)

    test_data = read_client_data(dataset, c, is_train=False)
    test_loader = DataLoader(test_data, batch_size=len(test_data), shuffle=False)

    print("Start evaluating client {}".format(c))

    model_path = os.path.join('models', dataset, "pFedBayes_clients_2", "pFedBayes_client_{}".format(c) + ".pt")
    local_model = torch.load(model_path)
    local_model = local_model.to(device)

    stats = evaluate(local_model, test_loader)

    y_prob.append(stats[1])
    y_true.append(stats[2])

y_prob = torch.cat(y_prob, axis=0)
y_true = torch.cat(y_true, axis=0)

ece = multiclass_calibration_error(y_prob, y_true, num_classes=10, n_bins=15, norm="l1")
mce = multiclass_calibration_error(y_prob, y_true, num_classes=10, n_bins=15, norm="max")

make_model_diagrams(y_prob, y_true, ece, mce)

print(ece, mce)



