import os
import copy
import random
import numpy as np
from flcore.trainmodel.pFBModel import *
from torch.utils.data import DataLoader
from utils.data_utils import read_client_data
from torchmetrics.functional.classification import multiclass_calibration_error


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

algorithm = "pFedBayes"
dataset = "Cifar10-test-dir-0.75"
model_path = os.path.join('models', "Cifar10-dir-0.1", algorithm + "_server" + ".pt")
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


def train_model(network, client_id, epochs, learning_rate):

    train_data = read_client_data(dataset, client_id, is_train=True)
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

    optimizer = torch.optim.SGD(network.parameters(), lr=learning_rate, momentum=0.9)

    for step in range(epochs):
        network.train()

        x, y = get_next_batch(train_loader)
        x, y = x.to(device), y.to(device)
        y_onehot = F.one_hot(y, num_classes=10)

        epsilons = network.sample_epsilons(network.layer_param_shapes)
        layer_params = network.transform_gaussian_samples(network.mus, network.rhos, epsilons)

        output = network.net(x, layer_params)

        # calculate the loss
        loss = network.combined_loss_personalized(
            output, y_onehot, network.mus, network.sigmas,
            copy.deepcopy(global_model.mus), [t.clone().detach() for t in global_model.sigmas], len(train_data))

        optimizer.zero_grad()
        loss.backward()
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

        ece = multiclass_calibration_error(y_prob, y_true, num_classes=10, n_bins=15, norm="l1")
        mce = multiclass_calibration_error(y_prob, y_true, num_classes=10, n_bins=15, norm="max")

        print(ece, mce)

    return eval_cor, eval_num, y_prob, y_true

test_accs = []
test_eces = []
test_mces = []

c_start = 0
c_end = 10

for i in range(5):

    test_cors = []
    test_nums = []
    test_probs = []
    test_trues = []

    for c in range(c_start, c_end):

        test_data = read_client_data(dataset, c, is_train=False)
        test_loader = DataLoader(test_data, batch_size=len(test_data), shuffle=False)

        print("Start evaluating client {}".format(c))

        network = pBNN(device=device, output_dim=10).to(device)

        network = train_model(network, client_id=c, epochs=300, learning_rate=0.001)

        stats = evaluate(network, test_loader)

        test_cors.append(stats[0])
        test_nums.append(stats[1])
        test_probs.append(stats[2])
        test_trues.append(stats[3])

    test_trues = torch.cat(test_trues, axis=0)
    test_probs = torch.cat(test_probs, axis=0)

    test_acc = sum(test_cors) / sum(test_nums)
    test_ece = multiclass_calibration_error(test_probs, test_trues, num_classes=10, n_bins=15, norm="l1")
    test_mce = multiclass_calibration_error(test_probs, test_trues, num_classes=10, n_bins=15, norm="max")

    print(test_acc, test_ece, test_mce)

    test_accs.append(test_acc)
    test_eces.append(test_ece)
    test_mces.append(test_mce)

print(sum(test_accs) / len(test_accs))
print(sum(test_eces) / len(test_eces))
print(sum(test_mces) / len(test_mces))





