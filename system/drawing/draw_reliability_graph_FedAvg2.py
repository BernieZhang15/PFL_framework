import os
import random
from torch.utils.data import DataLoader
from flcore.trainmodel.be_models import *
from utils.data_utils import read_client_data
from visualize.plot_reliability_diagram import make_model_diagrams
from torchmetrics.functional.classification import multiclass_calibration_error

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

algorithm = "FedAvg"
train_one_step = False
dataset = "Cifar10-pat-2S"
model_path = os.path.join('models', dataset, algorithm + "_server" + ".pt")


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

            outputs  = network(x)

            outputs = F.softmax(outputs, dim=1)

            eval_cor += (torch.sum(torch.argmax(outputs, dim=1) == y)).item()
            eval_num += y.shape[0]

            y_prob.append(outputs.detach().cpu())
            y_true.append(y.cpu())

        y_prob = torch.cat(y_prob, axis=0)
        y_true = torch.cat(y_true, axis=0)

        return eval_cor / eval_num, y_prob, y_true


c_start = 5
c_end = 6

probs = []
trues = []

for c in range(c_start, c_end):

    print("Start evaluating client {}".format(c))

    train_data = read_client_data(dataset, c, is_train=True)
    train_loader = DataLoader(train_data, batch_size=32, shuffle=False)

    network = torch.load(model_path)
    network = network.to(device)

    if train_one_step:

        optimizer = torch.optim.SGD(network.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

        network.train()

        x, y = get_next_batch(train_loader)
        x, y = x.to(device), y.to(device)

        pred = network(x)
        optimizer.zero_grad()
        loss = F.cross_entropy(pred, y)
        loss.backward()
        optimizer.step()

    test_data = read_client_data(dataset, c, is_train=False)
    test_loader = DataLoader(test_data, batch_size=128, shuffle=False)

    stats = evaluate(network, test_loader)

    probs.append(stats[1])
    trues.append(stats[2])

probs = torch.concat(probs, axis=0)
trues = torch.concat(trues, axis=0)

ece = multiclass_calibration_error(probs, trues, num_classes=10, n_bins=15, norm="l1")
mce = multiclass_calibration_error(probs, trues, num_classes=10, n_bins=15, norm="max")

make_model_diagrams(probs, trues, ece, mce)

print(ece, mce)









