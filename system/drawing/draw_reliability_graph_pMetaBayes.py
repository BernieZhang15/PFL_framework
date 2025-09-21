import os
import random
from collections import OrderedDict
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

algorithm = "FedMetaBayes"
dataset = "Cifar10-pat-2S"
model_path = os.path.join('models', "Cifar10-pat-2S", algorithm + "_server_4_0.005" + ".pt")
model = torch.load(model_path)
model = model.to(device)


def get_next_batch(dataloader):
    iter_dataloader = iter(dataloader)
    try:
        x_input, label = next(iter_dataloader)
    except StopIteration:
        iter_trainloader = iter(dataloader)
        x_input, label = next(iter_trainloader)
    return x_input, label


def fine_tune(net, trainloader, testloader, train_nums, update_step=None, learning_rate=None):

    lr_matrix = OrderedDict((n, p) for (n, p) in net.named_parameters() if  any(name in n for name in ["alpha", "gamma"]))

    for e in range(update_step):

        net.train()

        x, y = get_next_batch(trainloader)
        x, y = x.to(device), y.to(device)

        outputs, kl = net(x, lr_matrix)

        outputs = F.softmax(outputs, dim=1).reshape(4, x.shape[0], 10)
        outputs = torch.log(torch.mean(outputs, dim=0))

        # Calculate NLL loss
        nll_loss = F.nll_loss(outputs, y)

        loss = nll_loss + kl / train_nums

        grads = torch.autograd.grad(loss, lr_matrix.values())
        lr_matrix = OrderedDict((n, p - learning_rate * grad) for ((n, p), grad) in zip(lr_matrix.items(), grads))

    return evaluate(net, lr_matrix, testloader)


def evaluate(net, lr_matrices, dataloader):

    net.eval()

    eval_cor = 0
    eval_num = 0
    y_prob = []
    y_true = []

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)

            outputs, kl = net(x, lr_matrices)

            outputs = F.softmax(outputs, dim=1).reshape(4, x.shape[0], 10)
            outputs = torch.mean(outputs, dim=0)

            eval_cor += (torch.sum(torch.argmax(outputs, dim=1) == y)).item()
            eval_num += y.shape[0]

            y_prob.append(outputs.detach().cpu())
            y_true.append(y.cpu())

        y_prob = torch.cat(y_prob, axis=0)
        y_true = torch.cat(y_true, axis=0)

        ece = multiclass_calibration_error(y_prob, y_true, num_classes=10, n_bins=15, norm="l1")
        mce = multiclass_calibration_error(y_prob, y_true, num_classes=10, n_bins=15, norm="max")

        make_model_diagrams(y_prob, y_true, ece, mce)

        print(ece, mce)

        return eval_cor, eval_num, y_prob, y_true

c_start = 0
c_end = 10

test_accs = []
test_eces = []
test_mces = []

for i in range(1):

    set_seed(i + 5173)

    test_cors = []
    test_nums = []
    test_probs = []
    test_trues = []

    for c in range(c_start, c_end):

        print("Start evaluating client {}".format(c))

        train_data = read_client_data(dataset, c, is_train=True)
        train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

        test_data = read_client_data(dataset, c, is_train=False)
        test_loader = DataLoader(test_data, batch_size=128, shuffle=False)

        stats = fine_tune(model, train_loader, test_loader, len(train_data), update_step=60, learning_rate=0.001)

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







