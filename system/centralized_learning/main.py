import torch
import numpy as np
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from models.FourierFTModel import FourierFTModel
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

freq = 5
lr = 1e-3
batch_size = 16
num_epochs = 4000

device = 'cuda' if torch.cuda.is_available() else 'cpu'

seed = 675
cudnn.benchmark = True
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

writer = SummaryWriter(comment=' Bayes low')

class SimpleDataset(Dataset):
    def __init__(self, X, y):

        self.X = torch.as_tensor(X, dtype=torch.float32).view(-1, 1)
        self.y = torch.as_tensor(y, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def f_true(x, eps=0.5):
    return np.sin(2 * np.pi * x) + eps * np.sin(20 * np.pi * x)

def make_data(n_train=40, n_test=400, eps=0.5, noise_std=0.1, seed=0):
    rng = np.random.default_rng(seed)
    x_train = rng.uniform(low=0, high=1, size=n_train)
    y_train = f_true(x_train, eps=eps) + rng.normal(0, noise_std, size=n_train)

    x_test = np.linspace(0,1, n_test)
    y_test = f_true(x_test, eps=eps)

    return x_train, y_train, x_test, y_test


x_train, y_train, x_test, y_test = make_data()
trainloader = DataLoader(SimpleDataset(x_train, y_train), batch_size=batch_size, shuffle=True, num_workers=0)
testloader = DataLoader(SimpleDataset(x_test, y_test), batch_size=400, shuffle=False, num_workers=0)

print('\n[Phase 2] : Model setup')
net = FourierFTModel(ens_num=4).to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)

print('\n[Phase 3] : Training model')
print('| Training Epochs = ' + str(num_epochs))
print('| Initial Learning Rate = ' + str(lr))

for i in range(num_epochs):

    losses = 0
    net.train()

    for X, Y in trainloader:

        x, y = X.to(device), Y.to(device)

        optimizer.zero_grad()

        output, kl = net(x)

        output = torch.mean(output, dim=0)
        loss = F.mse_loss(output, y)

        loss += kl / 32 * 0.0001

        loss.backward()

        losses += loss.item()

        optimizer.step()

    print('Train Loss at epoch {}: {:.4f}'.format(i, losses / len(trainloader)))

    if i % freq == 0:

        net.eval()

        with torch.no_grad():

            for x, y in testloader:

                x, y = x.to(device), y.to(device)

                output, kl = net(x)

                mu = torch.mean(output, dim=0)
                sigma = torch.std(output, dim=0)

                var = sigma**2 + 0.01

                nll = 0.5 * torch.mean(torch.log(2 * torch.pi * var) + (y - mu)**2 / var)

            writer.add_scalar('Val/NLL', nll, i)














