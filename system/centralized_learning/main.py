import torch
import torchvision
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torchvision.transforms as transforms
from models.FourierFTModel import FourierFTFedAvg

freq = 10
lr = 0.05
batch_size = 256
num_epochs = 100

device = 'cuda' if torch.cuda.is_available() else 'cpu'

seed = 675
cudnn.benchmark = True
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

print("| Preparing CIFAR-10 dataset...")

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10("./data", train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10("./data", train=False, download=True, transform=transform)

train_samples = len(trainset)
test_samples = len(testset)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)

print('\n[Phase 2] : Model setup')
net = FourierFTFedAvg().to(device)
ce_criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.995)

print('\n[Phase 3] : Training model')
print('| Training Epochs = ' + str(num_epochs))
print('| Initial Learning Rate = ' + str(lr))

best_acc = 0

for i in range(num_epochs):

    losses = 0
    net.train()

    for X, Y in trainloader:

        x, y = X.to(device), Y.to(device)

        optimizer.zero_grad()

        output, kl = net(x)

        output = F.softmax(output, dim=2)
        output = torch.log(torch.mean(output, dim=0))
        loss = F.nll_loss(output, y)

        loss += kl / train_samples

        loss.backward()

        losses += loss.item()

        optimizer.step()

    scheduler.step()

    print('Train Loss at epoch {}: {:.4f}'.format(i, losses / len(trainloader)))

    if i % freq == 0:

        net.eval()

        test_cor = 0
        test_num = 0

        y_prob = []
        y_true = []

        with torch.no_grad():

            for x, y in testloader:

                x, y = x.to(device), y.to(device)

                output, kl = net(x)

                output = torch.mean(F.softmax(output, dim=2), dim=0)

                test_cor += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                test_num += y.shape[0]

                y_prob.append(output.detach().cpu())
                y_true.append(y.cpu())

        test_acc = test_cor / test_num

        print("Evaluation results for round {}".format(i))
        print("Averaged Test Accuracy: {:.4f}".format(test_acc))

        if test_acc > best_acc:
            best_acc = test_acc

print(best_acc)






