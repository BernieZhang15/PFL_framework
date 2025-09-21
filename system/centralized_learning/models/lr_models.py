import torch
import torch.nn as nn
import torch.nn.functional as F

class lowRankLinear(nn.Module):
    def __init__(self, input_size, output_size, rank):
        super(lowRankLinear, self).__init__()
        self.rank = rank
        self.U = nn.Parameter(torch.randn(input_size, rank))
        self.V = nn.Parameter(torch.randn(rank, output_size))
        self.bias = nn.Parameter(torch.zeros(output_size))

        self.orthogonal()

    def orthogonal(self):
        with torch.no_grad():
            self.U.data = torch.linalg.qr(self.U.data, mode='reduced')[0]

    def forward(self, x):
        weight = self.U @ self.V
        return F.linear(x, weight.T, self.bias)

class lr_model(nn.Module):
    def __init__(self, input_size=3, output_size=10, rank=4):
        super(lr_model, self).__init__()
        self.rank = rank
        self.input_size = input_size
        self.output_size = output_size

        self.conv1 = nn.Conv2d(self.input_size, 32, 3, padding=1, stride=1, bias=True)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1, stride=1, bias=True)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1, stride=1, bias=True)

        self.fc1 = lowRankLinear(2048, 512, rank)
        self.fc2 = lowRankLinear(512, 256, rank)
        self.fc3 = lowRankLinear(256, output_size, rank)

    def orthogonal(self):
        self.fc1.orthogonal()
        self.fc2.orthogonal()
        self.fc3.orthogonal()

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(out)
        out = F.max_pool2d(out, 2)

        out = self.conv2(out)
        out = F.relu(out)
        out = F.max_pool2d(out, 2)

        out = self.conv3(out)
        out = F.relu(out)
        out = F.max_pool2d(out, 2)

        out = torch.flatten(out, 1)

        out = self.fc1(out)
        out = F.relu(out, inplace=True)

        out = self.fc2(out)
        out = F.relu(out, inplace=True)

        out = self.fc3(out)

        return out


class fr_model(nn.Module):
    def __init__(self, input_size=3, output_size=10):
        super(fr_model, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

        self.conv1 = nn.Conv2d(self.input_size, 32, 3, padding=1, stride=1, bias=True)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1, stride=1, bias=True)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1, stride=1, bias=True)

        self.fc1 = nn.Linear(2048, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, output_size)

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(out)
        out = F.max_pool2d(out, 2)

        out = self.conv2(out)
        out = F.relu(out)
        out = F.max_pool2d(out, 2)

        out = self.conv3(out)
        out = F.relu(out)
        out = F.max_pool2d(out, 2)

        out = torch.flatten(out, 1)

        out = self.fc1(out)
        out = F.relu(out, inplace=True)

        out = self.fc2(out)
        out = F.relu(out, inplace=True)

        out = self.fc3(out)

        return out


if __name__=="__main__":
    random_sample = torch.randn(10, 3, 32, 32)
    net = lr_model(3, 10, 4)
    out = net(random_sample)









