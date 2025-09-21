import numpy as np
import os
import sys
import random
import torch
import torchvision
import scipy.io as scio
import torchvision.transforms as transforms
from utils.dataset_utils import check, separate_data, split_data, save_file


random.seed(1)
np.random.seed(1)
num_clients = 50
num_classes = 10
dir_path = "mnist-0.1/"


# Allocate data to users
def generate_mnist(dir_path, num_clients, num_classes, niid, balance, partition, is_Noisy):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        
    # Setup directory for train/test data
    config_path = dir_path + "config.json"
    train_path = dir_path + "train/"
    test_path = dir_path + "test/"

    if check(config_path, train_path, test_path, num_clients, num_classes, niid, balance, partition):
        return

    # FIX HTTP Error 403: Forbidden
    from six.moves import urllib
    opener = urllib.request.build_opener()
    opener.addheaders = [('User-agent', 'Mozilla/5.0')]
    urllib.request.install_opener(opener)

    # Get MNIST data
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

    if not is_Noisy:
        trainset = torchvision.datasets.MNIST(
            root=dir_path+"rawdata", train=True, download=True, transform=transform)
        testset = torchvision.datasets.MNIST(
            root=dir_path+"rawdata", train=False, download=True, transform=transform)

        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=len(trainset.data), shuffle=False)
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=len(testset.data), shuffle=False)

        for _, train_data in enumerate(trainloader, 0):
            trainset.data, trainset.targets = train_data
        for _, test_data in enumerate(testloader, 0):
            testset.data, testset.targets = test_data

        dataset_image = []
        dataset_label = []

        dataset_image.extend(trainset.data.cpu().detach().numpy())
        dataset_image.extend(testset.data.cpu().detach().numpy())
        dataset_label.extend(trainset.targets.cpu().detach().numpy())
        dataset_label.extend(testset.targets.cpu().detach().numpy())
        dataset_image = np.array(dataset_image)
        dataset_label = np.array(dataset_label)
    else:
        data = scio.loadmat("./mnist-with-reduced-contrast-and-awgn.mat")

        dataset_image = []
        dataset_label = []

        dataset_image.extend(np.expand_dims(data['train_x'].reshape(-1, 28, 28), axis=1))
        dataset_image.extend(np.expand_dims(data['test_x'].reshape(-1, 28, 28), axis=1))
        dataset_label.extend(np.where(data['train_y']==1)[1])
        dataset_label.extend(np.where(data['test_y']==1)[1])

        dataset_image = np.array(dataset_image)
        dataset_label = np.array(dataset_label)

    X, y, statistic = separate_data((dataset_image, dataset_label), num_clients, num_classes, 
                                    niid, balance, partition)
    train_data, test_data = split_data(X, y)
    save_file(config_path, train_path, test_path, train_data, test_data, num_clients, num_classes, 
        statistic, niid, balance, partition)


if __name__ == "__main__":
    niid = True
    balance = False
    partition = "dir"
    is_Noisy = False

    generate_mnist(dir_path, num_clients, num_classes, niid, balance, partition, is_Noisy)