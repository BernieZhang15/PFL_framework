import os
import torch
import random
import torchvision
import numpy as np
import torchvision.transforms as transforms
from utils.dataset_utils import check, separate_data, split_data, save_file

random.seed(1)
np.random.seed(1)

data_root =  "../rawdata/fmnist"

def generate_fmnist(dir_path, num_clients, num_classes, niid, balance, partition):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        
    # Setup directory for train/test data
    config_path = dir_path + "config.json"
    train_path = dir_path + "train/"
    test_path = dir_path + "test/"

    if check(config_path, train_path, test_path, num_clients, num_classes, niid, balance, partition):
        return

    # Get FashionMNIST data
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

    trainset = torchvision.datasets.FashionMNIST(root=data_root, train=True, download=True, transform=transform)
    testset = torchvision.datasets.FashionMNIST(root=data_root, train=False, download=True, transform=transform)

    dataset_image = np.concatenate([trainset.data, testset.data], axis=0)
    dataset_label = np.concatenate([trainset.targets, testset.targets], axis=0)

    X, y, statistic = separate_data((dataset_image, dataset_label), num_clients, num_classes, 
                                    niid, balance, partition)
    train_data, test_data = split_data(X, y)
    save_file(config_path, train_path, test_path, train_data, test_data, num_clients, num_classes, 
        statistic, niid, balance, partition)


if __name__ == "__main__":
    niid = True
    balance = True
    partition = 'pat'

    num_clients = 200
    num_classes = 10
    dir_path = "fmnist-2/"

    generate_fmnist(dir_path, num_clients, num_classes, niid, balance, partition)