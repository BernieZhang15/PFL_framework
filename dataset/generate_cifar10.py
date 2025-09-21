import os
import random
import torchvision
import numpy as np
import torchvision.transforms as transforms
from utils.gen_noisy_data import corrupt_images
from utils.dataset_utils import check, separate_data, split_data, save_file

random.seed(52)
np.random.seed(52)

data_root =  "./rawdata/cifar10"

def generate_cifar10(dir_path, num_clients, num_classes, niid, balance, partition, is_Noisy, class_per_client):


    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # Setup directory for train/test data
    config_path = dir_path + "config.json"
    train_path = dir_path + "train/"
    test_path = dir_path + "test/"

    noisy_train_path = dir_path + "noisy-train/"
    noisy_test_path = dir_path + "noisy-test/"

    if check(config_path, train_path, test_path, num_clients, num_classes, niid, balance, partition):
        return

    # Get Cifar10 data
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root=data_root, train=False, download=True, transform=transform)

    dataset_image = np.concatenate([trainset.data, testset.data], axis=0)
    dataset_label = np.concatenate([trainset.targets, testset.targets], axis=0)

    x, y, statistic = separate_data((dataset_image, dataset_label), num_clients, num_classes, niid, balance,
                                    partition, class_per_client=class_per_client)

    train_data, test_data = split_data(x, y)

    if is_Noisy:
        corrupted_train_data, corrupted_test_data = corrupt_images(train_data, test_data)
        check(config_path, noisy_train_path, noisy_test_path, num_clients, num_classes, niid, balance, partition)
        save_file(config_path, noisy_train_path, noisy_test_path, corrupted_train_data, corrupted_test_data, num_clients,
                  num_classes, statistic, niid, balance, partition)

    save_file(config_path, train_path, test_path, train_data, test_data, num_clients, num_classes, statistic, niid,
              balance, partition)


if __name__ == "__main__":
    niid = True
    balance = True
    partition = 'pat'
    is_Noisy = False

    num_clients = 10
    num_classes = 10
    class_per_client = 5
    dir_path = "Cifar10-dir-test/"

    generate_cifar10(dir_path, num_clients, num_classes, niid, balance, partition, is_Noisy, class_per_client)
