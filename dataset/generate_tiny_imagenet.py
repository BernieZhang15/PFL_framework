import os
import torch
import random
import numpy as np
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from utils.dataset_utils import check, separate_data, split_data, save_file

random.seed(1)
np.random.seed(1)
torch.manual_seed(1)


class ImageFolderCustom(ImageFolder):
    def __init__(self, root, dataidxs=None, transform=None, target_transform=None):
        super().__init__(root, transform=transform, target_transform=target_transform)
        if dataidxs is not None:
            self.samples = [self.samples[i] for i in dataidxs]
            self.targets = [self.targets[i] for i in dataidxs]


def dataset_to_numpy(dataset):
    images, labels = [], []
    for path, target in dataset.samples:
        img = dataset.loader(path)
        if dataset.transform is not None:
            img = dataset.transform(img)
        images.append(img.numpy())
        labels.append(target)
    return np.stack(images), np.array(labels)


def generate_dataset(dir_path, num_clients, num_classes, niid, balance, partition):
    os.makedirs(dir_path, exist_ok=True)

    config_path = os.path.join(dir_path, "config.json")
    train_path = os.path.join(dir_path, "train")
    test_path = os.path.join(dir_path, "test")

    if check(config_path, train_path, test_path, num_clients, num_classes, niid, balance, partition):
        return

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = ImageFolderCustom('/home/largeDisk/boning/data/tiny-imagenet-200/train/', transform=transform)
    testset = ImageFolderCustom('/home/largeDisk/boning/data/tiny-imagenet-200/val/', transform=transform)

    train_images, train_labels = dataset_to_numpy(trainset)
    test_images, test_labels = dataset_to_numpy(testset)

    dataset_image = np.concatenate([train_images, test_images], axis=0)
    dataset_label = np.concatenate([train_labels, test_labels], axis=0)

    X, y, statistic = separate_data((dataset_image, dataset_label), num_clients, num_classes, niid, balance, partition)
    train_data, test_data = split_data(X, y)
    save_file(config_path, train_path, test_path, train_data, test_data, num_clients, num_classes, statistic, niid, balance, partition)


if __name__ == "__main__":
    dir_path = "./tiny-imagenet-fed/"
    num_clients = 10
    num_classes = 200
    niid = True
    balance = True
    partition = 'pat'
    generate_dataset(dir_path, num_clients, num_classes, niid, balance, partition)
