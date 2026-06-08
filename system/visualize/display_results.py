from matplotlib import pyplot as plt
from statistics import mean
import numpy as np
import h5py


def get_mean_std():

    file_name = input() + '.out'

    acc = []

    with open(file_name, 'r') as f:
        is_best = False
        for l in f.readlines():
            if is_best:
                acc.append(float(l))
                is_best = False
            elif 'Best accuracy' in l:
                is_best = True

    print(acc)
    print(mean(acc)*100, np.std(acc)*100)


def get_results(file_path, rs_key, index):
    color = ["red", "blue", "grey", "orange", "black", "yellow", "green"]
    for i, j in enumerate(index):
        f = h5py.File(file_path[j])

        test_acc = f[rs_key[0]][:]
        ax = plt.subplot(2, 1, 1)
        plt.plot(test_acc, color=color[i])
        print(test_acc)

        test_ece = f[rs_key[1]][:]
        ax = plt.subplot(2, 1, 2)
        plt.plot(test_ece, color=color[i])
        print(test_ece)

        test_mce = f[rs_key[2]][:]
        print(test_mce)

    plt.show()


if __name__ == "__main__":
    path = {
            1: "../../results/Cifar10-pat-5S_FedMetaBayes_0.005_50_0.2_8_2024_10_06_17_12_17.h5",
            2: "../../results/Cifar100-pat-5M_FedMetaBayes_0.001_40_0.3_4_2024_05_27_15_36_00.h5",
            3: "../results/Cifar10-pat-2M_FedMetaBayes_0.005_40_0.3_4_2024_05_28_00_16_04.h5",
            4: "../results/Cifar10-pat-2M_FedMetaBayes_0.005_40_0.3_8_2024_05_28_16_23_08.h5",
            5: "../results/Cifar10-pat-2M_FedMetaBayes_0.005_40_0.3_16_2024_05_28_05_18_32.h5",
            }

    key = ['rs_test_acc', 'rs_test_ece', 'rs_test_mce']
    get_results(path, key, [1])



