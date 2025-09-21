import os
import h5py
import copy
import time
import torch
import random
import numpy as np
from utils.data_utils import read_client_data
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.functional.classification import multiclass_calibration_error


class Server(object):
    def __init__(self, args, times):
        self.args = args
        self.seed = args.seed
        self.device = args.device
        self.dataset = args.dataset
        self.num_classes = args.num_classes
        self.global_rounds = args.global_rounds
        self.local_epochs = args.local_epochs
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.global_model = copy.deepcopy(args.model)
        self.num_clients = args.num_clients
        self.join_ratio = args.join_ratio
        self.random_join_ratio = args.random_join_ratio
        self.num_join_clients = int(self.num_clients * self.join_ratio)
        self.current_num_join_clients = self.num_join_clients
        self.algorithm = args.algorithm
        self.time_select = args.time_select
        self.time_threshold = args.time_threshold
        self.save_folder_name = args.save_folder_name
        self.top_cnt = 100
        self.auto_break = args.auto_break

        self.clients = []
        self.selected_clients = []
        self.train_slow_clients = []
        self.send_slow_clients = []

        self.uploaded_weights = []
        self.uploaded_ids = []
        self.uploaded_models = []

        self.rs_test_acc = []
        self.rs_test_ece = []
        self.rs_test_mce = []
        self.rs_train_loss = []

        self.times = times
        self.eval_gap = args.eval_gap
        self.client_drop_rate = args.client_drop_rate
        self.train_slow_rate = args.train_slow_rate
        self.send_slow_rate = args.send_slow_rate

        self.num_new_clients = args.num_new_clients
        self.new_clients = []
        self.eval_new_clients = False
        self.fine_tuning_epoch = args.fine_tuning_epoch

        comment = (f" {self.dataset} {self.algorithm}_{self.local_epochs}_{self.learning_rate}_{self.num_clients}_"
                   f"{self.join_ratio}_{args.seed}")
        self.writer = SummaryWriter(comment=comment)


    def set_clients(self, clientObj):
        for i, train_slow, send_slow in zip(range(self.num_clients), self.train_slow_clients, self.send_slow_clients):
            train_data = read_client_data(self.dataset, i, is_train=True)
            test_data = read_client_data(self.dataset, i, is_train=False)
            client = clientObj(self.args, id=i, train_samples=len(train_data), test_samples=len(test_data),
                               train_slow=train_slow, send_slow=send_slow)
            self.clients.append(client)

    def select_slow_clients(self, slow_rate):
        slow_clients = [False for i in range(self.num_clients)]
        idx = [i for i in range(self.num_clients)]
        idx_ = np.random.choice(idx, int(slow_rate * self.num_clients))
        for i in idx_:
            slow_clients[i] = True

        return slow_clients

    def set_slow_clients(self):
        self.train_slow_clients = self.select_slow_clients(self.train_slow_rate)
        self.send_slow_clients = self.select_slow_clients(self.send_slow_rate)

    def select_clients(self):
        self.current_num_join_clients = (
            np.random.choice(range(self.num_join_clients, self.num_clients + 1), 1, replace=False)[0]
            if self.random_join_ratio else self.num_join_clients
        )

        selected_clients = list(np.random.choice(self.clients, self.current_num_join_clients, replace=False))

        return selected_clients

    def send_models(self):
        assert (len(self.clients) > 0)

        for client in self.clients:
            start_time = time.time()
            
            client.set_parameters(self.global_model)

            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

    def receive_models(self):
        assert (len(self.selected_clients) > 0)

        active_clients = random.sample(
            self.selected_clients, int((1 - self.client_drop_rate) * self.current_num_join_clients))

        self.uploaded_ids = []
        self.uploaded_weights = []
        self.uploaded_models = []
        tot_samples = 0

        for client in active_clients:
            try:
                client_time_cost = client.train_time_cost['total_cost'] / client.train_time_cost['num_rounds'] + \
                        client.send_time_cost['total_cost'] / client.send_time_cost['num_rounds']
            except ZeroDivisionError:
                client_time_cost = 0

            if client_time_cost <= self.time_threshold:
                tot_samples += client.train_samples
                self.uploaded_ids.append(client.id)
                self.uploaded_weights.append(client.train_samples)
                self.uploaded_models.append(client.model)

        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples

    def aggregate_parameters(self):
        assert (len(self.uploaded_models) > 0)

        for param in self.global_model.parameters():
            param.data.zero_()
            
        for w, client_model in zip(self.uploaded_weights, self.uploaded_models):
            self.add_parameters(w, client_model)

    def add_parameters(self, w, client_model):
        for server_param, client_param in zip(self.global_model.parameters(), client_model.parameters()):
            server_param.data += client_param.data.clone() * w

    def save_global_model(self):
        model_path = os.path.join("models", self.dataset)
        os.makedirs(model_path, exist_ok=True)
        model_path = os.path.join(model_path, f"{self.algorithm}_server_{self.seed}_{self.learning_rate}.pt")
        torch.save(self.global_model, model_path)

    def load_model(self):
        model_path = os.path.join("models", self.dataset)
        model_path = os.path.join(model_path, f"{self.algorithm}_server.pt")
        assert (os.path.exists(model_path))
        self.global_model = torch.load(model_path)

    def model_exists(self):
        model_path = os.path.join("models", self.dataset, self.algorithm + "_server.pt")
        return os.path.exists(model_path)
        
    def save_results(self):

        timestamp = time.strftime("%Y_%m_%d_%H_%M_%S")

        algo = f"{self.dataset}_{self.algorithm}_{self.local_epochs}_{self.learning_rate}_{self.num_clients}"
        result_path = os.path.join("..", "results")
        os.makedirs(result_path, exist_ok=True)

        if self.rs_test_acc:
            algo = f"{algo}_{timestamp}"
            file_path = os.path.join(result_path, f"{algo}.h5")
            print(f"File path: {file_path}")

            with h5py.File(file_path, 'w') as hf:
                hf.create_dataset('rs_test_acc', data=self.rs_test_acc)
                hf.create_dataset('rs_test_ece', data=self.rs_test_ece)
                hf.create_dataset('rs_test_mce', data=self.rs_test_mce)

    def save_item(self, item, item_name):
        if not os.path.exists(self.save_folder_name):
            os.makedirs(self.save_folder_name)
        torch.save(item, os.path.join(self.save_folder_name, "server_" + item_name + ".pt"))

    def load_item(self, item_name):
        return torch.load(os.path.join(self.save_folder_name, "server_" + item_name + ".pt"))

    def test_metrics(self):
        if self.eval_new_clients and self.num_new_clients > 0:
            self.fine_tuning_new_clients()
            return self.test_clients_metrics(self.new_clients)

        return self.test_clients_metrics(self.clients)

    def test_clients_metrics(self, clients):
        num_samples = []
        tot_correct = []
        tot_prob = []
        tot_true = []
        for c in clients:
            ct, ns, prob, true = c.test_metrics(c.test_samples)
            num_samples.append(ns)
            tot_correct.append(ct * 1.0)
            tot_prob.append(prob)
            tot_true.append(true)

        tot_prob = torch.cat(tot_prob, dim=0)
        tot_true = torch.cat(tot_true, dim=0)

        ids = [c.id for c in self.clients]

        return ids, num_samples, tot_correct, tot_prob, tot_true

    def train_metrics(self):
        if self.eval_new_clients and self.num_new_clients > 0:
            return [0], [1], [0]
        
        num_samples = []
        losses = []
        for c in self.clients:
            cl, ns = c.train_metrics(c.train_samples)
            num_samples.append(ns)
            losses.append(cl * 1.0)

        ids = [c.id for c in self.clients]

        return ids, num_samples, losses

    def evaluate(self, global_round):
        stats = self.test_metrics()
        stats_train = self.train_metrics()

        test_acc = sum(stats[2]) * 1.0 / sum(stats[1])
        accs = [a / n for a, n in zip(stats[2], stats[1])]
        train_loss = sum(stats_train[2]) * 1.0 / sum(stats_train[1])

        test_ece = multiclass_calibration_error(stats[3], stats[4], num_classes=self.num_classes, n_bins=15, norm='l1')
        test_mce = multiclass_calibration_error(stats[3], stats[4], num_classes=self.num_classes, n_bins=15, norm='max')

        self.rs_test_acc.append(test_acc)
        self.rs_test_ece.append(test_ece)
        self.rs_test_mce.append(test_mce)

        self.writer.add_scalar("Train_loss", train_loss, global_round)
        self.writer.add_scalar("Test_acc", test_acc, global_round)

        print("Averaged Train Loss: {:.4f}".format(train_loss))
        print("Averaged Test Accuracy: {:.4f}".format(test_acc))
        print("Std Test Accuracy: {:.4f}".format(np.std(accs)))
        print("Test ECE: {:.4f}".format(test_ece))
        print("Test MCE: {:.4f}".format(test_mce))

    @staticmethod
    def print_(test_acc, test_ece, train_loss):
        print("Average Test Accuracy: {:.4f}".format(test_acc))
        print("Average Test ECE: {:.4f}".format(test_ece))
        print("Average Train Loss: {:.4f}".format(train_loss))


    def check_done(self, acc_lss, top_cnt=None, div_value=None):
        for acc_ls in acc_lss:
            if top_cnt is not None and div_value is not None:
                find_top = len(acc_ls) - torch.topk(torch.tensor(acc_ls), 1).indices[0] > top_cnt
                find_div = len(acc_ls) > 1 and np.std(acc_ls[-top_cnt:]) < div_value
                if find_top and find_div:
                    pass
                else:
                    return False
            elif top_cnt is not None:
                find_top = len(acc_ls) - torch.topk(torch.tensor(acc_ls), 1).indices[0] > top_cnt
                if find_top:
                    pass
                else:
                    return False
            elif div_value is not None:
                find_div = len(acc_ls) > 1 and np.std(acc_ls[-top_cnt:]) < div_value
                if find_div:
                    pass
                else:
                    return False
            else:
                raise NotImplementedError
        return True


    def set_new_clients(self, clientObj):
        for i in range(self.num_clients, self.num_clients + self.num_new_clients):
            train_data = read_client_data(self.dataset, i, is_train=True)
            test_data = read_client_data(self.dataset, i, is_train=False)
            client = clientObj(self.args, id=i, train_samples=len(train_data), test_samples=len(test_data),
                               train_slow=False, send_slow=False)
            self.new_clients.append(client)

    def fine_tuning_new_clients(self):
        for client in self.new_clients:
            client.set_parameters(self.global_model)

            if self.fine_tuning_epoch > 0:
                opt = torch.optim.SGD(client.model.parameters(), lr=self.learning_rate)
                ce_loss = torch.nn.CrossEntropyLoss()
                trainloader = client.load_train_data()
                client.model.train()
                for e in range(self.fine_tuning_epoch):
                    for i, (x, y) in enumerate(trainloader):
                        x, y = x.to(client.device), y.to(client.device)
                        output = client.model(x)
                        loss = ce_loss(output, y)
                        opt.zero_grad()
                        loss.backward()
                        opt.step()
