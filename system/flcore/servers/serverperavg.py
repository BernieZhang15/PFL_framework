import copy
import time
import numpy as np
from flcore.servers.serverbase import Server
from flcore.clients.clientperavg import clientPerAvg
from torchmetrics.functional.classification import multiclass_calibration_error


class PerAvg(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientPerAvg)
        self.selected_clients = None

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

    def train(self):
        for i in range(self.global_rounds + 1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()

            if i % self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model with one step update")
                self.evaluate_one_step(i)

            # choose several clients to send back updated model to server
            for client in self.selected_clients:
                client.train()

            self.receive_models()
            self.aggregate_parameters()

            print('-' * 25, 'time cost', '-' * 25, time.time() - s_t)

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\nBest accuracy.")
        print(max(self.rs_test_acc))

        self.save_results()
        self.save_global_model()

        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(clientPerAvg)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate(self.global_rounds + 1)

    def evaluate_one_step(self, global_round):
        models_temp = []
        for c in self.clients:
            models_temp.append(copy.deepcopy(c.model))
            c.train_one_step()

        stats = self.test_metrics()
        stats_train = self.train_metrics()

        # set the local model back on clients for training process
        for i, c in enumerate(self.clients):
            c.clone_model(models_temp[i], c.model)

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

    def fine_tuning_new_clients(self):
        for client in self.new_clients:
            client.set_parameters(self.global_model)
            client.train_one_step()