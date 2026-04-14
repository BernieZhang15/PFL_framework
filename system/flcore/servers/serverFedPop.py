"""
FedPop Server implementation.
Adapted from: https://github.com/nkotelevskii/FedPop
Paper: "FedPop: A Bayesian Approach for Personalised Federated Learning" (NeurIPS 2022)

Server pattern: Same as FedRep — only aggregate the shared base (feature extractor).
Personal head parameters stay local on each client.
"""

import random
import time
from flcore.servers.serverbase import Server
from flcore.clients.clientFedPop import clientFedPop


class FedPop(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        self.set_slow_clients()
        self.set_clients(clientFedPop)
        self.selected_clients = None

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print(f"FedPop: prior_sigma={args.prior_sigma}, n_inner_iters={args.n_inner_iters}, "
              f"burn_in={args.burn_in}, sgld_lr={args.sgld_lr}")
        print("Finished creating server and clients.")

        self.Budget = []

    def train(self):
        for i in range(self.global_rounds + 1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()

            if i % self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate personalized models")
                self.evaluate(i)

            for client in self.selected_clients:
                client.train()

            self.receive_models()
            self.aggregate_parameters()

            self.Budget.append(time.time() - s_t)
            print('-' * 25, 'time cost', '-' * 25, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\nBest accuracy.")
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:]) / len(self.Budget[1:]))

        self.save_results()
        self.save_global_model()

        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(clientFedPop)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate(self.global_rounds + 1)

    def receive_models(self):
        """Only collect base (shared) parameters from clients."""
        assert len(self.selected_clients) > 0

        active_clients = random.sample(
            self.selected_clients,
            int((1 - self.client_drop_rate) * self.current_num_join_clients)
        )

        self.uploaded_ids = []
        self.uploaded_weights = []
        self.uploaded_models = []
        tot_samples = 0

        for client in active_clients:
            try:
                client_time_cost = (
                    client.train_time_cost['total_cost'] / client.train_time_cost['num_rounds']
                    + client.send_time_cost['total_cost'] / client.send_time_cost['num_rounds']
                )
            except ZeroDivisionError:
                client_time_cost = 0

            if client_time_cost <= self.time_threshold:
                tot_samples += client.train_samples
                self.uploaded_ids.append(client.id)
                self.uploaded_weights.append(client.train_samples)
                self.uploaded_models.append(client.model.base)  # Only base

        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples
