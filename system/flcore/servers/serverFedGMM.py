"""
FedGMM Server implementation.
Adapted from: https://github.com/zshuai8/FedGMM_ICML2023
Paper: "Personalized Federated Learning under Mixture of Distributions" (ICML 2023)
"""

import copy
import time
import torch
import random
import numpy as np
from flcore.servers.serverbase import Server
from flcore.clients.clientFedGMM import clientFedGMM


class FedGMM(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        self.n_learners = args.n_learners
        self.n_gmm = args.n_gmm

        # K global models
        self.global_models = [copy.deepcopy(args.model) for _ in range(self.n_learners)]
        self.global_model = self.global_models[0]  # backward compat

        # Global GMM and autoencoder will be initialized from first round
        self.global_gmm_params = None  # (pi, mu, var)
        self.global_autoencoder = None

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientFedGMM)
        self.selected_clients = None

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print(f"FedGMM: n_learners={self.n_learners}, n_gmm={self.n_gmm}")
        print("Finished creating server and clients.")

        self.Budget = []

    def send_models(self):
        assert len(self.clients) > 0

        for client in self.clients:
            start_time = time.time()

            client.set_parameters(
                self.global_models,
                gmm_params=self.global_gmm_params,
                ae_model=self.global_autoencoder,
            )

            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

    def receive_models(self):
        assert len(self.selected_clients) > 0

        active_clients = random.sample(
            self.selected_clients,
            int((1 - self.client_drop_rate) * self.current_num_join_clients)
        )

        self.uploaded_ids = []
        self.uploaded_weights = []
        # Store lists of K models per client
        self.uploaded_models_list = []
        # Store GMM params per client
        self.uploaded_gmm_params = []
        # Store learners_weights per client
        self.uploaded_learners_weights = []
        # Store autoencoder per client
        self.uploaded_autoencoders = []
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
                self.uploaded_models_list.append(client.models)
                self.uploaded_gmm_params.append(client.gmm.get_all_parameter())
                self.uploaded_learners_weights.append(client.learners_weights)
                self.uploaded_autoencoders.append(client.autoencoder)

        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples

        # Also set uploaded_models for base class evaluate() compatibility
        self.uploaded_models = [m[0] for m in self.uploaded_models_list]

    def aggregate_parameters(self):
        assert len(self.uploaded_models_list) > 0

        n_clients = len(self.uploaded_weights)

        # Aggregate each learner separately (FedAvg)
        for lid in range(self.n_learners):
            for param in self.global_models[lid].parameters():
                param.data.zero_()

            for w, client_models in zip(self.uploaded_weights, self.uploaded_models_list):
                for server_param, client_param in zip(
                    self.global_models[lid].parameters(),
                    client_models[lid].parameters()
                ):
                    server_param.data += client_param.data.clone() * w

        self.global_model = self.global_models[0]

        # Aggregate GMM parameters: per-component weighted average
        # Weight each client's component k by (n_samples * pi_k), matching the
        # original FedGMM paper's aggregation strategy.
        if len(self.uploaded_gmm_params) > 0:
            pi_agg = None
            mu_agg = None
            var_agg = None

            for w, params in zip(self.uploaded_weights, self.uploaded_gmm_params):
                pi, mu, var = params
                if pi_agg is None:
                    pi_agg = w * pi.clone()
                    mu_agg = w * mu.clone()
                    var_agg = w * var.clone()
                else:
                    pi_agg += w * pi.clone()
                    mu_agg += w * mu.clone()
                    var_agg += w * var.clone()

            # Regularize aggregated covariance to prevent singularity
            if len(var_agg.shape) == 4:  # full covariance
                eps_mat = torch.eye(var_agg.shape[-1], device=var_agg.device) * 0.1
                var_agg = var_agg + eps_mat
            else:
                var_agg = torch.clamp(var_agg, min=0.1)

            # Clamp and renormalize pi to prevent log(0)
            pi_agg = torch.clamp(pi_agg, min=1e-6)
            pi_agg = pi_agg / pi_agg.sum()

            self.global_gmm_params = (pi_agg, mu_agg, var_agg)

        # Aggregate autoencoder parameters
        if len(self.uploaded_autoencoders) > 0:
            if self.global_autoencoder is None:
                self.global_autoencoder = copy.deepcopy(self.uploaded_autoencoders[0])

            for param in self.global_autoencoder.parameters():
                param.data.zero_()

            for w, ae in zip(self.uploaded_weights, self.uploaded_autoencoders):
                for server_param, client_param in zip(
                    self.global_autoencoder.parameters(), ae.parameters()
                ):
                    server_param.data += client_param.data.clone() * w

    def train(self):
        for i in range(self.global_rounds + 1):
            s_t = time.time()
            self.selected_clients = self.select_clients()

            # Only send global models after first round
            if i > 0:
                self.send_models()

            if i % self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
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
            self.set_new_clients(clientFedGMM)
            print("\nEvaluate new clients")
            self.evaluate(self.global_rounds + 1)
