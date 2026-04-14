"""
FedGMM Client implementation.
Adapted from: https://github.com/zshuai8/FedGMM_ICML2023
Paper: "Personalized Federated Learning under Mixture of Distributions" (ICML 2023)
"""

import copy
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from flcore.clients.clientbase import Client
from flcore.trainmodel.gmm import GaussianMixture, Autoencoder
from utils.data_utils import read_client_data


class clientFedGMM(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        self.n_learners = args.n_learners
        self.n_gmm = args.n_gmm
        self.embedding_dim = getattr(args, 'embedding_dim', 64)

        # Determine input image size
        if "imagenet" in self.dataset.lower():
            input_size = (3, 64, 64)
        elif "mnist" in self.dataset.lower() or "fmnist" in self.dataset.lower():
            input_size = (1, 28, 28)
        else:
            input_size = (3, 32, 32)

        # K learner models
        self.models = [copy.deepcopy(args.model) for _ in range(self.n_learners)]
        self.model = self.models[0]  # for base class compatibility

        # Optimizers for each learner
        self.optimizers = [
            torch.optim.SGD(m.parameters(), lr=self.learning_rate)
            for m in self.models
        ]
        self.lr_schedulers = [
            torch.optim.lr_scheduler.ExponentialLR(opt, gamma=args.learning_rate_decay_gamma)
            for opt in self.optimizers
        ]

        # Autoencoder
        self.autoencoder = Autoencoder(self.embedding_dim, input_size).to(self.device)
        self.ae_optimizer = torch.optim.Adam(self.autoencoder.parameters(), lr=1e-3)
        self.ae_criterion = nn.MSELoss(reduction='none')
        self.recon_weight = 10.0
        self.nll_weight = 1.0

        # GMM
        self.gmm = GaussianMixture(
            n_components=self.n_gmm,
            n_features=self.embedding_dim,
            device=self.device
        )

        # Learner weights: (n_gmm, n_learners)
        self.learners_weights = (
            torch.ones(self.n_gmm, self.n_learners, device=self.device)
            / (self.n_learners * self.n_gmm)
        )

        self.gmm_initialized = False

        # Override the single optimizer/scheduler
        self.optimizer = self.optimizers[0]
        self.learning_rate_scheduler = self.lr_schedulers[0]

    def _initialize_gmm(self, trainloader):
        """Initialize GMM with k-means on encoded training data."""
        self.autoencoder.eval()
        data_list = []
        with torch.no_grad():
            for x, y in trainloader:
                x = x.to(self.device)
                z = self.autoencoder.encode(x)
                data_list.append(z)
        data = torch.cat(data_list, dim=0)
        self.gmm.initialize_gmm(data)
        self.gmm_initialized = True

    def _predict_gmm(self, x):
        """
        Compute mixture assignment probabilities p(k,m|x).
        Returns: (batch, n_gmm, n_learners) normalized probabilities.
        """
        self.autoencoder.eval()
        with torch.no_grad():
            z = self.autoencoder.encode(x)
            log_prob_gmm = self.gmm.calc_log_prob(z).unsqueeze(2)  # (n, n_gmm, 1)
            weighted_log = log_prob_gmm + torch.log(
                self.learners_weights.unsqueeze(0) + 1e-12
            )  # (n, n_gmm, n_learners)
            flat = weighted_log.view(-1, self.n_gmm * self.n_learners)
            prob = torch.softmax(flat, dim=1)
        return prob.view(-1, self.n_gmm, self.n_learners)

    def _gather_losses(self, trainloader):
        """Compute per-sample loss for each learner. Returns (n_samples, n_learners)."""
        n_samples = self.train_samples
        all_losses = torch.zeros(n_samples, self.n_learners, device=self.device)

        for model in self.models:
            model.eval()

        sample_idx = 0
        with torch.no_grad():
            for x, y in trainloader:
                x, y = x.to(self.device), y.to(self.device)
                bs = x.shape[0]
                for lid, model in enumerate(self.models):
                    output = model(x)
                    loss_vec = F.cross_entropy(output, y, reduction='none')
                    all_losses[sample_idx:sample_idx + bs, lid] = loss_vec
                sample_idx += bs

        return all_losses

    def _calc_sample_weights(self, trainloader):
        """
        E-step: compute per-sample, per-component, per-learner weights.
        Returns: (n_samples, n_gmm, n_learners)
        """
        all_losses = self._gather_losses(trainloader)  # (n_samples, n_learners)

        n_samples = self.train_samples
        all_log_prob = torch.zeros(n_samples, self.n_gmm, device=self.device)

        self.autoencoder.eval()
        sample_idx = 0
        with torch.no_grad():
            for x, y in trainloader:
                x = x.to(self.device)
                bs = x.shape[0]
                z = self.autoencoder.encode(x)
                log_prob = self.gmm.calc_log_prob(z)  # (bs, n_gmm)
                all_log_prob[sample_idx:sample_idx + bs] = log_prob
                sample_idx += bs

            # (n, 1, n_learners) log weights + (n, n_gmm, 1) log_prob - (n, 1, n_learners) losses
            weighted_log = (
                torch.log(self.learners_weights + 1e-12).unsqueeze(0)
                + all_log_prob.unsqueeze(2)
                - all_losses.unsqueeze(1)
            )  # (n_samples, n_gmm, n_learners)

            # Clamp to prevent extreme values in softmax
            weighted_log = torch.clamp(weighted_log, min=-50.0, max=50.0)

            flat = weighted_log.view(n_samples, -1)
            sample_weights = F.softmax(flat, dim=1).view(
                n_samples, self.n_gmm, self.n_learners
            )

        return sample_weights

    def _m_step(self, sample_weights, trainloader):
        """M-step: update GMM parameters and learner weights."""
        self.autoencoder.eval()
        data_list = []
        with torch.no_grad():
            for x, y in trainloader:
                x = x.to(self.device)
                z = self.autoencoder.encode(x)
                data_list.append(z)
        data = torch.cat(data_list, dim=0)

        # GMM M-step uses the marginal over learners: sum over n_learners -> (n, n_gmm, 1)
        resp_gmm = sample_weights.sum(dim=2).unsqueeze(2)
        self.gmm.m_step_with_response(data, resp_gmm)

        # Update learner mixture weights
        self.learners_weights = sample_weights.mean(dim=0)  # (n_gmm, n_learners)

    def _train_autoencoder(self, trainloader):
        """Train autoencoder with reconstruction + NLL loss."""
        self.autoencoder.train()
        for x, y in trainloader:
            x = x.to(self.device)
            self.ae_optimizer.zero_grad()

            x_flat = x.view(x.size(0), -1)
            x_recon = self.autoencoder(x)
            recon_loss = self.ae_criterion(x_recon, x_flat).sum(dim=1).mean()

            z = self.autoencoder.encode(x)
            nll_loss = -self.gmm.score_samples(z).mean()

            loss = self.recon_weight * recon_loss + self.nll_weight * nll_loss

            # Skip update if loss is NaN to prevent corruption
            if torch.isnan(loss) or torch.isinf(loss):
                self.ae_optimizer.zero_grad()
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.autoencoder.parameters(), max_norm=5.0)
            self.ae_optimizer.step()

    def train(self):
        # Use a non-shuffled loader to keep sample ordering consistent for weights
        trainloader_ordered = self.load_train_data()
        trainloader = self.load_train_data()

        start_time = time.time()

        # Initialize GMM on first round
        if not self.gmm_initialized:
            self._initialize_gmm(trainloader_ordered)

        # EM step
        sample_weights = self._calc_sample_weights(trainloader_ordered)  # (n, n_gmm, n_learners)
        self._m_step(sample_weights, trainloader_ordered)

        # Per-learner weights: sum over GMM components -> (n_samples, n_learners)
        learner_weights = sample_weights.sum(dim=1)  # (n_samples, n_learners)

        # Train each learner
        for lid in range(self.n_learners):
            model = self.models[lid]
            optimizer = self.optimizers[lid]
            model.train()

            sample_idx = 0
            for step in range(self.local_epochs):
                sample_idx = 0
                for x, y in trainloader_ordered:
                    x, y = x.to(self.device), y.to(self.device)
                    bs = x.shape[0]

                    optimizer.zero_grad()
                    output = model(x)
                    loss_vec = F.cross_entropy(output, y, reduction='none')

                    # Apply per-sample weights
                    w = learner_weights[sample_idx:sample_idx + bs, lid].to(self.device)
                    loss = (loss_vec * w).sum() / (w.sum() + 1e-12)

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                    optimizer.step()
                    sample_idx += bs

        # Train autoencoder
        self._train_autoencoder(trainloader)

        if self.learning_rate_decay:
            for scheduler in self.lr_schedulers:
                scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    def set_parameters(self, models_list, gmm_params=None, ae_model=None):
        """Receive K models + GMM params + autoencoder from server."""
        for lid in range(self.n_learners):
            for new_param, old_param in zip(models_list[lid].parameters(), self.models[lid].parameters()):
                old_param.data = new_param.data.clone()

        if gmm_params is not None:
            pi, mu, var = gmm_params
            self.gmm.update_parameter(mu=mu, var=var, _pi=pi)

        if ae_model is not None:
            for new_param, old_param in zip(ae_model.parameters(), self.autoencoder.parameters()):
                old_param.data = new_param.data.clone()

    def test_metrics(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size

        testloader = self.load_test_data(batch_size=batch_size)

        for model in self.models:
            model.eval()
        self.autoencoder.eval()

        test_acc = 0
        test_num = 0
        y_prob = []
        y_true = []

        with torch.no_grad():
            for x, y in testloader:
                x, y = x.to(self.device), y.to(self.device)

                # Mixture prediction
                p_k_x = self._predict_gmm(x).sum(dim=1)  # (batch, n_learners)

                y_pred = torch.zeros(x.shape[0], self.num_classes, device=self.device)
                for lid, model in enumerate(self.models):
                    output = F.softmax(model(x), dim=1)
                    y_pred += p_k_x[:, lid].unsqueeze(1) * output

                y_pred = torch.clamp(y_pred, min=1e-12, max=1.0)

                test_acc += (torch.argmax(y_pred, dim=1) == y).sum().item()
                test_num += y.shape[0]

                y_prob.append(y_pred.detach().cpu())
                y_true.append(y.cpu())

        y_prob = torch.cat(y_prob, dim=0)
        y_true = torch.cat(y_true, dim=0)

        return test_acc, test_num, y_prob, y_true

    def train_metrics(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size

        trainloader = self.load_train_data(batch_size=batch_size)

        for model in self.models:
            model.eval()
        self.autoencoder.eval()

        train_num = 0
        losses = 0

        with torch.no_grad():
            for x, y in trainloader:
                x, y = x.to(self.device), y.to(self.device)

                p_k_x = self._predict_gmm(x).sum(dim=1)
                y_pred = torch.zeros(x.shape[0], self.num_classes, device=self.device)
                for lid, model in enumerate(self.models):
                    output = F.softmax(model(x), dim=1)
                    y_pred += p_k_x[:, lid].unsqueeze(1) * output

                y_pred = torch.clamp(y_pred, min=1e-12, max=1.0)
                loss = F.nll_loss(torch.log(y_pred), y)

                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]

        return losses, train_num
