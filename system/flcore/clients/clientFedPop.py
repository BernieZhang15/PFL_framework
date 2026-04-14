"""
FedPop Client implementation.
Adapted from: https://github.com/nkotelevskii/FedPop
Paper: "FedPop: A Bayesian Approach for Personalised Federated Learning" (NeurIPS 2022)

Key idea: Split model into shared base (aggregated) and personal head (sampled via SGLD).
Uses Gaussian prior on head parameters and Bayesian prediction averaging.
"""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from flcore.clients.clientbase import Client


class SGLD(Optimizer):
    """Stochastic Gradient Langevin Dynamics optimizer.

    Update rule: θ_{t+1} = θ_t - lr * ∇U(θ_t) + sqrt(2 * lr * temperature) * ε
    where ε ~ N(0, I), and U = -log p(D|θ) - log p(θ).
    """

    def __init__(self, params, lr=1e-2, temperature=1.0, num_burn_in_steps=0):
        defaults = dict(lr=lr, temperature=temperature,
                        num_burn_in_steps=num_burn_in_steps)
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            temperature = group['temperature']

            for p in group['params']:
                if p.grad is None:
                    continue

                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                state['step'] += 1

                # Only add noise after burn-in
                if state['step'] > group['num_burn_in_steps']:
                    noise_std = (2.0 * lr * temperature) ** 0.5
                    noise = torch.randn_like(p.data) * noise_std
                else:
                    noise = 0.0

                p.data.add_(-lr * p.grad.data + noise)

        return loss


class clientFedPop(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        self.prior_sigma = getattr(args, 'prior_sigma', 0.54)
        self.n_inner_iters = getattr(args, 'n_inner_iters', 10)
        self.burn_in = getattr(args, 'burn_in', 5)
        self.sgld_lr = getattr(args, 'sgld_lr', 0.01)
        self.plocal_steps = getattr(args, 'plocal_steps', 20)

        # Optimizer for base (shared features) - standard SGD
        self.optimizer = torch.optim.SGD(
            self.model.base.parameters(), lr=self.learning_rate
        )
        self.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer, gamma=args.learning_rate_decay_gamma
        )

        # SGLD optimizer for head (personal classifier)
        # Temperature scaled by 1/n to match the mean-loss gradient scaling.
        # This is equivalent to correct SGLD with effective lr = sgld_lr / n.
        self.sgld_optimizer = SGLD(
            self.model.head.parameters(),
            lr=self.sgld_lr,
            temperature=1.0 / max(self.train_samples, 1),
            num_burn_in_steps=0,
        )

        # Store theta samples for Bayesian prediction averaging
        self.theta_samples = []

    def _add_prior_grad(self):
        """Add Gaussian prior gradient to head params, scaled by 1/n.

        Since the loss uses reduction='mean', the data gradient is (1/B)*Σ∇L_j.
        For correct SGLD, the prior gradient must also be scaled to match:
            (1/n) * ∂/∂θ [-log p(θ)] = θ / (σ² * n)
        This ensures the prior does not dominate the data signal.
        """
        n = max(self.train_samples, 1)
        for p in self.model.head.parameters():
            if p.grad is not None:
                p.grad.data.add_(p.data / (self.prior_sigma ** 2 * n))

    def train(self):
        trainloader = self.load_train_data()
        start_time = time.time()

        self.model.train()

        # Phase 1: SGLD sampling of head parameters (with fixed base)
        for param in self.model.base.parameters():
            param.requires_grad = False
        for param in self.model.head.parameters():
            param.requires_grad = True

        self.theta_samples = []

        for m in range(self.n_inner_iters):
            x, y = self.get_next_batch(trainloader)
            self.sgld_optimizer.zero_grad()

            output = self.model(x)
            loss = self.loss(output, y)
            loss.backward()

            # Add prior gradient before SGLD step
            self._add_prior_grad()

            # SGLD step (gradient descent + noise injection)
            self.sgld_optimizer.step()

            # Collect samples after burn-in
            if m >= self.burn_in:
                sample = parameters_to_vector(
                    self.model.head.parameters()
                ).detach().clone()
                self.theta_samples.append(sample)

        # Phase 2: Train base with standard SGD (with fixed head)
        # Use the mean of collected theta samples for a less noisy head
        for param in self.model.base.parameters():
            param.requires_grad = True
        for param in self.model.head.parameters():
            param.requires_grad = False

        if self.theta_samples:
            mean_theta = torch.mean(torch.stack(self.theta_samples), dim=0)
            vector_to_parameters(mean_theta, self.model.head.parameters())

        for step in range(self.local_epochs):
            x, y = self.get_next_batch(trainloader)
            self.optimizer.zero_grad()

            output = self.model(x)
            loss = self.loss(output, y)
            loss.backward()
            self.optimizer.step()

        # Restore requires_grad
        for param in self.model.head.parameters():
            param.requires_grad = True

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    def set_parameters(self, model):
        """Only update base parameters from server (head stays personal)."""
        for new_param, old_param in zip(model.parameters(), self.model.base.parameters()):
            old_param.data = new_param.data.clone()

    def test_metrics(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size

        testloader = self.load_test_data(batch_size=batch_size)

        self.model.eval()

        test_acc = 0
        test_num = 0
        y_prob = []
        y_true = []

        with torch.no_grad():
            for x, y in testloader:
                x, y = x.to(self.device), y.to(self.device)

                if len(self.theta_samples) > 0:
                    # Bayesian prediction: average softmax over theta samples
                    y_pred = torch.zeros(x.shape[0], self.num_classes, device=self.device)

                    original_params = parameters_to_vector(
                        self.model.head.parameters()
                    ).detach().clone()

                    for theta in self.theta_samples:
                        vector_to_parameters(theta, self.model.head.parameters())
                        output = F.softmax(self.model(x), dim=1)
                        y_pred += output

                    y_pred /= len(self.theta_samples)

                    # Restore original params
                    vector_to_parameters(original_params, self.model.head.parameters())
                else:
                    y_pred = F.softmax(self.model(x), dim=1)

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

        self.model.eval()

        train_num = 0
        losses = 0

        with torch.no_grad():
            for x, y in trainloader:
                x, y = x.to(self.device), y.to(self.device)

                output = self.model(x)
                loss = self.loss(output, y)

                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]

        return losses, train_num
