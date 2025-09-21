import math
import time
import torch
import numpy as np
import torch.nn.functional as F
from torch import nn
from torch.nn import Parameter

seed = 65
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


class AutoLoRaQuantizer(nn.Module):
    def __init__(self, device=None, p_init=1.0, threshold=0.15):
        super().__init__()

        self.device = device
        self.threshold = threshold

        theta = torch.tensor([p_init, p_init, p_init, p_init], device=self.device)
        self.theta = nn.Parameter(theta)

    def forward(self, alpha_mu, alpha_sigma):
        diag = torch.softmax(self.theta, dim=0)
        gate_matrix = torch.diag(diag)

        v_alpha_mu = torch.mm(gate_matrix, alpha_mu)
        v_alpha_sigma = torch.mm(gate_matrix, alpha_sigma)

        return v_alpha_mu, v_alpha_sigma


def calculate_kl(mu_q, sig_q, mu_p, sig_p):
    kl = 0.5 * (2 * torch.log(sig_p / sig_q) - 1 + (sig_q / sig_p).pow(2) + ((mu_p - mu_q) / sig_p).pow(2)).sum()
    return kl

class ER1BayesLinear(nn.Module):
    def __init__(self, in_features, out_features, ensemble, priors=None, device=None):
        super(ER1BayesLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.device = device

        if priors is None:
            priors = {
                'prior_mu': 1,
                'prior_sigma': 0.1,
                'posterior_mu_initial': (1, 0.1),
                'posterior_rho_initial': (-3, 0.1),
            }
        self.W = Parameter(torch.empty((out_features, in_features), device=self.device))
        self.W_bias = Parameter(torch.empty(out_features, device=self.device))
        torch.nn.init.kaiming_normal_(self.W, a=0, mode='fan_in', nonlinearity='leaky_relu')
        torch.nn.init.zeros_(self.W_bias)

        self.alpha_sigma = None
        self.gamma_sigma = None
        self.ensemble = ensemble

        self.prior_mu = priors['prior_mu']
        self.prior_sigma = priors['prior_sigma']
        self.posterior_mu_initial = priors['posterior_mu_initial']
        self.posterior_rho_initial = priors['posterior_rho_initial']

        self.alpha_mu = Parameter(torch.empty(in_features, device=self.device))
        self.alpha_rho = Parameter(torch.empty(in_features, device=self.device))

        self.gamma_mu = Parameter(torch.empty(out_features, device=self.device))
        self.gamma_rho = Parameter(torch.empty(out_features, device=self.device))

        self.reset_parameters()

    def reset_parameters(self):

        self.alpha_mu.data.normal_(*self.posterior_mu_initial)
        self.alpha_rho.data.normal_(*self.posterior_rho_initial)

        self.gamma_mu.data.normal_(*self.posterior_mu_initial)
        self.gamma_rho.data.normal_(*self.posterior_rho_initial)

    def forward(self, input, weights=None, sample=True):
        num_sample = input.shape[0] // self.ensemble

        if weights is not None:
            alpha_mu = weights[0]
            alpha_rho = weights[1]
            gamma_mu = weights[2]
            gamma_rho = weights[3]
        else:
            alpha_mu = self.alpha_mu
            alpha_rho = self.alpha_rho
            gamma_mu = self.gamma_mu
            gamma_rho = self.gamma_rho

        if self.training or sample:
            alpha_eps = torch.randn((self.ensemble, alpha_mu.size()[0]), device=self.device)
            self.alpha_sigma = torch.log1p(torch.exp(alpha_rho))

            alpha_mu_expanded = alpha_mu.unsqueeze(0).expand(self.ensemble, -1)
            alpha_sigma_expanded = self.alpha_sigma.unsqueeze(0).expand(self.ensemble, -1)

            alpha = alpha_mu_expanded + alpha_eps * alpha_sigma_expanded

            gamma_eps = torch.randn((self.ensemble, gamma_mu.size()[0]), device=self.device)
            self.gamma_sigma = torch.log1p(torch.exp(gamma_rho))

            gamma_mu_expanded = gamma_mu.unsqueeze(0).expand(self.ensemble, -1)
            gamma_sigma_expanded = self.gamma_sigma.unsqueeze(0).expand(self.ensemble, -1)

            gamma = gamma_mu_expanded + gamma_eps * gamma_sigma_expanded

        else:
            alpha = self.alpha_mu
            gamma = self.gamma_mu

        alpha = alpha.repeat_interleave(num_sample, dim=0)
        gamma = gamma.repeat_interleave(num_sample, dim=0)

        return F.linear(input * alpha, self.W, self.W_bias) * gamma

    def kl_loss(self):
        kl = calculate_kl(self.prior_mu, self.prior_sigma, self.alpha_mu, self.alpha_sigma)
        kl += calculate_kl(self.prior_mu, self.prior_sigma, self.gamma_mu, self.gamma_sigma)
        return kl


class ER1BayesConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, ensemble, stride=1, padding=1, dilation=1, priors=None,
                 device=None):

        super(ER1BayesConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = 1
        self.device = device

        if priors is None:
            priors = {
                'prior_mu': 1,
                'prior_sigma': 0.1,
                'posterior_mu_initial': (1, 0.1),
                'posterior_rho_initial': (-3, 0.1),
            }

        self.W = Parameter(torch.empty((out_channels, in_channels, *self.kernel_size), device=self.device))
        self.W_bias = Parameter(torch.empty(out_channels, device=self.device))
        torch.nn.init.kaiming_normal_(self.W, a=0, mode='fan_in', nonlinearity='leaky_relu')
        torch.nn.init.zeros_(self.W_bias)

        self.prior_mu = priors['prior_mu']
        self.prior_sigma = priors['prior_sigma']
        self.posterior_mu_initial = priors['posterior_mu_initial']
        self.posterior_rho_initial = priors['posterior_rho_initial']

        self.alpha_sigma = None
        self.gamma_sigma = None
        self.ensemble = ensemble

        self.alpha_mu = Parameter(torch.empty(in_channels, device=self.device))
        self.alpha_rho = Parameter(torch.empty(in_channels, device=self.device))

        self.gamma_mu = Parameter(torch.empty(out_channels, device=self.device))
        self.gamma_rho = Parameter(torch.empty(out_channels, device=self.device))

        self.reset_parameters()

    def reset_parameters(self):
        self.alpha_mu.data.normal_(*self.posterior_mu_initial)
        self.alpha_rho.data.normal_(*self.posterior_rho_initial)

        self.gamma_mu.data.normal_(*self.posterior_mu_initial)
        self.gamma_rho.data.normal_(*self.posterior_rho_initial)

    def forward(self, data_input, weights=None, sample=True):

        num_sample = data_input.shape[0] // self.ensemble

        if weights is not None:
            alpha_mu = weights[0]
            alpha_rho = weights[1]
            gamma_mu = weights[2]
            gamma_rho = weights[3]
        else:
            alpha_mu = self.alpha_mu
            alpha_rho = self.alpha_rho
            gamma_mu = self.gamma_mu
            gamma_rho = self.gamma_rho

        if self.training or sample:
            alpha_eps = torch.randn((self.ensemble, alpha_mu.size()[0]), device=self.device)
            self.alpha_sigma = torch.log1p(torch.exp(alpha_rho))
            alpha_mu_expanded = alpha_mu.unsqueeze(0).expand(self.ensemble, -1)
            alpha_sigma_expanded = self.alpha_sigma.unsqueeze(0).expand(self.ensemble, -1)
            alpha = alpha_mu_expanded + alpha_eps * alpha_sigma_expanded

            gamma_eps = torch.randn((self.ensemble, gamma_mu.size()[0]), device=self.device)
            self.gamma_sigma = torch.log1p(torch.exp(gamma_rho))

            gamma_mu_expanded = gamma_mu.unsqueeze(0).expand(self.ensemble, -1)
            gamma_sigma_expanded = self.gamma_sigma.unsqueeze(0).expand(self.ensemble, -1)

            gamma = gamma_mu_expanded + gamma_eps * gamma_sigma_expanded

        else:
            alpha = self.alpha_mu
            gamma = self.gamma_mu

        alpha = alpha.repeat_interleave(num_sample, dim=0)[..., None, None]
        gamma = gamma.repeat_interleave(num_sample, dim=0)[..., None, None]

        return F.conv2d(data_input * alpha, self.W, self.W_bias, self.stride, self.padding, self.dilation,
                        self.groups) * gamma

    def kl_loss(self):
        kl = calculate_kl(self.prior_mu, self.prior_sigma, self.alpha_mu, self.alpha_sigma)
        kl += calculate_kl(self.prior_mu, self.prior_sigma, self.gamma_mu, self.gamma_sigma)
        return kl


class ER1FedAvgCNN(nn.Module):
    def __init__(self, in_features=3, num_classes=10, ens=4, dim=2048, device=None):
        super().__init__()
        self.ens = ens

        self.conv1 = ER1BayesConv2d(in_features, 32, kernel_size=3, ensemble=ens, device=device)

        self.conv2 = ER1BayesConv2d(32, 64, kernel_size=3, ensemble=ens, device=device)

        self.conv3 = ER1BayesConv2d(64, 128, kernel_size=3, ensemble=ens, device=device)

        self.fc1 = ER1BayesLinear(dim, 512, ensemble=ens, device=device)

        self.fc2 = ER1BayesLinear(512, 256, ensemble=ens, device=device)

        self.fc3 = ER1BayesLinear(256, num_classes, ensemble=ens, device=device)

    def reset_parameters(self):
        for module in self.children():
            module.reset_parameters()

    def forward(self, x, rank_1_matrix=None):
        x = torch.tile(x, (self.ens, 1, 1, 1))

        weights = {}
        if rank_1_matrix is not None:
            weights[0] = [rank_1_matrix['conv1.alpha_mu'], rank_1_matrix['conv1.alpha_rho'],
                          rank_1_matrix['conv1.gamma_mu'], rank_1_matrix['conv1.gamma_rho']]
            weights[1] = [rank_1_matrix['conv2.alpha_mu'], rank_1_matrix['conv2.alpha_rho'],
                          rank_1_matrix['conv2.gamma_mu'], rank_1_matrix['conv2.gamma_rho']]
            weights[2] = [rank_1_matrix['conv3.alpha_mu'], rank_1_matrix['conv3.alpha_rho'],
                          rank_1_matrix['conv3.gamma_mu'], rank_1_matrix['conv3.gamma_rho']]
            weights[3] = [rank_1_matrix['fc1.alpha_mu'], rank_1_matrix['fc1.alpha_rho'],
                          rank_1_matrix['fc1.gamma_mu'], rank_1_matrix['fc1.gamma_rho']]
            weights[4] = [rank_1_matrix['fc2.alpha_mu'], rank_1_matrix['fc2.alpha_rho'],
                          rank_1_matrix['fc2.gamma_mu'], rank_1_matrix['fc2.gamma_rho']]
            weights[5] = [rank_1_matrix['fc3.alpha_mu'], rank_1_matrix['fc3.alpha_rho'],
                          rank_1_matrix['fc3.gamma_mu'], rank_1_matrix['fc3.gamma_rho']]

        out = self.conv1(x, weights.get(0))
        out = F.relu(out, inplace=True)
        out = F.max_pool2d(out, kernel_size=(2, 2))

        out = self.conv2(out, weights.get(1))
        out = F.relu(out, inplace=True)
        out = F.max_pool2d(out, kernel_size=(2, 2))

        out = self.conv3(out, weights.get(2))
        out = F.relu(out, inplace=True)
        out = F.max_pool2d(out, kernel_size=(2, 2))

        out = torch.flatten(out, 1)

        out = self.fc1(out, weights.get(3))
        out = F.relu(out, inplace=True)

        out = self.fc2(out, weights.get(4))
        out = F.relu(out, inplace=True)

        out = self.fc3(out, weights.get(5))

        kl = 0.0
        for module in self.modules():
            if hasattr(module, 'kl_loss'):
                kl = kl + module.kl_loss()

        return out, kl

class LRBayesLinear(nn.Module):
    def __init__(self, in_features, out_features, ensemble=4, priors=None, device=None, rank=4, adaptive=False, quan_method=None):
        super(LRBayesLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.rank = rank

        if priors is None:
            priors = {
                'prior_mu': 1 / np.sqrt(rank),
                'prior_sigma': 0.1,
                'posterior_mu_initial': (1 / np.sqrt(rank), 0.1),
                'posterior_rho_initial': (-3, 0.1),
            }

        self.W = Parameter(torch.empty((out_features, in_features), device=self.device))
        self.W_bias = Parameter(torch.empty(out_features, device=self.device))
        torch.nn.init.kaiming_normal_(self.W, a=0, mode='fan_in', nonlinearity='leaky_relu')
        torch.nn.init.zeros_(self.W_bias)

        self.alpha_sigma = None
        self.gamma_sigma = None
        self.ensemble = ensemble

        self.prior_mu = priors['prior_mu']
        self.prior_sigma = priors['prior_sigma']
        self.posterior_mu_initial = priors['posterior_mu_initial']
        self.posterior_rho_initial = priors['posterior_rho_initial']

        self.alpha_mu = Parameter(torch.empty(self.rank, in_features, device=self.device))
        self.alpha_rho = Parameter(torch.empty(self.rank, in_features, device=self.device))

        self.gamma_mu = Parameter(torch.empty(self.rank, out_features, device=self.device))
        self.gamma_rho = Parameter(torch.empty(self.rank, out_features, device=self.device))

        self.adaptive = adaptive

        if adaptive:
            if quan_method == "BayesianBit":
                self.quantizer = BayesianBitsQuantizer(self.device, g2_init=6.0, g3_init=6.0, g4_init=6.0)
            elif quan_method == "AutoLoRa":
                self.quantizer = AutoLoRaQuantizer(self.device, p_init=1.0)
            elif quan_method == "HardConcrete":
                self.quantizer = HardConcreteQuantizer(self.device, rank=self.rank)

        self.reset_parameters()

    def reset_parameters(self):

        self.alpha_mu.data.normal_(*self.posterior_mu_initial)
        self.alpha_rho.data.normal_(*self.posterior_rho_initial)

        self.gamma_mu.data.normal_(*self.posterior_mu_initial)
        self.gamma_rho.data.normal_(*self.posterior_rho_initial)


    def forward(self, input, weights=None, sample=True):
        num_sample = input.shape[0] // self.ensemble

        if weights is not None:
            alpha_mu = weights[0]
            alpha_rho = weights[1]
            gamma_mu = weights[2]
            gamma_rho = weights[3]
        else:
            alpha_mu = self.alpha_mu
            alpha_rho = self.alpha_rho
            gamma_mu = self.gamma_mu
            gamma_rho = self.gamma_rho

        if self.training or sample:

            alpha_eps = torch.empty((alpha_mu.size()[0], self.ensemble, alpha_mu.size()[1]), device=self.device).normal_(0, 1)
            self.alpha_sigma = torch.log1p(torch.exp(alpha_rho))

            gamma_eps = torch.empty((gamma_mu.size()[0], self.ensemble, gamma_mu.size()[1]), device=self.device).normal_(0, 1)
            self.gamma_sigma = torch.log1p(torch.exp(gamma_rho))

            if self.adaptive:
                alpha_mu, gamma_mu = self.quantizer(alpha_mu, gamma_mu)

            alpha_mu_expanded = alpha_mu.unsqueeze(1).expand(-1, self.ensemble, -1)
            alpha_sigma_expanded = self.alpha_sigma.unsqueeze(1).expand(-1, self.ensemble, -1)
            alpha = alpha_mu_expanded + alpha_eps * alpha_sigma_expanded

            gamma_mu_expanded = gamma_mu.unsqueeze(1).expand(-1, self.ensemble, -1)
            gamma_sigma_expanded = self.gamma_sigma.unsqueeze(1).expand(-1, self.ensemble, -1)
            gamma = gamma_mu_expanded + gamma_eps * gamma_sigma_expanded

        else:
            alpha = self.alpha_mu.unsqueeze(1).expand(-1, self.ensemble, -1)
            gamma = self.gamma_mu.unsqueeze(1).expand(-1, self.ensemble, -1)

        alpha = alpha.repeat_interleave(num_sample, dim=1)
        gamma = gamma.repeat_interleave(num_sample, dim=1)

        input = torch.unsqueeze(input, dim=0)
        outputs = F.linear(input * alpha, self.W, self.W_bias)
        outputs = torch.sum(outputs * gamma, dim=0)

        return outputs

    def kl_loss(self):
        kl = 0
        for r in range(self.rank):
            kl += calculate_kl(self.prior_mu, self.prior_sigma, self.alpha_mu[r], self.alpha_sigma[r])
            kl += calculate_kl(self.prior_mu, self.prior_sigma, self.gamma_mu[r], self.gamma_sigma[r])
        return kl

class LRBayesConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, ensemble=4, stride=1, padding=1, dilation=1, priors=None,
                 device=None, rank=4, adaptive=False, quan_method=None):

        super(LRBayesConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = 1
        self.device = device
        self.rank = rank

        if priors is None:
            priors = {
                'prior_mu': 1 / np.sqrt(rank),
                'prior_sigma': 0.1,
                'posterior_mu_initial': (1 / np.sqrt(rank), 0.1),
                'posterior_rho_initial': (-3, 0.1),
            }

        self.W = Parameter(torch.empty((out_channels, in_channels, *self.kernel_size), device=self.device))
        self.W_bias = Parameter(torch.empty(out_channels, device=self.device))
        torch.nn.init.kaiming_normal_(self.W, a=0, mode='fan_in', nonlinearity='leaky_relu')
        torch.nn.init.zeros_(self.W_bias)

        self.prior_mu = priors['prior_mu']
        self.prior_sigma = priors['prior_sigma']
        self.posterior_mu_initial = priors['posterior_mu_initial']
        self.posterior_rho_initial = priors['posterior_rho_initial']

        self.alpha_sigma = None
        self.gamma_sigma = None
        self.ensemble = ensemble

        self.alpha_mu = Parameter(torch.empty(self.rank, in_channels, device=self.device))
        self.alpha_rho = Parameter(torch.empty(self.rank, in_channels, device=self.device))

        self.gamma_mu = Parameter(torch.empty(self.rank, out_channels, device=self.device))
        self.gamma_rho = Parameter(torch.empty(self.rank, out_channels, device=self.device))

        self.adaptive = adaptive

        if adaptive:
            if quan_method == "BayesianBit":
                self.quantizer = BayesianBitsQuantizer(self.device, g2_init=6.0, g3_init=6.0, g4_init=6.0)
            elif quan_method == "AutoLoRa":
                self.quantizer = AutoLoRaQuantizer(self.device, p_init=1.0)
            elif quan_method == "HardConcrete":
                self.quantizer = HardConcreteQuantizer(self.device, rank=self.rank)

        self.reset_parameters()

    def reset_parameters(self):
        self.alpha_mu.data.normal_(*self.posterior_mu_initial)
        self.alpha_rho.data.normal_(*self.posterior_rho_initial)

        self.gamma_mu.data.normal_(*self.posterior_mu_initial)
        self.gamma_rho.data.normal_(*self.posterior_rho_initial)

    def forward(self, data_input, weights=None, sample=True):

        num_sample = data_input.shape[0] // self.ensemble

        if weights is not None:
            alpha_mu = weights[0]
            alpha_rho = weights[1]
            gamma_mu = weights[2]
            gamma_rho = weights[3]
        else:
            alpha_mu = self.alpha_mu
            alpha_rho = self.alpha_rho
            gamma_mu = self.gamma_mu
            gamma_rho = self.gamma_rho

        if self.training or sample:
            alpha_eps = torch.randn((alpha_mu.size()[0], self.ensemble, alpha_mu.size()[1]), device=self.device)
            self.alpha_sigma = torch.log1p(torch.exp(alpha_rho))

            gamma_eps = torch.randn((gamma_mu.size()[0], self.ensemble, gamma_mu.size()[1]), device=self.device)
            self.gamma_sigma = torch.log1p(torch.exp(gamma_rho))

            if self.adaptive:
                alpha_mu, gamma_mu = self.quantizer(alpha_mu, gamma_mu)

            alpha_mu_expanded = alpha_mu.unsqueeze(1).expand(-1, self.ensemble, -1)
            alpha_sigma_expanded = self.alpha_sigma.unsqueeze(1).expand(-1, self.ensemble, -1)
            alpha = alpha_mu_expanded + alpha_eps * alpha_sigma_expanded

            gamma_mu_expanded = gamma_mu.unsqueeze(1).expand(-1, self.ensemble, -1)
            gamma_sigma_expanded = self.gamma_sigma.unsqueeze(1).expand(-1, self.ensemble, -1)
            gamma = gamma_mu_expanded + gamma_eps * gamma_sigma_expanded

        else:
            alpha = self.alpha_mu
            gamma = self.gamma_mu

        alpha = alpha.repeat_interleave(num_sample, dim=1)[..., None, None]
        gamma = gamma.repeat_interleave(num_sample, dim=1)[..., None, None]

        data_input = torch.unsqueeze(data_input, 0)
        perturb_input = data_input * alpha

        shape = tuple(torch.cat((torch.tensor([-1]), torch.tensor(perturb_input.shape[2:])), dim=0).tolist())
        perturb_input = perturb_input.reshape(shape)
        outputs = F.conv2d(perturb_input, self.W, self.W_bias, self.stride, self.padding, self.dilation, self.groups)
        shape = tuple(torch.cat((torch.tensor([self.rank, -1]), torch.tensor(outputs.shape[1:])), dim=0).tolist())
        outputs = outputs.reshape(shape)
        outputs = torch.sum(outputs * gamma, dim=0)

        return outputs

    def kl_loss(self):
        kl = 0
        for r in range(self.rank):
            kl += calculate_kl(self.prior_mu, self.prior_sigma, self.alpha_mu[r], self.alpha_sigma[r])
            kl += calculate_kl(self.prior_mu, self.prior_sigma, self.gamma_mu[r], self.gamma_sigma[r])
        return kl


class LRFedAvgCNN(nn.Module):
    def __init__(self, in_features=3, num_classes=10, ens=4, dim=2048, rank=8, device=None, ada_rank=False, quan_method=None):
        super().__init__()

        self.ens = ens

        self.conv1 = LRBayesConv2d(in_features, 32, kernel_size=3, ensemble=ens, rank=rank, device=device,
                                   adaptive=ada_rank, quan_method=quan_method)

        self.conv2 = LRBayesConv2d(32, 64, kernel_size=3, ensemble=ens, rank=rank, device=device,
                                   adaptive=ada_rank, quan_method=quan_method)

        self.conv3 = LRBayesConv2d(64, 128, kernel_size=3, ensemble=ens, rank=rank, device=device,
                                   adaptive=ada_rank, quan_method=quan_method)

        self.fc1 = LRBayesLinear(dim, 512, ensemble=ens, rank=rank, device=device, adaptive=ada_rank,
                                 quan_method=quan_method)

        self.fc2 = LRBayesLinear(512, 256, ensemble=ens, rank=rank, device=device,
                                 adaptive=ada_rank, quan_method=quan_method)

        self.fc3 = LRBayesLinear(256, num_classes, ensemble=ens, rank=rank, device=device,
                                 adaptive=ada_rank, quan_method=quan_method)

    def reset_parameters(self):
        for module in self.children():
            module.reset_parameters()

    def forward(self, x, rank_1_matrix=None):
        x = torch.tile(x, (self.ens, 1, 1, 1))

        weights = {}
        if rank_1_matrix is not None:
            weights[0] = [rank_1_matrix['conv1.alpha_mu'], rank_1_matrix['conv1.alpha_rho'],
                          rank_1_matrix['conv1.gamma_mu'], rank_1_matrix['conv1.gamma_rho']]
            weights[1] = [rank_1_matrix['conv2.alpha_mu'], rank_1_matrix['conv2.alpha_rho'],
                          rank_1_matrix['conv2.gamma_mu'], rank_1_matrix['conv2.gamma_rho']]
            weights[2] = [rank_1_matrix['conv3.alpha_mu'], rank_1_matrix['conv3.alpha_rho'],
                          rank_1_matrix['conv3.gamma_mu'], rank_1_matrix['conv3.gamma_rho']]
            weights[3] = [rank_1_matrix['fc1.alpha_mu'], rank_1_matrix['fc1.alpha_rho'],
                          rank_1_matrix['fc1.gamma_mu'], rank_1_matrix['fc1.gamma_rho']]
            weights[4] = [rank_1_matrix['fc2.alpha_mu'], rank_1_matrix['fc2.alpha_rho'],
                          rank_1_matrix['fc2.gamma_mu'], rank_1_matrix['fc2.gamma_rho']]
            weights[5] = [rank_1_matrix['fc3.alpha_mu'], rank_1_matrix['fc3.alpha_rho'],
                          rank_1_matrix['fc3.gamma_mu'], rank_1_matrix['fc3.gamma_rho']]

        out = self.conv1(x, weights.get(0))
        out = F.relu(out, inplace=True)
        out = F.max_pool2d(out, kernel_size=(2, 2))

        out = self.conv2(out, weights.get(1))
        out = F.relu(out, inplace=True)
        out = F.max_pool2d(out, kernel_size=(2, 2))

        out = self.conv3(out, weights.get(2))
        out = F.relu(out, inplace=True)
        out = F.max_pool2d(out, kernel_size=(2, 2))

        out = torch.flatten(out, 1)

        out = self.fc1(out, weights.get(3))
        out = F.relu(out, inplace=True)

        out = self.fc2(out, weights.get(4))
        out = F.relu(out, inplace=True)

        out = self.fc3(out, weights.get(5))

        kl = 0.0
        for module in self.modules():
            if hasattr(module, 'kl_loss'):
                kl = kl + module.kl_loss()

        return out, kl

def hc_prob_pos(p, beta=0.5, zeta=1.1, gamma=-0.1):
    return torch.sigmoid(p - beta * math.log(-gamma / zeta))

def l0_sample(log_alpha, device=None, beta=0.5, zeta=1.1, gamma=-0.1):
    u = torch.rand(1, device=device)
    sigm = torch.sigmoid((torch.log(u) - torch.log(1 - u) + log_alpha) / beta)
    sbar = sigm * (zeta - gamma) + gamma
    return torch.clamp(sbar, 0, 1)

def hard_concrete_func(x, zeta=1.1, gamma=-0.1, mult=1):
    return torch.clamp(torch.sigmoid(x * mult) * (zeta - gamma) + gamma, 0, 1)

def gate_kl_loss(network):
    loss = 0.0
    for n, module in network.named_modules():
        if isinstance(module, BayesianBitsQuantizer):
            loss += module.regularizer()
    return loss

class BayesianBitsQuantizer(nn.Module):
    def __init__(self, device=None, g2_init=6.0, g3_init=6.0, g4_init=6.0):
        super().__init__()
        self.device = device
        theta_2 = torch.tensor([g2_init], device=self.device)
        theta_3 = torch.tensor([g3_init], device=self.device)
        theta_4 = torch.tensor([g4_init], device=self.device)
        self.theta_2 = nn.Parameter(theta_2)
        self.theta_3 = nn.Parameter(theta_3)
        self.theta_4 = nn.Parameter(theta_4)

        self.hc_beta = 0.5
        self.hc_gamma, self.hc_zeta = -0.1, 1.1
        self.hc_threshold = 0.34

    def get_gate_matrix(self):
        beta, zeta, gamma = self.hc_beta, self.hc_zeta, self.hc_gamma
        offset = math.log(-self.hc_gamma / self.hc_zeta) * self.hc_beta
        args = []

        if self.training:
            gate_f = l0_sample
            args = self.device, beta, zeta, gamma
        else:
            gate_f = lambda x: (torch.sigmoid(offset - x) < self.hc_threshold).float()

        gate_2 = gate_f(self.theta_2, *args)
        gate_3 = gate_f(self.theta_3, *args)
        gate_4 = gate_f(self.theta_4, *args)

        gate_matrix = torch.diag(torch.cat([torch.tensor(1), gate_2, gate_3, gate_4]))

        return gate_matrix

    def regularizer(self):
        beta, zeta, gamma = self.hc_beta, self.hc_zeta, self.hc_gamma

        q_g2 = hc_prob_pos(self.theta_2, beta, zeta, gamma)
        q_g3 = hc_prob_pos(self.theta_3, beta, zeta, gamma)
        q_g4 = hc_prob_pos(self.theta_4, beta, zeta, gamma)

        kl_g2 = q_g2
        kl_g3 = q_g3 * q_g2
        kl_g4 = q_g4 * q_g3 * q_g2

        reg_i = kl_g2.sum() + kl_g3.sum() + kl_g4.sum()

        return reg_i

    def forward(self, alpha_mu, alpha_sigma):

        gate_matrix = self.get_gate_matrix()

        v_alpha_mu = torch.mm(gate_matrix, alpha_mu)
        v_alpha_sigma = torch.mm(gate_matrix, alpha_sigma)

        return v_alpha_mu, v_alpha_sigma


class HardConcreteQuantizer(nn.Module):
    def __init__(self, device=None, param_init=2.3, rank=4):
        super().__init__()
        self.device = device

        theta = torch.tensor([param_init] * (rank - 1), device=self.device)

        self.theta = nn.Parameter(theta)

        self.hc_gamma, self.hc_zeta, self.mult = -0.1, 1.1, 1
        self.hc_threshold = 0.95

    def get_gate_matrix(self):
        zeta, gamma, mult = self.hc_zeta, self.hc_gamma, self.mult

        gate_f = lambda x : hard_concrete_func(x, zeta, gamma, mult)

        gate = gate_f(self.theta)

        gate_matrix = torch.diag(torch.cat([torch.tensor([1], device=self.device), gate]))

        return gate_matrix


    def forward(self, alpha_mu, alpha_sigma):

        gate_matrix = self.get_gate_matrix()

        v_alpha_mu = torch.mm(gate_matrix, alpha_mu)
        v_alpha_sigma = torch.mm(gate_matrix, alpha_sigma)

        return v_alpha_mu, v_alpha_sigma


if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    random_sample = torch.randn([32, 3, 32, 32]).to(device)
    random_label = torch.randint(0, 10, (128,)).to(device)

    net = LRFedAvgCNN(device=device).to(device)

    total_params = sum(p.numel() for p in net.parameters())
    print(f"Total Parameters: {total_params}")

    w_params = []
    r_params = []
    for name, param in net.named_parameters():
        if "W" in name:
            w_params.append(param)
        elif any(n in name for n in ["alpha", "gamma"]):
            r_params.append(param)

    optim = torch.optim.SGD(w_params, lr=0.1, momentum=0.9)

    criterion = nn.CrossEntropyLoss()

    start_time = time.time()

    for i in range(1000):

        optim.zero_grad()

        outputs, _ = net(random_sample)

        loss = criterion(outputs, random_label)

        loss.backward()

        optim.step()

    print((time.time() - start_time) / 1000)




