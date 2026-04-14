"""
Gaussian Mixture Model and Autoencoder for FedGMM.
Adapted from: https://github.com/zshuai8/FedGMM_ICML2023
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


# ======================== GMM Utility Functions ========================

def calculate_matmul_n_times(n_components, mat_a, mat_b):
    """
    mat_a: (n, k, 1, d), mat_b: (1, k, d, d)
    """
    res = torch.zeros(mat_a.shape, device=mat_a.device)
    for i in range(n_components):
        mat_a_i = mat_a[:, i, :, :].squeeze(-2)
        mat_b_i = mat_b[0, i, :, :].squeeze()
        res[:, i, :, :] = mat_a_i.mm(mat_b_i).unsqueeze(1)
    return res


def calculate_matmul(mat_a, mat_b):
    """
    mat_a: (n, k, 1, d), mat_b: (n, k, d, 1)
    """
    assert mat_a.shape[-2] == 1 and mat_b.shape[-1] == 1
    return torch.sum(mat_a.squeeze(-2) * mat_b.squeeze(-1), dim=2, keepdim=True)


# ======================== Gaussian Mixture Model ========================

class GaussianMixture(nn.Module):
    def __init__(self, n_components, n_features, device,
                 covariance_type="full", eps=1e-1, init_params="kmeans"):
        super().__init__()
        self.n_components = n_components
        self.n_features = n_features
        self.device = device
        self.eps = eps
        self.covariance_type = covariance_type
        self.init_params = init_params
        self.log_likelihood = -np.inf
        self._init_params()

    def _init_params(self):
        self.mu = nn.Parameter(
            torch.randn(1, self.n_components, self.n_features), requires_grad=False
        )

        if self.covariance_type == "full":
            self.var = nn.Parameter(
                torch.eye(self.n_features).reshape(1, 1, self.n_features, self.n_features)
                .repeat(1, self.n_components, 1, 1) * 100,
                requires_grad=False
            )
        else:
            self.var = nn.Parameter(
                torch.ones(1, self.n_components, self.n_features), requires_grad=False
            )

        self.pi = nn.Parameter(
            torch.ones(1, self.n_components, 1) / self.n_components, requires_grad=False
        )

        for p in self.parameters():
            p.data = p.data.to(self.device)

    def check_size(self, x):
        if len(x.size()) == 2:
            x = x.unsqueeze(1)
        return x

    def initialize_gmm(self, x):
        if self.init_params == "kmeans":
            mu = self._get_kmeans_mu(x, n_centers=self.n_components)
            self.mu.data = mu
        _, log_resp = self._e_step(x)
        pi, mu, var = self._m_step(x, log_resp)
        self.mu.data = mu
        self.var.data = var

    def _estimate_log_prob(self, x):
        x = self.check_size(x)
        if self.covariance_type == "full":
            mu = self.mu
            var = self.var
            # Add jitter to diagonal for numerical stability before inversion
            eps_jitter = torch.eye(var.shape[-1], device=var.device) * 1e-4
            var_stable = var + eps_jitter
            precision = torch.inverse(var_stable)
            d = x.shape[-1]
            log_2pi = d * np.log(2.0 * np.pi)
            log_det = self._calculate_log_det(precision)
            x_mu_T = (x - mu).unsqueeze(-2)
            x_mu = (x - mu).unsqueeze(-1)
            x_mu_T_precision = calculate_matmul_n_times(self.n_components, x_mu_T, precision)
            x_mu_T_precision_x_mu = calculate_matmul(x_mu_T_precision, x_mu)
            return -0.5 * (log_2pi - log_det + x_mu_T_precision_x_mu)
        else:
            mu = self.mu
            prec = torch.rsqrt(self.var)
            log_p = torch.sum(
                (mu * mu + x * x - 2 * x * mu) * (prec ** 2), dim=2, keepdim=True
            )
            log_det = torch.sum(torch.log(prec), dim=2, keepdim=True)
            return -0.5 * (self.n_features * np.log(2.0 * np.pi) + log_p) + log_det

    def calc_log_prob(self, x):
        return self._estimate_log_prob(x).squeeze(2)

    def _calculate_log_det(self, var):
        log_det = torch.empty(size=(self.n_components,), device=var.device)
        for k in range(self.n_components):
            _, logabsdet = torch.linalg.slogdet(var[0, k])
            log_det[k] = logabsdet
        return log_det.unsqueeze(-1)

    def _e_step(self, x):
        x = self.check_size(x)
        weighted_log_prob = self._estimate_log_prob(x) + torch.log(self.pi + 1e-12)
        log_prob_norm = torch.logsumexp(weighted_log_prob, dim=1, keepdim=True)
        log_resp = weighted_log_prob - log_prob_norm
        return torch.mean(log_prob_norm), log_resp

    def _m_step(self, x, log_resp):
        x = self.check_size(x)
        resp = torch.exp(log_resp)
        pi = torch.sum(resp, dim=0, keepdim=True) + self.eps
        mu = torch.sum(resp * x, dim=0, keepdim=True) / pi

        if self.covariance_type == "full":
            eps_mat = (torch.eye(self.n_features) * self.eps).to(x.device)
            var = (
                torch.sum(
                    (x - mu).unsqueeze(-1).matmul((x - mu).unsqueeze(-2)) * resp.unsqueeze(-1),
                    dim=0, keepdim=True
                ) / pi.unsqueeze(3) + eps_mat
            )
        else:
            x2 = (resp * x * x).sum(0, keepdim=True) / pi
            mu2 = mu * mu
            xmu = (resp * mu * x).sum(0, keepdim=True) / pi
            var = x2 - 2 * xmu + mu2 + self.eps

        pi = pi / x.shape[0]
        return pi, mu, var

    def m_step_with_response(self, x, resp):
        x = self.check_size(x)
        pi = torch.sum(resp, dim=0, keepdim=True) + self.eps
        mu = torch.sum(resp * x, dim=0, keepdim=True) / pi

        if self.covariance_type == "full":
            eps_mat = (torch.eye(self.n_features) * self.eps).to(x.device)
            var = (
                torch.sum(
                    (x - mu).unsqueeze(-1).matmul((x - mu).unsqueeze(-2)) * resp.unsqueeze(-1),
                    dim=0, keepdim=True
                ) / pi.unsqueeze(3) + eps_mat
            )
        else:
            x2 = (resp * x * x).sum(0, keepdim=True) / pi
            mu2 = mu * mu
            xmu = (resp * mu * x).sum(0, keepdim=True) / pi
            var = x2 - 2 * xmu + mu2 + self.eps

        pi = pi / x.shape[0]
        self.pi.data = pi
        self.mu.data = mu
        self.var.data = var
        return pi, mu, var

    def score_samples(self, x):
        x = self.check_size(x)
        weighted_log_prob = self._estimate_log_prob(x) + torch.log(self.pi + 1e-12)
        per_sample_score = torch.logsumexp(weighted_log_prob, dim=1)
        return torch.squeeze(per_sample_score)

    def update_parameter(self, mu, var, _pi=None):
        if _pi is not None:
            self.pi.data = _pi.to(self.device)
        self.mu.data = mu.to(self.device)
        self.var.data = var.to(self.device)

    def get_all_parameter(self):
        return self.pi.data.clone(), self.mu.data.clone(), self.var.data.clone()

    def _get_kmeans_mu(self, x, n_centers, init_times=50, min_delta=1e-3):
        if len(x.size()) == 3:
            x = x.squeeze(1)
        x_min, x_max = x.min(), x.max()
        x = (x - x_min) / (x_max - x_min + 1e-12)

        min_cost = np.inf
        center = x[:n_centers].clone()

        for _ in range(init_times):
            indices = np.random.choice(x.shape[0], size=n_centers, replace=True)
            tmp_center = x[indices]
            l2_dis = torch.norm(x.unsqueeze(1) - tmp_center.unsqueeze(0), p=2, dim=2)
            l2_cls = torch.argmin(l2_dis, dim=1)
            cost = 0
            for c in range(n_centers):
                mask = l2_cls == c
                if not mask.any():
                    continue
                cost += torch.norm(x[mask] - tmp_center[c], p=2, dim=1).mean()
            if cost < min_cost:
                min_cost = cost
                center = tmp_center.clone()

        delta = np.inf
        while delta > min_delta:
            l2_dis = torch.norm(x.unsqueeze(1) - center.unsqueeze(0), p=2, dim=2)
            l2_cls = torch.argmin(l2_dis, dim=1)
            center_old = center.clone()
            for c in range(n_centers):
                mask = l2_cls == c
                if not mask.any():
                    center[c] = center_old[c]
                    continue
                center[c] = x[mask].mean(dim=0)
            delta = torch.norm(center_old - center, dim=1).max()

        return (center.unsqueeze(0) * (x_max - x_min) + x_min)


# ======================== Autoencoder ========================

class CNNEncoder(nn.Module):
    def __init__(self, embed_size, input_size=(3, 32, 32)):
        super().__init__()
        self.input_size = input_size
        ch = 16
        self.conv = nn.Sequential(
            nn.Conv2d(input_size[0], ch, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ch, ch * 2, 4, 2, 1),
            nn.BatchNorm2d(ch * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ch * 2, ch * 4, 4, 2, 1),
            nn.BatchNorm2d(ch * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ch * 4, ch * 8, 4, 2, 1),
            nn.BatchNorm2d(ch * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.flat_fts = self._get_flat_fts()
        self.linear = nn.Sequential(
            nn.Linear(self.flat_fts, embed_size),
            nn.BatchNorm1d(embed_size),
            nn.LeakyReLU(0.2),
        )

    def _get_flat_fts(self):
        self.eval()
        f = self.conv(Variable(torch.ones(2, *self.input_size)))
        self.train()
        return int(np.prod(f.size()[1:]))

    def forward(self, x):
        x = self.conv(x.view(-1, *self.input_size))
        x = x.view(-1, self.flat_fts)
        return self.linear(x)


class CNNDecoder(nn.Module):
    def __init__(self, embed_size, input_size=(3, 32, 32)):
        super().__init__()
        self.input_channel = input_size[0]
        self.input_height = input_size[1]
        self.input_width = input_size[2]
        ch = 16
        fc_dim = 512

        self.fc = nn.Sequential(
            nn.Linear(embed_size, fc_dim),
            nn.BatchNorm1d(fc_dim),
            nn.ReLU(True),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(fc_dim, ch * 4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ch * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ch * 4, ch * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ch * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ch * 2, ch, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ch),
            nn.ReLU(True),
            nn.ConvTranspose2d(ch, self.input_channel, 4, 2, 1, bias=False),
            nn.Sigmoid(),
        )
        self.fc_dim = fc_dim

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, self.fc_dim, 1, 1)
        x = self.deconv(x)
        return x.view(-1, self.input_channel * self.input_width * self.input_height)


class Autoencoder(nn.Module):
    def __init__(self, embed_size, input_size=(3, 32, 32)):
        super().__init__()
        self.embed_size = embed_size
        self.input_size = input_size
        self.encoder = CNNEncoder(embed_size, input_size)
        self.decoder = CNNDecoder(embed_size, input_size)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)
