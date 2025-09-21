import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch.distributions.normal import Normal
from torch.distributions.kl import kl_divergence

def calculate_kl(mu_q, sig_q, mu_p, sig_p):
    kl = 0.5 * (2 * torch.log(sig_p / sig_q) - 1 + (sig_q / sig_p).pow(2) + ((mu_p - mu_q) / sig_p).pow(2)).sum()
    return kl

class MMR1BayesLinear(nn.Module):
    def __init__(self, in_features, out_features, ensemble, priors=None, device=None):
        super(MMR1BayesLinear, self).__init__()

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

        self.alpha_sigma1 = None
        self.gamma_sigma1 = None
        self.alpha_sigma2 = None
        self.gamma_sigma2 = None
        self.ensemble = ensemble

        self.prior_mu = priors['prior_mu']
        self.prior_sigma = priors['prior_sigma']
        self.posterior_mu_initial = priors['posterior_mu_initial']
        self.posterior_rho_initial = priors['posterior_rho_initial']

        self.alpha_mu1 = Parameter(torch.empty(in_features, device=self.device))
        self.alpha_rho1 = Parameter(torch.empty(in_features, device=self.device))

        self.gamma_mu1 = Parameter(torch.empty(out_features, device=self.device))
        self.gamma_rho1 = Parameter(torch.empty(out_features, device=self.device))

        self.alpha_mu2 = Parameter(torch.empty(in_features, device=self.device))
        self.alpha_rho2 = Parameter(torch.empty(in_features, device=self.device))

        self.gamma_mu2 = Parameter(torch.empty(out_features, device=self.device))
        self.gamma_rho2 = Parameter(torch.empty(out_features, device=self.device))

        self.reset_parameters()

    def reset_parameters(self):

        self.alpha_mu1.data.normal_(*self.posterior_mu_initial)
        self.alpha_rho1.data.normal_(*self.posterior_rho_initial)

        self.gamma_mu1.data.normal_(*self.posterior_mu_initial)
        self.gamma_rho1.data.normal_(*self.posterior_rho_initial)

        self.alpha_mu2.data.normal_(*self.posterior_mu_initial)
        self.alpha_rho2.data.normal_(*self.posterior_rho_initial)

        self.gamma_mu2.data.normal_(*self.posterior_mu_initial)
        self.gamma_rho2.data.normal_(*self.posterior_rho_initial)

    def forward(self, input, sample=True):
        num_sample = input.shape[0] // (self.ensemble * 2)

        if self.training or sample:
            alpha_eps1 = torch.empty((self.ensemble, self.alpha_mu1.size()[0])).normal_(0, 1).to(self.device)
            self.alpha_sigma1 = torch.log1p(torch.exp(self.alpha_rho1))
            alpha1 = self.alpha_mu1.tile(self.ensemble, 1) + alpha_eps1 * self.alpha_sigma1.tile(self.ensemble, 1)

            alpha_eps2 = torch.empty((self.ensemble, self.alpha_mu2.size()[0])).normal_(0, 1).to(self.device)
            self.alpha_sigma2 = torch.log1p(torch.exp(self.alpha_rho2))
            alpha2 = self.alpha_mu2.tile(self.ensemble, 1) + alpha_eps2 * self.alpha_sigma2.tile(self.ensemble, 1)

            alpha = torch.cat((alpha1, alpha2), dim=0)

            gamma_eps1 = torch.empty((self.ensemble, self.gamma_mu1.size()[0])).normal_(0, 1).to(self.device)
            self.gamma_sigma1 = torch.log1p(torch.exp(self.gamma_rho1))
            gamma1 = self.gamma_mu1.tile(self.ensemble, 1) + gamma_eps1 * self.gamma_sigma1.tile(self.ensemble, 1)

            gamma_eps2 = torch.empty((self.ensemble, self.gamma_mu2.size()[0])).normal_(0, 1).to(self.device)
            self.gamma_sigma2 = torch.log1p(torch.exp(self.gamma_rho2))
            gamma2 = self.gamma_mu2.tile(self.ensemble, 1) + gamma_eps2 * self.gamma_sigma2.tile(self.ensemble, 1)

            gamma = torch.cat((gamma1, gamma2), dim=0)

        else:
            alpha = torch.cat((self.alpha_mu1, self.alpha_mu2), dim=0)
            gamma = torch.cat((self.gamma_mu1, self.gamma_mu2), dim=0)

        alpha = alpha.repeat_interleave(num_sample, dim=0)
        gamma = gamma.repeat_interleave(num_sample, dim=0)

        return F.linear(input * alpha, self.W, self.W_bias) * gamma

    def kl_loss(self):
        kl = calculate_kl(self.prior_mu, self.prior_sigma, self.alpha_mu1, self.alpha_sigma1)
        kl += calculate_kl(self.prior_mu, self.prior_sigma, self.gamma_mu1, self.gamma_sigma1)
        kl += calculate_kl(self.prior_mu, self.prior_sigma, self.alpha_mu2, self.alpha_sigma2)
        kl += calculate_kl(self.prior_mu, self.prior_sigma, self.gamma_mu2, self.gamma_sigma2)

        kl -= torch.sum(kl_divergence(Normal(self.alpha_mu1, self.alpha_sigma1),
                                      Normal(self.alpha_mu2, self.alpha_sigma1)))
        kl -= torch.sum(kl_divergence(Normal(self.alpha_mu2, self.alpha_sigma2),
                                      Normal(self.alpha_mu2, self.alpha_sigma2)))
        return kl


class MMR1BayesConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, ensemble, stride=1, padding=1, dilation=1, priors=None,
                 device=None):

        super(MMR1BayesConv2d, self).__init__()
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

        self.alpha_sigma1 = None
        self.gamma_sigma1 = None
        self.alpha_sigma2 = None
        self.gamma_sigma2 = None
        self.ensemble = ensemble

        self.alpha_mu1 = Parameter(torch.empty(in_channels, device=self.device))
        self.alpha_rho1 = Parameter(torch.empty(in_channels, device=self.device))

        self.gamma_mu1 = Parameter(torch.empty(out_channels, device=self.device))
        self.gamma_rho1 = Parameter(torch.empty(out_channels, device=self.device))

        self.alpha_mu2 = Parameter(torch.empty(in_channels, device=self.device))
        self.alpha_rho2 = Parameter(torch.empty(in_channels, device=self.device))

        self.gamma_mu2 = Parameter(torch.empty(out_channels, device=self.device))
        self.gamma_rho2 = Parameter(torch.empty(out_channels, device=self.device))

        self.reset_parameters()

    def reset_parameters(self):
        self.alpha_mu1.data.normal_(*self.posterior_mu_initial)
        self.alpha_rho1.data.normal_(*self.posterior_rho_initial)

        self.gamma_mu1.data.normal_(*self.posterior_mu_initial)
        self.gamma_rho1.data.normal_(*self.posterior_rho_initial)

        self.alpha_mu2.data.normal_(*self.posterior_mu_initial)
        self.alpha_rho2.data.normal_(*self.posterior_rho_initial)

        self.gamma_mu2.data.normal_(*self.posterior_mu_initial)
        self.gamma_rho2.data.normal_(*self.posterior_rho_initial)

    def forward(self, data_input, sample=True):

        num_sample = data_input.shape[0] // (self.ensemble * 2)

        if self.training or sample:
            alpha_eps1 = torch.empty((self.ensemble, self.alpha_mu1.size()[0])).normal_(0, 1).to(self.device)
            self.alpha_sigma1 = torch.log1p(torch.exp(self.alpha_rho1))
            alpha1 = self.alpha_mu1.tile(self.ensemble, 1) + alpha_eps1 * self.alpha_sigma1.tile(self.ensemble, 1)

            alpha_eps2 = torch.empty((self.ensemble, self.alpha_mu2.size()[0])).normal_(0, 1).to(self.device)
            self.alpha_sigma2 = torch.log1p(torch.exp(self.alpha_rho2))
            alpha2 = self.alpha_mu2.tile(self.ensemble, 1) + alpha_eps2 * self.alpha_sigma2.tile(self.ensemble, 1)

            alpha = torch.cat((alpha1, alpha2), dim=0)

            gamma_eps1 = torch.empty((self.ensemble, self.gamma_mu1.size()[0])).normal_(0, 1).to(self.device)
            self.gamma_sigma1 = torch.log1p(torch.exp(self.gamma_rho1))
            gamma1 = self.gamma_mu1.tile(self.ensemble, 1) + gamma_eps1 * self.gamma_sigma1.tile(self.ensemble, 1)

            gamma_eps2 = torch.empty((self.ensemble, self.gamma_mu2.size()[0])).normal_(0, 1).to(self.device)
            self.gamma_sigma2 = torch.log1p(torch.exp(self.gamma_rho2))
            gamma2 = self.gamma_mu2.tile(self.ensemble, 1) + gamma_eps2 * self.gamma_sigma2.tile(self.ensemble, 1)

            gamma = torch.cat((gamma1, gamma2), dim=0)

        else:
            alpha = torch.cat((self.alpha_mu1, self.alpha_mu2), dim=0)
            gamma = torch.cat((self.gamma_mu1, self.gamma_mu2), dim=0)

        alpha = alpha.repeat_interleave(num_sample, dim=0)[..., None, None]
        gamma = gamma.repeat_interleave(num_sample, dim=0)[..., None, None]

        return F.conv2d(data_input * alpha, self.W, self.W_bias, self.stride, self.padding, self.dilation,
                        self.groups) * gamma

    def kl_loss(self):
        kl = calculate_kl(self.prior_mu, self.prior_sigma, self.alpha_mu1, self.alpha_sigma1)
        kl += calculate_kl(self.prior_mu, self.prior_sigma, self.gamma_mu1, self.gamma_sigma1)
        kl += calculate_kl(self.prior_mu, self.prior_sigma, self.alpha_mu2, self.alpha_sigma2)
        kl += calculate_kl(self.prior_mu, self.prior_sigma, self.gamma_mu2, self.gamma_sigma2)

        kl -= torch.sum(kl_divergence(Normal(self.alpha_mu1, self.alpha_sigma1),
                                      Normal(self.alpha_mu2, self.alpha_sigma1)))
        kl -= torch.sum(kl_divergence(Normal(self.alpha_mu2, self.alpha_sigma2),
                                      Normal(self.alpha_mu2, self.alpha_sigma2)))
        return kl


class MMR1FedAvgCNN(nn.Module):
    def __init__(self, in_features=3, num_classes=10, ens=4, dim=2048, device=None):
        super().__init__()
        self.ens = ens

        self.conv1 = MMR1BayesConv2d(in_features, 32, kernel_size=3, ensemble=ens, device=device)

        self.conv2 = MMR1BayesConv2d(32, 64, kernel_size=3, ensemble=ens, device=device)

        self.conv3 = MMR1BayesConv2d(64, 128, kernel_size=3, ensemble=ens, device=device)

        self.fc1 = MMR1BayesLinear(dim, 512, ensemble=ens, device=device)

        self.fc2 = MMR1BayesLinear(512, 256, ensemble=ens, device=device)

        self.fc3 = MMR1BayesLinear(256, num_classes, ensemble=ens, device=device)


    def forward(self, x):
        x = torch.tile(x, (self.ens * 2, 1, 1, 1))

        out = self.conv1(x)
        out = F.relu(out, inplace=True)
        out = F.max_pool2d(out, kernel_size=(2, 2))

        out = self.conv2(out)
        out = F.relu(out, inplace=True)
        out = F.max_pool2d(out, kernel_size=(2, 2))

        out = self.conv3(out)
        out = F.relu(out, inplace=True)
        out = F.max_pool2d(out, kernel_size=(2, 2))

        out = torch.flatten(out, 1)

        out = self.fc1(out)
        out = F.relu(out, inplace=True)

        out = self.fc2(out)
        out = F.relu(out, inplace=True)

        out = self.fc3(out)

        kl = 0.0
        for module in self.modules():
            if hasattr(module, 'kl_loss'):
                kl = kl + module.kl_loss()

        return out, kl



if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MMR1FedAvgCNN(device=device).to(device)
    input_data = torch.randn((2, 3, 32, 32)).to(device)
    out = model(input_data)

