import math
import torch
from torch import Tensor
import torch.nn.functional as F
torch.set_default_dtype(torch.float32)

def calculate_kl(mu_q, sig_q, mu_p, sig_p):
    kl = 0.5 * (2 * torch.log(sig_p / sig_q) - 1 + (sig_q / sig_p).pow(2) + ((mu_p - mu_q) / sig_p).pow(2)).sum()
    return kl

class Bayes_Linear(torch.nn.Module):

    def __init__(self, in_features: int, out_features: int, bias: bool=True, device=None) -> None:

        factory_kwargs = {'device': device}
        super(Bayes_Linear, self).__init__()

        self.device = device
        self.in_features = in_features
        self.out_features = out_features

        priors = {
            'prior_mu': 0,
            'prior_sigma': 0.1,
            'posterior_mu_initial': (0, 0.1),
            'posterior_rho_initial': (-5, 0.1),
        }

        self.prior_mu = priors['prior_mu']
        self.prior_sigma = priors['prior_sigma']
        self.posterior_mu_initial = priors['posterior_mu_initial']
        self.posterior_rho_initial = priors['posterior_rho_initial']

        self.weight_mu = torch.nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        self.weight_rho = torch.nn.Parameter(torch.empty((out_features, in_features)))

        if bias:
            self.bias = torch.nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.weight_mu.data.normal_(*self.posterior_mu_initial)
        self.weight_rho.data.normal_(*self.posterior_rho_initial)

        if self.bias is not None:
            self.bias.data.normal_(*self.posterior_mu_initial)

    def forward(self, input: Tensor) -> Tensor:
        """
        forward phase for the layer
        """
        sigma = F.softplus(self.weight_rho)
        epsilon = torch.empty(self.weight_rho.shape).normal_(0, 1).to(self.device)
        weight = self.weight_mu + sigma * epsilon

        return F.linear(input, weight, self.bias)


    def kl_loss(self):

        weight_sigma = F.softplus(self.weight_rho)
        kl = calculate_kl(self.prior_mu, self.prior_sigma, self.weight_mu, weight_sigma)

        return kl


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    random_sample = torch.randn([8, 256]).to(device)
    network = Bayes_Linear(in_features=256, out_features=64, bias=True, device=device).to(device)

    network(random_sample)