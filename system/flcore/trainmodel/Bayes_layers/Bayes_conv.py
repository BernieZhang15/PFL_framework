import math
import torch
import warnings
import numpy as np
import torch.nn.init as init
import torch.nn.functional as F
warnings.filterwarnings("ignore", category=Warning)
torch.set_default_dtype(torch.float32)

def calculate_kl(mu_q, sig_q, mu_p, sig_p):
    kl = 0.5 * (2 * torch.log(sig_p / sig_q) - 1 + (sig_q / sig_p).pow(2) + ((mu_p - mu_q) / sig_p).pow(2)).sum()
    return kl

class Bayes_Conv2d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, padding=0, stride=1, bias=True, device=None) -> None:

        factory_kwargs = {'device': device}
        super(Bayes_Conv2d, self).__init__()


        self.device = device
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = [kernel_size, kernel_size] if isinstance(kernel_size, int) else kernel_size
        self.kernel_size_number = self.kernel_size[0] * self.kernel_size[1]
        self.dilation = dilation if type(dilation)==tuple else (dilation, dilation)
        self.padding = padding if type(padding) == tuple else(padding, padding)
        self.stride = (stride if type(stride)==tuple else (stride, stride))

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

        self.weight_mu = torch.nn.Parameter(
            torch.empty(tuple([self.out_channels, self.in_channels] + list(self.kernel_size)), **factory_kwargs))
        self.weight_rho = torch.nn.Parameter(
            torch.empty(tuple([self.out_channels, self.in_channels] + list(self.kernel_size))))

        if bias:
            self.bias = torch.nn.Parameter(torch.empty(self.out_channels, **factory_kwargs))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.weight_mu.data.normal_(*self.posterior_mu_initial)
        self.weight_rho.data.normal_(*self.posterior_rho_initial)

        if self.bias is not None:
            self.bias.data.normal_(*self.posterior_mu_initial)

    def forward(self, input):

        sigma = F.softplus(self.weight_rho)
        epsilon = torch.empty(self.weight_rho.shape).normal_(0, 1).to(self.device)
        weight = self.weight_mu + sigma * epsilon

        return F.conv2d(input=input, weight=weight, bias=self.bias, stride=self.stride, padding=self.padding, dilation=self.dilation)

    def kl_loss(self):
        weight_sigma = F.softplus(self.weight_rho)
        kl = calculate_kl(self.prior_mu, self.prior_sigma, self.weight_mu, weight_sigma)

        return kl

if __name__=="__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random_sample = torch.randn([32, 3, 32, 32]).to(device)

    layer = Bayes_Conv2d(in_channels=3, out_channels=16, kernel_size=(3, 3), device=device).to(device)

    output = layer(random_sample)

    print(output.shape)
