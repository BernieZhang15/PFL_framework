import math
import torch
import warnings
import torch.nn as nn
import torch.nn.functional as F
warnings.filterwarnings("ignore", message="Can't initialize NVML")

device = "cuda" if torch.cuda.is_available() else "cpu"
def calculate_kl(mu_p, sig_p, mu_q, sig_q):
    kl = 0.5 * (2 * torch.log(sig_p / sig_q) - 1 + (sig_q / sig_p).pow(2) + ((mu_p - mu_q) / sig_p).pow(2)).sum()
    return kl


class FourierFTLayer(nn.Module):
    def __init__(self, base_layer, n_frequency, scaling, init_weights, random_loc_seed, **kwargs) -> None:
        super().__init__()
        self.base_layer = base_layer
        self.n_frequency = n_frequency
        self.scaling = scaling

        self.in_features, self.out_features = base_layer.in_features, base_layer.out_features

        indices = torch.randperm(self.out_features * self.in_features,
                                     generator=torch.Generator().manual_seed(random_loc_seed))[:n_frequency]
        self.indices = torch.stack([indices // self.in_features, indices % self.in_features], dim=0).to(device)

        self.spectrum_mu = nn.Parameter(0.1 * torch.randn(n_frequency, device=device), requires_grad=True)
        self.spectrum_rho = nn.Parameter(torch.empty(n_frequency, device=device).normal_(-5, 0.1), requires_grad=True)

        self.bayes_mask = self.frequency_mask(0.5, mode="low").float().to(device)

        if init_weights:
            nn.init.zeros_(self.spectrum_mu)

    def frequency_mask(self, radius=0.5, mode='high'):

        H, W = int(self.out_features), int(self.in_features)
        center = torch.tensor([H / 2, W / 2], device=device).view(2, 1)
        max_radius = math.hypot(H / 2.0, W / 2.0)
        thresh =  max_radius * radius
        dists = torch.linalg.norm(self.indices - center, dim=0)

        return dists >= thresh if mode == 'high' else dists < thresh


    def get_delta_weight(self) -> torch.Tensor:
        spectrum_eps = torch.randn(self.n_frequency, device=device)
        spectrum_sigma = torch.log1p(torch.exp(self.spectrum_rho))
        spectrum = self.spectrum_mu + spectrum_sigma * spectrum_eps * self.bayes_mask

        dense_spectrum = torch.zeros(self.out_features, self.in_features, device=device)
        dense_spectrum[self.indices[0, :], self.indices[1, :]] = spectrum.float()
        dense_spectrum = torch.fft.ifftshift(dense_spectrum)
        delta_weight = torch.fft.ifft2(dense_spectrum).real * self.scaling

        return delta_weight

    def kl_loss(self):

        mu_p = torch.tensor(0.0, device=device)
        sig_p = torch.tensor(1.0, device=device)

        m = self.bayes_mask.bool()
        mu_q = self.spectrum_mu[m]
        sig_q = torch.log1p(torch.exp(self.spectrum_rho))[m]

        return calculate_kl(mu_p, sig_p, mu_q, sig_q)

class FourierFTLinear(FourierFTLayer):
    def __init__(self, linear_layer, n_frequency=1000, scaling=150.0, init_weights=False, random_loc_seed=777, **kwargs):
        super().__init__(linear_layer, n_frequency, scaling, init_weights, random_loc_seed, **kwargs)
        self.ens_num = kwargs.get('ens_num', 1)

    def forward(self, data_input: torch.Tensor) -> torch.Tensor:

        delta_stack = torch.stack([self.get_delta_weight() for _ in range(self.ens_num)], dim=0)

        base_weight = self.base_layer.weight.unsqueeze(0).expand(self.ens_num, -1, -1)

        # print(torch.norm(delta_stack[0]) / torch.norm(base_weight[0]))

        agg_weight = delta_stack + base_weight

        ens_out = torch.einsum("ebi, eoi->ebo", data_input, agg_weight) # (ens, batch, out)

        return ens_out

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "fourierft." + rep


class FourierFTModel(nn.Module):
    def __init__(self, ens_num=4):
        super().__init__()
        self.ens_num = ens_num

        self.fc1 = nn.Linear(1, 64, bias=True)
        self.fc2 = nn.Linear(64, 64, bias=True)
        self.fc3 = nn.Linear(64, 64, bias=True)
        self.fc4 = nn.Linear(64, 1, bias=True)

        if ens_num > 1:
            self.fc2 = FourierFTLinear(self.fc2, n_frequency=1024, scaling=12, ens_num=ens_num, device=device)
            self.fc3 = FourierFTLinear(self.fc3, n_frequency=1024, scaling=12, ens_num=ens_num, device=device)


    def forward(self, data_input: torch.Tensor) -> torch.Tensor:
        if self.ens_num > 1:
            out = torch.tile(data_input, (self.ens_num, 1, 1))
        else:
            out = data_input

        out = self.fc1(out)
        out = F.relu(out, inplace=True)

        out = self.fc2(out)
        out = F.relu(out, inplace=True)

        out = self.fc3(out)
        out = F.relu(out, inplace=True)

        out = self.fc4(out)

        kl = 0.0
        for module in self.modules():
            if hasattr(module, 'kl_loss'):
                kl = kl + module.kl_loss()

        return out, kl




