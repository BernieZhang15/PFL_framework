import torch
import warnings
import torch.nn as nn
import torch.nn.functional as F
warnings.filterwarnings("ignore", message="Can't initialize NVML")

def calculate_kl(mu_q, sig_q, mu_p, sig_p):
    kl = 0.5 * (2 * torch.log(sig_p / sig_q) - 1 + (sig_q / sig_p).pow(2) + ((mu_p - mu_q) / sig_p).pow(2)).sum()
    return kl

class FourierFTLayer(nn.Module):
    def __init__(self, base_layer, n_frequency, scaling, init_weights, random_loc_seed, **kwargs) -> None:
        super().__init__()
        self.base_layer = base_layer
        self.n_frequency = n_frequency
        self.scaling = scaling
        self.device = None
        self.freq_bias = kwargs.get('freq_bias', False)
        self.bandwidth = kwargs.get('bandwidth', 200)
        self.fc = kwargs.get('fc', 200)

        if isinstance(base_layer, nn.Linear):
            self.in_features, self.out_features = base_layer.in_features, base_layer.out_features
        else:
            raise ValueError(f"Unsupported layer type {type(base_layer)}")

        self.prior_mu = 0
        self.prior_sigma = 1

        if not self.freq_bias:
            indices = torch.randperm(self.out_features * self.in_features,
                                     generator=torch.Generator().manual_seed(random_loc_seed))[:n_frequency]
        else:
            P = self.indices_sampling(self.fc, self.bandwidth)
            indices = torch.multinomial(P, num_samples=n_frequency, replacement=False, generator=torch.Generator().manual_seed(random_loc_seed))

        self.indices = torch.stack([indices // self.in_features, indices % self.in_features], dim=0).to(self.device)

        self.spectrum_mu = nn.Parameter(torch.randn(n_frequency, device=self.device), requires_grad=True)
        self.spectrum_rho = nn.Parameter(torch.empty(n_frequency, device=self.device).normal_(-3, 0.1), requires_grad=True)
        self.spectrum_sigma = None

        if init_weights:
            nn.init.zeros_(self.spectrum_mu)

    def indices_sampling(self, fc, bandwidth):
        H, W = self.out_features, self.in_features
        u = torch.arange(H).unsqueeze(1).expand(H, W)
        v = torch.arange(W).unsqueeze(0).expand(H, W)

        u_center, v_center = H // 2, W // 2
        D = torch.sqrt((u - u_center) ** 2 + (v - v_center) ** 2)

        P = torch.exp(- ((D ** 2 - fc ** 2) / (D * bandwidth + 1e-8)) ** 2)  # [H, W]
        P = P.flatten()

        P = P / P.sum()

        return P

    def get_delta_weight(self) -> torch.Tensor:
        self.device = self.base_layer.weight.device
        spectrum_eps = torch.randn(self.n_frequency, device=self.device)
        self.spectrum_sigma = torch.log1p(torch.exp(self.spectrum_rho))
        spectrum = self.spectrum_mu + self.spectrum_sigma * spectrum_eps

        indices = self.indices.to(self.device)
        dense_spectrum = torch.zeros(self.out_features, self.in_features, device=self.device)
        dense_spectrum[indices[0, :], indices[1, :]] = spectrum.float()

        # By default, ifft treat (0, 0) as low frequency
        dense_spectrum = torch.fft.ifftshift(dense_spectrum)
        delta_weight = torch.fft.ifft2(dense_spectrum).real * self.scaling

        return delta_weight

    def kl_loss(self):
        self.spectrum_sigma = torch.log1p(torch.exp(self.spectrum_sigma))
        kl = calculate_kl(self.prior_mu, self.prior_sigma, self.spectrum_mu, self.spectrum_sigma)
        return kl

class FourierFTLinear(FourierFTLayer):
    def __init__(self, linear_layer, n_frequency=1000, scaling=150.0, init_weights=False, random_loc_seed=777, **kwargs):
        super().__init__(linear_layer, n_frequency, scaling, init_weights, random_loc_seed, **kwargs)
        self.ens_num = kwargs.get('ens_num', 1)

    def forward(self, data_input: torch.Tensor) -> torch.Tensor:

        delta_stack = torch.stack([self.get_delta_weight() * self.scaling for _ in range(self.ens_num)], dim=0)
        base_weight = self.base_layer.weight.unsqueeze(0).expand(self.ens_num, -1, -1)

        agg_weight = delta_stack + base_weight

        ens_out = torch.einsum("ebi, eoi->ebo", data_input, agg_weight) # (ens, batch, out)

        if self.base_layer.bias is not None:
            bias = self.base_layer.bias.unsqueeze(0).unsqueeze(0)
            ens_out += bias

        return ens_out

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "fourierft." + rep


class FTFedAvgCNN(nn.Module):
    def __init__(self, in_features=3, num_classes=10, ens_num=4, dim=2048):
        super().__init__()
        self.ens_num = ens_num

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_features, 32, kernel_size=3, padding=1, stride=1, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=1, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=1, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        fc1 = nn.Linear(dim, 512, bias=True)
        fc2 = nn.Linear(512, 256, bias=True)
        fc3 = nn.Linear(256, num_classes, bias=True)

        self.fc1 = FourierFTLinear(fc1, n_frequency=1000, scaling=150, ens_num=ens_num, freq_bias=True, fc=600, bandwidth=200)
        self.fc2 = FourierFTLinear(fc2, n_frequency=1000, scaling=150, ens_num=ens_num, freq_bias=True, fc=200, bandwidth=100)
        self.fc3 = FourierFTLinear(fc3, n_frequency=200, scaling=10, ens_num=ens_num, freq_bias=True, fc=80, bandwidth=40)


    def forward(self, data_input: torch.Tensor) -> torch.Tensor:
        batch_size = data_input.shape[0]
        out = torch.tile(data_input, (self.ens_num, 1, 1, 1))

        out = self.conv1(out)
        out = self.conv2(out)
        out = self.conv3(out)

        out = out.reshape(self.ens_num, batch_size, -1)
        out = torch.flatten(out, start_dim=2)

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


if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = FTFedAvgCNN().to(device)

    x = torch.randn(20, 3, 32, 32).to(device)
    target = torch.randn(20, 10).to(device)

    ens_y, kl_loss = model(x)
    ens_y = torch.mean(ens_y, dim=0)
    loss = F.mse_loss(ens_y, target)
    loss.backward()






