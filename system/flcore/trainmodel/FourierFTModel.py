import torch
import warnings
import torch.nn as nn
import torch.nn.functional as F
warnings.filterwarnings("ignore", message="Can't initialize NVML")

# def calculate_kl(mu_p, sig_p, mu_q, sig_q):
#     kl = 0.5 * (2 * torch.log(sig_p / sig_q) - 1 + (sig_q / sig_p).pow(2) + ((mu_p - mu_q) / sig_p).pow(2)).sum()
#     return kl
# def calculate_kl(mu_p, sig_p, mu_q, sig_q):
#     kl = 0.5 * (2 * torch.log(sig_p / sig_q) - 1 + (sig_q / sig_p).pow(2) + ((mu_q - mu_p) / sig_p).pow(2)).sum()
#     return kl 
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

    def indices_sampling(self, fc_norm, bw_norm):
        # H, W = self.out_features, self.in_features
        # u = torch.arange(H).unsqueeze(1).expand(H, W)
        # v = torch.arange(W).unsqueeze(0).expand(H, W)

        # u_center, v_center = H // 2, W // 2
        # D = torch.sqrt((u - u_center) ** 2 + (v - v_center) ** 2)

        # P = torch.exp(- ((D ** 2 - fc ** 2) / (D * bandwidth + 1e-8)) ** 2)  # [H, W]
        # P = P.flatten()

        # P = P / P.sum()

        # return P
        H, W = self.out_features, self.in_features
        device = self.base_layer.weight.device

        u = torch.arange(H, dtype=torch.float32, device=device).unsqueeze(1).expand(H, W)
        v = torch.arange(W, dtype=torch.float32, device=device).unsqueeze(0).expand(H, W)

        uc, vc = H / 2.0, W / 2.0
        D = torch.sqrt((u - uc) ** 2 + (v - vc) ** 2)     
        D_norm = D / D.max()                              

        fc = fc_norm
        bw = bw_norm
        P = torch.exp(- ((D_norm - fc) / (bw + 1e-6)) ** 2)
        P = P.flatten()
        P = P / P.sum().clamp_min(1e-12)
        return P

    def get_delta_weight(self) -> torch.Tensor:
        # timing disabled
        # import time
        self.device = self.base_layer.weight.device
        # t0 = time.time()
        spectrum_eps = torch.randn(self.n_frequency, device=self.device)
        self.spectrum_sigma = torch.log1p(torch.exp(self.spectrum_rho))
        spectrum = self.spectrum_mu + self.spectrum_sigma * spectrum_eps
        # t1 = time.time()

        indices = self.indices.to(self.device)
        dense_spectrum = torch.zeros(self.out_features, self.in_features, device=self.device)
        dense_spectrum[indices[0, :], indices[1, :]] = spectrum.float()
        # t2 = time.time()

        # By default, ifft treat (0, 0) as low frequency
        # dense_spectrum = torch.fft.ifftshift(dense_spectrum)
        # t3 = time.time()
        delta_weight = torch.fft.ifft2(dense_spectrum).real * self.scaling
        # t4 = time.time()
        # print("Weight ratio:", torch.norm(self.base_layer.weight)/torch.norm(delta_weight))
        # timing print disabled
        return delta_weight

    def get_dense_spectrum(self, sample: bool = True) -> torch.Tensor:
        self.device = self.base_layer.weight.device
        self.spectrum_sigma = torch.log1p(torch.exp(self.spectrum_rho))
        if sample:
            spectrum_eps = torch.randn(self.n_frequency, device=self.device)
            spectrum = self.spectrum_mu + self.spectrum_sigma * spectrum_eps
        else:
            spectrum = self.spectrum_mu

        indices = self.indices.to(self.device)
        dense_spectrum = torch.zeros(self.out_features, self.in_features, device=self.device)
        dense_spectrum[indices[0, :], indices[1, :]] = spectrum.float()
        return dense_spectrum

    def kl_loss(self):
        # timing disabled
        # import time
        # t0 = time.time()
        self.spectrum_sigma = torch.log1p(torch.exp(self.spectrum_rho))
        kl = calculate_kl(self.spectrum_mu, self.spectrum_sigma, self.prior_mu, self.prior_sigma)
        # t1 = time.time()
        # print(f"    [FourierFTLayer] KL散度计算: {t1-t0:.6f}s")
        return kl

class FourierFTLinear(FourierFTLayer):
    def __init__(self, linear_layer, n_frequency=1000, scaling=150.0, init_weights=False, random_loc_seed=777, **kwargs):
        super().__init__(linear_layer, n_frequency, scaling, init_weights, random_loc_seed, **kwargs)
        self.ens_num = kwargs.get('ens_num', 1)

    def forward(self, data_input: torch.Tensor) -> torch.Tensor:

        delta_stack = torch.stack([self.get_delta_weight() for _ in range(self.ens_num)], dim=0)
        base_weight = self.base_layer.weight.unsqueeze(0).expand(self.ens_num, -1, -1)

        agg_weight = base_weight + delta_stack
        # ratio = delta_stack.norm() / base_weight.norm()
        # print(f"[{self.base_layer}] ||ΔW||/||W|| = {ratio.item():.4f}")

        ens_out = torch.einsum("ebi, eoi->ebo", data_input, agg_weight) # (ens, batch, out)

        if self.base_layer.bias is not None:
            bias = self.base_layer.bias.unsqueeze(0).unsqueeze(0)
            ens_out += bias

        return ens_out

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "fourierft." + rep


class FTFedAvgCNN(nn.Module):
    def __init__(self, in_features=3, num_classes=10, ens_num=4, dim=2048, freq_ratio=1.0, freq_bias=True):
        super().__init__()
        self.ens_num = ens_num
        self.freq_ratio = freq_ratio
        self.freq_bias = freq_bias

        # 基础 n_frequency 值，按 freq_ratio 比例调整
        base_freq1, base_freq2, base_freq3 = 1024, 512, 256
        n_freq1 = max(1, int(base_freq1 * freq_ratio))
        n_freq2 = max(1, int(base_freq2 * freq_ratio))
        n_freq3 = max(1, int(base_freq3 * freq_ratio))

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

        self.fc1 = FourierFTLinear(fc1, n_frequency=n_freq1, scaling=350, ens_num=ens_num, freq_bias=freq_bias, fc=0.8, bandwidth=0.25)
        self.fc2 = FourierFTLinear(fc2, n_frequency=n_freq2, scaling=200, ens_num=ens_num, freq_bias=freq_bias, fc=0.8, bandwidth=0.15)
        self.fc3 = FourierFTLinear(fc3, n_frequency=n_freq3, scaling=100, ens_num=ens_num, freq_bias=freq_bias, fc=0.8, bandwidth=0.2)


    def forward(self, data_input: torch.Tensor) -> torch.Tensor:
        # timing disabled
        # import time
        # start_time = time.time()

        batch_size = data_input.shape[0]
        # t0 = time.time()
        out = torch.tile(data_input, (self.ens_num, 1, 1, 1))
        # t1 = time.time()
        # print(f"[Forward] tile输入: {t1-t0:.6f}s")

        # t0 = time.time()
        out = self.conv1(out)
        # t1 = time.time()
        # print(f"[Forward] conv1: {t1-t0:.6f}s")

        # t0 = time.time()
        out = self.conv2(out)
        # t1 = time.time()
        # print(f"[Forward] conv2: {t1-t0:.6f}s")

        # t0 = time.time()
        out = self.conv3(out)
        # t1 = time.time()
        # print(f"[Forward] conv3: {t1-t0:.6f}s")

        # t0 = time.time()
        out = out.reshape(self.ens_num, batch_size, -1)
        out = torch.flatten(out, start_dim=2)
        # t1 = time.time()
        # print(f"[Forward] flatten: {t1-t0:.6f}s")

        # t0 = time.time()
        out = self.fc1(out)
        # t1 = time.time()
        # print(f"[Forward] fc1(FourierFT): {t1-t0:.6f}s")

        # t0 = time.time()
        out = F.relu(out, inplace=True)
        # t1 = time.time()
        # print(f"[Forward] relu1: {t1-t0:.6f}s")

        # t0 = time.time()
        out = self.fc2(out)
        # t1 = time.time()
        # print(f"[Forward] fc2(FourierFT): {t1-t0:.6f}s")

        # t0 = time.time()
        out = F.relu(out, inplace=True)
        # t1 = time.time()
        # print(f"[Forward] relu2: {t1-t0:.6f}s")

        # t0 = time.time()
        out = self.fc3(out)
        # t1 = time.time()
        # print(f"[Forward] fc3(FourierFT): {t1-t0:.6f}s")

        # t0 = time.time()
        kl = 0.0
        for module in self.modules():
            if hasattr(module, 'kl_loss'):
                kl = kl + module.kl_loss()
        # t1 = time.time()
        # print(f"[Forward] KL总计: {t1-t0:.6f}s")

        # end_time = time.time()
        # print(f"[Client Forward] 总用时: {end_time - start_time:.6f} seconds")

        return out, kl

class FTTransformerBlock(nn.Module):
    """Transformer block with shared attention and FourierFT MLP."""
    def __init__(self, dim, num_heads, mlp_ratio=4.0, ens_num=4, freq_ratio=1.0,
                 freq_bias=True, drop=0.0, seed_offset=0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=drop, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)

        mlp_hidden = int(dim * mlp_ratio)
        n_freq1 = max(1, int(1024 * freq_ratio))
        n_freq2 = max(1, int(512 * freq_ratio))

        fc1 = nn.Linear(dim, mlp_hidden)
        fc2 = nn.Linear(mlp_hidden, dim)
        self.mlp_fc1 = FourierFTLinear(fc1, n_frequency=n_freq1, scaling=200, ens_num=ens_num,
                                       freq_bias=freq_bias, random_loc_seed=777 + seed_offset,
                                       fc=0.8, bandwidth=0.2)
        self.mlp_fc2 = FourierFTLinear(fc2, n_frequency=n_freq2, scaling=200, ens_num=ens_num,
                                       freq_bias=freq_bias, random_loc_seed=888 + seed_offset,
                                       fc=0.8, bandwidth=0.2)

    def forward(self, x, ens_num, batch_size):
        # x: (ens*batch, seq_len, dim)
        seq_len = x.shape[1]
        dim = x.shape[2]

        # Self-attention (shared weights)
        residual = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x)
        x = residual + x

        # MLP with FourierFT
        residual = x
        x = self.norm2(x)
        # Reshape: (ens*batch, seq_len, dim) -> (ens, batch*seq_len, dim)
        x = x.reshape(ens_num, batch_size * seq_len, dim)
        x = self.mlp_fc1(x)
        x = F.gelu(x)
        x = self.mlp_fc2(x)
        # Reshape back: (ens, batch*seq_len, dim) -> (ens*batch, seq_len, dim)
        x = x.reshape(ens_num * batch_size, seq_len, dim)

        x = residual + x
        return x


class FTFedViT(nn.Module):
    def __init__(self, num_classes=10, ens_num=4, dim=256, freq_ratio=1.0, freq_bias=True,
                 in_channels=3, img_size=32, patch_size=4, depth=4, num_heads=4, mlp_ratio=2.0):
        super().__init__()
        self.ens_num = ens_num
        self.freq_ratio = freq_ratio
        self.freq_bias = freq_bias
        self.dim = dim
        self.num_patches = (img_size // patch_size) ** 2

        # Patch embedding (shared)
        self.patch_embed = nn.Conv2d(in_channels, dim, kernel_size=patch_size, stride=patch_size)

        # CLS token + positional embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            FTTransformerBlock(dim, num_heads, mlp_ratio, ens_num, freq_ratio, freq_bias,
                               seed_offset=i * 100)
            for i in range(depth)
        ])

        self.norm = nn.LayerNorm(dim)

        # Classification head (FourierFT)
        head_linear = nn.Linear(dim, num_classes)
        n_freq_head = max(1, int(256 * freq_ratio))
        self.head = FourierFTLinear(head_linear, n_frequency=n_freq_head, scaling=100,
                                    ens_num=ens_num, freq_bias=freq_bias, fc=0.8, bandwidth=0.2)

    def forward(self, data_input: torch.Tensor) -> torch.Tensor:
        batch_size = data_input.shape[0]

        # Tile input for ensemble: (batch, C, H, W) -> (ens*batch, C, H, W)
        x = torch.tile(data_input, (self.ens_num, 1, 1, 1))

        # Patch embedding: (ens*batch, dim, H/P, W/P) -> (ens*batch, num_patches, dim)
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)

        # Prepend CLS token + add positional embedding
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed

        # Transformer blocks
        for block in self.blocks:
            x = block(x, self.ens_num, batch_size)

        x = self.norm(x)

        # CLS token output: (ens*batch, dim)
        x = x[:, 0]

        # Reshape for FourierFT head: (ens, batch, dim)
        x = x.reshape(self.ens_num, batch_size, self.dim)

        # Classification
        out = self.head(x)  # (ens, batch, num_classes)

        # KL loss
        kl = 0.0
        for module in self.modules():
            if hasattr(module, 'kl_loss'):
                kl = kl + module.kl_loss()

        return out, kl


if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Test FTFedAvgCNN
    model = FTFedAvgCNN().to(device)
    x = torch.randn(20, 3, 32, 32).to(device)
    target = torch.randn(20, 10).to(device)
    ens_y, kl_loss = model(x)
    ens_y = torch.mean(ens_y, dim=0)
    loss = F.mse_loss(ens_y, target)
    loss.backward()
    print("FTFedAvgCNN OK")

    # Test FTFedViT
    vit = FTFedViT(num_classes=10, ens_num=4, dim=256, depth=4, num_heads=4,
                   img_size=32, patch_size=4, mlp_ratio=2.0).to(device)
    x2 = torch.randn(8, 3, 32, 32).to(device)
    target2 = torch.randn(8, 10).to(device)
    ens_y2, kl2 = vit(x2)
    ens_y2 = torch.mean(ens_y2, dim=0)
    loss2 = F.mse_loss(ens_y2, target2) + 1e-4 * kl2
    loss2.backward()
    print(f"FTFedViT OK, output shape: {ens_y2.shape}, kl: {kl2.item():.4f}")






