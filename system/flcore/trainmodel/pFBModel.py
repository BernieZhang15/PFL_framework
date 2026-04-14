import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions.normal import Normal
from torch.distributions.kl import kl_divergence


class pBNN(nn.Module):
    def __init__(self, device=None, input_dim=3, output_dim=10, weight_scale=0.1, rho_offset=-3, zeta=1, dim=2048):
        super(pBNN, self).__init__()
        self.device = device

        self.dim = dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layer_param_shapes = self.get_layer_param_shapes()

        self.mus = nn.ParameterList()
        self.rhos = nn.ParameterList()

        self.weight_scale = weight_scale
        self.rho_offset = rho_offset
        self.zeta = torch.tensor(zeta, device=self.device)
        self.sigmas = torch.tensor([1.] * len(self.layer_param_shapes), device=self.device)

        for shape in self.layer_param_shapes:
            mu = nn.Parameter(torch.normal(mean=torch.zeros(shape), std=self.weight_scale * torch.ones(shape)))
            rho = nn.Parameter(self.rho_offset + torch.zeros(shape))
            self.mus.append(mu)
            self.rhos.append(rho)

    def get_layer_param_shapes(self):
        layer_param_shapes = []

        W_shape = (32, self.input_dim, 3, 3)
        b_shape = (32,)
        layer_param_shapes.extend([W_shape, b_shape])

        W_shape = (64, 32, 3, 3)
        b_shape = (64,)
        layer_param_shapes.extend([W_shape, b_shape])

        W_shape = (128, 64, 3, 3)
        b_shape = (128,)
        layer_param_shapes.extend([W_shape, b_shape])

        W_shape = (512, self.dim)
        b_shape = (512,)
        layer_param_shapes.extend([W_shape, b_shape])

        W_shape = (256, 512)
        b_shape = (256,)
        layer_param_shapes.extend([W_shape, b_shape])

        W_shape = (self.output_dim, 256)
        b_shape = (self.output_dim,)
        layer_param_shapes.extend([W_shape, b_shape])

        return layer_param_shapes

    def transform_rhos(self, rhos):
        self.sigmas = [F.softplus(rho) for rho in rhos]

    def transform_gaussian_samples(self, mus, rhos, epsilons):
        self.transform_rhos(rhos)
        samples = []
        for j in range(len(mus)):
            a = self.sigmas[j] * epsilons[j]
            sample = mus[j] + a
            samples.append(sample)
        return samples

    def sample_epsilons(self, param_shapes):
        epsilons = [torch.normal(mean=torch.zeros(shape), std=torch.ones(shape)).to(self.device) for shape in param_shapes]
        return epsilons

    def net(self, X, layer_params):
        # import time
        # start_time = time.time()

        out = F.conv2d(X, layer_params[0], layer_params[1], stride=1, padding=1, dilation=1)
        out = F.relu(out, inplace=True)
        out = F.max_pool2d(out, kernel_size=(2, 2))

        out = F.conv2d(out, layer_params[2], layer_params[3], stride=1, padding=1, dilation=1)
        out = F.relu(out, inplace=True)
        out = F.max_pool2d(out, kernel_size=(2, 2))

        out = F.conv2d(out, layer_params[4], layer_params[5], stride=1, padding=1, dilation=1)
        out = F.relu(out, inplace=True)
        out = F.max_pool2d(out, kernel_size=(2, 2))

        out = torch.flatten(out, 1)

        out = F.linear(out, layer_params[6], layer_params[7])
        out = F.relu(out, inplace=True)

        out = F.linear(out, layer_params[8], layer_params[9])
        out = F.relu(out, inplace=True)

        out = F.linear(out, layer_params[10], layer_params[11])

        # end_time = time.time()
        # print(f"[pBNN.net] Forward Time: {end_time - start_time:.6f} seconds")
        return out

    def log_softmax_likelihood(self, outputs, y_onehot):
        return torch.nansum(y_onehot * F.log_softmax(outputs, dim=1), dim=0)

    def combined_loss_personalized(self, output, y_onehot, mus, sigmas, mus_local, sigmas_local, sample_num):
        # Calculate data likelihood
        log_likelihood = torch.mean(self.log_softmax_likelihood(output, y_onehot), dim=0)

        KL_q_w = sum([torch.sum(kl_divergence(
            Normal(mus[i], sigmas[i]), Normal(mus_local[i].detach(), sigmas_local[i].detach())))
            for i in range(len(self.layer_param_shapes))])

        return KL_q_w * self.zeta / sample_num - log_likelihood

    def combined_loss_local(self, mus, sigmas, mus_local, sigmas_local, sample_num):
        KL_q_w = sum([torch.sum(kl_divergence(Normal(mus[i].detach(), sigmas[i].detach()),
                        Normal(mus_local[i], sigmas_local[i]))) for i in range(len(self.layer_param_shapes))])
        return KL_q_w * self.zeta / sample_num


class pBNN_ViT(nn.Module):
    """Bayesian Vision Transformer for pFedBayes.

    Uses the same mu/rho reparameterisation as pBNN so that
    clientpFedBayes can drive it without any changes.
    """

    def __init__(self, device=None, num_classes=200, dim=256, in_channels=3,
                 img_size=64, patch_size=4, depth=6, num_heads=4,
                 mlp_ratio=2.0, weight_scale=0.1, rho_offset=-3, zeta=1):
        super().__init__()
        self.device = device
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.depth = depth
        self.mlp_hidden = int(dim * mlp_ratio)
        self.num_patches = (img_size // patch_size) ** 2
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.patch_size = patch_size

        self.layer_param_shapes = self._build_param_shapes()

        self.mus = nn.ParameterList()
        self.rhos = nn.ParameterList()

        self.weight_scale = weight_scale
        self.rho_offset = rho_offset
        self.zeta = torch.tensor(zeta, device=self.device)
        self.sigmas = [torch.tensor(1.0, device=self.device) for _ in range(len(self.layer_param_shapes))]

        for shape in self.layer_param_shapes:
            mu = nn.Parameter(torch.normal(
                mean=torch.zeros(shape),
                std=self.weight_scale * torch.ones(shape)))
            rho = nn.Parameter(self.rho_offset + torch.zeros(shape))
            self.mus.append(mu)
            self.rhos.append(rho)

    # ------------------------------------------------------------------
    # Parameter-shape bookkeeping
    # ------------------------------------------------------------------
    def _build_param_shapes(self):
        s = []
        # 0,1  patch_embed conv weight + bias
        s.append((self.dim, self.in_channels, self.patch_size, self.patch_size))
        s.append((self.dim,))
        # 2    cls_token
        s.append((1, 1, self.dim))
        # 3    pos_embed
        s.append((1, self.num_patches + 1, self.dim))

        # Each transformer block: 12 tensors
        for _ in range(self.depth):
            # norm1 weight, bias
            s.append((self.dim,))
            s.append((self.dim,))
            # attn: in_proj_weight, in_proj_bias, out_proj_weight, out_proj_bias
            s.append((3 * self.dim, self.dim))
            s.append((3 * self.dim,))
            s.append((self.dim, self.dim))
            s.append((self.dim,))
            # norm2 weight, bias
            s.append((self.dim,))
            s.append((self.dim,))
            # mlp fc1 weight, bias, fc2 weight, bias
            s.append((self.mlp_hidden, self.dim))
            s.append((self.mlp_hidden,))
            s.append((self.dim, self.mlp_hidden))
            s.append((self.dim,))

        # final norm weight, bias
        s.append((self.dim,))
        s.append((self.dim,))
        # head weight, bias
        s.append((self.num_classes, self.dim))
        s.append((self.num_classes,))
        return s

    # ------------------------------------------------------------------
    # Bayesian helpers  (same API as pBNN)
    # ------------------------------------------------------------------
    def transform_rhos(self, rhos):
        self.sigmas = [F.softplus(rho) for rho in rhos]

    def transform_gaussian_samples(self, mus, rhos, epsilons):
        self.transform_rhos(rhos)
        samples = []
        for j in range(len(mus)):
            samples.append(mus[j] + self.sigmas[j] * epsilons[j])
        return samples

    def sample_epsilons(self, param_shapes):
        return [torch.normal(mean=torch.zeros(shape),
                             std=torch.ones(shape)).to(self.device)
                for shape in param_shapes]

    # ------------------------------------------------------------------
    # Functional forward pass
    # ------------------------------------------------------------------
    def _layer_norm(self, x, weight, bias):
        return F.layer_norm(x, (self.dim,), weight, bias)

    def _mha(self, x, in_proj_w, in_proj_b, out_proj_w, out_proj_b):
        B, T, _ = x.shape
        qkv = F.linear(x, in_proj_w, in_proj_b)          # (B, T, 3*dim)
        qkv = qkv.reshape(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)                 # (3, B, H, T, d)
        q, k, v = qkv.unbind(0)

        scale = self.head_dim ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(B, T, self.dim)
        return F.linear(out, out_proj_w, out_proj_b)

    def net(self, X, p):
        idx = 0
        # --- patch embedding ---
        x = F.conv2d(X, p[idx], p[idx + 1], stride=self.patch_size)
        idx += 2
        x = x.flatten(2).transpose(1, 2)                  # (B, N, dim)

        # --- prepend cls token + add pos embed ---
        cls = p[idx].expand(x.shape[0], -1, -1);  idx += 1
        x = torch.cat([cls, x], dim=1)
        x = x + p[idx];  idx += 1

        # --- transformer blocks ---
        for _ in range(self.depth):
            # norm1 → mha → residual
            residual = x
            x = self._layer_norm(x, p[idx], p[idx + 1]);  idx += 2
            x = self._mha(x, p[idx], p[idx + 1],
                          p[idx + 2], p[idx + 3]);          idx += 4
            x = residual + x

            # norm2 → mlp → residual
            residual = x
            x = self._layer_norm(x, p[idx], p[idx + 1]);  idx += 2
            x = F.linear(x, p[idx], p[idx + 1]);           idx += 2
            x = F.gelu(x)
            x = F.linear(x, p[idx], p[idx + 1]);           idx += 2
            x = residual + x

        # --- final norm + head ---
        x = self._layer_norm(x, p[idx], p[idx + 1]);      idx += 2
        x = x[:, 0]                                        # cls token
        x = F.linear(x, p[idx], p[idx + 1])
        return x

    # ------------------------------------------------------------------
    # Loss helpers  (same API as pBNN)
    # ------------------------------------------------------------------
    def log_softmax_likelihood(self, outputs, y_onehot):
        return torch.nansum(y_onehot * F.log_softmax(outputs, dim=1), dim=0)

    def combined_loss_personalized(self, output, y_onehot,
                                   mus, sigmas, mus_local, sigmas_local,
                                   sample_num):
        log_likelihood = torch.mean(
            self.log_softmax_likelihood(output, y_onehot), dim=0)
        KL_q_w = sum(
            torch.sum(kl_divergence(
                Normal(mus[i], sigmas[i]),
                Normal(mus_local[i].detach(), sigmas_local[i].detach())))
            for i in range(len(self.layer_param_shapes)))
        return KL_q_w * self.zeta / sample_num - log_likelihood

    def combined_loss_local(self, mus, sigmas, mus_local, sigmas_local,
                            sample_num):
        KL_q_w = sum(
            torch.sum(kl_divergence(
                Normal(mus[i].detach(), sigmas[i].detach()),
                Normal(mus_local[i], sigmas_local[i])))
            for i in range(len(self.layer_param_shapes)))
        return KL_q_w * self.zeta / sample_num


if __name__ == '__main__':

    random_sample = torch.randn([32, 3, 32, 32])

    model = pBNN(output_dim=200)

    epsilons = model.sample_epsilons(model.layer_param_shapes)
    layer_params = model.transform_gaussian_samples(model.mus, model.rhos, epsilons)

    p_output = model.net(random_sample, layer_params)

    for n, p in model.named_parameters():
        print(n)

    print("\n--- pBNN_ViT test ---")
    vit_sample = torch.randn([4, 3, 64, 64])
    vit_model = pBNN_ViT(num_classes=200)
    eps = vit_model.sample_epsilons(vit_model.layer_param_shapes)
    lp = vit_model.transform_gaussian_samples(vit_model.mus, vit_model.rhos, eps)
    vit_out = vit_model.net(vit_sample, lp)
    print("output shape:", vit_out.shape)
    for n, p in vit_model.named_parameters():
        print(n, p.shape)


