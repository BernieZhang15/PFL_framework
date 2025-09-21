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


if __name__ == '__main__':

    random_sample = torch.randn([32, 3, 32, 32])

    model = pBNN(output_dim=200)

    epsilons = model.sample_epsilons(model.layer_param_shapes)
    layer_params = model.transform_gaussian_samples(model.mus, model.rhos, epsilons)

    p_output = model.net(random_sample, layer_params)

    for n, p in model.named_parameters():
        print(n)


