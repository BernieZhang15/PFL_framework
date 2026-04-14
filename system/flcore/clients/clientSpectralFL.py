import time
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from flcore.clients.clientbase import Client


def compute_fft_of_weights(weight_tensor):
    """Compute FFT of a weight tensor and return flattened complex array."""
    if len(weight_tensor.shape) > 2:
        weight_tensor = weight_tensor.view(weight_tensor.shape[0], -1)

    if len(weight_tensor.shape) == 2:
        fft_weights = np.fft.fft2(weight_tensor.cpu().numpy(), axes=[0, 1])
    elif len(weight_tensor.shape) == 1:
        fft_weights = np.fft.fft(weight_tensor.cpu().numpy())
    else:
        raise ValueError("Unsupported tensor shape for FFT: {}".format(weight_tensor.shape))

    return fft_weights.flatten()


def fft_weights_cal(model):
    """Compute FFT-transformed weights for all layers with weights."""
    all_fft_weights = []
    for name, layer in model.named_modules():
        if hasattr(layer, 'weight') and layer.weight is not None:
            weight_tensor = layer.weight.data
            fft_weights = compute_fft_of_weights(weight_tensor)
            all_fft_weights.extend(fft_weights)
    return np.array(all_fft_weights)


def spectral_cal(model, as_log_prob=True):
    """
    Compute spectral distribution from model weights via FFT.
    
    Args:
        model: neural network model
        as_log_prob: if True, return log_softmax (for KL input); 
                     if False, return softmax (for KL target)
    """
    f_weights = fft_weights_cal(model)
    prob_dist = np.abs(f_weights)
    prob_dist /= np.sum(prob_dist)
    prob_dist = torch.tensor(prob_dist, dtype=torch.float)

    if as_log_prob:
        prob_dist = F.log_softmax(prob_dist, dim=0)
    return prob_dist


class clientSpectralFL(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        self.beta = args.beta            # proximal term coefficient for personal model
        self.ratio = args.sd_ratio       # spectral truncation ratio
        self.lambda_g = args.lambda_g    # spectral regularizer coefficient (global model)
        self.lambda_l = args.lambda_l    # spectral regularizer coefficient (personal model)

        # Personal model (separate copy)
        self.model_per = copy.deepcopy(self.model)
        self.optimizer_per = torch.optim.SGD(
            self.model_per.parameters(), lr=self.learning_rate, momentum=0.5
        )
        self.learning_rate_scheduler_per = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer_per,
            gamma=args.learning_rate_decay_gamma
        )

        self.loss_func_kl = nn.KLDivLoss(reduction="batchmean")

    def train(self):
        """Train the global model with spectral distillation from server's global model."""
        trainloader = self.load_train_data()
        start_time = time.time()
        self.model.train()

        # Keep reference to global model (set by server via set_parameters)
        global_model = copy.deepcopy(self.model)
        global_model.eval()

        for step in range(self.local_epochs):
            for batch_idx, (x, y) in enumerate(trainloader):
                x, y = x.to(self.device), y.to(self.device)
                y = y.long()

                self.optimizer.zero_grad()
                output = self.model(x)
                loss = self.loss(output, y)

                # Spectral distillation regularization (after first batch)
                if batch_idx > 0:
                    spec_w = spectral_cal(self.model, as_log_prob=True)
                    spec_l = spectral_cal(global_model, as_log_prob=False)

                    len_trunc_w = int(len(spec_w) * self.ratio)
                    len_trunc_l = int(len(spec_l) * self.ratio)
                    trunc_spec_w = spec_w[:len_trunc_w]
                    trunc_spec_l = spec_l[:len_trunc_l]

                    # Ensure same length for KL divergence
                    min_len = min(len(trunc_spec_w), len(trunc_spec_l))
                    trunc_spec_w = trunc_spec_w[:min_len]
                    trunc_spec_l = trunc_spec_l[:min_len]

                    l_reg = self.loss_func_kl(trunc_spec_w, trunc_spec_l)
                    loss += self.lambda_g * l_reg

                loss.backward()
                self.optimizer.step()

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    def ptrain(self):
        """Train the personal model with spectral distillation + proximal regularization."""
        trainloader = self.load_train_data()
        start_time = time.time()
        self.model_per.train()

        # Reference to server's global model (already set via set_parameters on self.model)
        global_model = copy.deepcopy(self.model)
        global_model.eval()

        for step in range(self.local_epochs):
            for batch_idx, (x, y) in enumerate(trainloader):
                x, y = x.to(self.device), y.to(self.device)
                y = y.long()

                self.optimizer_per.zero_grad()
                output = self.model_per(x)
                loss = self.loss(output, y)

                # Spectral distillation regularization (reversed direction)
                if batch_idx > 0:
                    spec_w = spectral_cal(self.model_per, as_log_prob=False)
                    spec_l = spectral_cal(global_model, as_log_prob=True)

                    l_reg = self.loss_func_kl(spec_l, spec_w)
                    loss += self.lambda_l * l_reg

                # Proximal term: ||w_global - w_personal||
                if self.beta > 0 and batch_idx > 0:
                    w_diff = torch.tensor(0.0).to(self.device)
                    for w_g, w_p in zip(global_model.parameters(), self.model_per.parameters()):
                        w_diff += torch.pow(torch.norm(w_g - w_p), 2)
                    w_diff = torch.sqrt(w_diff)
                    loss += self.beta * w_diff

                loss.backward()
                self.optimizer_per.step()

        if self.learning_rate_decay:
            self.learning_rate_scheduler_per.step()

        self.train_time_cost['total_cost'] += time.time() - start_time

    def test_metrics(self, batch_size=None):
        """Evaluate using the personal model."""
        if batch_size is None:
            batch_size = self.batch_size

        testloader = self.load_test_data(batch_size=batch_size)
        self.model_per.eval()

        test_acc = 0
        test_num = 0
        y_prob = []
        y_true = []

        with torch.no_grad():
            for x, y in testloader:
                x, y = x.to(self.device), y.to(self.device)
                output = self.model_per(x)

                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                test_num += y.shape[0]

                y_prob.append(output.detach().cpu())
                y_true.append(y.cpu())

        y_prob = torch.cat(y_prob, dim=0)
        y_true = torch.cat(y_true, dim=0)

        return test_acc, test_num, y_prob, y_true

    def train_metrics(self, batch_size=None):
        """Evaluate training loss using the personal model."""
        if batch_size is None:
            batch_size = self.batch_size

        trainloader = self.load_train_data(batch_size=batch_size)
        self.model_per.eval()

        train_num = 0
        losses = 0
        with torch.no_grad():
            for x, y in trainloader:
                x, y = x.to(self.device), y.to(self.device)
                output = self.model_per(x)
                loss = self.loss(output, y)
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]

        return losses, train_num
