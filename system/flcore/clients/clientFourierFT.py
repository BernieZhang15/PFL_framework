import time
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
from typing import Optional
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from flcore.trainmodel.FourierFTModel import FourierFTLinear
from flcore.clients.clientbase import Client


class clientFourierFT(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        self.spec_lambda = args.spec_lambda
        self.delta_lr_ratio = getattr(args, "delta_lr_ratio", 1.0)
        self._delta_lr_last_decay_round = None

        base_params = []
        delta_params = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if 'spectrum' in name:
                delta_params.append(param)
            else:
                base_params.append(param)

        self.optimizer = torch.optim.SGD(
            [
                {"params": base_params, "lr": self.learning_rate},
                {"params": delta_params, "lr": self.learning_rate * self.delta_lr_ratio},
            ]
        )
        self.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer,
            gamma=args.learning_rate_decay_gamma
        )


    def train(self):
        train_loader = self.load_train_data()
        start_time = time.time()
        self.model.train()

        # if self.train_time_cost['num_rounds'] >= 400 and len(self.optimizer.param_groups) > 1:
        #     current_round = self.train_time_cost['num_rounds']
        #     target_round = 400 + ((current_round - 400) // 100) * 100
        #     if self._delta_lr_last_decay_round is None or target_round > self._delta_lr_last_decay_round:
        #         self.optimizer.param_groups[1]['lr'] *= 0.001
        #         self._delta_lr_last_decay_round = target_round

        for step in range(self.local_epochs):
            x, y = self.get_next_batch(train_loader)
            self.optimizer.zero_grad()
            output, kl = self.model(x)
            # Only apply ensemble averaging if output has ensemble dimension (3D)
            if output.dim() == 3:
                output = torch.mean(output, dim=0)  # (ens, batch, num_classes) -> (batch, num_classes)
            loss = self.loss(output, y)
            spec_loss = 0.0
            for name, module in self.model.named_modules():
                if isinstance(module, nn.Linear):
                    spec_loss += self.spectrum_regularization(module.weight, mode='high', radius=0.4, lambd=1e-3)
            # print(f"Step {step}: Spec Loss={spec_loss.item():.4f}")
            lambda_kl = 0.9
            loss += self.spec_lambda * spec_loss + lambda_kl * kl / self.train_samples
            # print(f"CE={loss.item():.4f}, KLps={(kl/self.train_samples).item():.4f}")

            loss.backward()
            self.optimizer.step()
            # for i, (x, y) in enumerate(train_loader):
            #     x, y =  x.to(self.device), y.to(self.device)

            #     self.optimizer.zero_grad()

            #     output, kl = self.model(x)
            #     output = torch.mean(F.softmax(output, dim=2), dim=0)
            #     loss = self.loss(output, y)

            #     spec_loss = 0.0
            #     for name, module in self.model.named_modules():
            #         if isinstance(module, nn.Linear):
            #             spec_loss += self.spectrum_regularization(module.weight, mode='high', radius=0.3, lambd=1e-3)

                # loss += spec_loss + kl / self.train_samples

                # loss.backward()
                # self.optimizer.step()

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    @staticmethod
    def spectrum_regularization(weight, mode="high", radius=0.3, lambd=1e-3):
        W_fft = torch.fft.fftshift(torch.fft.fft2(weight))
        h, w = W_fft.shape
        cy, cx = h // 2, w // 2

        Y, X = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
        dist = torch.sqrt(((Y - cy) / h) ** 2 + ((X - cx) / w) ** 2)

        if mode == "low":
            mask = dist <= radius
        elif mode == "high":
            mask = dist >= radius
        else:
            raise ValueError("mode must be 'low' or 'high'")
        power = W_fft.abs() ** 2
        # power = torch.nan_to_num(power, nan=0.0, posinf=0.0, neginf=0.0)

        reg_loss = (power[mask]).mean() * lambd

        return reg_loss


    def evaluate(self, dataloader):

        self.model.eval()

        eval_cor = 0
        eval_num = 0
        y_prob = []
        y_true = []

        with torch.no_grad():
            for x, y in dataloader:
                x, y = x.to(self.device), y.to(self.device)

                output, kl = self.model(x)

                # Only apply ensemble averaging if output has ensemble dimension (3D)
                if output.dim() == 3:
                    output = torch.mean(output, dim=0)  # (ens, batch, num_classes) -> (batch, num_classes)

                eval_cor += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                eval_num += y.shape[0]

                y_prob.append(output.detach().cpu())
                y_true.append(y.cpu())

            y_prob = torch.cat(y_prob, axis=0)
            y_true = torch.cat(y_true, axis=0)

        return eval_cor, eval_num, y_prob, y_true

    def test_metrics(self, batch_size=None):

        if batch_size is None:
            batch_size = self.batch_size

        testloader = self.load_test_data(batch_size)

        test_acc, test_num, test_prob, test_true = self.evaluate(testloader)

        return test_acc, test_num, test_prob, test_true

    def train_metrics(self, batch_size=None):

        if batch_size is None:
            batch_size = self.batch_size

        trainloader = self.load_train_data(batch_size=batch_size)

        self.model.eval()

        train_num = 0
        losses = 0
        with torch.no_grad():
            for x, y in trainloader:
                x, y = x.to(self.device), y.to(self.device)

                output, kl = self.model(x)

                # Only apply ensemble averaging if output has ensemble dimension (3D)
                if output.dim() == 3:
                    output = torch.mean(output, dim=0)  # (ens, batch, num_classes) -> (batch, num_classes)
                
                loss = self.loss(output, y)

                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]

        return losses, train_num

    def set_parameters(self, model):
        for (module_name, new_param), old_param in zip(model.named_parameters(), self.model.parameters()):
            if 'spectrum' not in module_name:
                old_param.data = new_param.data.clone()

    # def save_fft_spectra(self, round_idx: int, output_dir: Optional[str] = None):
    #     if output_dir is None:
    #         output_dir = os.path.join(self.save_folder_name, "fft_visuals")
    #     os.makedirs(output_dir, exist_ok=True)

    #     linear_layers = [(name, module) for name, module in self.model.named_modules()
    #                      if isinstance(module, FourierFTLinear)]
    #     if not linear_layers:
    #         print("[FourierFT] No FourierFTLinear layers found for visualization.")
    #         return

    #     target_layers = linear_layers[-3:]

    #     with torch.no_grad():
    #         for name, module in target_layers:
    #             base_weight = module.base_layer.weight.detach()
    #             base_fft = torch.fft.fftshift(torch.fft.fft2(base_weight))
    #             delta_spectrum = module.get_dense_spectrum(sample=True)
    #             delta_weight = torch.fft.ifft2(delta_spectrum).real * module.scaling
    #             delta_re_fft = torch.fft.fftshift(torch.fft.fft2(delta_weight))

    #             safe_name = name.replace(".", "_")
    #             base_fft_path = os.path.join(
    #                 output_dir,
    #                 f"round_{round_idx}_client_{self.id}_{safe_name}_base_fft.npy"
    #             )
    #             delta_spec_path = os.path.join(
    #                 output_dir,
    #                 f"round_{round_idx}_client_{self.id}_{safe_name}_delta_spectrum.npy"
    #             )
    #             delta_weight_path = os.path.join(
    #                 output_dir,
    #                 f"round_{round_idx}_client_{self.id}_{safe_name}_delta_weight.npy"
    #             )
    #             delta_re_fft_path = os.path.join(
    #                 output_dir,
    #                 f"round_{round_idx}_client_{self.id}_{safe_name}_delta_re_fft.npy"
    #             )

    #             np.save(base_fft_path, base_fft.detach().cpu().numpy())
    #             np.save(delta_spec_path, delta_spectrum.detach().cpu().numpy())
    #             np.save(delta_weight_path, delta_weight.detach().cpu().numpy())
    #             np.save(delta_re_fft_path, delta_re_fft.detach().cpu().numpy())

    #             base_vis = torch.log1p(base_fft.abs()).detach().cpu().numpy()
    #             delta_vis = torch.log1p(delta_spectrum.abs()).detach().cpu().numpy()
    #             delta_re_fft_vis = torch.log1p(delta_re_fft.abs()).detach().cpu().numpy()

    #             plt.figure(figsize=(5, 4))
    #             plt.imshow(base_vis, cmap="viridis", aspect="auto")
    #             plt.colorbar()
    #             plt.title(f"Base FFT | {safe_name}")
    #             plt.tight_layout()
    #             plt.savefig(
    #                 os.path.join(
    #                     output_dir,
    #                     f"round_{round_idx}_client_{self.id}_{safe_name}_base_fft.png"
    #                 ),
    #                 dpi=200
    #             )
    #             plt.close()

    #             dv_min, dv_max = np.percentile(delta_vis, [5, 99])
    #             plt.figure(figsize=(5, 4))
    #             plt.imshow(delta_vis, cmap="magma", aspect="auto", vmin=dv_min, vmax=dv_max)
    #             plt.colorbar()
    #             plt.title(f"Delta Spectrum | {safe_name}")
    #             plt.tight_layout()
    #             plt.savefig(
    #                 os.path.join(
    #                     output_dir,
    #                     f"round_{round_idx}_client_{self.id}_{safe_name}_delta_spectrum.png"
    #                 ),
    #                 dpi=300
    #             )
    #             plt.close()

    #             dr_min, dr_max = np.percentile(delta_re_fft_vis, [5, 99])
    #             plt.figure(figsize=(5, 4))
    #             plt.imshow(delta_re_fft_vis, cmap="magma", aspect="auto", vmin=dr_min, vmax=dr_max)
    #             plt.colorbar()
    #             plt.title(f"Delta Re-FFT | {safe_name}")
    #             plt.tight_layout()
    #             plt.savefig(
    #                 os.path.join(
    #                     output_dir,
    #                     f"round_{round_idx}_client_{self.id}_{safe_name}_delta_re_fft.png"
    #                 ),
    #                 dpi=300
    #             )
    #             plt.close()


