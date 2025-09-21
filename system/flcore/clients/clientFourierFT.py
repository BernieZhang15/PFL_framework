import time
import torch
from torch.nn import nn
import torch.nn.functional as F
from flcore.clients.clientbase import Client


class clientFourierFT(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)


    def train(self):
        train_loader = self.load_train_data()
        start_time = time.time()
        self.model.train()

        for step in range(self.local_epochs):
            for i, (x, y) in enumerate(train_loader):
                x, y =  x.to(self.device), y.to(self.device)

                self.optimizer.zero_grad()

                output, kl = self.model(x)

                loss = self.loss(output, y)

                spec_loss = 0.0
                for name, module in self.model.named_modules():
                    if isinstance(module, nn.Linear):
                        spec_loss += self.spectrum_regularization(module.weight, mode='high', radius=0.3, lambd=1e-3)

                loss += spec_loss + kl / self.train_samples

                loss.backward()
                self.optimizer.step()

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

        reg_loss = (W_fft[mask].abs() ** 2).sum() * lambd

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

                output = torch.mean(F.softmax(output, dim=2), dim=0)

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

                output = torch.mean(F.softmax(output, dim=2), dim=0)
                loss = F.cross_entropy(output, y)

                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]

        return losses, train_num

    def set_parameters(self, model):
        for (module_name, new_param), old_param in zip(model.named_parameters(), self.model.parameters()):
            if 'spectrum' not in module_name:
                old_param.data = new_param.data.clone()


