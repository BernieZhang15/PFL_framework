import time
import torch
import torch.nn.functional as F
from collections import OrderedDict
from torch.utils.data import DataLoader
from flcore.clients.clientbase import Client
from utils.data_utils import read_client_data


class clientMetaBAYES(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        self.lr_matrix = None
        self.sep_data = args.sep_data
        self.adaptive_rank = args.adaptive_rank

        self.w_params = []
        self.r_params = []
        for name, param in self.model.named_parameters():
            if "W" in name:
                self.w_params.append(param)
            elif any(n in name for n in ["alpha", "gamma"]):
                self.r_params.append(param)

        self.temperature = args.temperature
        self.lr_learning_rate = 0.01

        self.optimizer_W = torch.optim.SGD(self.w_params, self.learning_rate, momentum=0.9, weight_decay=5e-4)
        self.optimizer_R = torch.optim.SGD(self.r_params, self.lr_learning_rate, momentum=0.9, weight_decay=5e-4)

    def train(self):

        self.model.train()

        start_time = time.time()

        if self.sep_data:
            train_loader, val_loader = self.load_meta_data()
        else:
            train_loader = self.load_train_data()
            val_loader = train_loader

        for step in range(self.local_epochs):

            # Inner loop updating low-rank matrices
            lr_matrix = self.update_lr_matrices(train_loader)

            # Meta Update
            self.optimizer_W.zero_grad()

            loss = self.meta_forward_pass(val_loader, lr_matrix)

            loss.backward()

            self.optimizer_W.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    def meta_forward_pass(self, dataloader, lr_matrix=None):

        x, y = self.get_next_batch(dataloader)

        output, kl = self.model(x, lr_matrix)

        output = F.softmax(output, dim=1).reshape(self.num_ensemble, x.shape[0], self.num_classes)
        output = torch.log(torch.mean(output, dim=0))

        loss = F.nll_loss(output, y)

        return loss

    def update_lr_matrices(self, dataloader, update_step=None, learning_rate=None):

        update_step = self.num_update_rs if update_step is None else update_step
        learning_rate = self.lr_learning_rate if learning_rate is None else learning_rate

        lr_matrix = OrderedDict((name, param) for (name, param) in self.model.named_parameters()
                                if any(n in name for n in ["alpha", "gamma"]))

        for i in range(update_step):

            x, y = self.get_next_batch(dataloader)

            output, kl = self.model(x, lr_matrix)

            output = F.softmax(output, dim=1).reshape(self.num_ensemble, x.shape[0], self.num_classes)
            output = torch.log(torch.mean(output, dim=0))

            # Calculate NLL loss
            nll_loss = F.nll_loss(output, y)

            loss = nll_loss + kl / self.train_samples * self.temperature

            grads = torch.autograd.grad(loss, lr_matrix.values())
            lr_matrix = OrderedDict((name, param - learning_rate * grad) for ((name, param), grad)
                                    in zip(lr_matrix.items(), grads))

        return lr_matrix

    def load_meta_data(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size

        meta_data = read_client_data(self.dataset, self.id, is_train=True)
        num_data = len(meta_data)
        return (DataLoader(meta_data[:num_data // 2], batch_size, drop_last=False, shuffle=True),
                DataLoader(meta_data[num_data // 2:], batch_size, drop_last=False, shuffle=True))


    def evaluate(self, dataloader, lr_matrices=None):

        self.model.eval()

        eval_acc = 0
        eval_num = 0
        y_prob = []
        y_true = []

        with torch.no_grad():
            for x, y in dataloader:
                x, y = x.to(self.device), y.to(self.device)

                output, kl = self.model(x, lr_matrices)

                output = F.softmax(output, dim=1).reshape(self.num_ensemble, x.shape[0], self.num_classes)
                output = torch.mean(output, dim=0)

                eval_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                eval_num += y.shape[0]

                y_prob.append(output.detach().cpu())
                y_true.append(y.cpu())

            y_prob = torch.cat(y_prob, axis=0)
            y_true = torch.cat(y_true, axis=0)

        return eval_acc, eval_num, y_prob, y_true

    def test_metrics(self):
        # Local adaption
        trainloader = self.load_train_data()
        self.lr_matrix = self.update_lr_matrices(trainloader, update_step=40, learning_rate=0.005)

        testloader = self.load_test_data(32)
        test_acc, test_num, test_prob, test_true = self.evaluate(testloader, self.lr_matrix)

        return test_acc, test_num, test_prob, test_true

    def train_metrics(self, batch_size=None):

        self.model.eval()

        trainloader = self.load_train_data(batch_size)

        train_num = 0
        losses = 0
        with torch.no_grad():
            for x, y in trainloader:
                x, y = x.to(self.device), y.to(self.device)

                output, kl = self.model(x, self.lr_matrix)

                output = F.softmax(output, dim=1).reshape(self.num_ensemble, x.shape[0], self.num_classes)
                output = torch.log(torch.mean(output, dim=0))

                loss = F.nll_loss(output, y)
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]

        return losses, train_num

    def set_parameters(self, model):
        for (module_name, new_param), old_param in zip(model.named_parameters(), self.model.parameters()):
            if 'W' in module_name:
                old_param.data = new_param.data.clone()


