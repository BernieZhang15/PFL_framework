import time
import torch
import torch.nn.functional as F
from flcore.clients.clientbase import Client


class  clientMetaBAYES(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        w_params = []
        r_params = []
        for name, param in self.model.named_parameters():
            if any(k in name for k in ['W_sgd', 'W_bias']):
                w_params.append(param)
            else:
                r_params.append(param)

        self.optimizer = torch.optim.SGD([{'params': w_params, 'lr': self.learning_rate},
                                          {'params': r_params, 'lr': self.learning_rate}])

        self.loss = torch.nn.NLLLoss()

    def train(self):
        trainloader = self.load_train_data()

        self.model.train()

        start_time = time.time()

        for step in range(self.local_epochs):

            x, y = self.get_next_batch(trainloader)

            self.optimizer.zero_grad()

            x, y = x.to(self.device), y.to(self.device)

            output, kl = self.model(x)

            output = F.softmax(output, dim=1).reshape(self.num_ensemble, x.shape[0], self.num_classes)
            output = torch.log(torch.mean(output, dim=0))

            nll_loss = self.loss(output, y)

            loss = nll_loss + kl / self.train_samples * 0.1

            loss.backward()
            self.optimizer.step()

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    def test_metrics(self):
        testloader = self.load_test_data(self.test_samples)

        self.model.eval()

        test_acc = 0
        test_num = 0
        y_prob = []
        y_true = []

        with torch.no_grad():
            for x, y in testloader:
                x, y = x.to(self.device), y.to(self.device)

                output, kl = self.model(x)

                output = F.softmax(output, dim=1).reshape(self.num_ensemble, x.shape[0], self.num_classes)
                output = torch.log(torch.mean(output, dim=0))

                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                test_num += y.shape[0]

                y_prob.append(output.detach().cpu())
                y_true.append(y.cpu())

        y_prob = torch.cat(y_prob, axis=0)
        y_true = torch.cat(y_true, axis=0)

        return test_acc, test_num, y_prob, y_true

    def train_metrics(self, batch_size=None):
        trainloader = self.load_train_data(batch_size)
        self.model.eval()

        train_num = 0
        losses = 0
        with torch.no_grad():
            for x, y in trainloader:
                x, y = x.to(self.device), y.to(self.device)

                output, kl = self.model(x)

                output = F.softmax(output, dim=1).reshape(self.num_ensemble, x.shape[0], self.num_classes)
                output = torch.log(torch.mean(output, dim=0))

                loss = self.loss(output, y)
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]

        return losses, train_num

    def set_parameters(self, model):
        for (module_name, new_param), old_param in zip(model.named_parameters(), self.model.parameters()):
            if 'W' in module_name:
                old_param.data = new_param.data.clone()

