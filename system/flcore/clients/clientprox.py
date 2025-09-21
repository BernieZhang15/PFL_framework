import torch
import time
import copy
import torch.nn as nn
from flcore.clients.clientbase import Client
from flcore.optimizers.fedoptimizer import PerturbedGradientDescent


class clientProx(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        self.mu = args.mu

        self.global_params = copy.deepcopy(list(self.model.parameters()))

        self.loss = nn.CrossEntropyLoss()
        self.optimizer = PerturbedGradientDescent(self.model.parameters(), lr=self.learning_rate, mu=self.mu)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[500, 800], gamma=0.1)

    def train(self):
        trainloader = self.load_train_data()
        start_time = time.time()

        self.model.train()

        for step in range(self.local_epochs):
            x, y =  self.get_next_batch(trainloader)
            output = self.model(x)
            loss = self.loss(output, y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step(self.global_params, self.device)

        if self.learning_rate_decay:
            self.scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time


    def set_parameters(self, model):
        for new_param, global_param, param in zip(model.parameters(), self.global_params, self.model.parameters()):
            global_param.data = new_param.data.clone()
            param.data = new_param.data.clone()

    def train_metrics(self, batch_size):
        trainloader = self.load_train_data(batch_size)

        self.model.eval()

        train_num = 0
        losses = 0

        with torch.no_grad():
            for x, y in trainloader:
                x, y = x.to(self.device), y.to(self.device)
                output = self.model(x)
                loss = self.loss(output, y)

                gm = torch.cat([p.data.view(-1) for p in self.global_params], dim=0)
                pm = torch.cat([p.data.view(-1) for p in self.model.parameters()], dim=0)
                loss += 0.5 * self.mu * torch.norm(gm-pm, p=2)

                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]

        return losses, train_num
