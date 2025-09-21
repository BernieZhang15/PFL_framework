import time
import copy
import torch
from flcore.clients.clientbase import Client
from flcore.optimizers.fedoptimizer import PerAvgOptimizer


class clientPerAvg(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        # self.beta = args.beta
        self.beta = self.learning_rate

        self.optimizer = PerAvgOptimizer(self.model.parameters(), lr=self.learning_rate)
        self.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer,
            gamma=args.learning_rate_decay_gamma
        )

    def train(self):
        trainloader = self.load_train_data(self.batch_size * 2)
        start_time = time.time()

        self.model.train()

        for step in range(self.local_epochs):  # local update
            X, Y = self.get_next_batch(trainloader)

            temp_model = copy.deepcopy(list(self.model.parameters()))

            # step 1
            x = X[:self.batch_size].to(self.device)
            y = Y[:self.batch_size].to(self.device)

            output = self.model(x)
            loss = self.loss(output, y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # step 2
            x = X[self.batch_size:].to(self.device)
            y = Y[self.batch_size:].to(self.device)

            self.optimizer.zero_grad()
            output = self.model(x)
            loss = self.loss(output, y)
            loss.backward()

            # restore the model parameters to the one before first update
            for old_param, new_param in zip(self.model.parameters(), temp_model):
                old_param.data = new_param.data.clone()

            self.optimizer.step(beta=self.beta)

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    def train_one_step(self):
        self.model.train()

        trainloader = self.load_train_data()
        iter_loader = iter(trainloader)
        x, y = next(iter_loader)
        x, y = x.to(self.device), y.to(self.device)

        output = self.model(x)
        loss = self.loss(output, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train_one_epoch(self):
        self.model.train()

        trainloader = self.load_train_data(self.batch_size)
        for i, (x, y) in enumerate(trainloader):
            x, y = x.to(self.device), y.to(self.device)
            output = self.model(x)
            loss = self.loss(output, y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()




