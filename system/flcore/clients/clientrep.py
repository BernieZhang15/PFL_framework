import time
import torch
import numpy as np
from flcore.clients.clientbase import Client


class clientRep(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        
        self.optimizer = torch.optim.SGD(self.model.base.parameters(), lr=self.learning_rate)
        self.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer, 
            gamma=args.learning_rate_decay_gamma
        )
        self.optimizer_per = torch.optim.SGD(self.model.head.parameters(), lr=self.learning_rate)
        self.learning_rate_scheduler_per = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer_per, 
            gamma=args.learning_rate_decay_gamma
        )

        self.plocal_steps = args.plocal_steps

    def train(self):
        trainloader = self.load_train_data()
        
        start_time = time.time()

        self.model.train()

        for param in self.model.base.parameters():
            param.requires_grad = False
        for param in self.model.head.parameters():
            param.requires_grad = True

        for step in range(self.plocal_steps):
            x, y = self.get_next_batch(trainloader)

            x, y = x.to(self.device), y.to(self.device)

            output = self.model(x)
            loss = self.loss(output, y)
            self.optimizer_per.zero_grad()
            loss.backward()
            self.optimizer_per.step()
                
        max_local_epochs = self.local_epochs

        for param in self.model.base.parameters():
            param.requires_grad = True
        for param in self.model.head.parameters():
            param.requires_grad = False

        for step in range(max_local_epochs):

            x, y = self.get_next_batch(trainloader)

            x, y = x.to(self.device), y.to(self.device)

            output = self.model(x)
            loss = self.loss(output, y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()
            self.learning_rate_scheduler_per.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time
        
            
    def set_parameters(self, base):
        for new_param, old_param in zip(base.parameters(), self.model.base.parameters()):
            old_param.data = new_param.data.clone()