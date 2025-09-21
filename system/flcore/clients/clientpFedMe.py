import time
import copy
import torch
from flcore.clients.clientbase import Client
from flcore.optimizers.fedoptimizer import pFedMeOptimizer


class clientpFedMe(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        self.lamda = args.lamda
        self.K = args.K
        self.personalized_learning_rate = args.p_learning_rate

        # these parameters are for personalized federated learning.
        self.local_params = copy.deepcopy(list(self.model.parameters()))
        self.personalized_params = copy.deepcopy(list(self.model.parameters()))

        self.optimizer = pFedMeOptimizer(
            self.model.parameters(), lr=self.personalized_learning_rate, lamda=self.lamda)

        self.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer, 
            gamma=args.learning_rate_decay_gamma
        )

    def train(self):
        trainloader = self.load_train_data()

        start_time = time.time()

        self.model.train()

        for step in range(self.local_epochs):  # local update

            x, y = self.get_next_batch(trainloader)

            # K is number of personalized steps
            for i in range(self.K):
                self.optimizer.zero_grad()
                output = self.model(x)
                loss = self.loss(output, y)
                loss.backward()

                # finding approximate theta
                self.personalized_params = self.optimizer.step(self.local_params, self.device)

            # update local weight after finding approximate theta
            for new_param, local_weight in zip(self.personalized_params, self.local_params):
                local_weight = local_weight.to(self.device)
                local_weight.data = local_weight.data - self.lamda * self.learning_rate * (local_weight.data - new_param.data)

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.update_parameters(self.model, self.local_params)

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time


    def set_parameters(self, model):
        for new_param, old_param, local_param in zip(model.parameters(), self.model.parameters(), self.local_params):
            old_param.data = new_param.data.clone()
            local_param.data = new_param.data.clone()

