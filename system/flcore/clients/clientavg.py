import time
from flcore.clients.clientbase import Client


class clientAVG(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

    def train(self):
        trainloader = self.load_train_data()

        self.model.train()

        start_time = time.time()

        for step in range(self.local_epochs):
            x, y = self.get_next_batch(trainloader)

            self.optimizer.zero_grad()
            output = self.model(x)
            loss = self.loss(output, y)
            loss.backward()
            self.optimizer.step()

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

