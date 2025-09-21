import os
import time
import copy
import torch
import torch.nn.functional as F
from flcore.clients.clientbase import Client


class clientpFedBayes(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        self.personalized_learning_rate = args.p_learning_rate

        # these parameters are for personalized federated learning.
        self.per_model = copy.deepcopy(self.model)

        self.optimizer1 = torch.optim.SGD(self.per_model.parameters(), lr=self.personalized_learning_rate)
        self.optimizer2 = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)


    def train(self):

        Round = 5

        start_time = time.time()

        self.model.train()
        self.per_model.train()

        trainloader = self.load_train_data()

        for step in range(self.local_epochs):

            x, y = self.get_next_batch(trainloader)
            y_onehot = F.one_hot(y, num_classes=self.num_classes)

            # personalized model updating
            for r in range(Round):

                epsilons = self.per_model.sample_epsilons(self.model.layer_param_shapes)
                layer_params = self.per_model.transform_gaussian_samples(
                    self.per_model.mus, self.per_model.rhos, epsilons)

                p_output = self.per_model.net(x, layer_params)

                # calculate the loss
                p_loss = self.per_model.combined_loss_personalized(
                    p_output, y_onehot, self.per_model.mus, self.per_model.sigmas,
                    copy.deepcopy(self.model.mus), [t.clone().detach() for t in self.model.sigmas], self.train_samples)

                self.optimizer1.zero_grad()
                p_loss.backward()
                self.optimizer1.step()

            # update personalized model sigmas
            self.per_model.transform_rhos(self.per_model.rhos)

            # localized global model update
            model_loss = self.model.combined_loss_local(
                copy.deepcopy(self.per_model.mus), [t.clone().detach() for t in self.per_model.sigmas],
                self.model.mus, self.model.sigmas, self.train_samples)

            self.optimizer2.zero_grad()
            model_loss.backward()
            self.optimizer2.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time


    def test_metrics(self, batch_size=None):

        if batch_size is None:
            batch_size = self.batch_size

        self.per_model.eval()

        test_acc = 0
        eval_num = 0
        y_prob = []
        y_true = []

        testloader = self.load_test_data(batch_size)

        with torch.no_grad():
            for x, y in testloader:
                x, y = x.to(self.device), y.to(self.device)

                outputs = []
                for _ in range(self.num_ensemble):
                    # personalized model
                    epsilons = self.per_model.sample_epsilons(self.per_model.layer_param_shapes)
                    layer_params = self.per_model.transform_gaussian_samples(self.per_model.mus, self.per_model.rhos,
                                                                             epsilons)
                    # forward-propagate the batch
                    output = self.per_model.net(x, layer_params)
                    outputs.append(output)

                outputs = torch.stack(outputs, dim=0)
                outputs = F.softmax(outputs, dim=2)
                output = torch.mean(outputs, dim=0)

                preds = torch.argmax(output, dim=1)
                test_acc += (torch.sum(preds == y)).item()

                y_prob.append(output.detach().cpu())
                y_true.append(y.cpu())
                eval_num += y.shape[0]

            y_prob = torch.cat(y_prob, dim=0)
            y_true = torch.cat(y_true, dim=0)

        return test_acc, eval_num, y_prob, y_true

    def train_metrics(self, batch_size=None):

        self.per_model.eval()

        train_num = 0
        losses = 0
        testloader = self.load_test_data(batch_size)

        with torch.no_grad():
            for x, y in testloader:
                x, y = x.to(self.device), y.to(self.device)
                y_onehot = F.one_hot(y, num_classes=self.num_classes)

                # personalized model
                epsilons = self.per_model.sample_epsilons(self.model.layer_param_shapes)
                layer_params = self.per_model.transform_gaussian_samples(self.per_model.mus,
                                                                         self.per_model.rhos, epsilons)
                # forward-propagate the batch
                outputs = self.per_model.net(x, layer_params)

                loss = self.per_model.combined_loss_personalized(outputs, y_onehot, self.per_model.mus, self.per_model.sigmas,
                    copy.deepcopy(self.model.mus), [t.clone().detach() for t in self.model.sigmas], self.train_samples)

                losses += loss.item()
                train_num += y.shape[0]

        return losses, train_num

    def save_personalized_model(self):
        model_path = os.path.join("models", self.dataset, "pFedBayes_clients_{}".format(str(self.seed)))
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_path = os.path.join(model_path, self.algorithm + "_client_{}_{}".format(self.learning_rate, self.id) + ".pt")
        torch.save(self.per_model, model_path)
