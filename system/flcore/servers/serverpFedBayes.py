import copy
import torch
import torch.nn.functional as F
from flcore.servers.serverbase import Server
from flcore.clients.clientpFedBayes import clientpFedBayes


class pFedBayes(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.selected_clients = None
        self.set_slow_clients()
        self.set_clients(clientpFedBayes)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

    def train(self):

        for i in range(self.global_rounds + 1):

            self.selected_clients = self.select_clients()
            self.send_models()

            if i % self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate model")
                self.evaluate(i)

            for client in self.selected_clients:
                client.train()

            self.receive_models()
            self.aggregate_parameters()

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc_per], top_cnt=self.top_cnt):
                break

        print("\nBest accuracy.")
        print(max(self.rs_test_acc))

        self.save_results()
        self.save_global_model()
        self.save_personalized_model()

        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(clientpFedBayes)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate(self.global_rounds + 1)

    def save_personalized_model(self):
        for c in self.clients:
            c.save_personalized_model()

    def fine_tuning_new_clients(self):

        Round = 5

        self.global_model.transform_rhos(self.global_model.rhos)

        for client in self.new_clients:

            optimizer = torch.optim.SGD(client.per_model.parameters(), lr=client.personalized_learning_rate)

            trainloader = client.load_train_data()
            client.per_model.train()

            for i in range(self.fine_tuning_epoch):

                for (x, y) in trainloader:
                    x, y = x.to(self.device), y.to(self.device)
                    y_onehot = F.one_hot(y, num_classes=self.num_classes).to(self.device)

                    for r in range(Round):
                        epsilons = client.per_model.sample_epsilons(client.per_model.layer_param_shapes)
                        layer_params = client.per_model.transform_gaussian_samples(
                            client.per_model.mus, client.per_model.rhos, epsilons, client.train_samples)

                        outputs = client.per_model.net(x, layer_params)

                        # calculate the loss
                        p_loss = client.per_model.combined_loss_personalized(
                            outputs, y_onehot, client.per_model.mus, client.per_model.sigmas,
                            copy.deepcopy(self.global_model.mus), [t.clone().detach() for t in self.global_model.sigmas],
                            len(trainloader))

                        optimizer.zero_grad()
                        p_loss.backward()
                        optimizer.step()


