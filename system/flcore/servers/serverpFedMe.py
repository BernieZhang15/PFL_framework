
import copy
from flcore.servers.serverbase import Server
from flcore.clients.clientpFedMe import clientpFedMe



class pFedMe(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.selected_clients = None
        self.set_slow_clients()
        self.set_clients(clientpFedMe)

        self.previous_global_model = None

        self.beta = args.beta

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

    def train(self):
        for i in range(self.global_rounds + 1):

            self.selected_clients = self.select_clients()
            self.send_models()

            for client in self.selected_clients:
                client.train()

            self.previous_global_model = copy.deepcopy(list(self.global_model.parameters()))

            self.receive_models()
            self.aggregate_parameters()
            self.beta_aggregate_parameters()

            if i % self.eval_gap == 0:

                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate personalized model")
                for client in self.clients:
                    client.update_parameters(client.model, client.personalized_params)

                self.evaluate(i)

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc_per], top_cnt=self.top_cnt):
                break

        print("\nBest accuracy.")
        print(max(self.rs_test_acc))

        self.save_results()
        self.save_global_model()

        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(clientpFedMe)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate(self.global_rounds + 1)

    def beta_aggregate_parameters(self):
        # aggregate average model with previous model using parameter beta
        for pre_param, param in zip(self.previous_global_model, self.global_model.parameters()):
            param.data = (1 - self.beta) * pre_param.data + self.beta * param.data

