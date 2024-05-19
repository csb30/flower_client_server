import flwr as fl
import numpy as np
import customtkinter as ctk


class SaveModelStrategy(fl.server.strategy.FedAvg):
    def __init__(self, label: ctk.CTkLabel, progress: ctk.CTkProgressBar, rounds=3, init_param=None):
        super().__init__(on_fit_config_fn=fit_config, on_evaluate_config_fn=eval_config, initial_parameters=init_param)
        self.label = label
        self.progress = progress
        self.rounds = rounds

    def aggregate_fit(
            self,
            rnd,
            results,
            failures
    ):
        self.update_status(rnd)
        aggregated_weights = super().aggregate_fit(rnd, results, failures)
        if aggregated_weights is not None:
            # Save aggregated_weights
            print(f"Saving round {rnd} aggregated_weights...")
            np.savez(f"round-{rnd}-weights.npz", *aggregated_weights)
        return aggregated_weights

    def update_status(self, rnd):
        text = "Round " + str(rnd) + " / " + str(self.rounds)
        self.label.configure(text=text)
        x = rnd / self.rounds
        self.progress.set(x)

class FedAvgGUI(fl.server.strategy.FedAvg):
    def __init__(self, label: ctk.CTkLabel, progress: ctk.CTkProgressBar, rounds = 3, init_param=None):
        super().__init__(on_fit_config_fn=fit_config, on_evaluate_config_fn=eval_config, initial_parameters=init_param)
        self.label = label
        self.progress = progress
        self.rounds = rounds

    def aggregate_fit(
            self,
            rnd,
            results,
            failures
    ):
        self.update_status(rnd)
        return super().aggregate_fit(rnd, results, failures)

    def update_status(self, rnd):
        text = "Round " + str(rnd) + " / " + str(self.rounds)
        self.label.configure(text=text)
        x = rnd / self.rounds
        self.progress.set(x)

class FedAdamGUI(fl.server.strategy.FedAdagrad):
    def __init__(self, label: ctk.CTkLabel, progress: ctk.CTkProgressBar, rounds = 3, init_param=None):
        super().__init__(on_fit_config_fn=fit_config, on_evaluate_config_fn=eval_config, initial_parameters=init_param)
        self.label = label
        self.progress = progress
        self.rounds = rounds

    def aggregate_fit(
            self,
            rnd,
            results,
            failures
    ):
        self.update_status(rnd)
        return super().aggregate_fit(rnd, results, failures)

    def update_status(self, rnd):
        text = "Round " + str(rnd) + " / " + str(self.rounds)
        self.label.configure(text=text)
        x = rnd / self.rounds
        self.progress.set(x)

class FedProxGUI(fl.server.strategy.FedProx):
    def __init__(self, label: ctk.CTkLabel, progress: ctk.CTkProgressBar, rounds = 3, init_param = None, proximal_mu = 0.1):
        super().__init__(on_fit_config_fn=fit_config, on_evaluate_config_fn=eval_config, initial_parameters=init_param, proximal_mu=proximal_mu)
        self.label = label
        self.progress = progress
        self.rounds = rounds

    def aggregate_fit(
            self,
            rnd,
            results,
            failures
    ):
        self.update_status(rnd)
        return super().aggregate_fit(rnd, results, failures)

    def update_status(self, rnd):
        text = "Round " + str(rnd) + " / " + str(self.rounds)
        self.label.configure(text=text)
        x = rnd / self.rounds
        self.progress.set(x)

def fit_config(server_round: int):
    config = {
        "server_round": server_round,  # The current round of federated learning
    }
    return config

def eval_config(server_round: int):
    config = {
        "server_round": server_round,  # The current round of federated learning
    }
    return config