import flwr as fl
import sys

import flwr.server
import numpy as np
import customtkinter as ctk
import threading
import time

# https://github.com/PratikGarai/MNIST-Federated/tree/master/03_Non%20IID%20Demo
from flwr.common import Parameters, Scalar

f = open("server.txt", "wt")
f.close()
fl.common.logger.configure(identifier="server", filename="server.txt")


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


class FlowerServer:
    def __init__(self):
    #init
        self.strategy = None
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("dark-blue")

        self.root = ctk.CTk()
        self.root.geometry("1000x500")
        self.root.title("Flower Server")

        self.frame = ctk.CTkFrame(self.root)

        self.history = flwr.server.History()

        self.total_rounds = 3

    #title
        self.label = ctk.CTkLabel(self.frame, text="Flower Server GUI", font=('Roboto', 24))

    #Status, scroll, log
        self.status = ctk.CTkLabel(self.frame, text="Press Start", font=('Roboto', 15))
        self.progress = ctk.CTkProgressBar(self.frame)
        self.progress.set(0)

        self.scroll = ctk.CTkScrollableFrame(self.frame)
        self.log = ctk.CTkLabel(self.scroll, text="Press Start", font=('Roboto', 15), justify="left")

    #threads
        self.th_server = threading.Thread(target=self.start_server)
        self.th_log = threading.Thread(target=self.update_log)


    #Options
        self.ipinput = ctk.CTkTextbox(self.frame, height=10)
        self.ipinput.insert(ctk.END, "localhost:8500")

        self.stratSTR = ctk.StringVar(self.root)
        self.stratSTR.set("Select Strategy")
        self.stratOPT = ctk.CTkOptionMenu(self.frame, variable=self.stratSTR,
                                          values=['SaveModelStrategy', 'FedAvgGUI', 'FedAdamGUI'])

        self.rounds = ctk.CTkTextbox(self.frame, height=10)
        self.rounds.insert(ctk.END, "3")

    #Button
        self.button = ctk.CTkButton(self.frame, text="Start Server",
                                    command=lambda: [self.th_server.start(), self.th_log.start()])

    def finish(self):
        text = ""
        for i in self.history.losses_distributed:
            round, loss = i
            text += "round " + str(round) + ": " + str(loss * 100) + "%\n"
        self.status.configure(text=text)

    def start_server(self):
        # delete log
        f = open("server.txt", "wt")
        f.close()

        #show
        self.status.pack(padx=10, pady=12)
        self.progress.pack(padx=10, pady=12)
        self.scroll.pack(padx=20, pady=20, fill="both", expand=True)
        self.log.pack(padx=10, pady=12, anchor="w")

        #hide
        self.button.pack_forget()
        self.ipinput.pack_forget()
        self.rounds.pack_forget()
        self.stratOPT.pack_forget()

        ip = self.ipinput.get("0.0", ctk.END)
        self.total_rounds = int(self.rounds.get("0.0", ctk.END))

        self.status.configure(text="Waiting for results...")

        if self.stratSTR.get() == 'SaveModelStrategy':
            self.strategy = SaveModelStrategy(self.status, self.progress, self.total_rounds)
        elif self.stratSTR.get() == 'FedAdamGUI':
            self.strategy = FedAdamGUI(self.status, self.progress, self.total_rounds)
        else:
            self.strategy = FedAvgGUI(self.status, self.progress, self.total_rounds)

        self.history = fl.server.start_server(
            server_address=ip,
            config=fl.server.ServerConfig(num_rounds=self.total_rounds),
            grpc_max_message_length=1024 * 1024 * 1024,
            strategy=self.strategy,
        )
        self.finish()

        #show
        self.ipinput.pack(padx=10, pady=12)
        self.rounds.pack(padx=10, pady=12)
        self.stratOPT.pack(padx=10, pady=12)
        self.button.pack(padx=10, pady=12)

        #hide
        self.log.pack_forget()
        self.scroll.pack_forget()
        self.progress.pack_forget()

        self.th_server = threading.Thread(target=self.start_server)

    def update_log(self):
        f = open("server.txt", "rt")
        text = ""
        while self.th_server.is_alive():
            line = f.readline()
            while line != "":
                if len(line.split("|")) >= 4:
                    text += line.split("|")[3]
                    self.log.configure(text=text)
                line = f.readline()
            time.sleep(1)
        line = f.readline()
        while line != "":
            if len(line.split("|")) >= 4:
                text += line.split("|")[3]
                self.log.configure(text=text)
            line = f.readline()
        f.close()
        self.th_log = threading.Thread(target=self.update_log)

    def start_gui(self):
        self.frame.pack(padx=20, pady=20, fill="both", expand=True)

        self.label.pack(padx=10, pady=12)
        self.ipinput.pack(padx=10, pady=12)
        self.rounds.pack(padx=10, pady=12)
        self.stratOPT.pack(padx=10, pady=12)
        self.button.pack(padx=10, pady=12)

        self.root.mainloop()


server = FlowerServer()
server.start_gui()
