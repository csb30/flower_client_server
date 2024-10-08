import sys
from collections import OrderedDict
from typing import List, Tuple

import flwr.server
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
# from datasets.utils.logging import disable_progress_bar
from torch.utils.data import DataLoader
import time

import flwr as fl
from flwr.common import Metrics

from typing import Dict, Optional

import customtkinter as ctk
import threading

from model import *

DEVICE = torch.device("cpu")

LOGFILE = "client" + sys.argv[1] + ".txt"

# delete previous log
f = open(LOGFILE, "wt")
f.close()
fl.common.logger.configure(identifier="client", filename=LOGFILE)


# Define Flower client with GUI access
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid, net, trainloader, valloader, label: ctk.CTkLabel):
        self.cid = cid
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.label = label
        self.results = [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]

    def get_parameters(self, config=None):
        print(f"[Client {self.cid}] get_parameters")
        self.label.configure(text="Sending parameters...")
        return self.net.get_parameters_n()

    def fit(self, parameters, config):
        # Read values from config
        server_round = config["server_round"]
        # local_epochs = config["local_epochs"]

        # Use values provided by the config
        print(f"[Client {self.cid}] fit, round {server_round}, config: {config}")
        self.label.configure(text=f"Round {server_round}: \nTraining model locally...")
        self.net.set_parameters_n(parameters)
        self.net.train_n(self.trainloader, 1, DEVICE)
        return self.net.get_parameters_n(), len(self.trainloader), {}

    def evaluate(self, parameters, config):
        self.net.set_parameters_n(parameters)
        loss, accuracy = self.net.test_n(self.valloader, DEVICE)
        server_round = config["server_round"]
        self.results[server_round - 1][0] = loss
        self.results[server_round - 1][1] = accuracy
        print(f"[Client {self.cid}] evaluate, config: {config}, loss: {loss}, accuracy: {accuracy}")
        self.label.configure(text=f"Round\t| 1\t| 2\t| 3\n"
                                  f"Loss\t| {self.results[0][0]:.2f}\t| {self.results[1][0]:.2f}\t| {self.results[2][0]:.2f}\n"
                                  f"Acc\t| {self.results[0][1]:.2f}\t| {self.results[1][1]:.2f}\t| {self.results[2][1]:.2f}")
        return float(loss), len(self.valloader), {"accuracy": float(accuracy)}


# Start Flower client

class FlowerClientGUI:
    def __init__(self, clientnum: int):
        # init
        self.trainloaders = None
        self.valloaders = None
        self.testloader = None
        self.clientnum = clientnum

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("dark-blue")

        self.root = ctk.CTk()
        self.root.geometry("500x350")
        self.root.title("Flower Client " + str(clientnum))

        self.frame = ctk.CTkFrame(self.root)

        # title
        title = "Flower Client " + str(clientnum) + " GUI"
        self.label = ctk.CTkLabel(self.frame, text=title, font=('Roboto', 24))

        # status, progress
        self.status = ctk.CTkLabel(self.frame, text="Press Start", font=('Roboto', 15))
        self.progress = ctk.CTkProgressBar(self.frame)
        self.progress.set(0)

        # log
        self.scroll = ctk.CTkScrollableFrame(self.frame)
        self.log = ctk.CTkLabel(self.scroll, text="Press Start", font=('Roboto', 10), justify="left")

        # threads
        self.th_client = threading.Thread(target=self.start_client)
        self.th_log = threading.Thread(target=self.update_log)

        # options
        self.ipinput = ctk.CTkTextbox(self.frame, height=10)
        self.ipinput.insert(ctk.END, "localhost:8500")

        # button
        self.button = ctk.CTkButton(self.frame, text="Loading datasets...",
                                    command=lambda: [self.th_client.start(), self.th_log.start()], state="disabled")

        th_load = threading.Thread(target=self.load_datasets)
        th_load.start()

    def start_client(self):
        # delete log
        f = open(LOGFILE, "wt")
        f.close()

        self.status.pack(padx=10, pady=12)
        # self.progress.pack(padx=10, pady=12)
        self.scroll.pack(padx=20, pady=20, fill="both", expand=True)
        self.log.pack(padx=10, pady=12, anchor="w")

        self.button.pack_forget()
        ip = self.ipinput.get("0.0", ctk.END)
        self.ipinput.pack_forget()

        self.status.configure("Connecting to server...")

        try:
            fl.client.start_client(
                server_address=ip,
                client=self.client_fn(self.clientnum, DEVICE, self.status).to_client(),
                grpc_max_message_length=1024 * 1024 * 1024
            )
        except:
            print("AN error occured")

        finally:
            #reset UI
            self.ipinput.pack(padx=10, pady=12)
            self.button.pack(padx=10, pady=12)
            self.log.pack_forget()
            self.scroll.pack_forget()
            # self.status.pack_forget()
            # self.progress.pack_forget()
            self.th_client = threading.Thread(target=self.start_client)

    def start_gui(self):
        self.frame.pack(padx=20, pady=20, fill="both", expand=True)

        self.label.pack(padx=10, pady=12)

        self.ipinput.pack(padx=10, pady=12)
        self.button.pack(padx=10, pady=12)

        self.root.mainloop()

    def load_datasets(self, NUM_CLIENTS=10, BATCH_SIZE=32):
        from flwr_datasets import FederatedDataset

        fds = FederatedDataset(dataset="cifar10", partitioners={"train": NUM_CLIENTS})

        def apply_transforms(batch):
            # Instead of passing transforms to CIFAR10(..., transform=transform)
            # we will use this function to dataset.with_transform(apply_transforms)
            # The transforms object is exactly the same
            transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            )
            batch["img"] = [transform(img) for img in batch["img"]]
            return batch

        # Create train/val for each partition and wrap it into DataLoader
        self.trainloaders = []
        self.valloaders = []
        for partition_id in range(NUM_CLIENTS):
            partition = fds.load_partition(partition_id, "train")
            partition = partition.with_transform(apply_transforms)
            partition = partition.train_test_split(train_size=0.8)
            self.trainloaders.append(DataLoader(partition["train"], batch_size=BATCH_SIZE))
            self.valloaders.append(DataLoader(partition["test"], batch_size=BATCH_SIZE))
        testset = fds.load_split("test").with_transform(apply_transforms)
        self.testloader = DataLoader(testset, batch_size=BATCH_SIZE)

        #enable the start button
        self.button.configure(state="normal", text="Start client")

    def client_fn(self, cid, DEVICE, label) -> FlowerClient:
        net = Net().to(DEVICE)
        trainloader = self.trainloaders[int(cid)]
        valloader = self.valloaders[int(cid)]
        return FlowerClient(cid, net, trainloader, valloader, label)

    def update_log(self):
        #updates the log UI element from log file
        f = open(LOGFILE, "rt")
        text = ""
        while self.th_client.is_alive():
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


client = FlowerClientGUI(int(sys.argv[1]))
client.start_gui()
