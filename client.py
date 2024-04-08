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
from datasets.utils.logging import disable_progress_bar
from torch.utils.data import DataLoader

import flwr as fl
from flwr.common import Metrics
from flwr_datasets import FederatedDataset

from typing import Dict, Optional

DEVICE = torch.device("cpu")

class Net(nn.Module):
    def __init__(self) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def train_n(self, trainloader, epochs: int, DEVICE, verbose=False):
        """Train the network on the training set."""
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters())
        self.train()
        for epoch in range(epochs):
            correct, total, epoch_loss = 0, 0, 0.0
            for batch in trainloader:
                images, labels = batch["img"].to(DEVICE), batch["label"].to(DEVICE)
                optimizer.zero_grad()
                outputs = self(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                # Metrics
                epoch_loss += loss
                total += labels.size(0)
                correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
            epoch_loss /= len(trainloader.dataset)
            epoch_acc = correct / total
            if verbose:
                print(f"Epoch {epoch + 1}: train loss {epoch_loss}, accuracy {epoch_acc}")

    def test_n(self, testloader, DEVICE):
        """Evaluate the network on the entire test set."""
        criterion = torch.nn.CrossEntropyLoss()
        correct, total, loss = 0, 0, 0.0
        self.eval()
        with torch.no_grad():
            for batch in testloader:
                images, labels = batch["img"].to(DEVICE), batch["label"].to(DEVICE)
                outputs = self(images)
                loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        loss /= len(testloader.dataset)
        accuracy = correct / total
        return loss, accuracy

    def set_parameters_n(self, parameters: List[np.ndarray]):
        params_dict = zip(self.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        self.load_state_dict(state_dict, strict=True)

    def get_parameters_n(self) -> List[np.ndarray]:
        return [val.cpu().numpy() for _, val in self.state_dict().items()]


# Define Flower client
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid, net, trainloader, valloader):
        self.cid = cid
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader

    def get_parameters(self, config=None):
        print(f"[Client {self.cid}] get_parameters")
        return self.net.get_parameters_n()

    def fit(self, parameters, config):
        # Read values from config
        server_round = config["server_round"]
        #local_epochs = config["local_epochs"]

        # Use values provided by the config
        print(f"[Client {self.cid}] fit, round {server_round}, config: {config}")
        self.net.set_parameters_n(parameters)
        self.net.train_n(self.trainloader, 1, DEVICE)
        return self.net.get_parameters_n(), len(self.trainloader), {}

    def evaluate(self, parameters, config):
        self.net.set_parameters_n(parameters)
        loss, accuracy = self.net.test_n(self.valloader, DEVICE)
        print(f"[Client {self.cid}] evaluate, config: {config}, loss: {loss}, accuracy: {accuracy}")
        return float(loss), len(self.valloader), {"accuracy": float(accuracy)}


def load_datasets(NUM_CLIENTS=10, BATCH_SIZE=32):
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
    trainloaders = []
    valloaders = []
    for partition_id in range(NUM_CLIENTS):
        partition = fds.load_partition(partition_id, "train")
        partition = partition.with_transform(apply_transforms)
        partition = partition.train_test_split(train_size=0.8)
        trainloaders.append(DataLoader(partition["train"], batch_size=BATCH_SIZE))
        valloaders.append(DataLoader(partition["test"], batch_size=BATCH_SIZE))
    testset = fds.load_split("test").with_transform(apply_transforms)
    testloader = DataLoader(testset, batch_size=BATCH_SIZE)
    return trainloaders, valloaders, testloader

trainloaders, valloaders, testloader = load_datasets()
net = Net()

def client_fn(cid, DEVICE) -> FlowerClient:
    net = Net().to(DEVICE)
    trainloader = trainloaders[int(cid)]
    valloader = valloaders[int(cid)]
    return FlowerClient(cid, net, trainloader, valloader)

# Start Flower client
fl.client.start_client(
    server_address="localhost:" + str(sys.argv[1]),
    client=client_fn(sys.argv[2], DEVICE).to_client(),
    grpc_max_message_length=1024 * 1024 * 1024
)
