import flwr as fl
import sys
import numpy as np


#https://github.com/PratikGarai/MNIST-Federated/tree/master/03_Non%20IID%20Demo

def fit_config(server_round: int):
    """Return training configuration dict for each round.

    Perform two rounds of training with one local epoch, increase to two local
    epochs afterwards.
    """
    config = {
        "server_round": server_round,  # The current round of federated learning
    }
    return config


class SaveModelStrategy(fl.server.strategy.FedAvg):
    def __init__(self):
        super().__init__(on_fit_config_fn=fit_config)

    def aggregate_fit(
            self,
            rnd,
            results,
            failures
    ):
        aggregated_weights = super().aggregate_fit(rnd, results, failures)
        if aggregated_weights is not None:
            # Save aggregated_weights
            print(f"Saving round {rnd} aggregated_weights...")
            np.savez(f"round-{rnd}-weights.npz", *aggregated_weights)
        return aggregated_weights


# Create strategy and run server
strategy = SaveModelStrategy()

# Start Flower server for three rounds of federated learning
fl.server.start_server(
    server_address='localhost:' + str(sys.argv[1]),
    config=fl.server.ServerConfig(num_rounds=3),
    grpc_max_message_length=1024 * 1024 * 1024,
    strategy=strategy
)
