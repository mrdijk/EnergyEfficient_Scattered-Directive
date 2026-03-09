import argparse
from collections import Counter, OrderedDict
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from hfl_data.svhn_dataset import SVHNDataset, get_svhn_transforms
from numpy.typing import test
from pandas.core.config_init import data_manager_doc
from torch.utils.data import DataLoader

# ---------------------- SET SEED FOR REPRODUCIBILITY ----------------------
SEED = 0
np.random.seed(SEED)
torch.manual_seed(SEED)
# ---------------------------------------------------------------------------------------------

SERVER_PATH = "/home/maurits/EnergyEfficient_Scattered-Directive/datasets/train.tar.gz"
TRAINING_PATH = (
    "/home/maurits/EnergyEfficient_Scattered-Directive/datasets/train.tar.gz"
)

# ----------------- Parse command line arguments -----------------
parser = argparse.ArgumentParser(description="Run HFL with dynamic number of clients")
# parser.add_argument(
#     "--clients",
#     type=int,
#     default=len(DATA_PROVIDERS.keys()),
#     help="Number of clients to use in this run",
# )
parser.add_argument(
    "--rounds", type=int, default=20, help="Number of rounds in this run"
)
args = parser.parse_args()

# NOF_CLIENTS = args.clients
# print(f"Using {NOF_CLIENTS} clients for this run.")
TOTAL_ROUNDS = args.rounds
SERVER_CHECKPOINT_PATH = "server_state_hfl.pth"


# ---------------- MODEL ----------------
class SVHN_Model(nn.Module):
    def __init__(self):
        super(SVHN_Model, self).__init__()
        self.fc3 = nn.Linear(3072, 512)  # 32x32x3
        self.fc5 = nn.Linear(512, 10)
        self.size = float(6.1)  # Mb todo

    def forward(self, xb):
        out = xb.view(-1, 3072)
        out = self.fc3(out)
        out = F.relu(out)
        out = self.fc5(out)
        return F.log_softmax(out, dim=1)

    def get_size(self):
        return self.size


# ---------------- SERIALIZATION ----------------
def parameters_to_ndarrays(state_dict):
    return [(k, v.detach().cpu().numpy()) for k, v in state_dict.items()]


def ndarrays_to_state_dict(nd_list):
    sd = OrderedDict()
    for k, nd in nd_list:
        sd[k] = torch.from_numpy(nd).float()
    return sd


# ---------------- CLIENT ----------------
class HFLClient:
    def __init__(
        self,
        file_path: str,
        row_ids: list[int] = [],
        zipf_rank: int = 0,
        row_count: int = 0,
        learning_rate: float = 0.1,
        batch_size: int = 128,
        model_state=None,
    ):
        """
        Args:
            file_path: Path to .tar.gz file containing images
            row_ids: List of specific row indices to use for this partition
            zipf_rank: Rank of the partition (1 to N, N being the number of total partitions)
            row_count: Number of rows in partition
            learning_rate: Learning rate for optimizer based on Drainakis et al.
            batch_size: batch size for training based on Drainakis et al.
            model_state: Optional pre-trained model state dict
        """
        transform = get_svhn_transforms()

        self.dataset = SVHNDataset(file_path, transform=transform, row_ids=row_ids)
        self.rank = zipf_rank
        self.row_count = row_count
        self.row_ids = row_ids

        print(f"Client initialized with {len(self.dataset)} samples")

        # Initialize model
        self.model = SVHN_Model()

        if model_state is not None:
            self.model.load_state_dict(model_state)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

    def train_local(self, epochs: int = 25, batch_size: int = 128):
        """Perform local training on partitioned data."""
        self.model.train()

        loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            total_loss = 0
            for images, labels in loader:
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(loader):.4f}")

    def evaluate(self):
        """Evaluate on partitioned data."""
        self.model.eval()
        correct = 0
        total = 0

        loader = torch.utils.data.DataLoader(
            self.dataset, batch_size=128, shuffle=False
        )

        with torch.no_grad():
            for images, labels in loader:
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total if total > 0 else 0
        return accuracy

    def get_update(self):
        return {
            "num_samples": len(self.dataset),
            "params": parameters_to_ndarrays(self.model.state_dict()),
        }

    def load_global(self, global_params):
        sd = ndarrays_to_state_dict(global_params)
        self.model.load_state_dict(sd)

    def get_data_size(self):
        return len(self.dataset)


# ---------------- SERVER ----------------
class HFLServer:
    def __init__(
        self,
        # dataset: SVHNDataset,
        file_path: str,
        row_ids: list[int] = [],
        zipf_rank: int = 0,
        row_count: int = 0,
        learning_rate: float = 0.1,
        batch_size: int = 128,
        model_state=None,
    ):
        """
        Args:
            file_path: Path to .tar.gz file containing images
            row_ids: List of specific row indices to use for this partition
            learning_rate: Learning rate for optimizer
            model_state: Optional pre-trained model state dict
        """
        transform = get_svhn_transforms()

        self.dataset = SVHNDataset(file_path, transform=transform, row_ids=row_ids)
        self.rank = zipf_rank
        self.row_count = row_count
        self.row_ids = row_ids

        # Initialize model
        self.model = SVHN_Model()

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

    def get_model_size(self):
        param_size = 0
        for param in self.model.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in self.model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        # size_all_mb = (param_size + buffer_size) / 1024**2
        size_all_kb = (param_size + buffer_size) / 1024
        print("model size: {:.3f}KB".format(size_all_kb))

    def aggregate_fit(self, client_updates):
        total_samples = sum(upd["num_samples"] for upd in client_updates)
        keys = [k for k, _ in client_updates[0]["params"]]
        accum = {
            k: np.zeros_like(client_updates[0]["params"][i][1], dtype=np.float64)
            for i, k in enumerate(keys)
        }

        for upd in client_updates:
            weight = upd["num_samples"] / total_samples
            for k, nd in upd["params"]:
                accum[k] += nd.astype(np.float64) * weight

        averaged = [(k, accum[k].astype(np.float32)) for k in keys]
        state_dict = OrderedDict()
        for k, nd in averaged:
            state_dict[k] = torch.from_numpy(nd)
        self.model.load_state_dict(state_dict)
        # self.model.load_state_dict(ndarrays_to_state_dict(averaged))
        return averaged

    def evaluate(self):
        """Evaluate on partitioned data."""
        self.model.eval()
        correct = 0
        total = 0

        loader = torch.utils.data.DataLoader(
            self.dataset, batch_size=128, shuffle=False
        )

        with torch.no_grad():
            for images, labels in loader:
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total if total > 0 else 0
        return accuracy

    def save_state(self, filepath):
        torch.save({"model_state_dict": self.model.state_dict()}, filepath)
        print(f"HFL Server state saved to {filepath}")

    def load_state(self, filepath):
        state = torch.load(filepath)
        self.model.load_state_dict(state["model_state_dict"])
        print(f"HFL Server state loaded from {filepath}")


# ---------------- DATA LOADING ----------------
# Split the data in across the client
# Load the full dataset once (no row_ids)
# def load_data(file_path: str) -> SVHNDataset:
#     """Load SVHN dataset from tar.gz file."""
#     extract_dir = "./svhn_data_extracted"

#     if not os.path.exists(extract_dir):
#         print(f"Extracting {file_path}...")
#         tar = tarfile.open(file_path, "r:gz")
#         for member in tar:

#         with tarfile.open(file_path, "r:gz") as tar:
#             tar.extractall(path=extract_dir)
#         print("Extraction complete")

#     image_dir = os.path.join(extract_dir, "train")

#     print(f"Looking for images in: {image_dir}")

#     transform = get_svhn_transforms()
#     dataset = SVHNDataset(image_dir, transform=transform)

#     print(f"Dataset created with {len(dataset)} images")
#     return dataset


# full_dataset = load_data(
#     "/home/maurits/EnergyEfficient_Scattered-Directive/datasets/train.tar.gz"
# )

train_dataset = "/home/maurits/EnergyEfficient_Scattered-Directive/python/hfl-train/datasets/extra_32x32.mat"
test_dataset = "/home/maurits/EnergyEfficient_Scattered-Directive/python/hfl-train-model/datasets/test_32x32.mat"
client_datasets: list[SVHNDataset] = []
row_ids = [
    list(range(0, 5000)),
    list(range(5000, 10000)),
    list(range(10000, 15000)),
    list(range(15000, 20000)),
    list(range(20000, 25000)),
]

# ---------------- CREATE CLIENTS ----------------

# Create clients with subsets
clients: List[HFLClient] = []
for i in range(5):
    clients.append(
        HFLClient(
            # dataset=client_datasets[i],  # Pass the Subset
            file_path=train_dataset,
            row_ids=row_ids[i],
            zipf_rank=i + 1,
            row_count=len(row_ids[i]),
            learning_rate=0.01,
        )
    )
# Print how many samples each client has per class
for i, client in enumerate(clients):
    labels = client.dataset.labels  # Already a list of ints
    print(f"Client {i}: {Counter(labels)}")
    print(f"Client {i} total samples: {len(labels)}")
    # Should see all 10 digits represented reasonably

# ---------------- CREATE SERVER ----------------
# Create server with full dataset
server = HFLServer(
    file_path=test_dataset, row_ids=list(range(26032)), zipf_rank=0, row_count=26032
)
print(f"Server: {Counter(server.dataset.labels)}")
print(f"Server total samples: {len(server.dataset.labels)}")
# ---------------- TRAINING LOOP ----------------
train_results = []

for rnd in range(TOTAL_ROUNDS):
    print(f"\n--- Round {rnd + 1} ---")
    client_updates = []
    for client in clients:
        client.train_local(epochs=1)
        updates = client.get_update()
        # print(f"client updates: {updates}")
        client_updates.append(updates)

    global_params = server.aggregate_fit(client_updates)
    # print(f"Global_params: {global_params}")
    server.get_model_size()
    # broadcast global model back
    for client in clients:
        client.load_global(global_params)

    # evaluation
    client_accs = [client.evaluate() for client in clients]
    # print(f"Client accs: {client_accs}")
    server_acc = server.evaluate()
    client_sizes = [client.get_data_size() for client in clients]
    # , Client Accs: {[round(a,2) for a in client_accs]}
    print(f"Server Acc: {server_acc:.2f}")
    print(f"Client Accs: {[round(a, 2) for a in client_accs]}")
    train_results.append(
        {
            "round": rnd + 1,
            "server_accuracy": server_acc,
            "client_accuracies": client_accs,
            # "num_clients": NOF_CLIENTS
        }
    )
