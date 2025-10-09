import pandas as pd
import numpy as np
import json
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict
from sklearn.preprocessing import StandardScaler
import os

# ---------------------- SET SEED FOR REPRODUCIBILITY ----------------------
SEED = 0
np.random.seed(SEED)
torch.manual_seed(SEED)
# ---------------------------------------------------------------------------------------------

SERVER_PATH = "/home/maurits/EnergyEfficient_Scattered-Directive/python/hfl-train-model/datasets/titanic_training.csv"
TRAINING_PATH = '/home/maurits/EnergyEfficient_Scattered-Directive/python/hfl-train/datasets/titanic_training.csv'

# ----------------- Parse command line arguments -----------------
parser = argparse.ArgumentParser(description="Run HFL with dynamic number of clients")
parser.add_argument(
    "--clients", 
    type=int, 
    default=3,  # default value if not specified
    help="Number of clients to use in this run"
)
parser.add_argument(
    "--rounds",
    type=int,
    default=10,
    help="Number of rounds in this run"
    )
args = parser.parse_args()

NOF_CLIENTS = args.clients
print(f"Using {NOF_CLIENTS} clients for this run.")
TOTAL_ROUNDS = args.rounds
SERVER_CHECKPOINT_PATH = "server_state_hfl.pth"


# ---------------- MODEL ----------------
class Model(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc = nn.Linear(input_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.fc(x))

# ---------------- SERIALIZATION ----------------
def parameters_to_ndarrays(state_dict):
    return [(k, v.detach().cpu().numpy()) for k, v in state_dict.items()]

def ndarrays_to_state_dict(nd_list, device):
    sd = OrderedDict()
    for k, nd in nd_list:
        sd[k] = torch.from_numpy(nd).to(device)
    return sd

# ---------------- CLIENT ----------------
class HFLClient:
    def __init__(self, data, labels, device="cpu"):
        self.device = device
        X = StandardScaler().fit_transform(data)
        self.data = torch.tensor(X).float().to(device)
        self.labels = torch.tensor(labels.values).float().unsqueeze(1).to(device)
        self.model = Model(self.data.shape[1]).to(device)
        self.criterion = nn.BCELoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)

    def train_local(self, epochs=1, batch_size=32):
        dataset = torch.utils.data.TensorDataset(self.data, self.labels)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.model.train()
        for _ in range(epochs):
            for X, y in loader:
                self.optimizer.zero_grad()
                out = self.model(X)
                loss = self.criterion(out, y)
                loss.backward()
                self.optimizer.step()

    def evaluate(self):
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(self.data)
            preds = (outputs > 0.5).float()
            acc = (preds == self.labels).sum().item() / len(self.labels) * 100
        return acc

    def get_update(self):
        return {
            "num_samples": len(self.data),
            "params": parameters_to_ndarrays(self.model.state_dict())
        }

    def load_global(self, global_params):
        sd = ndarrays_to_state_dict(global_params, self.device)
        self.model.load_state_dict(sd)

# ---------------- SERVER ----------------
class HFLServer:
    def __init__(self, model_class, input_size, device="cpu"):
        self.device = device
        self.model = model_class(input_size).to(device)

    def aggregate_fit(self, client_updates):
        total_samples = sum(upd["num_samples"] for upd in client_updates)
        keys = [k for k, _ in client_updates[0]["params"]]
        accum = {k: np.zeros_like(client_updates[0]["params"][i][1], dtype=np.float64)
                 for i, k in enumerate(keys)}

        for upd in client_updates:
            weight = upd["num_samples"] / total_samples
            for k, nd in upd["params"]:
                accum[k] += nd.astype(np.float64) * weight

        averaged = [(k, accum[k].astype(np.float32)) for k in keys]
        self.model.load_state_dict(ndarrays_to_state_dict(averaged, self.device))
        return averaged

    def evaluate(self, data, labels):
        self.model.eval()
        X = torch.tensor(StandardScaler().fit_transform(data)).float().to(self.device)
        y = torch.tensor(labels.values).float().unsqueeze(1).to(self.device)
        with torch.no_grad():
            out = self.model(X)
            preds = (out > 0.5).float()
            acc = (preds == y).sum().item() / len(y) * 100
        return acc

    def save_state(self, filepath):
        torch.save({'model_state_dict': self.model.state_dict()}, filepath)
        print(f"HFL Server state saved to {filepath}")

    def load_state(self, filepath):
        state = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(state['model_state_dict'])
        print(f"HFL Server state loaded from {filepath}")

# ---------------- DATA LOADING ----------------
server_data = pd.read_csv(SERVER_PATH, delimiter=',', index_col=0) 

# Split the data in across the client
# TODO also split data with server
client_datasets = []
offset = 1
for i in range(NOF_CLIENTS):
    # Number of rows per client
    num_rows = round(889/NOF_CLIENTS)
    # Read num_rows, starting from the offset
    client_datasets.append(pd.read_csv(TRAINING_PATH, delimiter=',', index_col=0, nrows=num_rows, skiprows=range(1,(1 + offset))))
    # Increase the offset with the number of rows per client 
    offset += num_rows

# ---------------- CREATE CLIENTS ----------------
clients = []
for df in client_datasets[:NOF_CLIENTS]:
    if "Survived" not in df.columns:
        raise ValueError("Dataset missing 'Survived' column after encoding")

    X = df.drop("Survived", axis=1)
    y = df["Survived"]
    clients.append(HFLClient(X, y))

# ---------------- CREATE SERVER ----------------
if "Survived" not in server_data.columns:
    raise ValueError("Server dataset missing 'Survived' column after encoding")

#TODO Perhaps use this as validaton
X_server = server_data.drop("Survived", axis=1)
y_server = server_data["Survived"]

server = HFLServer(Model, input_size=X_server.shape[1])

# ---------------- TRAINING LOOP ----------------
train_results = []

for rnd in range(TOTAL_ROUNDS):
    print(f"\n--- Round {rnd+1} ---")
    client_updates = []
    for client in clients:
        client.train_local(epochs=1)
        updates = client.get_update()
        # print(f"client updates: {updates}")
        client_updates.append(updates)

    global_params = server.aggregate_fit(client_updates)
    # print(f"Global_params: {global_params}")

    # broadcast global model back
    for client in clients:
        client.load_global(global_params)

    # evaluation
    client_accs = [client.evaluate() for client in clients]
    # print(f"Client accs: {client_accs}")
    server_acc = server.evaluate(X_server, y_server)

    print(f"Server Acc: {server_acc:.2f}, Client Accs: {[round(a,2) for a in client_accs]}")

    train_results.append({
        "round": rnd + 1,
        "server_accuracy": server_acc,
        "client_accuracies": client_accs,
        "num_clients": NOF_CLIENTS
    })

# ---------------- SAVE RESULTS ----------------
payload = {
    "metadata": {
        "total_rounds": TOTAL_ROUNDS,
        "num_clients": NOF_CLIENTS
    },
    "results": train_results
}

os.makedirs("./run_dumps", exist_ok=True)
filename = f"./run_dumps/hfl_test_results_{NOF_CLIENTS}_clients_{TOTAL_ROUNDS}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json"
print(f"Saving to file {filename}")
with open(filename, "w") as f:
    json.dump(payload, f, indent=2)

# Save final server state
server.save_state(SERVER_CHECKPOINT_PATH)