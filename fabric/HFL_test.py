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
NOF_SURVIVORS = 340


# ---------------- MODEL ----------------
class Model(nn.Module):
    def __init__(self, input_size):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x) 

# ---------------- SERIALIZATION ----------------
def parameters_to_ndarrays(state_dict):
    return [(k, v.detach().cpu().numpy()) for k, v in state_dict.items()]

def ndarrays_to_state_dict(nd_list):
    sd = OrderedDict()
    for k, nd in nd_list:
        sd[k] = torch.from_numpy(nd)
    return sd

# ---------------- CLIENT ----------------
class HFLClient:
    def __init__(self, data, learning_rate=0.01, model_state=None):
        self.data = torch.tensor(data.drop("Survived", axis=1).values).float() 
        self.labels = torch.tensor(data["Survived"].values).float().unsqueeze(1)

        self.model = Model(self.data.shape[1])
        if model_state is not None:
            self.model.load_state_dict(model_state)

        # Loss with class balancing
        pos_weight = torch.tensor([len(self.labels) / sum(self.labels) - 1])  # roughly inverse class ratio
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

    def train_local(self, epochs=1, batch_size=32):
        """Perform local training."""
        if self.labels is None:
            print("Client has no labels for training.")
            return
        
        self.model.train()
        dataset = torch.utils.data.TensorDataset(self.data, self.labels)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for _ in range(epochs):
            for X_batch, y_batch in loader:
                self.optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
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
        sd = ndarrays_to_state_dict(global_params)
        self.model.load_state_dict(sd)

# ---------------- SERVER ----------------
class HFLServer:
    def __init__(self, data):
        self.data = torch.tensor(data.drop("Survived", axis=1).values).float() 
        self.labels = torch.tensor(data["Survived"].values).float().unsqueeze(1)
        self.model = Model(self.data.shape[1])
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

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
        self.model.load_state_dict(ndarrays_to_state_dict(averaged))
        return averaged

    def evaluate(self):
        self.model.eval()
        with torch.no_grad():
            logits = self.model(self.data)
            loss = self.criterion(logits, self.labels)
            preds = torch.sigmoid(logits)
            predicted = (preds > 0.5)
            acc = (predicted == self.labels).sum().item() / len(self.labels)
            print(f"Predicted survivors: {(predicted.sum().item())} compared to {NOF_SURVIVORS} actual survivors")
        return acc, loss

    def save_state(self, filepath):
        torch.save({'model_state_dict': self.model.state_dict()}, filepath)
        print(f"HFL Server state saved to {filepath}")

    def load_state(self, filepath):
        state = torch.load(filepath)
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
    
    clients.append(HFLClient(df))

# ---------------- CREATE SERVER ----------------
if "Survived" not in server_data.columns:
    raise ValueError("Server dataset missing 'Survived' column after encoding")

server = HFLServer(server_data)

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
    server_acc, server_loss = server.evaluate()

    print(f"Server Acc: {server_acc:.2f}, Server Loss: {server_loss:.2f}, Client Accs: {[round(a,2) for a in client_accs]}")

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

# os.makedirs("./run_dumps", exist_ok=True)
# filename = f"./run_dumps/hfl_test_results_{NOF_CLIENTS}_clients_{TOTAL_ROUNDS}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json"
# print(f"Saving to file {filename}")
# with open(filename, "w") as f:
#     json.dump(payload, f, indent=2)

# Save final server state
# server.save_state(SERVER_CHECKPOINT_PATH)