import os
import time
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from sklearn.preprocessing import StandardScaler
import torch

from vertical_fl.task import ClientModel, load_data


checkpoint_dir = "/app/model/clients/"

# note: there might be some issues with permissions in docker when creating the dir
# Create directory if it doesn't exist
os.makedirs(checkpoint_dir, exist_ok=True)


class FlowerClient(NumPyClient):
    def __init__(self, v_split_id, data, lr):
        self.v_split_id = v_split_id
        self.data = torch.tensor(StandardScaler().fit_transform(data)).float()
        self.model = ClientModel(input_size=self.data.shape[1])
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)

    def get_parameters(self, config):
        pass

    def fit(self, parameters, config):
        embedding = self.model(self.data)

        print("Saving checkpoint for server...")
        self.save_checkpoint()

        return [embedding.detach().numpy()], 1, {}

    def evaluate(self, gradients, config):
        self.model.zero_grad()
        embedding = self.model(self.data)
        embedding.backward(torch.from_numpy(gradients[int(self.v_split_id)]))
        self.optimizer.step()
        return 0.0, 1, {}

    def save_checkpoint(self):
        checkpoint_path = os.path.join(
            checkpoint_dir, f'checkpoint_{str(time.time())}.pth')

        # Save checkpoint
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            # Add any other info you want to save (e.g., loss)
        }, checkpoint_path)

        print(f'Checkpoint saved (client) at {checkpoint_path}')


def client_fn(context: Context):
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    partition, v_split_id = load_data(
        partition_id, num_partitions=num_partitions)
    lr = context.run_config["learning-rate"]
    return FlowerClient(v_split_id, partition, lr).to_client()


app = ClientApp(
    client_fn=client_fn,
)
