import base64
# import gzip
import json
import os
import sys
import tarfile
import threading
import time
from collections import OrderedDict

import microserviceCommunication_pb2 as msCommTypes
import numpy as np
# import pandas as pd
import rabbitMQ_pb2 as rabbitTypes
import torch
import torch.nn as nn
import torch.nn.functional as F
from dynamos.logger import InitLogger
from dynamos.ms_init import NewConfiguration
from dynamos.signal_flow import signal_continuation, signal_wait
from google.protobuf.empty_pb2 import Empty
from google.protobuf.json_format import MessageToDict
from google.protobuf.struct_pb2 import Struct
from hfl_data import SVHNDataset, get_svhn_transforms
from opentelemetry.context.context import Context

np.set_printoptions(threshold=sys.maxsize)

# --- DYNAMOS Interface code At the TOP ---------------------------
if os.getenv("ENV") == "PROD":
    import config_prod as config
else:
    import config_local as config

logger = InitLogger()
# tracer = InitTracer(config.service_name, config.tracing_host)

# Events to start the shutdown of this Microservice, can be used to call 'signal_shutdown'
stop_event = threading.Event()
stop_microservice_condition = threading.Condition()

# Events to make sure all services have started before starting to process a message
# Might be overkill, but good practice
wait_for_setup_event = threading.Event()
wait_for_setup_condition = threading.Condition()

ms_config = None

# --- END DYNAMOS Interface code At the TOP ----------------------


def load_data(file_path: str):
    """Load SVHN dataset from tar.gz file."""
    extract_dir = "./svhn_data_extracted"

    if not os.path.exists(extract_dir):
        logger.info(f"Extracting {file_path}...")
        with tarfile.open(file_path, "r:gz") as tar:
            tar.extractall(path=extract_dir)
        logger.info("Extraction complete")

    transform = get_svhn_transforms(train=True)
    dataset = SVHNDataset(extract_dir, transform=transform)

    return dataset


class SVHN_Model(nn.Module):
    def __init__(self):
        super(SVHN_Model, self).__init__()
        self.fc3 = nn.Linear(3072, 512)  # 32x32x3
        self.fc5 = nn.Linear(512, 10)
        self.size = float(6.1)  # Mb

    def forward(self, xb):
        out = xb.view(-1, 3072)
        out = self.fc3(out)
        out = F.relu(out)
        out = self.fc5(out)
        return F.log_softmax(out, dim=1)

    def get_size(self):
        return self.size


# def serialise_array(array):
#     return json.dumps([str(array.dtype), array.tobytes().decode("latin1"), array.shape])


# def deserialise_array(string, hook=None):
#     encoded_data = json.loads(string, object_pairs_hook=hook)
#     dataType = np.dtype(encoded_data[0])
#     dataArray = np.frombuffer(encoded_data[1].encode("latin1"), dataType)


#     if len(encoded_data) > 2:
#         return dataArray.reshape(encoded_data[2])
#     return dataArray
#
def serialise_array(array):
    """Serialize numpy array more efficiently using base64."""
    return json.dumps(
        [
            str(array.dtype),
            base64.b64encode(array.tobytes()).decode("ascii"),
            array.shape,
        ]
    )


def deserialise_array(string, hook=None):
    """Deserialize numpy array from base64."""
    encoded_data = json.loads(string, object_pairs_hook=hook)
    dataType = np.dtype(encoded_data[0])
    dataArray = np.frombuffer(
        base64.b64decode(encoded_data[1].encode("ascii")), dataType
    )
    if len(encoded_data) > 2:
        return dataArray.reshape(encoded_data[2])
    return dataArray


class HFLClient:
    """
    Horizontal Federated Learning Client:
    - Trains locally on its own data
    - Returns serialized model parameters
    - Loads the global model from the server
    """

    def __init__(
        self,
        file_path: str,
        row_ids: list[int] = [],
        zipf_rank: int = 0,
        row_count: int = 0,
        learning_rate: float = 0.1,
        iid : int = 10,
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
            iid: Numbber of classes selected for training
            batch_size: batch size for training based on Drainakis et al.
            model_state: Optional pre-trained model state dict
        """
        transform = get_svhn_transforms()

        self.data = SVHNDataset(file_path, transform=transform, row_ids=row_ids)
        self.rank = zipf_rank
        self.row_count = row_count
        self.row_ids = row_ids


        # Initialize model
        self.model = SVHN_Model()

        if model_state is not None:
            self.model.load_state_dict(model_state)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

    def train_local(self, epochs: int = 1, batch_size: int = 128):
        """Perform local training on partitioned data."""
        self.model.train()

        loader = torch.utils.data.DataLoader(
            self.data, batch_size=batch_size, shuffle=True
        )

        for epoch in range(epochs):
            total_loss = 0
            for images, labels in loader:
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            logger.debug(
                f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(loader):.4f}"
            )

    def evaluate(self):
        """Evaluate on partitioned data."""
        self.model.eval()
        correct = 0
        total = 0

        loader = torch.utils.data.DataLoader(self.data, batch_size=128, shuffle=False)

        with torch.no_grad():
            for images, labels in loader:
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total if total > 0 else 0
        return accuracy

    def get_model_updates(self):
        """Serialize model parameters and sample count."""
        state_dict = self.model.state_dict()
        params = []
        for k, v in state_dict.items():
            nd = v.detach().cpu().numpy()
            params.append([k, serialise_array(nd)])
        update = {"num_samples": self.row_count, "params": params}
        return json.dumps(update)

    def load_global_model(self, global_params_json):
        """Load global model parameters from server."""
        try:
            np_params = json.loads(global_params_json)
            state_dict = OrderedDict()
            for entry in np_params:
                k = entry.get("key")
                serialized = entry.get("value")
                nd = deserialise_array(serialized)
                state_dict[k] = torch.from_numpy(nd.copy()).float()
            self.model.load_state_dict(state_dict)
            logger.info("Global model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load global model: {e}")


def request_handler(msComm: msCommTypes.MicroserviceCommunication, ctx: Context = None):
    global ms_config
    global hfl_client
    logger.info(f"Received original request type: {msComm.request_type}")

    signal_wait(wait_for_setup_event, wait_for_setup_condition)

    try:
        request = rabbitTypes.Request()
        msComm.original_request.Unpack(request)
    except Exception as e:
        logger.error(f"Unexpected original request: {e}")
        ms_config.next_client.ms_comm.send_data(msComm, msComm.data, {})
        return Empty()

    DATA_STEWARD_NAME = os.getenv("DATA_STEWARD_NAME", "").lower()

    if DATA_STEWARD_NAME == "server":
        # Relay server messages
        if request.type == "hflShutdownRequest":
            logger.info("Received hflShutdownRequest, shutting down.")
            ms_config.next_client.ms_comm.send_data(msComm, msComm.data, {})
            signal_continuation(stop_event, stop_microservice_condition)
        else:
            logger.info("Server relaying request.")
            ms_config.next_client.ms_comm.send_data(msComm, msComm.data, {})
    else:
        # Client side
        if request is not None:
            if request.type == "hflTrainRequest":
                start = time.perf_counter()
                logger.info(f"{start}: Received hflTrainRequest (client training).")

                # Check if client is initialized
                if hfl_client is None:
                    logger.error("Client not initialized.")
                    return Empty()

                batch_size = int(request.data.get("batch_size", 128))
                epochs = int(request.data.get("epochs", 1))

                hfl_client.train_local(epochs=epochs, batch_size=batch_size)
                model_update_json = hfl_client.get_model_updates()
                acc = hfl_client.evaluate()
                logger.info(f"Local model accuracy is {acc}")

                data = Struct()
                data.update({"model_update": model_update_json})
                data.update({"accuracy": acc})
                training_time = (time.perf_counter() - start) * 1000
                logger.info(f"Training time: {training_time:.2f}ms")
                data.update({"t_train": training_time})

                ms_config.next_client.ms_comm.send_data(msComm, data, {})

            elif request.type == "hflLoadGlobalModel":
                start = time.perf_counter()
                logger.info("Received hflLoadGlobalModel (update local model).")
                try:
                    global_params_json = request.data["global_params"].string_value
                    global_params = hfl_client.load_global_model(global_params_json)
                    data = Struct()
                    data.update({"global_params": global_params})
                    loading_time = (time.perf_counter() - start) * 1000
                    logger.info(f"Loading time: {loading_time}ms")
                    data.update({"t_load": loading_time})
                except Exception as e:
                    logger.error(f"Failed to load global model: {e}")
                    data = Struct()
                ms_config.next_client.ms_comm.send_data(msComm, data, {})

            elif request.type == "hflShutdownRequest":
                logger.info("Received hflShutdownRequest, stopping client.")
                signal_continuation(stop_event, stop_microservice_condition)

            elif request.type == "hflPingRequest":
                logger.info("Received hflPingRequest.")

                logger.info(f"request: {request.data}")
                partition_config = request.data.get("partition", {})
                learning_rate = request.data.get("learning_rate", 0.1)
                iid = int(request.data.get("iid", 10))
                partition_dict = MessageToDict(partition_config.struct_value)

                # Access partition fields
                zipf_rank = int(partition_dict.get("zipf_rank", 0))
                row_count = int(partition_dict.get("row_count", 0))
                row_ids = [int(r) for r in partition_dict.get("row_ids", [])]
                logger.info(f"row_ids: {row_ids}")
                logger.info(f"Received partition {zipf_rank} with {row_count} rows")

                if hfl_client is None:
                    try:
                        logger.info(
                            f"Initializing HFL client with partition {zipf_rank}"
                        )
                        # Initialize or reinitialize client with partition
                        hfl_client = HFLClient(
                            file_path=config.dataset_filepath,
                            row_count=row_count,
                            zipf_rank=zipf_rank,
                            row_ids=row_ids,
                            iid=iid,
                            learning_rate=learning_rate,
                        )
                        logger.info(
                            f"Client initialized successfully with {hfl_client.row_count} samples"
                        )
                    except Exception as e:
                        logger.error(f"Error initializing HFL client: {e}")
                        raise
                else:
                    logger.info("HFL client already initialized")

                # Send response back
                ms_config.next_client.ms_comm.send_data(msComm, msComm.data, {})

            else:
                logger.error(f"Unknown HFL request type: {request.type}")

            return Empty()


def main():
    global config
    global ms_config
    global hfl_client
    global initial_data

    hfl_client = None

    ms_config = NewConfiguration(config.service_name, config.grpc_addr, request_handler)
    signal_continuation(wait_for_setup_event, wait_for_setup_condition)

    try:
        signal_wait(stop_event, stop_microservice_condition)
    except KeyboardInterrupt:
        logger.debug("KeyboardInterrupt, stopping client...")
        signal_continuation(stop_event, stop_microservice_condition)

    ms_config.stop(2)
    logger.debug(f"Exiting {config.service_name}")
    sys.exit(0)


# ---  END DYNAMOS Interface code At the Bottom -----------------

if __name__ == "__main__":
    main()
