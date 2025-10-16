import pandas as pd
import numpy as np
import sys
import os
import json
import torch
import torch.nn as nn
from collections import OrderedDict
from sklearn.preprocessing import StandardScaler
from google.protobuf.struct_pb2 import Struct
from dynamos.ms_init import NewConfiguration
from dynamos.signal_flow import signal_continuation, signal_wait
from dynamos.logger import InitLogger
import rabbitMQ_pb2 as rabbitTypes

from google.protobuf.empty_pb2 import Empty
import microserviceCommunication_pb2 as msCommTypes
import threading
from opentelemetry.context.context import Context

np.set_printoptions(threshold=sys.maxsize)

# --- DYNAMOS Interface code At the TOP ---------------------------
if os.getenv('ENV') == 'PROD':
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


def load_data(file_path):
    DATA_STEWARD_NAME = os.getenv("DATA_STEWARD_NAME").lower()
    file_name = f"{file_path}/{DATA_STEWARD_NAME}Data.csv"

    if DATA_STEWARD_NAME == "":
        logger.error("DATA_STEWARD_NAME not set.")
        file_name = f"{file_path}Data.csv"

    try:
        data = pd.read_csv(file_name, delimiter=',')
    except FileNotFoundError:
        logger.error(f"CSV file for table {file_name} not found.")
        return None

    return data


class ClientModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc = nn.Linear(input_size, 4)

    def forward(self, x):
        return self.fc(x)


def serialise_array(array):
    return json.dumps([
        str(array.dtype),
        array.tobytes().decode("latin1"),
        array.shape])


def deserialise_array(string, hook=None):
    encoded_data = json.loads(string, object_pairs_hook=hook)
    dataType = np.dtype(encoded_data[0])
    dataArray = np.frombuffer(encoded_data[1].encode("latin1"), dataType)

    if len(encoded_data) > 2:
        return dataArray.reshape(encoded_data[2])
    return dataArray


class HFLClient:
    """
    Horizontal Federated Learning Client:
    - trains locally on its own data
    - returns serialized model parameters
    - can load the global model from the server
    """

    def __init__(self, data, learning_rate=0.01, model_state=None, optimiser_state=None):
        try:
            scaled = StandardScaler().fit_transform(data)
            self.data = torch.tensor(scaled).float()
        except Exception as e:
            logger.error(f"StandardScaler failed in client init: {e}")
            raise

        if "Survived" in data.columns:
            self.labels = torch.tensor(data["Survived"].values).float().unsqueeze(1)
        else:
            self.labels = None

        self.model = ClientModel(data.shape[1])
        if model_state is not None:
            self.model.load_state_dict(model_state)

        self.optimiser = None
        self.learning_rate = learning_rate
        self.criterion = nn.BCELoss()

    def create_optimiser(self, learning_rate=None):
        if learning_rate is None:
            learning_rate = self.learning_rate
        if self.optimiser is None:
            self.optimiser = torch.optim.SGD(self.model.parameters(), lr=learning_rate)

    def train_local(self, epochs=1, batch_size=32):
        """Perform local training."""
        if self.labels is None:
            logger.error("Client has no labels for training.")
            return

        self.create_optimiser(self.learning_rate)
        dataset = torch.utils.data.TensorDataset(self.data, self.labels)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.model.train()
        for _ in range(epochs):
            for X, y in loader:
                self.optimiser.zero_grad()
                outputs = self.model(X)
                loss = self.criterion(outputs, y)
                loss.backward()
                self.optimiser.step()

    def evaluate(self):
        """Evaluate on local data."""
        if self.labels is None:
            return None
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(self.data)
            preds = (outputs > 0.5).float()
            acc = (preds == self.labels).sum().item() / len(self.labels) * 100
        return acc

    def get_model_update(self):
        """Serialize model parameters and sample count."""
        state_dict = self.model.state_dict()
        params = []
        for k, v in state_dict.items():
            nd = v.detach().cpu().numpy()
            params.append([k, serialise_array(nd)])
        update = {"num_samples": len(self.data), "params": params}
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
                state_dict[k] = torch.from_numpy(nd).float()
            self.model.load_state_dict(state_dict)
            logger.info("Global model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load global model: {e}")


def request_handler(msComm: msCommTypes.MicroserviceCommunication,
                    ctx: Context = None):
    global ms_config
    logger.info(f"Received original request type: {msComm.request_type}")

    signal_wait(wait_for_setup_event, wait_for_setup_condition)

    try:
        request = rabbitTypes.Request()
        msComm.original_request.Unpack(request)
    except Exception as e:
        logger.error(f"Unexpected original request: {e}")
        ms_config.next_client.ms_comm.send_data(msComm, msComm.data, {})
        return Empty()

    DATA_STEWARD_NAME = os.getenv("DATA_STEWARD_NAME").lower()

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
                logger.info("Received hflTrainRequest (client training).")
                try:
                    epochs = int(request.data.get("epochs").number_value) if "epochs" in request.data else 1
                except Exception:
                    epochs = 1

                hfl_client.train_local(epochs=epochs)
                model_update_json = hfl_client.get_model_update()

                data = Struct()
                data.update({"model_update": model_update_json})
                ms_config.next_client.ms_comm.send_data(msComm, data, {})

            elif request.type == "hflLoadGlobalModel":
                logger.info("Received hflLoadGlobalModel (update local model).")
                try:
                    global_params_json = request.data["global_params"].string_value
                    global_params = hfl_client.load_global_model(global_params_json)
                    data = Struct()
                    data.update({"global_params": global_params })
                except Exception as e:
                    logger.error(f"Failed to load global model: {e}")
                    data = Struct()
                ms_config.next_client.ms_comm.send_data(msComm, data, {})

            elif request.type == "hflShutdownRequest":
                logger.info("Received hflShutdownRequest, stopping client.")
                signal_continuation(stop_event, stop_microservice_condition)

            elif request.type == "hflPingRequest":
                logger.info("Received hflPingRequest.")
                ms_config.next_client.ms_comm.send_data(msComm, msComm.data, {})

            else:
                logger.error(f"Unknown HFL request type: {request.type}")

            return Empty()


def main():
    global config
    global ms_config
    global hfl_client

    try:
        data = load_data(config.dataset_filepath)
        hfl_client = HFLClient(data)
    except Exception as e:
        logger.error(f"Error initializing HFL client: {e}")
        raise

    ms_config = NewConfiguration(
        config.service_name, config.grpc_addr, request_handler)

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
