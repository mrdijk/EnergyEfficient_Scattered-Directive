import pandas as pd
import numpy as np
import sys
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from google.protobuf.struct_pb2 import Struct
from dynamos.ms_init import NewConfiguration
from dynamos.signal_flow import signal_continuation, signal_wait
from dynamos.logger import InitLogger
import rabbitMQ_pb2 as rabbitTypes

from google.protobuf.empty_pb2 import Empty
import microserviceCommunication_pb2 as msCommTypes
import threading
from opentelemetry.context.context import Context
from collections import OrderedDict

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
hfl_server = None

# --- END DYNAMOS Interface code At the TOP ----------------------

# ---- LOCAL TEST SETUP OPTIONAL!

# Go into local test code with flag '-t'
# parser = argparse.ArgumentParser()
# parser.add_argument("-t", "--test", action='store_true')
# args = parser.parse_args()
# test = args.test

# --------------------------------


def load_data(file_path):
    DATA_STEWARD_NAME = os.getenv("DATA_STEWARD_NAME").lower()
    file_name = f"{file_path}/titanic_training.csv"

    if DATA_STEWARD_NAME == "":
        logger.error("DATA_STEWARD_NAME not set.")
        file_name = f"{file_path}Data.csv"

    try:
        data = pd.read_csv(file_name, delimiter=',')
        logger.debug("Loaded server dataset successfully.")
    except FileNotFoundError:
        logger.error(f"CSV file {file_name} not found.")
        return None

    return data


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

class ServerModel(nn.Module):
    def __init__(self, input_size):
        super(ServerModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 12)
        self.fc2 = nn.Linear(12,1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return self.sigmoid(x)


class HFLServer:
    def __init__(self, data):
        self.device = "cpu"

        # if "Survived" not in data.columns:
        #     raise ValueError("Dataset must contain 'Survived' column.")

        self.labels = torch.tensor(data["Survived"].values).float().unsqueeze(1)
        self.data = torch.tensor(
            data.drop("Survived", axis=1).values
        ).float()
        
        self.model = ServerModel(self.data.shape[1])


    def aggregate_fit(self, client_updates):
        """
        Perform FedAvg aggregation of client model updates.
        client_updates: list of dicts with keys {num_samples, params}
        """
        try:
            total_samples = sum(update["num_samples"] for update in client_updates)
            keys = [k for k, _ in client_updates[0]["params"]]
            accum = {k: np.zeros_like(client_updates[0]["params"][i][1], dtype=np.float64)
                     for i, k in enumerate(keys)}

            for update in client_updates:
                weight = update["num_samples"] / total_samples
                for k, nd in update["params"]:
                    accum[k] += nd.astype(np.float64) * weight

            averaged = [(k, accum[k].astype(np.float32)) for k in keys]
            state_dict = OrderedDict()
            for k, nd in averaged:
                state_dict[k] = torch.from_numpy(nd).to(self.device)
            self.model.load_state_dict(state_dict)

        except Exception as e:
            logger.error(f"FedAvg aggregation failed: {e}")
            raise e

        # Evaluate accuracy on server dataset
        with torch.no_grad():
            self.model.eval()
            preds = (self.model(self.data) > 0.5).float()
            acc = (preds == self.labels).sum().item() / len(self.labels) * 100

        data = Struct()
        data.update({"accuracy": acc})
        logger.info(f"Aggregated global model accuracy: {acc:.2f}%")

        # Serialize averaged model parameters for clients
        np_params = []
        for k, v in self.model.state_dict().items():
            np_params.append({
                "key": k,
                "value": serialise_array(v.detach().cpu().numpy())
            })
        data.update({"global_params": json.dumps(np_params)})
        return data


def handleAggregateRequest(msComm):
    global ms_config
    global hfl_server

    request = rabbitTypes.Request()
    msComm.original_request.Unpack(request)

    try:
        data = request.data["model_updates"]
        client_updates = []
        for update_struct in data.list_value.values:
            update_obj = json.loads(update_struct.string_value)
            upd = {
                "num_samples": update_obj["num_samples"],
                "params": [(k, deserialise_array(v)) for k, v in update_obj["params"]]
            }
            client_updates.append(upd)
    except Exception as e:
        logger.error(f"Error deserializing client model updates: {e}")
        return

    logger.info("Performing FedAvg aggregation from client updates.")
    agg_result = hfl_server.aggregate_fit(client_updates)

    ms_config.next_client.ms_comm.send_data(msComm, agg_result, {})


def request_handler(msComm: msCommTypes.MicroserviceCommunication,
                    ctx: Context = None):
    global ms_config

    logger.info(f"Received original request type: {msComm.request_type}")
    signal_wait(wait_for_setup_event, wait_for_setup_condition)

    try:
        request = rabbitTypes.Request()
        msComm.original_request.Unpack(request)
    except Exception as e:
        logger.error(f"Unexpected original request received: {e}")
        ms_config.next_client.ms_comm.send_data(msComm, msComm.data, {})
        return Empty()

    DATA_STEWARD_NAME = os.getenv("DATA_STEWARD_NAME").lower()

    if DATA_STEWARD_NAME != "server":
        if request.type == "hflShutdownRequest":
            logger.info("Received hflShutdownRequest, shutting down service.")
            ms_config.next_client.ms_comm.send_data(msComm, msComm.data, {})
            signal_continuation(stop_event, stop_microservice_condition)
        else:
            logger.info("This is the server microservice, forwarding request.")
            ms_config.next_client.ms_comm.send_data(msComm, msComm.data, {})
    else:
        if request.type == "hflAggregateRequest":
            logger.info("Received hflAggregateRequest.")
            handleAggregateRequest(msComm)

        elif request.type == "hflPingRequest":
            logger.info("Received hflPingRequest.")
            ms_config.next_client.ms_comm.send_data(msComm, msComm.data, {})

        elif request.type == "hflShutdownRequest":
            logger.info("Received hflShutdownRequest.")
            signal_continuation(stop_event, stop_microservice_condition)

        return Empty()


def main():
    global config
    global ms_config
    global hfl_server

    data = load_data(config.dataset_filepath)
    hfl_server = HFLServer(data)

    ms_config = NewConfiguration(
        config.service_name, config.grpc_addr, request_handler)

    # Signal the message handler that all connections have been created
    signal_continuation(wait_for_setup_event, wait_for_setup_condition)

    # Wait for the end of processing to shutdown this Microservice
    try:
        signal_wait(stop_event, stop_microservice_condition)
    except KeyboardInterrupt:
        logger.debug("KeyboardInterrupt received, stopping server...")
        signal_continuation(stop_event, stop_microservice_condition)

    ms_config.stop(2)
    logger.debug(f"Exiting {config.service_name}")
    sys.exit(0)

# ---  END DYNAMOS Interface code At the Bottom -----------------

if __name__ == "__main__":
    main()
