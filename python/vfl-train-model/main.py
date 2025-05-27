import pandas as pd
import numpy as np
import sys
import os
import io
import json
import torch
import torch.nn as nn
import torch.optim as optim
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
server_configuration = {}
server_data = None
clients_embeddings = []

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

    file_name = f"{file_path}/outcomeData.csv"

    if DATA_STEWARD_NAME == "":
        logger.error("DATA_STEWARD_NAME not set.")
        file_name = f"{file_path}Data.csv"

    try:
        data = pd.read_csv(file_name, delimiter=',')
        logger.debug("after read csv")
    except FileNotFoundError:
        logger.error(f"CSV file for table {file_name} not found.")
        return None

    return data


class ServerModel(nn.Module):
    def __init__(self, input_size):
        super(ServerModel, self).__init__()
        self.fc = nn.Linear(input_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc(x)
        return self.sigmoid(x)


def server_fn(context: Context):
    """Construct components that set the ServerApp behaviour."""
    # get data
    # define strategy
    # get number of server rounds


class ClientModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc = nn.Linear(input_size, 4)

    def forward(self, x):
        return self.fc(x)


def serialise_dictionary(dictionary):
    buffer = io.BytesIO()
    torch.save(dictionary, buffer)

    return buffer.getvalue().decode("latin1")


def deserialise_dictionary(dictionary):
    data = json.loads(dictionary, object_pairs_hook=OrderedDict)

    return torch.load(io.BytesIO(data.encode("latin1")))


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


def aggregate_fit(results):
    global server_configuration

    # Convert results
    embedding_results = [
        torch.from_numpy(embedding)
        for embedding in results
    ]
    embeddings_aggregated = torch.cat(embedding_results, dim=1)
    embedding_server = embeddings_aggregated.detach().requires_grad_()
    output = server_configuration["model"](embedding_server)
    loss = server_configuration["criterion"](
        output, server_configuration["labels"])
    loss.backward()

    server_configuration["optimizer"].step()
    server_configuration["optimizer"].zero_grad()

    grads = embedding_server.grad.split([4, 4, 4], dim=1)
    np_gradients = [grad.numpy() for grad in grads]

    with torch.no_grad():
        correct = 0
        output = server_configuration["model"](embedding_server)
        predicted = (output > 0.5).float()

        correct += (predicted == server_configuration["labels"]).sum().item()

        accuracy = correct / len(server_configuration["labels"]) * 100

    metrics_aggregated = {"accuracy": accuracy}

    return serialise_array(np_gradients), metrics_aggregated


def handleVflTrainModelRequest(msComm):
    global ms_config
    global server_configuration

    request = rabbitTypes.Request()
    msComm.original_request.Unpack(request)

    try:
        learning_rate = request.data["learning_rate"].number_value
    except Exception:
        learning_rate = 0.01

    server_configuration["learning_rate"] = learning_rate

    try:
        cycles = request.data["cycles"].number_value
    except Exception:
        cycles = 10

    server_configuration["cycles"] = cycles

    data = Struct()
    data.update({
        "learning_rate": server_configuration["learning_rate"]
    })

    msComm.request_type = "vflTrainRequest"
    # request.data = data
    # msComm.original_request.Pack(request)

    logger.debug("Handling VFL Model Training request")
    # TODO: Send msComm message to run a training round
    ms_config.next_client.ms_comm.send_data(msComm, data, {})

    # We do not shut off the microservice, as we want this to
    # be a persistent microservice (or "ephemeral but long-lived")
    # signal_continuation(stop_event, stop_microservice_condition)


def handleVflClientTrainingCompleteRequest(msComm):
    global ms_config
    global server_configuration
    global clients_embeddings
    global clients_model_state

    request = rabbitTypes.Request()
    msComm.original_request.Unpack(request)

    try:
        data = request.data["embeddings"].string_value
        clients_embeddings += [deserialise_array(data)]
    except Exception as e:
        logger.error(f"Errored when deserialising client data: {e}")

    try:
        data = request.data["model_state"].string_value
        clients_model_state += [data]
    except Exception as e:
        logger.error(
            f"Errored when deserialising client model state: {e}")

    # Hardcoded the number of clients for now
    if len(clients_embeddings) == 3:
        gradients, accuracy = aggregate_fit(clients_embeddings)

        server_configuration["cycles"] -= 1

        if server_configuration["cycles"] == 0:
            ms_config.next_client.ms_comm.send_data(msComm, None, {})

        # TODO: Send msComm message to run a new training round
        # With gradients this time (also give back model state)


# ---  DYNAMOS Interface code At the Bottom --------

def request_handler(msComm: msCommTypes.MicroserviceCommunication,
                    ctx: Context = None):
    global ms_config
    global server_configuration
    global clients_embeddings
    global clients_model_state
    logger.info(f"Received original request type: {msComm.request_type}")
    logger.debug(msComm)

    # Ensure all connections have finished setting up before processing data
    signal_wait(wait_for_setup_event, wait_for_setup_condition)

    try:
        # This is the entry-point from the user request
        if msComm.request_type == "vflTrainModelRequest":
            logger.info("Received a vflTrainModelRequest.")
            handleVflTrainModelRequest(msComm)

        # Receiving data from clients -> Run aggregation if all are received
        elif msComm.request_type == "vflClientTrainingCompleteRequest":
            logger.info("Received a vflClientTrainingCompleteRequest.")
            handleVflClientTrainingCompleteRequest(msComm)

        return Empty()
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        return Empty()


def main():
    global config
    global ms_config
    global server_configuration
    global server_data

    DATA_STEWARD_NAME = os.getenv("DATA_STEWARD_NAME").lower()

    if DATA_STEWARD_NAME != "server":
        logger.info("This is not the server, shutting down.")
        sys.exit(0)

    server_data = load_data(config.dataset_filepath)

    server_configuration["model"] = ServerModel(12)
    # server_configuration["initial_parameters"] = ndarrays_to_parameters(
    #     [val.cpu().numpy()
    #      for _, val in server_configuration.model.state_dict().items()]
    # )
    server_configuration["optimizer"] = optim.SGD(
        server_configuration["model"].parameters(), lr=0.01)
    server_configuration["criterion"] = nn.BCELoss()
    server_configuration["labels"] = torch.tensor(
        server_data["Survived"].values).float().unsqueeze(1)

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
