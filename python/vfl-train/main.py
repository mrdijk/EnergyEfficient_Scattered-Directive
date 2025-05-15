import pandas as pd
import numpy as np
import sys
import os
import io
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

# ---- LOCAL TEST SETUP OPTIONAL!

# Go into local test code with flag '-t'
# parser = argparse.ArgumentParser()
# parser.add_argument("-t", "--test", action='store_true')
# args = parser.parse_args()
# test = args.test

# --------------------------------


def load_data(file_path):
    DATA_STEWARD_NAME = os.getenv("DATA_STEWARD_NAME").lower()

    file_name = f"{file_path}/{DATA_STEWARD_NAME}Data.csv"

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


def train_model(data, model):
    embedding = model(data)
    return embedding.detach().numpy()


def vfl_evaluate(data, model, optimiser, gradients):
    logger.debug("Start vfl_evaluate")

    try:
        model.zero_grad()
        embedding = model(data)
        embedding.backward(torch.from_numpy(gradients))
        optimiser.step()
    except Exception as e:
        logger.error(f"Error occurred: {e}")


# Note: Gradients sent by server are for this client only to preserve privacy
def vfl_train(learning_rate, model_state, gradients):
    global config

    try:
        data = load_data(config.dataset_filepath)
    except Exception as e:
        logger.error(f"Error occurred: {e}")

        # If data does not exist, shut down service
        logger.error("Shutting down the service")
        signal_continuation(stop_event, stop_microservice_condition)
        return None, None

    data = torch.tensor(StandardScaler().fit_transform(data)).float()
    model = ClientModel(data.shape[1])

    if model_state is not None:
        print(model_state)
        model.load_state_dict(model_state)

    optimiser = torch.optim.SGD(model.parameters(), lr=learning_rate)

    if gradients is not None:
        vfl_evaluate(data, model, optimiser, gradients)

    embeddings = train_model(data, model)
    model_state = model.state_dict()

    buffer = io.BytesIO()
    torch.save(model_state, buffer)

    data = Struct()
    data.update({"embeddings": serialise_array(embeddings),
                 "model_state": buffer.getvalue().decode("latin1")})

    return data


# ---  DYNAMOS Interface code At the Bottom --------

def request_handler(msComm: msCommTypes.MicroserviceCommunication, ctx: Context):
    global ms_config
    logger.info(f"Received original request type: {msComm.request_type}")
    logger.debug(msComm)

    # Ensure all connections have finished setting up before processing data
    signal_wait(wait_for_setup_event, wait_for_setup_condition)

    try:
        if msComm.request_type == "vflTrainRequest":
            request = rabbitTypes.Request()
            msComm.original_request.Unpack(request)

            try:
                learning_rate = request.data["learning_rate"].number_value
            except Exception:
                learning_rate = 0.05

            try:
                gradients = request.data["gradients"].string_value
                gradients = deserialise_array(gradients)
            except Exception as e:
                print(e, request.data["gradients"])
                gradients = None

            try:
                model_state = request.data["model_state"].string_value
                model_state = deserialise_dictionary(model_state)
            except Exception as e:
                print(e, request.data["model_state"])
                model_state = None

            logger.debug("Handling VFL Training request")
            data = vfl_train(
                learning_rate, model_state, gradients)
            logger.debug("Received data from VFL Training:")
            logger.debug(data)

            # Ignore metadata
            ms_config.next_client.ms_comm.send_data(msComm, data, {})
            signal_continuation(stop_event, stop_microservice_condition)

        else:
            logger.error(f"An unknown request_type: {msComm.request_type}")
            # If not recognised, this service should ceize to exist.
            signal_continuation(stop_event, stop_microservice_condition)

        return Empty()
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        return Empty()


def main():
    global config
    global ms_config

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
