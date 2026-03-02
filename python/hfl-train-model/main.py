import base64
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

# ---- LOCAL TEST SETUP OPTIONAL!

# Go into local test code with flag '-t'
# parser = argparse.ArgumentParser()
# parser.add_argument("-t", "--test", action='store_true')
# args = parser.parse_args()
# test = args.test

# --------------------------------


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


# def serialise_array(array):
#     return json.dumps([str(array.dtype), array.tobytes().decode("latin1"), array.shape])


# def deserialise_array(string, hook=None):
#     encoded_data = json.loads(string, object_pairs_hook=hook)
#     dataType = np.dtype(encoded_data[0])
#     dataArray = np.frombuffer(encoded_data[1].encode("latin1"), dataType)


#     if len(encoded_data) > 2:
#         return dataArray.reshape(encoded_data[2])
#     return dataArray
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


class HFLServer:
    """
    Horizontal Federated Learning Server:
    - Aggregates all client models
    - Averages all model parameters
    - Evaluates aggregates model performance
    - Sends back the averaged model parameters to the clients
    """

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
            zipf_rank: Rank of the partition (1 to N, N being the number of total partitions)
            row_count: Number of  rows in partition
            learning_rate: Learning rate for optimizer based on Drainakis et al.
            batch_size: Batch size for training based on Drainakis et al.
            model_state: Optional pre-trained model state dict
        """
        transform = get_svhn_transforms()

        self.data = SVHNDataset(file_path, transform=transform, row_ids=row_ids)
        self.rank = zipf_rank
        self.row_count = row_count
        self.row_ids = row_ids

        # Initialize model
        self.model = SVHN_Model()

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

    def aggregate_fit(self, client_updates):
        """Perform FedAvg aggregation of client model updates."""
        try:
            logger.info(f"Received {len(client_updates)} client updates")
            logger.debug(
                f"First update keys: {client_updates[0].keys() if len(client_updates) > 0 else 'no updates'}"
            )

            if len(client_updates) == 0:
                logger.error("No client updates to aggregate")
                return

            # Debug: check structure of params
            logger.debug(f"Type of params: {type(client_updates[0]['params'])}")
            logger.debug(
                f"Params content: {client_updates[0]['params'] if isinstance(client_updates[0]['params'], dict) else 'not a dict'}"
            )

            total_samples = sum(update["num_samples"] for update in client_updates)

            # Deserialize all updates
            deserialized_updates = []
            for i, update in enumerate(client_updates):
                logger.debug(
                    f"Processing update {i}: num_samples={update['num_samples']}, params type={type(update['params'])}"
                )

                # Check if params is already a dict or needs conversion
                if isinstance(update["params"], dict):
                    params_dict = {
                        k: deserialise_array(serialized)
                        if isinstance(serialized, str)
                        else serialized
                        for k, serialized in update["params"].items()
                    }
                elif isinstance(update["params"], list):
                    # Params is a list of [key, value] or [key, serialized]
                    params_dict = {}
                    for entry in update["params"]:
                        if isinstance(entry, (list, tuple)) and len(entry) == 2:
                            k, val = entry
                            params_dict[k] = (
                                deserialise_array(val) if isinstance(val, str) else val
                            )
                        else:
                            logger.error(f"Unexpected params entry format: {entry}")
                else:
                    logger.error(f"Unexpected params type: {type(update['params'])}")
                    continue

                deserialized_updates.append(
                    {"num_samples": update["num_samples"], "params": params_dict}
                )

            if len(deserialized_updates) == 0:
                logger.error("No valid updates after deserialization")
                return

            # Get keys
            keys = list(deserialized_updates[0]["params"].keys())
            logger.info(f"Model has {len(keys)} parameters: {keys}")

            # Weighted average
            accum = {
                k: np.zeros_like(deserialized_updates[0]["params"][k], dtype=np.float64)
                for k in keys
            }

            for update in deserialized_updates:
                weight = update["num_samples"] / total_samples
                for k in keys:
                    accum[k] += update["params"][k].astype(np.float64) * weight

            # Load into model
            state_dict = OrderedDict(
                (k, torch.from_numpy(accum[k].astype(np.float32))) for k in keys
            )
            self.model.load_state_dict(state_dict)

            logger.info(f"Successfully aggregated {len(client_updates)} updates")

        except Exception as e:
            logger.error(f"Aggregation failed: {e}")
            logger.error(f"Error at line: {e.__traceback__.tb_lineno}")
            import traceback

            logger.error(f"Traceback: {traceback.format_exc()}")
            raise

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

    def get_model_params(self):
        """Get serialized global model parameters."""
        state_dict = self.model.state_dict()
        params = []
        for k, v in state_dict.items():
            nd = v.detach().cpu().numpy()
            params.append({"key": k, "value": serialise_array(nd)})
        return params


def handleAggregateRequest(msComm):
    global ms_config
    global hfl_server
    request = rabbitTypes.Request()
    msComm.original_request.Unpack(request)
    try:
        start = time.perf_counter()
        data = request.data["model_updates"]
        logger.info(f"Received model_updates, type: {type(data)}")
        logger.debug(f"List values count: {len(data.list_value.values)}")
        client_updates = []
        for idx, update_struct in enumerate(data.list_value.values):
            update_obj = json.loads(update_struct.string_value)
            
            if len(update_obj["params"]) > 0:
                logger.debug(f"First param entry: {update_obj['params'][0]}")
            
            # params is a list of [key, serialized_value]
            # Convert to dict of {key: deserialized_value}
            params_dict = {}
            for entry_idx, entry in enumerate(update_obj["params"]):
                try:
                    k = entry[0]  # key
                    serialized = entry[1]  # serialized array
                    params_dict[k] = deserialise_array(serialized)
                    logger.debug(
                        f"Deserialized param {entry_idx}: {k}, shape: {params_dict[k].shape}"
                    )
                except Exception as e:
                    logger.error(f"Error deserializing entry {entry_idx}: {e}")
                    logger.error(f"Entry content: {entry}")
                    raise
            
            upd = {
                "num_samples": update_obj["num_samples"],
                "params": params_dict,
            }
            client_updates.append(upd)
            
    except Exception as e:
        logger.error(f"Error deserializing client model updates: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return
    
    logger.info("Performing FedAvg aggregation from client updates.")
    hfl_server.aggregate_fit(client_updates)
    agg_duration = (time.perf_counter() - start) * 1000
    accuracy = hfl_server.evaluate()
    logger.info(f"Aggregation duration: {agg_duration:.2f}ms")
    
    # Get global model params
    global_params = hfl_server.get_model_params()
    
    if global_params is None or len(global_params) == 0:
        logger.error("No global parameters returned from get_model_params!")
        return
    
    logger.info(f"Got {len(global_params)} parameters from global model")
    
    # CRITICAL FIX: Serialize to JSON string
    global_params_json = json.dumps(global_params)
    logger.info(f"Serialized global params: {len(global_params_json)} bytes")
    
    # Send as JSON string, not list
    data = Struct()
    data.update({"accuracy": accuracy})
    data.update({"global_params": global_params_json})  # Send as JSON string
    data.update({"t_agg": agg_duration})
    
    ms_config.next_client.ms_comm.send_data(msComm, data, {})


def request_handler(msComm: msCommTypes.MicroserviceCommunication, ctx: Context = None):
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

    DATA_STEWARD_NAME = os.getenv("DATA_STEWARD_NAME", "").lower()

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

    # data = load_data(config.dataset_filepath)
    row_ids = list(range(26032))
    hfl_server = HFLServer(
        config.dataset_filepath, row_ids=row_ids, row_count=len(row_ids), zipf_rank=1
    )

    ms_config = NewConfiguration(config.service_name, config.grpc_addr, request_handler)

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
