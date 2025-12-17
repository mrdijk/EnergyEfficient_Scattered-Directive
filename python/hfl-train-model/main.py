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
from datetime import datetime
import rabbitMQ_pb2 as rabbitTypes

from google.protobuf.empty_pb2 import Empty
import microserviceCommunication_pb2 as msCommTypes
import threading
from opentelemetry.context.context import Context
from collections import OrderedDict
import time

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
    file_name = f"{file_path}/courseData.csv"

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
    def __init__(self, input_size, device="cpu"):
        super(ServerModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x) 

class HFLServer:
    """
    Horizontal Federated Learning Server:
    - Aggregates all client models
    - Averages all model parameters
    - Evaluates aggregates model performance 
    - Sends back the averaged model parameters to the clients
    """
    def __init__(self, data):
        self.labels = torch.tensor((data["Completed"] == "Completed").astype(int).values).float().unsqueeze(1)
        self.feature_cols = ['Age', 'Login_Frequency', 'Average_Session_Duration_Min', 'Video_Completion_Rate',
									'Discussion_Participation', 'Time_Spent_Hours', 'Days_Since_Last_Login',
									'Notifications_Checked', 'Peer_Interaction_Score', 'Assignments_Submitted',
									'Assignments_Missed', 'Quiz_Attempts', 'Quiz_Score_Avg', 'Project_Grade',
									'Progress_Percentage', 'Rewatch_Count', 'Payment_Amount', 'App_Usage_Percentage',
									'Reminder_Emails_Clicked', 'Support_Tickets_Raised', 'Satisfaction_Rating',
									'Course_Duration_Days', 'Instructor_Rating'] +  ['Gender_Encoded', 'Education_Encoded', 'Employment_Encoded',
                                'Device_Encoded', 'Internet_Encoded', 'Level_Encoded',
                                'Category_Encoded', 'Payment_Encoded']
        
        self.data = torch.tensor(data[self.feature_cols].values).float() 
        self.model = ServerModel(self.data.shape[1])
        # Loss with class balancing
        pos_weight = torch.tensor([len(self.labels) / sum(self.labels) - 1])  # roughly inverse class ratio
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)

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
                state_dict[k] = torch.from_numpy(nd)
            self.model.load_state_dict(state_dict)

        except Exception as e:
            logger.error(f"FedAvg aggregation failed: {e}")
            raise e

        logger.info("FedAvg Succesful, evaluating results")

        # Evaluate accuracy on server dataset
        self.model.eval()
        with torch.no_grad():
            logits = self.model(self.data)
            loss = self.criterion(logits, self.labels)
            preds = torch.sigmoid(logits)
            predicted = (preds > 0.5)
            accuracy = (predicted == self.labels).sum().item() / len(self.labels)

        data = Struct()
        data.update({"accuracy": accuracy})

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
    start = time.perf_counter()
    time.perf_counter()
    agg_result = hfl_server.aggregate_fit(client_updates)
    agg_duration =  (time.perf_counter() - start) * 1000
    logger.info(f"Aggregation duration: {agg_duration:.2f}ms")
    agg_result.update({"ts": agg_duration})
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
            timestamp = datetime.now().strftime("%H:%M:%S")
            logger.info(f"{timestamp}: Received hflAggregateRequest.")
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
