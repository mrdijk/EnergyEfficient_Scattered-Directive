import pandas as pd
import sys
import os
from google.protobuf.struct_pb2 import Struct
from dynamos.ms_init import NewConfiguration
from dynamos.signal_flow import signal_continuation, signal_wait
from dynamos.logger import InitLogger

from google.protobuf.empty_pb2 import Empty
import microserviceCommunication_pb2 as msCommTypes
import threading
from opentelemetry.context.context import Context


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
    DATA_STEWARD_NAME = os.getenv("DATA_STEWARD_NAME")
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


# def dataframe_to_protobuf(df):
#     # Convert the DataFrame to a dictionary of lists (one for each column)
#     data_dict = df.to_dict(orient='list')
#
#     # Convert the dictionary to a Struct
#     data_struct = Struct()
#
#     # Iterate over the dictionary and add each value to the Struct
#     for key, values in data_dict.items():
#         # Pack each item of the list into a Value object
#         value_list = [Value(string_value=str(item)) for item in values]
#         # Pack these Value objects into a ListValue
#         list_value = ListValue(values=value_list)
#         # Add the ListValue to the Struct
#         data_struct.fields[key].CopyFrom(Value(list_value=list_value))
#
#     # Create the metadata
#     # Infer the data types of each column
#     data_types = df.dtypes.apply(lambda x: x.name).to_dict()
#     # Convert the data types to string values
#     metadata = {k: str(v) for k, v in data_types.items()}
#
#     return data_struct, metadata


def vfl_train(requestData, ctx):
    global config
    logger.debug("Start vfl_train")

    try:
        result = load_data(config.dataset_filepath)
        logger.debug(result)
        logger.debug("after load data")

        # embeddings =
        embeddings = ["this is not an embedding", "nor is this",
                      "but it gets the ~point~ data across."]

        # data, metadata = dataframe_to_protobuf(result)
        data = Struct()
        data.update({"embeddings": embeddings})

        return data
    except FileNotFoundError:
        logger.error(f"File not found at path {config.dataset_filepath}")
        return None
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        return None


# ---  DYNAMOS Interface code At the Bottom --------

def request_handler(msComm: msCommTypes.MicroserviceCommunication, ctx: Context):
    global ms_config
    logger.info(f"Received original request type: {msComm.request_type}")

    # Ensure all connections have finished setting up before processing data
    signal_wait(wait_for_setup_event, wait_for_setup_condition)

    try:
        if msComm.request_type == "vflTrainRequest":
            logger.debug("Handling VFL Training request")
            requestData = msComm.data
            logger.debug(requestData)
            data = vfl_train(requestData, ctx)
            logger.debug("Received data from VFL Training:")
            logger.debug(data)

            # logger.debug(f"Forwarding result, metadata: {metadata}")
            # Ignore metadata
            ms_config.next_client.ms_comm.send_data(msComm, data, {})
            signal_continuation(stop_event, stop_microservice_condition)

        else:
            logger.error(f"An unknown request_type: {msComm.request_type}")

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
