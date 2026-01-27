import argparse

# import csv
import os
import random
import time
from datetime import datetime, timedelta

import constants
import pandas as pd
import requests


def get_energy_consumption_precise(start, end):
    # Query at start time
    response_start = requests.get(
        f"{constants.PROMETHEUS_URL}/api/v1/query",
        params={
            "query": f"sum({constants.PROM_KEPLER_ENERGY_METRIC}{constants.PROM_CONTAINER_NS}) by ({constants.PROM_KEPLER_CONTAINER_LABEL})",
            "time": start,
        },
    )

    # Query at end time
    response_end = requests.get(
        f"{constants.PROMETHEUS_URL}/api/v1/query",
        params={
            "query": f"sum({constants.PROM_KEPLER_ENERGY_METRIC}{constants.PROM_CONTAINER_NS}) by ({constants.PROM_KEPLER_CONTAINER_LABEL})",
            "time": end,
        },
    )

    # Calculate difference manually
    start_data = {
        (
            r["metric"]["container_namespace"],
            r["metric"]["pod_name"],
            r["metric"]["container_name"],
        ): float(r["value"][1])
        for r in response_start.json()["data"]["result"]
    }
    end_data = {
        (
            r["metric"]["container_namespace"],
            r["metric"]["pod_name"],
            r["metric"]["container_name"],
        ): float(r["value"][1])
        for r in response_end.json()["data"]["result"]
    }

    energy_data = []
    for key, end_value in end_data.items():
        start_value = start_data.get(key, 0)
        increase = end_value - start_value
        # if increase > 0:  # Only include if there was an increase
        energy_data.append(
            {
                "namespace": key[0],
                "pod_name": key[1],
                "container_name": key[2],
                "joules": increase,
            }
        )

    return energy_data


# Function to query Prometheus for energy consumption
def get_energy_consumption(start, end):
    # Parse RFC3339 strings
    start_dt = datetime.fromisoformat(start.replace("Z", "+00:00"))
    end_dt = datetime.fromisoformat(end.replace("Z", "+00:00"))

    # Calculate duration in seconds
    duration_seconds = int((end_dt - start_dt).total_seconds())

    # Use a minimum lookback window of 1 minute, or the experiment duration
    lookback_window = max(duration_seconds, 60)

    # For very short experiments (< 1 min), use precise more precise method
    # We don't use the precise method it queries at the exact start and end times and this doesn't account for variable container start times
    # increase() better accounts for different container runtimes but is suited for very short measurments
    # if duration_seconds < 60:
    #     return get_energy_consumption_precise(start, end)

    query = f"sum(increase({constants.PROM_KEPLER_ENERGY_METRIC}{constants.PROM_CONTAINER_NS}[{lookback_window}s])) by ({constants.PROM_KEPLER_CONTAINER_LABEL})"

    response = requests.get(
        f"{constants.PROMETHEUS_URL}/api/v1/query",
        params={"query": query, "time": end},
    )
    # Parse the response JSON
    response_json = response.json()

    # Extract the energy data
    energy_data = []
    # If the query was successful, return the results
    if response.status_code == 200:
        # Construct as readable energy data for each container
        for result in response_json["data"]["result"]:
            # Extract the container name
            energy_data.append(
                {
                    "namespace": result["metric"]["container_namespace"],
                    "pod_name": result["metric"]["pod_name"],
                    "container_name": result["metric"]["container_name"],
                    "joules": float(result["value"][1]),
                }
            )
        # Return result
        return energy_data

    # If request failed, return empty
    return []


# Main function to execute the experiment
def run_experiment(output_dir, providers):
    results = []
    requests_url = constants.APPROVAL_URL

    # Construct HFL request body
    hfl_request_body = constants.HFL_REQUEST
    headers = constants.HEADERS.copy()

    print(f"Using the followiong clients for training {providers}")
    hfl_request_body["dataProviders"] = providers
    hfl_request_body["data_request"]["data"]["cycles"] = exp_cycles

    # Execute HFL request, using specific headers created for FABRIC
    request_response = requests.post(
        requests_url, json=hfl_request_body, headers=headers
    )
    hfl_request_json = request_response.json()

    # Get request id and status from response to use for polling
    request_id = hfl_request_json["request_id"]
    status = hfl_request_json["status"]

    # Poll status until training is done to collect the datas
    status_url = constants.STATUS_URL + request_id
    print("Polling training request with id: ", request_id)

    # Poll the api-gateway until the training status is done or failed
    # When the training is done, save the results from the response
    while True:
        poll_response = requests.post(status_url, headers=headers)
        status = poll_response.json()["status"]
        print(f"Current training status: {status}")

        if status == "done":
            results = poll_response.json()["data"]
            break
        elif status == "failed":
            print("Training failed!")
            return
        else:
            print("Waiting 60 seconds")
            time.sleep(60)

    # Put all the global statistics of the HFL in one csv file
    # [GlobalAccuracy  AggregationTime  TotalTrainingTime  RoundDuration]
    df_global = pd.DataFrame(results["rounds"])
    # Drop the ClientMetrics column, those are saved seperately
    df_global_clean = df_global.drop("ClientMetrics", axis=1)
    print(df_global_clean)
    # Ensure the output directory exists and save data as csv in output dir
    os.makedirs(output_dir, exist_ok=True)
    global_stats_path = os.path.join(output_dir, "global_stats.csv")
    df_global_clean.to_csv(global_stats_path)
    print(f"Global results saved to {global_stats_path}")

    # Do the same for all the client statistics
    rows = []
    for round_idx, round_data in enumerate(results["rounds"]):
        for client in round_data["ClientMetrics"]:
            rows.append(
                {
                    "Round": round_idx,
                    "ClientID": client["ClientID"],
                    "ClientAccuracy": client["Accuracy"],
                    "ClientTrainingTime": client["TrainingTime"],
                }
            )

    client_df_flattened = pd.DataFrame(rows)
    print(client_df_flattened)
    client_stats_path = os.path.join(output_dir, "client_stats.csv")
    client_df_flattened.to_csv(client_stats_path, index=False)
    print(f"Client results saved to {client_stats_path}")

    energy_data = get_energy_consumption(results["start_time"], results["end_time"])

    if len(energy_data) == 0:
        print("No energy data")
        return

    energy_df = pd.DataFrame(energy_data)
    energy_path = os.path.join(output_dir, "energy_consumption.csv")
    print(energy_df.groupby(["namespace", "pod_name"]).sum())
    energy_df.to_csv(energy_path)
    print(f"Energy results saved to {energy_path}")


def format_timestamp():
    # Generate the current timestamp
    timestamp = datetime.now().strftime("%d-%m-%y-%H%M%S")
    return timestamp


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run energy efficiency experiment")
    parser.add_argument(
        "--exp_clients",
        type=int,
        required=False,
        help="The number of expected clients')",
    )
    parser.add_argument(
        "--exp_cycles", type=int, help="The number of rounds that are performed"
    )
    parser.add_argument(
        "--exp_size",
        type=str,
        required=False,
        help="Select 'exp_clients' smallest or largest clients",
    )
    args = parser.parse_args()

    exp_clients: int = args.exp_clients
    exp_cycles: int = args.exp_cycles
    exp_size: str = args.exp_size

    output_dir = os.path.join(
        "experiments/data",
        f"{exp_clients}",
        f"{exp_cycles}",
        f"{format_timestamp()}_{exp_size}",
    )

    sorted_clients: list[tuple[str, int]] = sorted(
        constants.DATA_PROVIDERS.items(), key=lambda x: x[1]
    )

    if exp_size == "large":
        providers: list[str] = ["server"] + [
            client for client, _ in sorted_clients[-exp_clients:]
        ]
    elif exp_size == "small":
        providers = ["server"] + [client for client, _ in sorted_clients[:exp_clients]]
    else:
        sample = random.sample(list(constants.DATA_PROVIDERS.keys()), exp_clients)
        providers = ["server"] + sample
    print("Running experiments with the following clients\n")
    print("Client: #Rows")
    for p in providers:
        if p == "server":
            continue
        print(p, ": ", constants.DATA_PROVIDERS[p])

    print("\nStarting experiment")
    run_experiment(output_dir, providers)
    print("Restart of DYNAMOS required!")
