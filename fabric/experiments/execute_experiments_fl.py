import requests
import time
import csv
import constants
import argparse
import os
import re
import subprocess
import random
from kubernetes import client, config
from datetime import datetime
import pandas as pd

# Function to query Prometheus for energy consumption
def get_energy_consumption(start, end):
    # Query Prometheus
    response = requests.get(
        f"{constants.PROMETHEUS_URL}/api/v1/query",
        params={
            # Use range query, as we found that this was the most reliable in our thesis
            "query": constants.PROM_ENERGY_QUERY_RANGE,
            "start": start,
            "end": end
        },
    )
    # Parse the response JSON
    response_json = response.json()

    # Extract the energy data
    energy_data = []
    # If the query was successful, return the results
    if response.status_code == 200:
        # Construct as readable energy data for each container
        for result in response_json['data']['result']:
            # Extract the container name
            energy_data.append({
                'namespace': result['metric']['container_namespace'],
                'pod_name': result['metric']['pod_name'],
                'container_name': result['metric']['container_name'],
                'joules': float(result['value'][1])
            })
        # Return result
        return energy_data

    # If request failed, return empty
    return []

def calculate_and_save_energy_difference(idle_energy, active_energy, output_dir):
    """
    Calculate energy difference and save to CSV.
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create lookup dict from idle measurements
    idle_lookup = {
        (item['namespace'], item['pod_name'], item['container_name']): item['joules']
        for item in idle_energy
    }
    
    # Calculate differences
    energy_data = []
    for active in active_energy:
        key = (active['namespace'], active['pod_name'], active['container_name'])
        idle_joules = idle_lookup.get(key, 0)
        
        energy_data.append({
            'namespace': active['namespace'],
            'pod_name': active['pod_name'],
            'container_name': active['container_name'],
            'idle_joules': idle_joules,
            'active_joules': active['joules'],
            'difference_joules': active['joules'] - idle_joules
        })
    
    # Save to CSV
    df = pd.DataFrame(energy_data)
    csv_path = os.path.join(output_dir, "energy_consumption.csv")
    sorted_df = df.groupby(['namespace','pod_name', 'container_name']).sum().sort_values('difference_joules', ascending=False)
    sorted_df.to_csv(csv_path, index=False)
    
    print(f"Results saved to {csv_path}")

def get_logs():
    #Load kubernetes configuration
    config.load_kube_config()

    #Create Kubernetes API client
    v1 = client.CoreV1Api()

    # Read the results from the logs of the api-gateway
    namespace='api-gateway'
    container_name = 'api-gateway'
    # Get the name of the current api-gateway pod
    pod_name = subprocess.getoutput(r'kubectl get pods -n api-gateway | grep api-gateway | sed "s/^\(api-gateway[a-zA-Z0-9-]\+\).*/\1/"')

    logs = v1.read_namespaced_pod_log(name=pod_name, namespace=namespace, container=container_name, since_seconds=constants.ACTIVE_PERIOD)
    # print(logs)

    return logs.splitlines()

def parse_logs(lines):
    results = []
    first_ts = None

    # Regex for accuracy logs
    regex = re.compile(
        r'(?P<ts>[0-9.e+-]+).*accuracy achieved:\s+(?P<acc>[0-9.]+).*round\s+(?P<round>\d+)',
        re.IGNORECASE,
    )

    for line in lines:
        m = regex.search(line)
        if not m:
            continue

        ts_raw = float(m.group("ts"))
        acc = float(m.group("acc"))
        rnd = int(m.group("round"))

        # Convert timestamp
        sec = int(ts_raw)
        ms = int((ts_raw - sec) * 1000)

        if first_ts is None:
            first_ts = ts_raw
        rel = ts_raw - first_ts

        results.append((f"{rel:.2f}", rnd, acc))

    return results

def save_accuracies(accuracies: list[str], output_dir: str):
    print("Saving accuracies to file...")
    
   # Ensure the output directory exists
    accuracies_file = os.path.join(output_dir, "accuracies.txt")
    os.makedirs(output_dir, exist_ok=True)
    
    with open( accuracies_file, "w") as f:
        f.write("# Accuracy results\n")
        f.write("# Columns: [time_relative_sec] [round] [accuracy]\n")
        for row in accuracies:
            f.write("  ".join(map(str, row)) + "\n")

# Main function to execute the experiment
def run_experiment(output_dir, providers):
    results = []
    requests_url = constants.APPROVAL_URL

    # Phase 1: Idle period
    # Wait idle period
    # print(f"Waiting for idle period ({constants.IDLE_PERIOD}s)")
    # time.sleep(constants.IDLE_PERIOD)
    # Measure energy after idle (end_idle/start_active)
    # idle_energy = get_energy_consumption()
    # print(f"Idle Energy: {idle_energy} (in J)")

    # Phase 2: Active period
    runs = {}
    # Record the start time of the active period
    active_start_time = time.time()
    
    # Construct HFL request body
    hfl_request_body = constants.HFL_REQUEST
    headers = constants.HEADERS.copy()
    # Select a sample from data providers (+ the server)
    # providers = ["server"] + random.sample(list(constants.DATA_PROVIDERS.keys()), exp_clients)
    print(f"Using the followiong clients for training {providers}")
    hfl_request_body["dataProviders"] =  providers
    hfl_request_body["data_request"]["data"]["cycles"] = exp_cycles

    # Execute HFL request, using specific headers created for FABRIC
    # print("Request time: ", format_timestamp())
    request_respons = requests.post(requests_url, json=hfl_request_body, headers=headers)
    hfl_request_json = request_respons.json()

    # get request id and status from respons to use for polling
    request_id = hfl_request_json['request_id']
    status = hfl_request_json['status']

    # Poll status until training is done to collect the data
    status_url = constants.STATUS_URL + request_id
    print("Polling training request with id: ", request_id)
    
    while True:
        poll_respons = requests.post(status_url, headers=headers)
        status = poll_respons.json()["status"]
        print(f"Current training status: ", status)
        if status is "done":
            results = poll_respons.json()["data"]
            break
        print("Waiting 60 seconds")
        time.sleep(60)

    # print(results)
    
    # print("Active query time: ", format_timestamp())
    active_energy = get_energy_consumption(results["start"], results["end"])
    print(f"Active Energy: {active_energy} (in J)")
    
    # print(f"Saved accuracy logs to {output_dir}")

    # Calculate the difference between idle and active energy consumption
    # calculate_and_save_energy_difference(idle_energy, active_energy, output_dir)

def save_results(results, output_dir):
    print("Saving experiment results to file...")
    
    # Ensure the output directory exists
    output_dir_exp = os.path.join(output_dir)
    os.makedirs(output_dir_exp, exist_ok=True)

    # Save full active and idle energy values to a text file
    full_energy_file = os.path.join(output_dir_exp, "full_energy_values.txt")
    with open(full_energy_file, mode="w") as file:
        file.write("Idle Energy:\n")
        for container, value in results["idle_energy"].items():
            file.write(f"{container}: {value}\n")
        file.write("\nActive Energy:\n")
        for container, value in results["active_energy"].items():
            file.write(f"{container}: {value}\n")
        file.write("\nDifference in Energy:\n")
        for container, value in results["difference"].items():
            file.write(f"{container}: {value}\n")
    # Output file location that is clickable for the user
    print(f"Full energy values saved to {os.path.join(os.getcwd(), full_energy_file)}")


def format_timestamp():
    # Generate the current timestamp
    timestamp = datetime.now().strftime("%d-%m-%y-%H%M%S")
    return timestamp


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run energy efficiency experiment")
    parser.add_argument("exp_clients", type=int, help="The number of expected clients')")
    parser.add_argument("exp_cycles", type=int, help="The number of rounds that are performed")
    # Parse args
    args = parser.parse_args()

    exp_clients = args.exp_clients
    exp_cycles = args.exp_cycles
    output_dir = os.path.join('data', f'{exp_clients}', f'{exp_cycles}', f'{format_timestamp()}')
    sample = random.sample(list(constants.DATA_PROVIDERS.keys()), exp_clients)
    providers = ["server"] + sample
    print("Running experiments with the following clients\n")
    print("Client: #Rows")
    for s in sample:
        print(s, ": ",constants.DATA_PROVIDERS[s])

    os.makedirs(output_dir, exist_ok=True)
    rows_file = os.path.join(output_dir, "number_of_rows.txt")
    with open(rows_file, mode="w") as file:
        file.write("Client: #Rows\n")
        for s in sample:
            file.write(f"{s} : {constants.DATA_PROVIDERS[s]}")

    print(f"\nStarting experiment")
    run_experiment(output_dir, providers)
