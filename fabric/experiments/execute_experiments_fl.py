import requests
import time
import csv
import constants
import argparse
import os
import re
import subprocess
from kubernetes import client, config
from datetime import datetime, timezone

# Function to query Prometheus for energy consumption
def get_energy_consumption():
    # Query Prometheus
    response = requests.get(
        f"{constants.PROMETHEUS_URL}/api/v1/query",
        params={
            # Use range query, as we found that this was the most reliable in our thesis
            "query": constants.PROM_ENERGY_QUERY_RANGE
        },
    )
    # Parse the response JSON
    response_json = response.json()
    # print(f"Prometheus response status code: {response.status_code}")
    # print(f"Prometheus response: {response_json}")

    # Extract the energy data
    energy_data = {}
    # If the query was successful, return the results
    if response.status_code == 200:
        # Construct as readable energy data for each container
        for result in response_json['data']['result']:
            # Extract the container name
            container_name = result['metric'][constants.PROM_KEPLER_CONTAINER_LABEL]
            # Extract the actual result (value[0] is the timestamp)
            value = result['value'][1]
            energy_data[container_name] = float(value)
        # Return result
        return energy_data

    # If request failed, return empty
    return {}
    
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
def run_experiment(output_dir, exp_clients, exp_cycles):
    results = []
    requests_url = constants.APPROVAL_URL

    # Phase 1: Idle period
    # Wait idle period
    print("Waiting for idle period...")
    time.sleep(constants.IDLE_PERIOD)
    # Measure energy after idle (end_idle/start_active)
    idle_energy = get_energy_consumption()
    print(f"Idle Energy: {idle_energy} (in J)")

    # Phase 2: Active period
    # Record the start time of the active period
    active_start_time = time.time()

    # Construct HFL request body
    hfl_request_body = constants.HFL_REQUEST
    headers = constants.HEADERS_APPROVAL.copy()
    # Select number of nodes from data providers (+1 for the server)
    print(f"Using the followiong clients for training {constants.DATA_PROVIDERS[1:exp_clients+1]}")
    hfl_request_body["dataProviders"] = constants.DATA_PROVIDERS[:exp_clients+1]
    hfl_request_body["data_request"]["data"]["cycles"] = exp_cycles

    # Execute HFL request, using specific headers created for FABRIC
    requests.post(requests_url, json=hfl_request_body, headers=headers)

    print("Waiting for HFL to run")
    print("Waiting 120 seconds")
    time.sleep(30)
    print("Waiting 90 seconds")
    time.sleep(30)
    print("Waiting 60 seconds")
    time.sleep(30)
    print("Waiting 30 seconds")
    time.sleep(30)
    # time.sleep(120)

    # Get the results from the logs of the api-gateway
    logs = get_logs()
    accuracies = parse_logs(logs)
    save_accuracies(accuracies, output_dir)
    print(f"Saved accuracy logs to {output_dir}")

    # Save run data
    # runs[run] = {
    #     "appr_status_code": status_code_approval,
    #     # "appr_exec_time": execution_time_approval,
    #     "data_status_code": status_code_data_request,
    #     "data_req_exec_time": execution_time_data_request,
    # }
    # Apply interval between requests (if not last run of sequence) 
    # if (run + 1) != constants.NUM_EXP_ACTIONS:
    #     print("Waiting before next action...")
    #     time.sleep(8)


    # Before measuring the active energy, make sure the active period has passed for equal comparisons
    elapsed_time = time.time() - active_start_time
    # Add a few seconds to make sure a new Prometheus scrape is present
    remaining_time = (constants.ACTIVE_PERIOD + 2) - elapsed_time
    # If still time left to wait, sleep until the 2 minutes have passed
    if remaining_time > 0:
        print(f"Waiting for the remaining {remaining_time} seconds...")
        time.sleep(remaining_time)
    # Measure energy after active period (end_active) after the active period
    active_energy = get_energy_consumption()
    print(f"Active Energy: {active_energy} (in J)")

    energy_difference = {}
    for container, value in active_energy.items():
        energy_difference[container] = value - idle_energy[container]
    
    # Extract results for this run
    results = {
        # "runs": runs,
        "idle_energy": idle_energy,
        "active_energy": active_energy,
        "difference": energy_difference
    }

    # Save experiment results to files
    save_results(results, output_dir)


def save_results(results, output_dir):
    print("Saving experiment results to file...")
    
    # Ensure the output directory exists
    output_dir_exp = os.path.join(output_dir)
    os.makedirs(output_dir_exp, exist_ok=True)

    # Save runs results to CSV
    # csv_file = os.path.join(output_dir_exp, "results.csv")
    # Add the file
    # with open(csv_file, mode="w", newline="") as file:
        # Add run_nr as field and each key from the the runs list 
        # fieldnames = list(results.keys())
        # writer = csv.DictWriter(file, fieldnames=fieldnames)
        # writer.writeheader()
        # total_exec_time = 0
        # For each run, write the data
        # for run_nr, run_data in results["runs"].items():
        #     row = {"run_nr": run_nr}
        #     row.update(run_data)
        #     total_exec_time += run_data["appr_exec_time"] + run_data["data_req_exec_time"]
        #     writer.writerow(row)
    # Output file location that is clickable for the user
    # print(f"Runs results saved to {os.path.join(os.getcwd(), csv_file)}")

    # Calculate average execution times
    # average_exec_time = total_exec_time / len(results["runs"])

    # # Save experiment results to CSV
    # experiment_csv_file = os.path.join(output_dir_exp, "full_experiment_results.csv")
    # with open(experiment_csv_file, mode="w", newline="") as file:
    #     fieldnames = ["idle_energy_total", "active_energy_total", "total_energy_difference", "average_exec_time"]
    #     writer = csv.DictWriter(file, fieldnames=fieldnames)
    #     writer.writeheader()
    #     # Calculate total idle, active and difference energy consumption
    #     total_idle_energy = sum(float(value) for value in results["idle_energy"].values())
    #     total_active_energy = sum(float(value) for value in results["active_energy"].values())
    #     total_difference = total_active_energy - total_idle_energy
        
    #     # Add energy data
    #     writer.writerow({
    #         "idle_energy_total": total_idle_energy,
    #         "active_energy_total": total_active_energy,
    #         "total_energy_difference": total_difference,
    #         "average_exec_time": average_exec_time
    #     })
    # # Output file location that is clickable for the user
    # print(f"Full experiment results saved to {os.path.join(os.getcwd(), experiment_csv_file)}")

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
        for container, value in results["difference"].item():
            file.write(f"{container}: {value}\n")
    # Output file location that is clickable for the user
    print(f"Full energy values saved to {os.path.join(os.getcwd(), full_energy_file)}")


def format_timestamp():
    # Generate the current timestamp
    timestamp = datetime.now().strftime("%y%m%d-%H%M")
    return timestamp


if __name__ == "__main__":
    # Add argument parser
    parser = argparse.ArgumentParser(description="Run energy efficiency experiment")
    parser.add_argument("exp_clients", type=int, help="The number of expected clients")
    parser.add_argument("exp_cycles", type=int, help="The number of training rounds performed")
    # Parse args
    args = parser.parse_args()

    exp_clients = args.exp_clients
    exp_cycles = args.exp_cycles
    output_dir = os.path.join('data', f'{exp_clients}_{exp_cycles}_{format_timestamp()}')

    print(f"\nStarting experiment")
    run_experiment(output_dir, exp_clients, exp_cycles)
