import os
import re
import argparse
from kubernetes import client, config
from datetime import datetime, timezone
"""
Parse the accuracies from the api-gateway logs.

"""

def get_logs():
    #Load kubernetes configuration
    config.load_kube_config()

    #Create Kubernetes API client
    v1 = client.CoreV1Api()

    # Read the results from the logs of the api-gateway
    namespace = 'api-gateway'
    container_name = 'api-gateway'
    pod_name = 'api-gateway-6bf76d45b7-vjkc8'

    logs = v1.read_namespaced_pod_log(name=pod_name, namespace=namespace, container=container_name, since_seconds=500)
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
        kind = "Intermediate" if "Intermediate" in line else "Final"

        # Convert timestamp
        sec = int(ts_raw)
        ms = int((ts_raw - sec) * 1000)
        dt = datetime.fromtimestamp(sec, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        readable = f"{dt}.{ms:03d}"

        if first_ts is None:
            first_ts = ts_raw
        rel = ts_raw - first_ts

        results.append((readable, f"{rel:.2f}", rnd, kind, acc))

    return results

def save_accuracies(accuracies: list[str], output_dir: str, exp_cycles):
    print("Saving accuracies to file...")
    
    # Ensure the output directory exists
    output_dir_exp = os.path.join(output_dir, f'exp_{(exp_cycles)}')
    accuracies_file = os.path.join(output_dir_exp, "accuracies.txt")
    os.makedirs(output_dir_exp, exist_ok=True)
    
    with open( accuracies_file, "w") as f:
        f.write("# Accuracy results\n")
        f.write("# Columns: [time_readable] [time_relative_sec] [round] [type] [accuracy]\n")
        for row in accuracies:
            f.write("  ".join(map(str, row)) + "\n")

def format_timestamp():
    # Generate the current timestamp
    timestamp = datetime.now().strftime("%y%m%d-%H%M")
    return timestamp

if __name__ == "__main__":
    # Add argument parser
    parser = argparse.ArgumentParser(description="Run energy efficiency experiment")
    parser.add_argument("exp_nodes", type=int, help="The number of nodes in the node configuration')")
    parser.add_argument("exp_cycles", type=int, help="The number of rounds that are performed")
    # Parse args
    args = parser.parse_args()

    exp_nodes = args.exp_nodes
    exp_cycles = args.exp_cycles
    output_dir = os.path.join('results', f'{exp_nodes}_{exp_cycles}_{format_timestamp()}')

    # Get the results from the logs of the api-gateway
    logs = get_logs()
    print(f"Logs: {logs}")
    accuracies = parse_logs(logs)
    print(f"Accuracies: {accuracies}")
    save_accuracies(accuracies, output_dir, exp_cycles)
    print(f"✅ Saved accuracy logs to {output_dir}")
