import json
import os
import re
from pathlib import Path

import pandas as pd


def parse_experiment_path(path):
    parts = Path(path).parts
    print(parts)
    # 1. Precise exp extraction
    exp_match = [p for p in parts if re.fullmatch(r"exp\d+", p)]
    exp = exp_match[0] if exp_match else "unknown_exp"

    # 2. Extract K
    k_part = [p for p in parts if p.startswith("K")][0]
    K = int(k_part[1:])

    # 3. Extract Z, ed, iid
    config_part = [p for p in parts if p.startswith("Z")][0]
    z_str, ed_str, iid_str = config_part.split("_")
    Z = int(z_str[1:])
    ed = float(ed_str[2:].replace("p", "."))
    iid = int(iid_str[3:])

    # 4. Timestamp (The very last part of your provided path)
    timestamp_part = parts[-1]

    return {
        "exp": exp,
        "K": K,
        "Z": Z,
        "sigma_ed": ed,
        "sigma_iid": iid,
        "timestamp": timestamp_part,
        "path": str(path),
    }


def read_bandwidth_files(exp_dir):
    """Read all bandwidth JSON files in an experiment directory."""
    bandwidth_data = {}

    for file in Path(exp_dir).glob("bandwidth_*.json"):
        service_name = file.stem.replace("bandwidth_", "")

        with open(file, "r") as f:
            data = json.load(f)
            bandwidth_data[service_name] = data

    return bandwidth_data


def read_csv_file(exp_dir, filename):
    """Read a CSV file if it exists."""
    file_path = Path(exp_dir) / filename

    if file_path.exists():
        return pd.read_csv(file_path)
    return None


def combine_all_experiments(data_root):
    """Combine all experimental results into structured DataFrames."""

    # all_experiments = []
    all_client_stats = []
    all_global_stats = []
    all_energy_data = []
    # all_bandwidth_data = []

    # Find all experiment directories (those with timestamps)
    for root, dirs, files in os.walk(data_root):
        print("Current directory:", root)
        print("Subdirectories:", dirs)
        print("Files:", files)
        print("----------------")
        # Check if this is an experiment directory (contains result files)
        if "K" in root and any(f.endswith(".csv") or f.endswith(".json") for f in files):
            exp_dir = Path(root)
            # Parse experiment parameters
            exp_params = parse_experiment_path(exp_dir)
            # print(exp_params)

            # Read client stats
            client_stats = read_csv_file(exp_dir, "client_stats.csv")
            if client_stats is not None:
                for col in ["exp", "K", "Z", "sigma_ed", "sigma_iid", "timestamp"]:
                    # for col in ["K", "timestamp"]:
                    client_stats[col] = exp_params[col]
                all_client_stats.append(client_stats)

            # Read global stats
            global_stats = read_csv_file(exp_dir, "global_stats.csv")
            if global_stats is not None:
                # for col in ["K", "timestamp"]:
                for col in ["exp", "K", "Z", "sigma_ed", "sigma_iid", "timestamp"]:
                    global_stats[col] = exp_params[col]
                all_global_stats.append(global_stats)

            # Read energy consumption
            energy_data = read_csv_file(exp_dir, "energy_consumption.csv")
            if energy_data is not None:
                # for col in ["K", "timestamp"]:
                for col in ["exp", "K", "Z", "sigma_ed", "sigma_iid", "timestamp"]:
                    energy_data[col] = exp_params[col]
                all_energy_data.append(energy_data)

            # Read bandwidth data
            # bandwidth = read_bandwidth_files(exp_dir)
            # if bandwidth:
            #     for service, data in bandwidth.items():
            #         bw_row = {**exp_params, "service": service, "bandwidth_data": data}
            #         all_bandwidth_data.append(bw_row)

            # Track experiment
            # all_experiments.append(exp_params)

    # Combine into DataFrames
    # df_experiments = pd.DataFrame(all_experiments)
    df_client_stats = (
        pd.concat(all_client_stats, ignore_index=True)
        if all_client_stats
        else pd.DataFrame()
    )
    df_global_stats = (
        pd.concat(all_global_stats, ignore_index=True)
        if all_global_stats
        else pd.DataFrame()
    )
    df_energy = (
        pd.concat(all_energy_data, ignore_index=True)
        if all_energy_data
        else pd.DataFrame()
    )
    # df_bandwidth = (
    #     pd.DataFrame(all_bandwidth_data) if all_bandwidth_data else pd.DataFrame()
    # )
    return df_client_stats, df_global_stats, df_energy


if __name__ == "__main__":
    """Generate summary statistics across all experiments."""
    data_root = (
        "/home/maurits/EnergyEfficient_Scattered-Directive/fabric/experiments/data/"
    )

    (
        df_client_stats,
        df_global_stats,
        df_energy_stats,
    ) = combine_all_experiments(data_root)

    # print(f"Total experiments: {len(df_experiments)}")
    # print("\nExperiments by K (number of clients):")
    # print(df_experiments["K"].value_counts().sort_index())

    # print("\nExperiments by Z (number of partitions):")
    # print(df_experiments["Z"].value_counts().sort_index())

    # print("\nExperiments by sigma_ed:")
    # print(df_experiments["sigma_ed"].value_counts().sort_index())

    # print("\nExperiments by sigma_iid:")
    # print(df_experiments["sigma_iid"].value_counts().sort_index())

    # Global stats summary
    if not df_global_stats.empty:
        print(f"\n{'=' * 60}")
        print("GLOBAL STATISTICS SUMMARY")
        print(f"{'=' * 60}\n")

        # Group by Z and show mean accuracy
        if "GlobalAccuracy" in df_global_stats.columns:
            summary = df_global_stats["GlobalAccuracy"].agg(["mean", "std", "count"])
            print("\nAccuracy by Z (number of partitions):")
            print(summary)

        # Training time analysis
        if "ClientTrainingTime" in df_client_stats.columns:
            summary = df_client_stats["ClientTrainingTime"].agg(["mean", "std"])
            print("\nTraining time (ms) by Z:")
            print(summary)

    print(df_global_stats["exp"][0])
    Path("analysis_output/exp1").mkdir(parents=True, exist_ok=True)
    # 5. Export combined data
    # df_experiments.to_csv("analysis_output/exp3/combined_experiments.csv", index=True)
    if not df_global_stats.empty:
        df_global_stats.to_csv(
            "analysis_output/exp1/combined_global_stats.csv", index=False
        )
    if not df_client_stats.empty:
        df_client_stats.to_csv(
            "analysis_output/exp1/combined_client_stats.csv", index=False
        )
    if not df_energy_stats.empty:
        df_energy_stats.to_csv(
            "analysis_output/exp1/combined_energy_stats.csv", index=False
        )
    print("\nCombined CSVs saved!")
