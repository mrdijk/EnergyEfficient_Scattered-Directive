import argparse
import os
import shutil
from pathlib import Path

import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

PATH = "/home/maurits/EnergyEfficient_Scattered-Directive/fabric/experiments/data/exp1/K05/Z015_ed1000p0_iid10"

if __name__ == "__main__":
    df = pd.read_csv(
        "/home/maurits/EnergyEfficient_Scattered-Directive/fabric/experiments/data/exp1/K05/combined_energy_stats.csv",
        index_col=0,
    )
    # Step 1: average the 3 timestamps per (Z, container_name, round)
    avg_cols = ["exp", "K", "Z", "sigma_ed", "sigma_iid", "container_name", "round"]
    averaged = df.groupby(avg_cols, as_index=False)["joules"].mean()

    # Step 2: DBSCAN per (Z, container_name) across rounds
    group_cols = ["exp", "K", "Z", "sigma_ed", "sigma_iid", "container_name"]

    anomalous_rounds = []

    for group_keys, group in averaged.groupby(group_cols):
        if len(group) < 2:
            continue

        X = group[["joules"]].values
        X_scaled = StandardScaler().fit_transform(X)

        db = DBSCAN(eps=0.25, min_samples=2).fit(X_scaled)
        group = group.copy()
        group["dbscan_label"] = db.labels_

        # Only keep flagged rounds that are genuinely far out
        group["zscore"] = (group["joules"] - group["joules"].mean()) / group[
            "joules"
        ].std()
        outliers = group[(group["dbscan_label"] == -1) & (group["zscore"].abs() > 2.5)]

        # outliers = group[group["dbscan_label"] == -1]
        if not outliers.empty:
            anomalous_rounds.append(outliers)

    if anomalous_rounds:
        result = pd.concat(anomalous_rounds, ignore_index=True)
        print(f"Anomalous (container, round) pairs found: {len(result)}\n")
        print(result[group_cols + ["round", "joules"]].to_string(index=False))
        result.to_csv("anomalous_rounds.csv", index=False)
    else:
        print("No anomalies detected.")

    for _, row in result.iterrows():
        mask = (averaged["Z"] == row["Z"]) & (
            averaged["container_name"] == row["container_name"]
        )
        group = averaged[mask]["joules"]
        print(f"Z={row['Z']} {row['container_name']} round={row['round']}")
        print(
            f"  flagged: {row['joules']:.2f}  |  mean: {group.mean():.2f}  std: {group.std():.2f}  min: {group.min():.2f}  max: {group.max():.2f}\n"
        )
