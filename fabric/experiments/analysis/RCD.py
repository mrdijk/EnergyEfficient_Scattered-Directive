import csv

import numpy as np
import pandas as pd
from pyrca.analyzers.rcd import RCD

# Path to your combined energy CSV (long format)
ENERGY_DATA_FILE = "data/combined_energy_stats.csv"
# Path to idle baseline CSV collected from Prometheus
BASELINE_FILE = "data/baseline/idle_baseline.csv"
# Output path for RCD results
RCD_RESULTS_FILE = "data/RCD_RCA_results.csv"

# Number of top root causes to identify
K = 3

def load_baseline(filepath: str) -> pd.DataFrame:
    """
    Load idle baseline CSV and pivot to wide format (rows=rounds, cols=container_energy).
    """
    df = pd.read_csv(filepath)
    # Exclude training job containers — zero at idle, non-zero during training,
    # which would make them trivially dominant in root cause attribution
    # EXCLUDE_CONTAINERS = {"hfl-train", "hfl-train-model"}
    # df = df[~df["container_name"].isin(EXCLUDE_CONTAINERS)]

    wide = df.pivot_table(
        index="round",
        columns="container_name",
        values="joules",
        aggfunc="mean"
    )
    wide.columns = [f"{c}_energy" for c in wide.columns]
    return wide.reset_index(drop=True)


def load_and_pivot_test(filepath: str) -> dict:
    """
    Load long-format energy CSV and produce a wide-format DataFrame per
    (exp, K, Z, sigma_ed, sigma_iid), averaged across timestamps.
    """
    df = pd.read_csv(filepath, index_col=0)
    df = df[df["exp"] == "exp1"]

    avg_cols = ["exp", "K", "Z", "sigma_ed", "sigma_iid", "container_name", "round"]
    averaged = df.groupby(avg_cols, as_index=False)["joules"].mean()

    exp_cols = ["exp", "K", "Z", "sigma_ed", "sigma_iid"]
    test_dfs = {}

    for keys, group in averaged.groupby(exp_cols):
        wide = group.pivot_table(
            index="round",
            columns="container_name",
            values="joules",
            aggfunc="mean"
        )
        wide.columns = [f"{c}_energy" for c in wide.columns]
        test_dfs[keys] = wide.reset_index(drop=True)

    return test_dfs


def run_rcd(train_df: pd.DataFrame, test_df: pd.DataFrame) -> list:
    """
    Run RCD root cause analysis given training (normal) and test data.
    """
    model = RCD(config=RCD.config_class(
        start_alpha=0.05,
        k=K,
        bins=5,
        gamma=5,
        localized=True
    ))

    results = model.find_root_causes(train_df, test_df)
    return results.to_dict()


def main():
    print("Starting RCD Root Cause Analysis...")
    print(f"  Loading idle baseline from {BASELINE_FILE}...")
    train_df = load_baseline(BASELINE_FILE)
    print(f"  Baseline shape: {train_df.shape} ({len(train_df)} rounds, {len(train_df.columns)} containers)")

    print(f"  Loading test data from {ENERGY_DATA_FILE}...")
    test_dfs = load_and_pivot_test(ENERGY_DATA_FILE)

    all_results = []

    for (exp, K_val, Z, sigma_ed, sigma_iid), test_df in test_dfs.items():
        print(f"  Running RCD for exp={exp} K={K_val} Z={Z}...")

        common_cols = [c for c in train_df.columns if c in test_df.columns]
        if len(common_cols) == 0:
            print("    No common energy columns — skipping.")
            continue

        try:
            results = run_rcd(
                train_df[common_cols],
                test_df[common_cols]
            )
            nodes = [node[0] for node in results["root_cause_nodes"]]
        except Exception as e:
            print(f"    RCD failed for Z={Z}: {e}")
            nodes = []

        for rank, node in enumerate(nodes, start=1):
            all_results.append({
                "exp": exp,
                "K": K_val,
                "Z": Z,
                "sigma_ed": sigma_ed,
                "sigma_iid": sigma_iid,
                "rank": rank,
                "root_cause": node
            })

    if all_results:
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(RCD_RESULTS_FILE, index=False)
        print(f"\nRCD results saved to {RCD_RESULTS_FILE}")

        print("\n=== Root Cause Summary ===")
        summary = (results_df.groupby("root_cause")["Z"]
                   .count()
                   .sort_values(ascending=False)
                   .rename("times_flagged"))
        print(summary.to_string())
    else:
        print("No root causes identified.")


if __name__ == "__main__":
    main()
