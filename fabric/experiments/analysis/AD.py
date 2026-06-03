import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.spatial import distance
from sklearn import preprocessing
from sklearn.cluster import DBSCAN, Birch
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# Path to your combined energy CSV (long format)
ENERGY_DATA_FILE = "data/combined_energy_stats.csv"
# Output path for AD results
BIRCH_AD_RESULTS_FILE = "data/BIRCH_AD_results.csv"
DBSCAN_AD_RESULTS_FILE = "data/DBSCAN_AD_results.csv"


def load_and_pivot(filepath: str) -> dict[tuple, pd.DataFrame]:
    """
    Load the long-format energy CSV and pivot to wide format per
    (exp, K, Z, sigma_ed, sigma_iid, timestamp).

    Each run (timestamp) is kept separate, so AD runs on 25 rounds at a time
    rather than pooled across runs.

    Returns a dict keyed by (exp, K, Z, sigma_ed, sigma_iid, timestamp).
    """
    df = pd.read_csv(filepath, index_col=0)

    config_cols = ["exp", "K", "Z", "sigma_ed", "sigma_iid"]
    # config_cols = ["exp", "K", "Z", "sigma_ed", "sigma_iid", "timestamp"]
    pivoted = {}

    for config_key, config_group in df.groupby(config_cols):
        run_frames = []
        global_round_offset = 0

        # Sort timestamps so runs are concatenated in a consistent order
        for timestamp, ts_group in config_group.groupby("timestamp", sort=True):
            wide = ts_group.pivot_table(
                index="round",
                columns="container_name",
                values="joules",
                aggfunc="sum",
            )
            wide.columns = [f"{c}_energy" for c in wide.columns]
            wide = wide.reset_index()

            n_rounds = len(wide)
            wide["global_round"] = wide["round"] + global_round_offset
            wide["timestamp"] = timestamp

            run_frames.append(wide)
            global_round_offset += n_rounds

        combined = pd.concat(run_frames, ignore_index=True)
        pivoted[config_key] = combined
        # wide = config_group.pivot_table(
        #     index="round",
        #     columns="container_name",
        #     values="joules",
        #     aggfunc="sum",
        # )
        # wide.columns = [f"{c}_energy" for c in wide.columns]
        # wide = wide.reset_index()
        # wide["timestamp"] = config_key[-1]   # last element of key is timestamp
        # pivoted[config_key] = wide
    return pivoted


def run_BIRCH_AD(temp_df: pd.DataFrame, df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Run BIRCH anomaly detection on a single column (container energy series).
    Operates on a single run (25 rounds). Smoothing window kept short so it
    doesn't span more than the available rounds.
    """
    ad_threshold = 0.045
    smoothing_window = 12   # reduced from 12 — single run only has 25 points

    test_df = df[["round", column]].copy()

    for column_name, column_data in test_df.items():
        if column_name != "round":
            column_data = column_data.rolling(
                window=smoothing_window, min_periods=1
            ).mean()

            x = np.array(column_data)
            x = np.where(np.isnan(x), 0, x)

            if x.std() == 0:
                temp_df[f"{column}_Anomaly"] = 0
                temp_df[f"{column}_Anomaly_Score"] = 0.0
                continue

            normalized_x = preprocessing.normalize([x])
            X = normalized_x.reshape(-1, 1)

            birch = Birch(
                branching_factor=50,
                n_clusters=None,
                threshold=ad_threshold,
                compute_labels=True,
            )
            birch.fit(X)
            birch.predict(X)

            distances = distance.cdist(X, birch.subcluster_centers_)
            min_distances = np.min(distances, axis=1)

            threshold = np.percentile(min_distances, 99)
            test_df["anomaly_label"] = np.where(min_distances > threshold, 1, 0)

            temp_df = temp_df.assign(**{
                f"{column}_Anomaly": test_df["anomaly_label"].values,
                f"{column}_Anomaly_Score": min_distances,
            })

    return temp_df


def find_elbow(k_distances):
    diffs = np.diff(k_distances)
    elbow_idx = np.argmin(diffs)
    return k_distances[elbow_idx + 1]


def run_DBSCAN_AD(temp_df: pd.DataFrame, df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Run DBSCAN anomaly detection on a single column (container energy series).
    Points assigned to cluster -1 (noise) by DBSCAN are treated as anomalies.

    Operates on a single run (25 rounds). Smoothing window is reduced
    accordingly so it doesn't over-smooth the short series.
    """
    smoothing_window = 5   # reduced from 12 — single run only has 25 points
    min_samples = 3
    k = min_samples - 1

    test_df = df[["round", column]].copy()

    for column_name, column_data in test_df.items():
        if column_name != "round":
            column_data = column_data.rolling(
                window=smoothing_window, min_periods=1
            ).mean()

            x = np.array(column_data)
            x = np.where(np.isnan(x), 0, x)

            if x.std() == 0:
                temp_df[f"{column}_Anomaly"] = 0
                temp_df[f"{column}_Anomaly_Score"] = 0.0
                continue

            X = x.reshape(-1, 1)
            X_scaled = StandardScaler().fit_transform(X)

            nbrs = NearestNeighbors(n_neighbors=k).fit(X_scaled)
            distances, _ = nbrs.kneighbors(X_scaled)
            k_distances = np.sort(distances[:, -1])[::-1]
            eps = find_elbow(k_distances)

            if eps == 0.0 or np.isnan(eps):
                continue

            dbscan = DBSCAN(
                eps=eps,
                min_samples=min_samples,
                metric="euclidean",
            )
            labels = dbscan.fit_predict(X_scaled)

            anomaly_flags = np.where(labels == -1, 1, 0)

            if len(dbscan.core_sample_indices_) > 0:
                core_points = X_scaled[dbscan.core_sample_indices_]
                scores = np.min(distance.cdist(X_scaled, core_points), axis=1)
            else:
                scores = np.ones(len(X_scaled))

            temp_df = temp_df.assign(**{
                f"{column}_Anomaly": anomaly_flags,
                f"{column}_Anomaly_Score": scores,
            })

    return temp_df


def main():
    print("Starting Anomaly Detection (per run / timestamp)...")

    pivoted = load_and_pivot(ENERGY_DATA_FILE)
    all_results = []

    for (exp, K, Z, sigma_ed, sigma_iid), wide_df in pivoted.items():
        n_runs = wide_df["timestamp"].nunique()
        n_rows = len(wide_df)
        print(f"  Processing exp={exp} K={K} Z={Z}  ({n_runs} runs, {n_rows} rows)...")
    # for (exp, K, Z, sigma_ed, sigma_iid, timestamp), wide_df in pivoted.items():
    #     print(f"  Processing exp={exp} K={K} Z={Z} ts={timestamp} ({len(wide_df)} rounds)...")

        temp_df = wide_df.copy()
        energy_cols = [c for c in wide_df.columns if c.endswith("_energy")]

        # for col in energy_cols:
        #     temp_df = run_BIRCH_AD(temp_df, wide_df, col)
        for col in energy_cols:
            temp_df = run_DBSCAN_AD(temp_df, wide_df, col)

        temp_df["exp"]       = exp
        temp_df["K"]         = K
        temp_df["Z"]         = Z
        temp_df["sigma_ed"]  = sigma_ed
        temp_df["sigma_iid"] = sigma_iid
        # timestamp already present as a column

        all_results.append(temp_df)

    results = pd.concat(all_results, ignore_index=True)
    # results.to_csv(BIRCH_AD_RESULTS_FILE, index=False)
    # print(f"\nAD results saved to {BIRCH_AD_RESULTS_FILE}")
    results.to_csv(DBSCAN_AD_RESULTS_FILE, index=False)
    print(f"\nAD results saved to {DBSCAN_AD_RESULTS_FILE}")

    # Summary: anomaly flags per container, broken down by (Z, timestamp)
    anomaly_cols = [
        c for c in results.columns
        if c.endswith("_Anomaly") and not c.endswith("_Score")
    ]
    # 2. Group by configuration and timestamp (Run ID)
    # Note: Ensure your timestamp column name matches (e.g., 'timestamp' or 'ts')
    run_summary = results.groupby(['exp', 'K', 'Z', 'timestamp'])[anomaly_cols].sum()

    # 3. Filter for containers that had at least one anomaly
    ACTIVE_CLIENTS = [
        "client1", "client5", "client9", "client13", "client17",
        "api-gateway", "orchestrator", "policy-enforcer",
        "hfl-train", "hfl-train-model", "sidecar", "linker-proxy"]

    # # Create a regex pattern: 'client1|client5|orchestrator|...'
    # pattern = '|'.join(ACTIVE_CLIENTS)

    # # Filter columns that contain any of those strings
    # run_summary = run_summary.loc[:, run_summary.columns.str.contains(pattern)]

    # 1. Strip the suffix so the names match your ACTIVE_CLIENTS list
    run_summary.columns = run_summary.columns.str.replace('_energy_Anomaly', '')

    # 2. Now the exact match will work
    run_summary = run_summary[run_summary.columns.intersection(ACTIVE_CLIENTS)]
     # 4. Apply the Threshold (> 10 anomalies)
     # Filter Rows: Runs that have at least one container with > 10 anomalies
     # Filter Columns: Containers that have > 10 anomalies in at least one run
    filtered_summary = run_summary[(run_summary > 5).any(axis=1)]
    filtered_summary = filtered_summary.loc[:, (filtered_summary > 5).any(axis=0)]

    # 5. Visualize the "High-Anomaly" Matrix
    if not filtered_summary.empty:
        fig, ax = plt.subplots(figsize=(14, max(6, len(filtered_summary) * 0.6)))
        ax = sns.heatmap(filtered_summary,
                    cmap='YlOrRd',
                    annot=True,
                    fmt='g',
                    linewidths=.5,
                    cbar_kws={'label': 'Anomaly Count (> 10)'})
        ax.tick_params(axis='x', rotation=45)

    plt.title('High-Frequency Anomalies (>10 events per run)', fontsize=15)
    plt.ylabel('Experiment Run (Config | Timestamp)')
    plt.xlabel('Anomalous Containers')
    plt.tight_layout()
    plt.savefig("figures/DBSCAN_heatmap.png", dpi=300)


if __name__ == "__main__":
    main()
