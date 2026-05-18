import numpy as np
import pandas as pd
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
    (exp, K, Z, sigma_ed, sigma_iid), concatenating all timestamps so that
    repeated runs of the same configuration become a single contiguous series.

    Each timestamp's rounds (0..N-1) are offset by the cumulative round count
    of preceding timestamps, giving globally unique row indices across runs.
    This means BIRCH sees 75 data points per Z (3 runs × 25 rounds) rather
    than 25 per timestamp, making the 95th-percentile threshold more meaningful.

    The output DataFrame retains a 'timestamp' column so anomalous rounds can
    be traced back to the specific run they came from.

    Returns a dict keyed by (exp, K, Z, sigma_ed, sigma_iid).
    """
    df = pd.read_csv(filepath, index_col=0)
    df = df[df["exp"] == "exp1"]

    config_cols = ["exp", "K", "Z", "sigma_ed", "sigma_iid"]
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

    return pivoted


def run_BIRCH_AD(temp_df: pd.DataFrame, df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Run BIRCH anomaly detection on a single column (container energy series).
    Operates on the full pooled series (all timestamps for a given Z), so the
    95th-percentile threshold is computed over 75 points rather than 25.

    Smoothing window is scaled to span roughly one full run's worth of rounds,
    so transient within-run noise is suppressed while cross-run deviations remain
    detectable.
    """
    ad_threshold = 0.045
    smoothing_window = 12

    test_df = df[["global_round", column]].copy()

    for column_name, column_data in test_df.items():
        if column_name != "global_round":
            column_data = column_data.rolling(
                window=smoothing_window, min_periods=1
            ).mean()

            x = np.array(column_data)
            x = np.where(np.isnan(x), 0, x)

            # Skip zero-variance columns (e.g. linkerd-init always 0)
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

    eps controls the neighbourhood radius in normalised space; min_samples is
    kept low (2) since each series has ~75 points and genuine anomalies are
    expected to be isolated rather than clustered.
    """
    smoothing_window = 12
    eps = 0.2
    min_samples = 2
    k = 2

    test_df = df[["global_round", column]].copy()

    for column_name, column_data in test_df.items():
        if column_name != "global_round":
            column_data = column_data.rolling(
                window=smoothing_window, min_periods=1
            ).mean()

            x = np.array(column_data)
            x = np.where(np.isnan(x), 0, x)

            # Skip zero-variance columns (e.g. linkerd-init always 0)
            if x.std() == 0:
                temp_df[f"{column}_Anomaly"] = 0
                temp_df[f"{column}_Anomaly_Score"] = 0.0
                continue

            # normalized_x = preprocessing.normalize([x])
            # X = normalized_x.reshape(-1, 1)
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

            # DBSCAN marks noise points as -1 — these are our anomalies
            anomaly_flags = np.where(labels == -1, 1, 0)

            # Anomaly score: distance to nearest core point (or 0 if core itself)
            if len(dbscan.core_sample_indices_) > 0:
                core_points = X_scaled[dbscan.core_sample_indices_]
                scores = np.min(distance.cdist(X_scaled, core_points), axis=1)
            else:
                # Degenerate case: no core points found, flag everything
                scores = np.ones(len(X_scaled))

            temp_df = temp_df.assign(**{
                f"{column}_Anomaly": anomaly_flags,
                f"{column}_Anomaly_Score": scores,
            })

    return temp_df

def main():
    print("Starting Anomaly Detection (pooled across timestamps per Z)...")

    pivoted = load_and_pivot(ENERGY_DATA_FILE)
    all_results = []

    for (exp, K, Z, sigma_ed, sigma_iid), wide_df in pivoted.items():
        n_runs = wide_df["timestamp"].nunique()
        n_rows = len(wide_df)
        print(f"  Processing exp={exp} K={K} Z={Z}  ({n_runs} runs, {n_rows} rows)...")

        temp_df = wide_df.copy()
        energy_cols = [c for c in wide_df.columns if c.endswith("_energy")]

        # for col in energy_cols:
        #     temp_df = run_BIRCH_AD(temp_df, wide_df, col)
        for col in energy_cols:
            temp_df = run_DBSCAN_AD(temp_df, wide_df, col)


        # Tag with experiment identifiers
        temp_df["exp"] = exp
        temp_df["K"] = K
        temp_df["Z"] = Z
        temp_df["sigma_ed"] = sigma_ed
        temp_df["sigma_iid"] = sigma_iid
        # 'timestamp' is already a column from load_and_pivot

        all_results.append(temp_df)

    results = pd.concat(all_results, ignore_index=True)
    # results.to_csv(BIRCH_AD_RESULTS_FILE, index=False)
    # print(f"\nAD results saved to {BIRCH_AD_RESULTS_FILE}")
    results.to_csv(DBSCAN_AD_RESULTS_FILE, index=False)
    print(f"\nAD results saved to {DBSCAN_AD_RESULTS_FILE}")

    # Summary: anomaly flags per container, broken down by Z
    anomaly_cols = [
        c for c in results.columns
        if c.endswith("_Anomaly") and not c.endswith("_Score")
    ]
    print("\nAnomalous rounds per container (flagged > 0):")
    summary = results.groupby("Z")[anomaly_cols].sum()
    # Only print containers that were flagged at least once across all Z
    flagged_cols = summary.columns[summary.sum() > 0]
    print(summary[flagged_cols].to_string())
    # print(summary[flagged_cols])


if __name__ == "__main__":
    main()
