import numpy as np
import pandas as pd
from scipy.spatial import distance
from sklearn import preprocessing
from sklearn.cluster import DBSCAN, Birch
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("/home/maurits/EnergyEfficient_Scattered-Directive/fabric/experiments/data/baseline/idle_baseline.csv", index_col=0)

k = 2

def find_elbow(k_distances):
    diffs = np.diff(k_distances)
    elbow_idx = np.argmin(diffs)
    return k_distances[elbow_idx + 1]

# group_cols = ["exp", "K", "Z", "sigma_ed", "sigma_iid", "namespace", "pod_name", "container_name"]
group_cols = ["namespace", "pod_name", "container_name"]

anomalous = []

for group_keys, group in df.groupby(group_cols):
    if len(group) < k + 1:
        continue

    X = group[["joules"]].values

    if X.std() == 0:
        continue

    X_scaled = StandardScaler().fit_transform(X)

    nbrs = NearestNeighbors(n_neighbors=k).fit(X_scaled)
    distances, _ = nbrs.kneighbors(X_scaled)
    k_distances = np.sort(distances[:, -1])[::-1]
    eps = find_elbow(k_distances)

    if eps == 0.0 or np.isnan(eps):
        continue

    db = DBSCAN(eps=eps, min_samples=2).fit(X_scaled)
    group = group.copy()
    group["dbscan_label"] = db.labels_
    group["zscore"] = (group["joules"] - group["joules"].mean()) / group["joules"].std()

    outliers = group[(group["dbscan_label"] == -1) & (group["zscore"].abs() > 2.5)]
    if not outliers.empty:
        anomalous.append(outliers)

if anomalous:
    result = pd.concat(anomalous, ignore_index=True)
    print(f"Anomalous data points found: {len(result)}\n")
    print(result[group_cols + ["round", "timestamp", "joules", "zscore"]].to_string(index=False))
    print(result.groupby("container_name")["joules"].count().sort_values(ascending=False))
    print(result.groupby("Z")["joules"].count().sort_values(ascending=False))
    result.to_csv("anomalous_datapoints.csv", index=False)
else:
    print("No anomalies detected.")
