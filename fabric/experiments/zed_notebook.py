# %% Cell 1
import os

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# %%
energy = pd.read_csv(
    "/home/maurits/EnergyEfficient_Scattered-Directive/fabric/experiments/data/exp1/combined_energy_stats.csv",
    index_col=0,
)
global_stats = pd.read_csv(
    "/home/maurits/EnergyEfficient_Scattered-Directive/fabric/experiments/data/exp1/combined_global_stats.csv",
    index_col=0
)
# Rename the global_stats index to 'round' to match energy df
global_stats = global_stats.rename_axis("round").reset_index()

# Pivot energy: one column per container
join_cols = ["exp", "K", "Z", "sigma_ed", "sigma_iid", "timestamp", "round"]

energy_wide = energy.pivot_table(
    index=join_cols,
    columns=["namespace","pod_name","container_name"],
    values="joules",
    aggfunc="sum"  # sum across pod_name in case multiple pods per container type
).reset_index()

# Flatten column names
energy_wide.columns.name = None

# Merge with global stats
result = pd.merge(global_stats, energy_wide, on=join_cols, how="inner")

print(result.shape)  # should be (75, ...) per Z
print(result.head())

result.to_csv("rounds_dataset.csv", index=False)
# %%

# %%
k = 2  # min_samples - 1

container_types = averaged["container_name"].unique()
ncols = 3
nrows = -(-len(container_types) // ncols)  # ceiling division

fig, axes = plt.subplots(nrows, ncols, figsize=(15, nrows * 3))
axes = axes.flatten()

for i, container in enumerate(sorted(container_types)):
    subset = averaged[averaged["container_name"] == container][["joules"]].values

    if len(subset) < k + 1:
        axes[i].set_title(f"{container}\n(insufficient data)")
        continue

    X_scaled = StandardScaler().fit_transform(subset)

    nbrs = NearestNeighbors(n_neighbors=k).fit(X_scaled)
    distances, _ = nbrs.kneighbors(X_scaled)
    k_distances = np.sort(distances[:, -1])[::-1]

    axes[i].plot(k_distances)
    axes[i].axhline(y=1.5, color="red", linestyle="--", label="eps=1.5")
    axes[i].set_title(container)
    axes[i].set_xlabel("Points sorted by distance")
    axes[i].set_ylabel("2-NN distance (scaled)")
    axes[i].legend(fontsize=7)

# Hide unused subplots
for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.suptitle("k-distance plots per container type (scaled joules)", y=1.02)
plt.tight_layout()
# plt.savefig("kdistance_per_container.png", bbox_inches="tight")
plt.show()


# %%
def find_elbow(k_distances):
    # Largest drop in consecutive distances
    diffs = np.diff(k_distances)
    elbow_idx = np.argmin(diffs)  # steepest drop
    return k_distances[elbow_idx + 1]


for container in sorted(container_types):
    subset = averaged[averaged["container_name"] == container][["joules"]].values
    if len(subset) < k + 1:
        continue
    X_scaled = StandardScaler().fit_transform(subset)
    nbrs = NearestNeighbors(n_neighbors=k).fit(X_scaled)
    distances, _ = nbrs.kneighbors(X_scaled)
    k_distances = np.sort(distances[:, -1])[::-1]
    elbow = find_elbow(k_distances)
    print(f"{container:25s}  elbow eps ≈ {elbow:.4f}")
