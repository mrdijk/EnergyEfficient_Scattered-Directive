# %% Cell 1
import os

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# %% Cell
# Get the directories in the data folder
path = (
    "/home/maurits/EnergyEfficient_Scattered-Directive/fabric/experiments/data/exp1/K05"
)
data_folder_exp_dirs = []
directories = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
print("Directories:", directories)
# %%
df = pd.read_csv(
    "/home/maurits/EnergyEfficient_Scattered-Directive/fabric/experiments/data/exp1/K05/combined_energy_stats.csv",
    index_col=0,
)
# Step 1: average the 3 timestamps per (Z, container_name, round)
avg_cols = ["exp", "K", "Z", "sigma_ed", "sigma_iid", "container_name", "round"]
averaged = df.groupby(avg_cols, as_index=False)["joules"].mean()

# Step 2: DBSCAN per (Z, container_name) across rounds
group_cols = ["exp", "K", "Z", "sigma_ed", "sigma_iid", "container_name"]
averaged.groupby(group_cols).sum()

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
