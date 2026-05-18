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
    "/home/maurits/EnergyEfficient_Scattered-Directive/fabric/experiments/data/combined_energy_stats.csv",
    index_col=0,
)

df = energy[energy['exp'] == "exp1"]
grouped = df.groupby('container_name')['joules']
print()
fig, ax = plt.subplots(figsize=(15,10))
ax.boxplot(x=[group.values for name, group in grouped], tick_labels=grouped.groups.keys())
plt.xticks(rotation=45)
plt.show()
# df.boxplot(column='joules', by=['container_name'], rot=45, figsize=(15,10))

# %%
df['container_name'].unique()

# %%
df = pd.read_csv(
    "/home/maurits/EnergyEfficient_Scattered-Directive/fabric/experiments/data/combined_energy_stats.csv",
    index_col=0)
df = df[df["exp"] == "exp1"]

config_cols = ["exp", "K", "Z", "sigma_ed", "sigma_iid"]
pivoted = {}
k = 2  # min_samples - 1

for config_key, config_group in df.groupby(config_cols):

    container_types = config_group["container_name"].unique()
    ncols = 3
    nrows = -(-len(container_types) // ncols)  # ceiling division

    fig, axes = plt.subplots(nrows, ncols, figsize=(15, nrows * 3))
    axes = axes.flatten()

    for i, container in enumerate(sorted(container_types)):
        subset = config_group[config_group["container_name"] == container][["joules"]].values

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

df = df[df["exp"] == "exp1"]
for config_key, config_group in df.groupby(config_cols):

    container_types = config_group["container_name"].unique()
    for container in sorted(container_types):
        subset = config_group[config_group["container_name"] == container][["joules"]].values
        if len(subset) < k + 1:
            continue
        X_scaled = StandardScaler().fit_transform(subset)
        nbrs = NearestNeighbors(n_neighbors=k).fit(X_scaled)
        distances, _ = nbrs.kneighbors(X_scaled)
        k_distances = np.sort(distances[:, -1])[::-1]
        elbow = find_elbow(k_distances)
        print(f"{container:25s}  elbow eps ≈ {elbow:.4f}")

# %%
