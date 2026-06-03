# %% Cell 1
import math
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# %%
energy = pd.read_csv(
    "/home/maurits/EnergyEfficient_Scattered-Directive/fabric/experiments/data/combined_energy_stats.csv",
    index_col=0,
)

# %%
energy[energy['joules'] < 0]['pod_name']

energy[~(energy["pod_name"] == "client9-d68fb99b4-nkb65")]

# %%
ACTIVE_CLIENTS = ["server", "client1", "client5", "client9", "client13", "client17"]

def categorize_container(name):
    if "hfl-train" in name or "hfl-train-model" in name:
        return "#06A77D"
    if name.startswith("client") or name in "linkerd-proxy" or name in "server" or name in "sidecar":
        return "#E63946"
    if "policy" in name or "api-gateway" in name or "orchestrator" in name:
        return "#1f77b4"

components = [
    'linkerd-proxy',
    'sidecar',
    'client8',
    'client14',
         'client6',         'client4',        'client17',        'client15',
         'client9',         'client5',        'client12',        'client16',
         'client2',         'client3',        'client18',         'client7',
        'client11',        'client19',         'client1',        'client13',
        'client20',          'server',        'client10',     'api-gateway',
 'policy-enforcer',    'orchestrator', 'hfl-train-model',       'hfl-train']
components = sorted(components)
component_colors = {}
for c in components:
    component_colors[c] = categorize_container(c)

print(component_colors)
# %%
# Base colors
C_TRAIN = "#06A77D"
C_INFRA = "#E63946"
C_ORCH  = "#1f77b4"


# df = energy[energy['exp'] == "exp1"]
energy = energy[~(energy["pod_name"] == "client9-d68fb99b4-nkb65")]
df = energy[~energy["container_name"].isin(["linkerd-init"])]

# print(df['container_name'].unique())

grouped = df.groupby('container_name')['joules']
fig, ax = plt.subplots(figsize=(10,6))

bplot = ax.boxplot(
    x=[group.values for name, group in grouped],
    tick_labels=grouped.groups.keys(),
    patch_artist=True,
)

# fill with colors
for patch, color in zip(bplot['boxes'], component_colors.values()):
    # print(patch)
    patch.set_facecolor(color)

orch_patch =  mpatches.Patch(color='#1f77b4', label='Orchestration')
train_patch =  mpatches.Patch(color='#2ca02c', label='Training')
infra_patch =  mpatches.Patch(color='#d62728', label='Infrastructure')
fig.legend(handles=[orch_patch, train_patch, infra_patch])

ax.grid(True, alpha=0.3, axis="y")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_ylabel("Energy (J)")
ax.set_xlabel("Container")
plt.tight_layout()
plt.legend(loc="best")
plt.xticks(rotation=45,ha="right")
plt.tight_layout()
plt.savefig(
    "/home/maurits/EnergyEfficient_Scattered-Directive/fabric/experiments/figures/component_energy_bplot.png",
    dpi=300,
)
plt.show()
# df.boxplot(column='joules', by=['container_name'], rot=45, figsize=(15,10))

# %%
energy = pd.read_csv(
    "/home/maurits/EnergyEfficient_Scattered-Directive/fabric/experiments/data/combined_energy_stats.csv",
    index_col=0,
)
C_TRAIN = "#06A77D"   # base training green (per user palette)
DATA_DIR = Path(".")
OUT_DIR  = Path("figures")
FIG_EXT = "png"

def savefig(fig, name: str):
    out = OUT_DIR / f"{name}.{FIG_EXT}"
    fig.savefig(out)
    print(f"  Saved {out}")
    plt.close(fig)

def plot_rq1_hfltrain_boxplot(energy: pd.DataFrame):
    df = energy[
        (energy["exp"] == "exp1") &
        (energy["container_name"] == "hfl-train")
    ].copy()
    if df.empty:
        print("  [skip] no hfl-train rows in exp1")
        return

    z_vals = sorted(df["Z"].unique())
    n_z    = len(z_vals)
    n_cols = min(3, n_z)
    n_rows = math.ceil(n_z / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(4 * n_cols, 3.5 * n_rows),
                             sharey=True, squeeze=False)
    axes_flat = axes.flatten()
    for ax in axes_flat[n_z:]:   # hide unused cells
        ax.set_visible(False)

    for ax, z in zip(axes_flat, z_vals):
        sub       = df[df["Z"] == z]
        runs      = sorted(sub["timestamp"].unique())
        data      = [sub[sub["timestamp"] == ts]["joules"].values for ts in runs]
        # Short run labels: just index the runs 1, 2, …
        run_labels = [f"Run {i+1}" for i in range(len(runs))]

        bplot = ax.boxplot(
            data,
            tick_labels=run_labels,
            patch_artist=True,
            medianprops=dict(color="white", linewidth=1.8),
            whiskerprops=dict(linewidth=0.9),
            capprops=dict(linewidth=0.9),
            flierprops=dict(marker="o", markersize=2.5,
                            markerfacecolor=C_TRAIN, alpha=0.5,
                            linestyle="none"),
        )
        for patch in bplot["boxes"]:
            patch.set_facecolor(C_TRAIN)
            patch.set_alpha(0.80)

        ax.set_title(f"Z = {z}", fontsize=9)
        ax.set_xlabel("Run")
        if ax is axes_flat[0]:
            ax.set_ylabel("Energy per round (J)")
        ax.tick_params(axis="x", labelsize=8)
        ax.grid(axis="x", linestyle="-", alpha=0.35)
    fig.suptitle("RQ1 – hfl-train energy per round by Z and run (Exp 1)", fontsize=9)
    fig.tight_layout()
    # savefig(fig, "rq1_hfltrain_boxplot")
    plt.show()

# %%
plot_rq1_hfltrain_boxplot(energy)

# %%
df = pd.read_csv(
    "/home/maurits/EnergyEfficient_Scattered-Directive/fabric/experiments/data/combined_energy_stats.csv",
    index_col=0)
df = df[df["exp"] == "exp1"]

# Mean joules per (container_name, Z) across all rounds and timestamps
pivot = (
    df.groupby(["container_name", "Z"])["joules"]
    .sum()
    .unstack("Z")
)

# Sort containers by total mean energy descending so dominant ones are on top
pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).index]

z_vals = sorted(pivot.columns)
pivot  = pivot[z_vals]

fig_h = max(3, len(pivot) * 0.45)
fig_w = max(5, len(z_vals) * 1.2)
fig, ax = plt.subplots(figsize=(fig_w, fig_h))

im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrBr")

ax.set_xticks(range(len(z_vals)))
ax.set_xticklabels([f"Z={z}" for z in z_vals], fontsize=8)
ax.set_yticks(range(len(pivot)))
ax.set_yticklabels(pivot.index, fontsize=8)
ax.set_xlabel("Number of Partitions (Z)")
ax.set_title("RQ1 – Total energy per container by Z (Exp 1)", fontsize=9)


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
def darken(hex_color, factor=0.6):
    """Return a darkened version of hex_color (factor < 1 = darker)."""
    r, g, b = mcolors.to_rgb(hex_color)
    return mcolors.to_hex((r * factor, g * factor, b * factor))
# Base colors
C_INFRA = "#E63946"
COLOR_AGENT   = C_INFRA
COLOR_SIDECAR = darken(C_INFRA, 0.72)
COLOR_LINKERD = darken(C_INFRA, 0.50)
print(C_INFRA)
print(COLOR_SIDECAR)
print(COLOR_LINKERD)

# %%
df = pd.read_csv("/home/maurits/EnergyEfficient_Scattered-Directive/fabric/experiments/data/BIRCH_AD_results.csv")
# 1. Identify binary anomaly columns (ignoring scores)
anomaly_cols = [c for c in df.columns if '_Anomaly' in c and '_Score' not in c]

# 2. Group by configuration and timestamp (Run ID)
# Note: Ensure your timestamp column name matches (e.g., 'timestamp' or 'ts')
run_summary = df.groupby(['exp', 'K', 'Z', 'timestamp'])[anomaly_cols].sum()

# 3. Filter for containers that had at least one anomaly
run_summary = run_summary.loc[:, (run_summary > 0).any(axis=0)]

# 4. Clean column names for the heatmap
run_summary.columns = [c.replace('_energy_Anomaly', '') for c in run_summary.columns]

# Check total anomaly count per run across all containers
print(run_summary.sum(axis=1))

# 5. Visualize
plt.figure(figsize=(16, 12))
sns.heatmap(run_summary, cmap='YlOrRd', annot=True, fmt='g', cbar_kws={'label': 'Total Anomaly Events'})
plt.title('Anomaly Events per Individual Run (Timestamped)', fontsize=16)
plt.ylabel('Experiment Config & Timestamp')
plt.xlabel('Anomalous Containers')
plt.tight_layout()
plt.show()


# %%
df = pd.read_csv("/home/maurits/EnergyEfficient_Scattered-Directive/fabric/experiments/data/DBSCAN_AD_results.csv")
# Define the specific runs and containers
# Targets defined by your outlier selection
targets = [
    {'ts': '16-05-26-153111', 'container': 'orchestrator', 'name': 'Orchestrator'},
    {'ts': '11-05-26-174316', 'container': 'policy-enforcer', 'name': 'Policy-Enforcer'}
]

fig, axes = plt.subplots(1,2, figsize=(8,5))

for i, target in enumerate(targets):
    # Filter for the specific run and container
    run_data = df[df['timestamp'] == target['ts']].sort_values('round')
    energy_col = f"{target['container']}_energy"
    anomaly_col = f"{target['container']}_energy_Anomaly"
    # --- Right Column: QQ Plot ---
    # Compares the run's distribution against a Normal (Gaussian) distribution
    stats.probplot(run_data[energy_col], dist="norm", plot=axes[i])
    axes[i].set_title(f"Q-Q Plot: {target['name']} (Normality Check)")

plt.tight_layout()
plt.show()

# %%
# Define the specific runs and containers from your outliers selection
# Run 1: Orchestrator at 16-05-26-153111
# Run 2: Policy-Enforcer at 11-05-26-174316

plt.figure(figsize=(15, 7))

# 1. Process Orchestrator
df_orch = df[df['timestamp'] == '16-05-26-153111'].sort_values('round')
plt.plot(df_orch['round'], df_orch['orchestrator_energy'],
         label='Orchestrator Energy (Run: 153111)', color='blue', alpha=0.4)

orch_anoms = df_orch[df_orch['orchestrator_energy_Anomaly'] == 1]
plt.scatter(orch_anoms['round'], orch_anoms['orchestrator_energy'],
            color='blue', edgecolors='black', s=50, label='Orch Anomaly (19 flags)', zorder=5)

# 2. Process Policy-Enforcer
df_policy = df[df['timestamp'] == '11-05-26-174316'].sort_values('round')
plt.plot(df_policy['round'], df_policy['policy-enforcer_energy'],
         label='Policy-Enforcer Energy (Run: 174316)', color='green', alpha=0.4)

policy_anoms = df_policy[df_policy['policy-enforcer_energy_Anomaly'] == 1]
plt.scatter(policy_anoms['round'], policy_anoms['policy-enforcer_energy'],
            color='green', edgecolors='black', marker='X', s=70, label='Policy Anomaly (21 flags)', zorder=5)

# Formatting
plt.title("Combined Anomaly Timeline: Orchestrator vs. Policy-Enforcer", fontsize=14, fontweight='bold')
plt.xlabel("Round", fontsize=12)
plt.ylabel("Energy Consumption", fontsize=12)
plt.legend(frameon=True, loc='best')
plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()

# %%
def categorize_container(row):
    name = str(row["container_name"]).lower()
    namespace = str(row["namespace"]).lower()
    # Exclude inactive client namespaces entirely
    # if (
    #     "linkerd" in name
    #     and namespace.startswith("client")
    #     and namespace not in ACTIVE_CLIENTS.get(row["K"], [])
    # ):
    #     return "Idle Linkerd"
    if namespace.startswith("client") and namespace not in ACTIVE_CLIENTS.get(
        row["K"], []
    ):
        return "Idle clients"
    if "linkerd" in name:
        return "Linkerd"
    if "hfl-train" in name or "hfl-train-model" in name:
        return "Training"
    if "policy" in name or "api-gateway" in name or "orchestrator" in name:
        return "Coordination"
    # Only count client/server/sidecar as Infrastructure if it's an active client
    if (
        ("sidecar" in name and namespace in ACTIVE_CLIENTS.get(row["K"], []))
        or ("sidecar" in name and namespace in ["api-gateway", "orchestrator"])
        or name in ACTIVE_CLIENTS.get(row["K"], [])
        or "server" in name
    ):
        return "Infrastructure"
    return "Other"  # inactive clients → excluded

# %%
energy = pd.read_csv(
    "/home/maurits/EnergyEfficient_Scattered-Directive/fabric/experiments/data/combined_energy_stats.csv",
    index_col=0,
)
energy = energy[
    (energy['exp'] == 'exp3') &
    (energy['sigma_ed'] == 1000.0) &
    (energy['sigma_iid'] == 10)
]

summary = (
    energy.groupby(["sigma_ed", "sigma_iid", "timestamp", "round"])["joules"]
    .sum()
    .reset_index()
    .groupby(["sigma_ed", "sigma_iid", "round"])["joules"]
    .agg(["sum","mean", "std"])
    .div(1000)
    .reset_index()
)
summary
# %%
summary = (
    energy.groupby(["sigma_ed", "sigma_iid", "timestamp"])["joules"]
    .sum()
    .reset_index()
    .groupby(["sigma_ed", "sigma_iid"])["joules"]
    .agg(["sum","mean", "std"])
    .div(1000)
    .reset_index()
)
summary
