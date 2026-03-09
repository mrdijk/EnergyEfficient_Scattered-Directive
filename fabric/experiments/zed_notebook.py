# %% Cell 1
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# %% Cell 2
index = pd.MultiIndex.from_tuples(
    [(5, 90), (5, 120), (10, 90), (10, 120)], names=["K", "Z"]
)

energy_per_round = pd.Series(
    [57.693578, 57.326005, 69.528845, 61.375898],
    index=index,
    name="Energy per Round (kJ)",
)
energy_per_client = pd.Series(
    [15.182520, 15.085791, 15.802010, 13.949068],
    index=index,
    name="Energy per Client (kJ)",
)
energy_per_sample = pd.Series(
    [48.880895, 64.759293, 29.454164, 34.667161],
    index=index,
    name="Energy per Sample (J)",
)

df = pd.concat([energy_per_round, energy_per_client, energy_per_sample], axis=1)
print(df)
# %% Cell 3
z_values = [90, 120]
k_values = [5, 10]
metrics = df.columns.tolist()
# [(0.2980392156862745, 0.4470588235294118, 0.6901960784313725), (0.8666666666666667, 0.5176470588235295, 0.3215686274509804)]
colors = {
    "5": (0.2980392156862745, 0.4470588235294118, 0.6901960784313725),
    "10": (0.8666666666666667, 0.5176470588235295, 0.3215686274509804),
}
x = np.arange(len(z_values))
width = 0.35

fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.suptitle(
    "Energy Consumption by Partitions (Z), grouped by Clients (K)",
    fontsize=13,
    fontweight="bold",
)

# Compute consistent y-max per metric (max value + 15% headroom)
y_maxes = {metric: df[metric].max() * 1.15 for metric in metrics}

for ax, metric in zip(axes, metrics):
    for i, k in enumerate(k_values):
        values = [df.loc[(k, z), metric] for z in z_values]
        offset = (i - 0.5) * width
        bars = ax.bar(
            x + offset,
            values,
            width=width,
            label=f"K={k}",
            color=colors[str(k)],
            alpha=0.85,
            edgecolor="white",
            linewidth=0.5,
        )
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + y_maxes[metric] * 0.02,
                f"{bar.get_height():.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_ylim(0, y_maxes[metric])
    ax.set_title(metric, fontsize=11, fontweight="bold", pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Z={z}" for z in z_values], fontsize=10)
    ax.set_ylabel(metric, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig("energy_grouped_bar.png", dpi=150, bbox_inches="tight")
plt.show()
# %% Cell
df_energy = pd.read_csv(
    "/home/maurits/EnergyEfficient_Scattered-Directive/fabric/experiments/analysis_output/combined_energy_stats.csv",
    index_col=[0],
)
fig, ax = plt.subplots()
df_energy = df_energy[~df_energy["container_name"].isin(["linkerd-init"])]
# Create a copy with container_name replaced to "agent" for all client containers
df_plot = df_energy.copy()
df_plot.loc[
    df_plot["container_name"].str.contains("client|server"), "container_name"
] = "agent"
# Now group and sum
energy_by_container = (
    df_plot.groupby("container_name")["joules"].sum().sort_values(ascending=False)
)

(energy_by_container / 1000).plot(
    kind="barh", ax=ax, color=["#2875E2", "#2875E2", "#2875E2", "#06A77D", "#06A77D"]
)
ax.set_title(
    "Total Energy Consumption by Container Type", fontsize=12, fontweight="bold"
)
ax.set_xlabel("Energy (kJ)")
ax.set_ylabel("Container")
ax.grid(True, alpha=0.3, axis="x")
infra = mpatches.Patch(color="#2875E2", label="DYNAMOS")
train = mpatches.Patch(color="#06A77D", label="training")
ax.legend(handles=[infra, train])
plt.tight_layout()
# plt.show()
plt.savefig("analysis_output/plots/analysis_energy.png", dpi=300, bbox_inches="tight")

# %% Cell
df_client = pd.read_csv(
    "/home/maurits/EnergyEfficient_Scattered-Directive/fabric/experiments/analysis_output/combined_client_stats.csv",
    index_col=[0],
)
fig, ax = plt.subplots()
# plt.rcParams["figure.figsize"] = (12, 6)
if not df_client.empty:
    # 3.1 Average client training time by Z with error bars (grouped by K)
    time_stats = (
        df_client.groupby(["K", "Z"])["ClientTrainingTime"]
        .agg(["mean", "std"])
        .reset_index()
    )

    z_values = sorted(time_stats["Z"].unique())
    x = np.arange(len(z_values))
    width = 0.8 / len(k_values)

    for i, k in enumerate(k_values):
        k_data = time_stats[time_stats["K"] == k].set_index("Z").reindex(z_values)
        offset = (i - len(k_values) / 2 + 0.5) * width
        ax.bar(
            x + offset,
            k_data["mean"],
            width,
            yerr=k_data["std"],
            capsize=3,
            label=f"K={k}",
            alpha=0.8,
            color=(
                [
                    (0.2980392156862745, 0.4470588235294118, 0.6901960784313725),
                    (0.8666666666666667, 0.5176470588235295, 0.3215686274509804),
                ]
            )[i],
        )

    ax.set_xticks(x)
    ax.set_xticklabels(z_values)
    ax.set_title(
        "Average Client Training Time by Z and K (mean ± std)",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xlabel("Number of Partitions (Z)")
    ax.set_ylabel("Training Time (ms)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.savefig(
        "analysis_output/plots/training_time_by_KZ.png", dpi=300, bbox_inches="tight"
    )
# %% Cell
df_energy = pd.read_csv(
    "/home/maurits/EnergyEfficient_Scattered-Directive/fabric/experiments/analysis_output/combined_energy_stats.csv"
)

# Get unique K and Z values
k_values = sorted(df_energy["K"].unique()) if "K" in df_energy.columns else [5]
z_values = sorted(df_energy["Z"].unique())

print(f"\nNumber of clients (K): {k_values}")
print(f"Number of partitions (Z): {z_values}")

# Prepare data
df_plot = df_energy.copy()

# Exclude Z=330 if it's an anomaly
df_plot = df_plot[df_plot["Z"] != 330]
z_values = [z for z in z_values if z != 330]

# Exclude linkerd-init
df_plot = df_plot[~df_plot["container_name"].isin(["linkerd-init"])]

# Rename client/server containers to "agent"
df_plot.loc[
    df_plot["container_name"].str.contains("client|server", na=False), "container_name"
] = "agent"

# Group by K, Z, and container_name
grouped = df_plot.groupby(["K", "Z", "container_name"])["joules"].sum().reset_index()
grouped["kJ"] = grouped["joules"] / 1000

# Create pivot table
pivot = grouped.pivot_table(
    values="kJ", index=["K", "Z"], columns="container_name", aggfunc="sum", fill_value=0
)

# Sort containers by total energy
container_order = pivot.sum().sort_values(ascending=False).index
pivot = pivot[container_order]

# Define colors for containers
container_colors = {
    "sidecar": "#E63946",
    "linkerd-proxy": "#E63946",
    "agent": "#E63946",
    "hfl-train": "#06A77D",
    "hfl-train-model": "#06A77D",
    "api-gateway": "#E63946",
    "orchestrator": "#E63946",
    "policy-enforcer": "#FB5607",
}

colors = [container_colors.get(cont, "#6C757D") for cont in pivot.columns]

# Create figure
fig, ax = plt.subplots(figsize=(14, 6))

# Set up x-axis positions
n_z = len(z_values)
n_k = len(k_values)
bar_width = 0.8 / n_k
x = np.arange(n_z)

# Plot stacked bars for each K
for k_idx, k in enumerate(k_values):
    offset = (k_idx - n_k / 2 + 0.5) * bar_width
    bottom = np.zeros(n_z)

    for container_idx, container in enumerate(pivot.columns):
        values = []
        for z in z_values:
            if (k, z) in pivot.index:
                values.append(pivot.loc[(k, z), container])
            else:
                values.append(0)

        values = np.array(values)

        # Only add label for first K (to avoid duplicate legend entries)
        label = container if k_idx == 0 else None

        ax.bar(
            x + offset,
            values,
            bar_width,
            bottom=bottom,
            label=label,
            color=colors[container_idx],
            alpha=0.85,
            edgecolor="white",
            linewidth=0.5,
        )

        bottom += values

# Customize plot
ax.set_xticks(x)
ax.set_xticklabels([f"Z={z}" for z in z_values], rotation=45, ha="right")
ax.set_xlabel("Number of Partitions (Z)", fontsize=13, fontweight="bold")
ax.set_ylabel("Total Energy (kJ)", fontsize=13, fontweight="bold")
ax.set_title(
    "Energy Breakdown by Container Type (Stacked by Z, Grouped by K)",
    fontsize=15,
    fontweight="bold",
    pad=20,
)

# Add K labels as text annotations at the top
y_max = ax.get_ylim()[1]
for k_idx, k in enumerate(k_values):
    offset = (k_idx - n_k / 2 + 0.5) * bar_width
    # Place label above first bar
    ax.text(
        x[0] + offset,
        y_max * 1.02,
        f"K={k}",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
        bbox=dict(
            boxstyle="round,pad=0.4",
            facecolor="lightgray",
            alpha=0.7,
            edgecolor="black",
            linewidth=1,
        ),
    )

# Legend
handles, labels = ax.get_legend_handles_labels()
ax.legend(
    handles[::-1],
    labels[::-1],
    title="Container Type",
    bbox_to_anchor=(1.01, 1),
    loc="upper left",
    fontsize=10,
    frameon=True,
    shadow=True,
)

ax.grid(True, alpha=0.3, axis="y")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_linewidth(1.5)
ax.spines["bottom"].set_linewidth(1.5)

plt.tight_layout()
# plt.savefig(
#     "analysis_output/energy_stacked_by_z_grouped_by_k.png",
#     dpi=300,
#     bbox_inches="tight",
#     facecolor="white",
# )
# print("\nSaved: energy_stacked_by_z_grouped_by_k.png")

# Print summary statistics
print("\n" + "=" * 80)
print("SUMMARY STATISTICS")
print("=" * 80)

for k in k_values:
    print(f"\nK={k}:")
    k_data = pivot.loc[k]
    print(
        f"  Total energy range: {k_data.sum(axis=1).min():.2f} - {k_data.sum(axis=1).max():.2f} kJ"
    )
    print(f"  Average total: {k_data.sum(axis=1).mean():.2f} kJ")

    print("\n  Energy by container (average across Z):")
    for container in pivot.columns:
        avg_energy = k_data[container].mean()
        pct = (k_data[container].sum() / k_data.sum().sum()) * 100
        print(f"    {container:<20} {avg_energy:>8.2f} kJ ({pct:>5.1f}%)")

plt.show()
