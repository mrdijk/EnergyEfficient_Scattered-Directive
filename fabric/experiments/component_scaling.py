import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_style("white")
sns.set_palette("deep")
plt.rcParams["font.family"] = "sans-serif"

# Load data
df_global = pd.read_csv(
    "/home/maurits/EnergyEfficient_Scattered-Directive/fabric/experiments/data/combined_global_stats.csv"
)
df_global[df_global['exp'] == 'exp2']
# Group by K
accuracy_comparison = df_global.groupby(["K"])["GlobalAccuracy"].last().reset_index()

# Plot 1: Accuracy Comparison
# fig, ax = plt.subplots()

k_values = [5, 10]
x = np.arange(len(k_values))
width = 0.35
colors_k = sns.color_palette("deep", 2)

# Plot 2: Total Energy Scaling
ACTIVE_CLIENTS = {
    5: ["server", "client1", "client5", "client9", "client13", "client17"],
    10: [
        "server",
        "client1",
        "client2",
        "client5",
        "client6",
        "client9",
        "client10",
        "client13",
        "client14",
        "client17",
        "client18",
    ],
}

df_energy = pd.read_csv(
    "/home/maurits/EnergyEfficient_Scattered-Directive/fabric/experiments/data/combined_energy_stats.csv"
)
df_energy[df_energy['exp'] == "exp2"]
df_energy = df_energy[~df_energy["container_name"].isin(["linkerd-init"])]

# Calculate total energy per configuration
energy_totals = df_energy.groupby(["K"])["joules"].sum().reset_index()
energy_totals["total_kJ"] = energy_totals["joules"] / 1000

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
ax = axes[0, 0]

bars = ax.bar(x, energy_totals["total_kJ"], label=k_values, color=colors_k, alpha=0.85)

for bar in bars:
    height = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2.0,
        height + 0.01,
        f"{height:.3f}",
        ha="center",
        va="bottom",
        fontweight="bold",
        fontsize=10,
    )


ax.set_ylabel("Total Energy (kJ)", fontsize=12, fontweight="bold")
# ax.set_xlabel("Number of Clients", fontsize=12, fontweight="bold")
ax.set_title("Total Energy Consumption", fontsize=12, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels([f"{k}" for k in k_values])
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis="y")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
# plt.savefig(
# "analysis_output/energy_scaling_k5_vs_k10.png", dpi=300, bbox_inches="tight"
# )
# print("saved fig: energy_scaling_k5_vs_k10.png")
# Plot 3: Energy per Round, per Client, per Sample
ROUNDS = 25
TOTAL_SAMPLES = 531130

# Plot 1: Energy per Round
ax = axes[1, 1]

bars = ax.bar(
    x, energy_totals["total_kJ"] / ROUNDS, label=k_values, color=colors_k, alpha=0.85
)

for bar in bars:
    height = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2.0,
        height + 0.01,
        f"{height:.3f}",
        ha="center",
        va="bottom",
        fontweight="bold",
        fontsize=10,
    )

ax.set_ylabel("Energy per Round (kJ)", fontsize=11, fontweight="bold")
ax.set_title("Energy per Round", fontsize=12, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels([f"{k}" for k in k_values])
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis="y")

# Plot 2: Energy per Client
ax = axes[0, 1]
bars = ax.bar(
    x, energy_totals["total_kJ"] / k_values, label=k_values, color=colors_k, alpha=0.85
)

for bar in bars:
    height = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2.0,
        height + 0.01,
        f"{height:.3f}",
        ha="center",
        va="bottom",
        fontweight="bold",
        fontsize=10,
    )
ax.set_ylabel("Energy per Client (kJ)", fontsize=11, fontweight="bold")
ax.set_title("Energy per Client", fontsize=12, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels([f"{k}" for k in k_values])
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis="y")

# Plot 3: Energy per Sample
ax = axes[1, 0]
energy_totals["samples"] = energy_totals["K"] * (TOTAL_SAMPLES / 100)

energy_totals["energy_per_sample"] = (energy_totals["total_kJ"] * 1000) / energy_totals[
    "samples"
]
bars = ax.bar(
    x, energy_totals["energy_per_sample"], label=k_values, color=colors_k, alpha=0.85
)

for bar in bars:
    height = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2.0,
        height + 0.01,
        f"{height:.3f}",
        ha="center",
        va="bottom",
        fontweight="bold",
        fontsize=10,
    )

ax.set_ylabel("Energy per Sample (J)", fontsize=11, fontweight="bold")
ax.set_title("Energy per Sample", fontsize=12, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels([f"{k}" for k in k_values])
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis="y")

for ax in axes.flat:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

# plt.tight_layout()
# plt.savefig(
#     "analysis_output/energy_metrics_k5_vs_k10.png", dpi=300, bbox_inches="tight"
# )
# print("save fig: energy_metrics_k5_vs_k10.png")


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


df_energy["component_type"] = df_energy.apply(categorize_container, axis=1)
# print(df_energy.groupby(["K", "component_type", "timestamp"])["joules"].sum() / 1000)
# print(df_energy[df_energy["component_type"] == "Linkerd"])
# Drop rows that are not part of any category (inactive clients)
# Group by component type
component_energy = (
    df_energy.groupby(["K", "component_type", "timestamp", "round"])["joules"]
    .sum()
    .groupby(["K", "component_type"])
    .mean()
    .reset_index()
)
component_energy["kJ"] = component_energy["joules"] / 1000

metrics = []
for k in [5, 10]:
    total_energy = energy_totals[(energy_totals["K"] == k)]["total_kJ"].values[0]
    accuracy = accuracy_comparison[(accuracy_comparison["K"] == k)][
        "GlobalAccuracy"
    ].values[0]
    samples_per_partition = TOTAL_SAMPLES / 100
    total_samples = samples_per_partition * k
    metrics.append(
        {
            "K": k,
            "total_kJ": total_energy,
            "energy_per_round": total_energy / ROUNDS,
            "energy_per_client": total_energy / k,
            "energy_per_sample": (total_energy * 1000) / total_samples,  # Joules
            "accuracy": accuracy,
        }
    )
metrics_df = pd.DataFrame(metrics)

fig, ax = plt.subplots(figsize=(10, 7))

k5_data = component_energy[component_energy["K"] == 5]
k10_data = component_energy[component_energy["K"] == 10]

components = [
    "Infrastructure",
    "Linkerd",
    "Idle clients",
    # "Idle Linkerd",
    "Training",
    "Coordination",
    # "Other",
]

colors_comp = {
    "Training": "#06A77D",
    "Coordination": "#1f77b4",
    "Infrastructure": "#e06666",
    "Linkerd": "#cc0000",
    "Idle clients": "#990000",
    "Idle Linkerd": "#cc0000",
    # "Infrastructure": "#ff0000",
    # "Linkerd": "#df0000",
    # "Idle clients": "#c00000",
    # "Idle Linkerd": "#a00000",
    # "Other": "grey",
}

x_pos = [0, 1]
bottom_k5 = 0
bottom_k10 = 0
MIN_LABEL_HEIGHT = 50

for comp in components:
    k5_val = k5_data[k5_data["component_type"] == comp]["kJ"].values
    k5_val = k5_val[0] if len(k5_val) > 0 else 0

    k10_val = k10_data[k10_data["component_type"] == comp]["kJ"].values
    k10_val = k10_val[0] if len(k10_val) > 0 else 0

    ax.bar(
        0,
        k5_val,
        0.6,
        bottom=bottom_k5,
        label=comp,
        color=colors_comp[comp],
        alpha=0.85,
        edgecolor="white",
        linewidth=1,
    )
    ax.bar(
        1,
        k10_val,
        0.6,
        bottom=bottom_k10,
        color=colors_comp[comp],
        alpha=0.85,
        edgecolor="white",
        linewidth=1,
    )

    # Add percentage labels for large segments
    if k5_val > 1:
        k5_total = k5_data["kJ"].sum()
        pct = (k5_val / k5_total) * 100
        ax.text(
            0,
            bottom_k5 + k5_val / 2,
            # f"{pct:.1f}%",
            f"{k5_val:.2f}",
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            color="white",
        )
    else:
        mid = bottom_k5 + k5_val / 2
        # Place label outside to the right with a short line
        ax.annotate(
            f"{k5_val:.1f} kJ",
            xy=(0, mid),
            xytext=(0 + 0.40, mid),
            ha="left",
            va="center",
            fontsize=9,
            color="black",
            arrowprops=dict(arrowstyle="-", color="gray", lw=0.8),
        )
    if k10_val > 1:
        k10_total = k10_data["kJ"].sum()
        pct = (k10_val / k10_total) * 100
        ax.text(
            1,
            bottom_k10 + k10_val / 2,
            # f"{pct:.1f}%",
            f"{k10_val:.2f}",
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            color="white",
        )
    else:
        mid = bottom_k10 + k10_val / 2
        # Place label outside to the right with a short line
        ax.annotate(
            f"{k10_val:.1f} kJ",
            xy=(1, mid),
            xytext=(1 + 0.40, mid),
            ha="left",
            va="center",
            fontsize=9,
            color="black",
            arrowprops=dict(arrowstyle="-", color="gray", lw=0.8),
        )
    bottom_k5 += k5_val
    bottom_k10 += k10_val

ax.set_xticks([0, 1])
ax.set_xticklabels(["K=5", "K=10"])
ax.set_ylabel("Energy (kJ)", fontsize=11, fontweight="bold")
# ax.set_title("Component Breakdown", fontsize=12, fontweight="bold")
ax.grid(True, alpha=0.3, axis="y")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(fontsize=10, loc="upper left")

plt.tight_layout()
plt.savefig(
    "figures/component_scaling.png", dpi=300, bbox_inches="tight"
)
print("saved fig: component_scaling.png")

print("\n" + "=" * 80)
print("CLIENT SCALING ANALYSIS: K=5 vs K=10")
print("=" * 80)

k5_energy = energy_totals[(energy_totals["K"] == 5)]["total_kJ"].values[0]
k10_energy = energy_totals[(energy_totals["K"] == 10)]["total_kJ"].values[0]
k5_acc = accuracy_comparison[(accuracy_comparison["K"] == 5)]["GlobalAccuracy"].values[
    0
]
k10_acc = accuracy_comparison[(accuracy_comparison["K"] == 10)][
    "GlobalAccuracy"
].values[0]

k5_metrics = metrics_df[(metrics_df["K"] == 5)].iloc[0]
k10_metrics = metrics_df[(metrics_df["K"] == 10)].iloc[0]

print("\nTotal Energy:")
print(f"  K=5:  {k5_energy:,.2f} kJ")
print(f"  K=10: {k10_energy:,.2f} kJ")
print(
    f"  Increase: {k10_energy - k5_energy:+,.2f} kJ ({(k10_energy / k5_energy - 1) * 100:+.1f}%)"
)
print(f"  Scaling factor: {k10_energy / k5_energy:.2f}× (linear would be 2.0×)")

print("\nGlobal Accuracy:")
print(f"  K=5:  {k5_acc:.4f}")
print(f"  K=10: {k10_acc:.4f}")
print(f"  Improvement: {k10_acc - k5_acc:+.4f} ({(k10_acc / k5_acc - 1) * 100:+.1f}%)")

print("\nEnergy per Sample:")
print(f"  K=5:  {k5_metrics['energy_per_sample']:.4f} J/sample")
print(f"  K=10: {k10_metrics['energy_per_sample']:.4f} J/sample")
print(
    f"  Change: {k10_metrics['energy_per_sample'] - k5_metrics['energy_per_sample']:+.4f} J"
)

print("\nEnergy Efficiency (J per accuracy %):")
k5_eff = k5_metrics["energy_per_sample"] / (k5_acc * 100)
k10_eff = k10_metrics["energy_per_sample"] / (k10_acc * 100)
print(f"  K=5:  {k5_eff:.4f} J")
print(f"  K=10: {k10_eff:.4f} J")
if k10_eff < k5_eff:
    print("  → K=10 is MORE energy-efficient per accuracy point!")
else:
    print("  → K=5 is MORE energy-efficient per accuracy point!")

print("\n" + "=" * 80)
print("DETAILED INFRASTRUCTURE BREAKDOWN")
print("=" * 80)

components = [
    "Infrastructure",
    "Linkerd",
    "Idle clients",
    # "Idle Linkerd",
    "Training",
    "Coordination",
    "Other",
]

for component in components:
    k5_comp = component_energy[
        (component_energy["K"] == 5) & (component_energy["component_type"] == component)
    ]["kJ"].values
    k5_comp = k5_comp[0] if len(k5_comp) > 0 else 0

    k10_comp = component_energy[
        (component_energy["K"] == 10)
        & (component_energy["component_type"] == component)
    ]["kJ"].values
    k10_comp = k10_comp[0] if len(k10_comp) > 0 else 0

    scaling = k10_comp / k5_comp if k5_comp > 0 else 0

    print(f"\n{component}:")
    print(f"  K=5:  {k5_comp:,.2f} kJ ({k5_comp / k5_energy * 100:.1f}% of total)")
    print(f"  K=10: {k10_comp:,.2f} kJ ({k10_comp / k10_energy * 100:.1f}% of total)")
    print(f"  Scaling: {scaling:.2f}×")
    print(f"  Per client K=5:  {k5_comp / 5:.2f} kJ/client")
    print(f"  Per client K=10: {k10_comp / 10:.2f} kJ/client")

# Total infrastructure
k5_infra_total = component_energy[
    (component_energy["K"] == 5) & (component_energy["component_type"].isin(components))
]["kJ"].sum()

k10_infra_total = component_energy[
    (component_energy["K"] == 10)
    & (component_energy["component_type"].isin(components))
]["kJ"].sum()

print("\n" + "-" * 80)
print("TOTAL INFRASTRUCTURE:")
print(
    f"  K=5:  {k5_infra_total:,.2f} kJ ({k5_infra_total / k5_energy * 100:.1f}% of total)"
)
print(
    f"  K=10: {k10_infra_total:,.2f} kJ ({k10_infra_total / k10_energy * 100:.1f}% of total)"
)
print(f"  Scaling: {k10_infra_total / k5_infra_total:.2f}×")

# Training and Coordination
print("\n" + "=" * 80)
print("TRAINING AND COORDINATION")
print("=" * 80)

for component in ["Training", "Coordination"]:
    k5_comp = component_energy[
        (component_energy["K"] == 5) & (component_energy["component_type"] == component)
    ]["kJ"].values
    k5_comp = k5_comp[0] if len(k5_comp) > 0 else 0

    k10_comp = component_energy[
        (component_energy["K"] == 10)
        & (component_energy["component_type"] == component)
    ]["kJ"].values
    k10_comp = k10_comp[0] if len(k10_comp) > 0 else 0

    scaling = k10_comp / k5_comp if k5_comp > 0 else 0

    print(f"\n{component}:")
    print(f"  K=5:  {k5_comp:,.2f} kJ ({k5_comp / k5_energy * 100:.1f}% of total)")
    print(f"  K=10: {k10_comp:,.2f} kJ ({k10_comp / k10_energy * 100:.1f}% of total)")
    print(f"  Scaling: {scaling:.2f}×")
    if component == "Training":
        print(f"  Per client K=5:  {k5_comp / 5:.2f} kJ/client")
        print(f"  Per client K=10: {k10_comp / 10:.2f} kJ/client")
