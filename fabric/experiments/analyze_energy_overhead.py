import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# df = pd.DataFrame(data)
df = pd.read_csv(
    "/home/maurits/EnergyEfficient_Scattered-Directive/fabric/experiments/analysis_output/combined_energy_stats.csv",
    index_col=[0],
)
# Exclude Z=330 if it's an anomaly
df = df[df["Z"] != 330]
# z_values = [z for z in z_values if z != 330]
#
df_renamed = df[df["K"] == 5].copy()
df_renamed[~df_renamed["container_name"].isin(["linkerd-init"])]
df_renamed[~df_renamed["container_name"].isin(["linkerd-proxy"])]

df_renamed["container_name"] = df_renamed["container_name"].str.replace(
    r"^(client.*|server)$", "agent", regex=True
)
INFRA_CONTAINERS = {
    "orchestrator",
    "policy-enforcer",
    "api-gateway",
    "agent",
    # "linkerd-proxy",
    "sidecar",
}
TRAINING_CONTAINERS = {"hfl-train", "hfl-train-model"}


def categorize(name):
    if name in INFRA_CONTAINERS:
        return "Infrastructure_J"
    elif name in TRAINING_CONTAINERS:
        return "Training_J"
    else:
        return "other"


df_renamed["category"] = df_renamed["container_name"].apply(categorize)
config_cols = ["K", "Z"]

df_per_round = (
    df_renamed[df_renamed["category"] != "other"]
    .groupby(config_cols + ["timestamp", "category"])["joules"]
    .sum()
    .reset_index()
)

df = (
    df_per_round.groupby(config_cols + ["category"])["joules"]
    .mean()  # ← average over rounds/timestamps
    .unstack("category")
    .reset_index()
)

df["Total_J"] = df["Infrastructure_J"] + df["Training_J"]

df["Fixed_pct"] = (df["Infrastructure_J"] / df["Total_J"]) * 100
df["Variable_pct"] = (df["Training_J"] / df["Total_J"]) * 100

# Calculate samples per round
TOTAL_SAMPLES = 531130
df["Fixed_infrastructure_kJ"] = df["Infrastructure_J"] / 1000
df["Variable_training_kJ"] = df["Training_J"] / 1000
df["Total_kJ"] = df["Total_J"] / 1000

df["Samples_per_partition"] = TOTAL_SAMPLES / df["Z"]
df["Samples_per_round"] = df["Samples_per_partition"] * df["K"]

# Energy per sample
df["Energy_per_sample_J"] = (df["Total_kJ"] * 1000) / df["Samples_per_round"]
df["Fixed_per_sample_J"] = (df["Fixed_infrastructure_kJ"] * 1000) / df[
    "Samples_per_round"
]
df["Variable_per_sample_J"] = (df["Variable_training_kJ"] * 1000) / df[
    "Samples_per_round"
]

df["Energy_per_Client"] = (df["Total_kJ"] * 1000) / df["K"]
df["Fixed_energy_per_Client"] = (df["Fixed_infrastructure_kJ"] * 1000) / df["K"]
df["Var_Energy_per_Client"] = (df["Variable_training_kJ"] * 1000) / df["K"]

print("=" * 100)
print("CORRECTED OVERHEAD ANALYSIS")
print("=" * 100)
print("\nCategorization:")
print("  FIXED (Infrastructure):")
print("    - sidecar, linkerd-proxy (service mesh)")
print("    - api-gateway, orchestrator (coordination)")
print("    - client1-20 containers (infrastructure for clients)")
print("  VARIABLE (Training):")
print("    - hfl-train (client training)")
print("    - hfl-train-model (server/aggregation)")
print("=" * 100)

print(
    f"\n{'Z':<6} {'K':<4} {'Samples/Round':<15} {'Infrastructure (kJ/round)':<26} {'Training (kJ/round)':<20} "
    f"{'Total (kJ/round)':<18} {'Infra %':<10}"
)
print("-" * 100)

for idx, row in df.iterrows():
    print(
        f"{int(row['Z']):<6} {row['Samples_per_round']:<15,.0f} "
        f"{row['Fixed_infrastructure_kJ']:<20.2f} {row['Variable_training_kJ']:<15.2f} "
        f"{row['Total_kJ']:<12.2f} {row['Fixed_pct']:<10.1f}%"
    )

# fig, ax = plt.subplots(1, 1)

# # Plot 1: Stacked bar showing infrastructure vs training
# plot = "per_sample_overhead.png"
# # ax = axes[0, 0]
# ax.plot(
#     df["Z"],
#     df["Fixed_per_sample_J"],
#     "o-",
#     linewidth=2.5,
#     markersize=9,
#     label="DYNAMOS overhead/sample",
#     color="#2875E2",
#     markeredgewidth=1.5,
#     markeredgecolor="white",
# )
# ax.plot(
#     df["Z"],
#     df["Variable_per_sample_J"],
#     "s-",
#     linewidth=2.5,
#     markersize=9,
#     label="Training cost/sample",
#     color="#06A77D",
#     markeredgewidth=1.5,
#     markeredgecolor="white",
# )
# ax.plot(
#     df["Z"],
#     df["Energy_per_sample_J"],
#     "^-",
#     linewidth=3,
#     markersize=10,
#     label="Total energy/sample",
#     color="#E63946",
#     markeredgewidth=1.5,
#     markeredgecolor="white",
#     alpha=0.7,
# )

# ax.set_xlabel("Number of Partitions (Z)", fontsize=12, fontweight="bold")
# ax.set_ylabel("Energy Consumption (kJ)", fontsize=12, fontweight="bold")
# # ax.set_title(
# #     "Energy overhead per Sample",
# #     fontsize=13,
# #     fontweight="bold",
# # )
# ax.legend(fontsize=10, loc="upper right")
# ax.grid(True, alpha=0.3)
# ax.spines["top"].set_visible(False)
# ax.spines["right"].set_visible(False)
# plt.tight_layout()
# plt.savefig(f"analysis_output/{plot}", dpi=300, bbox_inches="tight")
# print(f"\nSaved: {plot}")

# # Plot 2: Energy per sample breakdown
# fig, ax = plt.subplots(1, 1)
# plot = "per_client_overhead.png"
# # ax = axes[0, 0]
# ax.plot(
#     df["Z"],
#     df["Fixed_energy_per_Client"],
#     "o-",
#     linewidth=2.5,
#     markersize=9,
#     label="DYNAMOS overhead/client",
#     color="#2875E2",
#     markeredgewidth=1.5,
#     markeredgecolor="white",
# )
# ax.plot(
#     df["Z"],
#     df["Var_Energy_per_Client"],
#     "s-",
#     linewidth=2.5,
#     markersize=9,
#     label="Training cost/client",
#     color="#06A77D",
#     markeredgewidth=1.5,
#     markeredgecolor="white",
# )
# ax.plot(
#     df["Z"],
#     df["Energy_per_Client"],
#     "^-",
#     linewidth=3,
#     markersize=10,
#     label="Total energy/client",
#     color="#E63946",
#     markeredgewidth=1.5,
#     markeredgecolor="white",
#     alpha=0.7,
# )

# ax.set_xlabel("Number of Partitions (Z)", fontsize=12, fontweight="bold")
# ax.set_ylabel("Energy Consumption (kJ)", fontsize=12, fontweight="bold")
# # ax.set_title(
# #     "Energy overhead per Client",
# #     fontsize=13,
# #     fontweight="bold",
# # )
# ax.legend(fontsize=10, loc="upper right")
# ax.grid(True, alpha=0.3)
# ax.spines["top"].set_visible(False)
# ax.spines["right"].set_visible(False)
# plt.savefig(f"analysis_output/{plot}", dpi=300, bbox_inches="tight")
# print(f"\nSaved: {plot}")

# Plot 3: Infrastructure percentage
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
ax = axes[0]

ax.plot(
    df["Z"],
    df["Fixed_infrastructure_kJ"] / df["K"],
    "o-",
    linewidth=2.5,
    markersize=9,
    label="Infrastructure energy",
    color="#2875E2",
    markeredgewidth=1.5,
    markeredgecolor="white",
)

ax2 = ax.twinx()
ax2.plot(
    df["Z"],
    df["Samples_per_round"] / df["K"],
    "s--",
    linewidth=2,
    markersize=8,
    label="Samples",
    color="#F77F00",
    alpha=0.7,
    markeredgewidth=1.5,
    markeredgecolor="white",
)

ax.set_xlabel("Number of Partitions (Z)", fontsize=12, fontweight="bold")
ax.set_ylabel(
    "Infrastructure Energy per Client (kJ)",
    fontsize=12,
    fontweight="bold",
    color="#2875E2",
)
ax2.set_ylabel("Samples per Client", fontsize=12, fontweight="bold", color="#F77F00")
ax.set_title(
    "Infrastructure Cost",
    fontsize=13,
    fontweight="bold",
)
ax.tick_params(axis="y", labelcolor="#06A77D")
ax2.tick_params(axis="y", labelcolor="#F77F00")

# Combine legends
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc="best", fontsize=10)

ax.grid(True, alpha=0.3)
ax.spines["top"].set_visible(False)
ax2.spines["top"].set_visible(False)

# Plot 4: Training energy detail
ax = axes[1]

ax.plot(
    df["Z"],
    df["Variable_training_kJ"] / df["K"],
    "o-",
    linewidth=2.5,
    markersize=9,
    label="Training energy",
    color="#06A77D",
    markeredgewidth=1.5,
    markeredgecolor="white",
)

ax2 = ax.twinx()
ax2.plot(
    df["Z"],
    df["Samples_per_round"] / df["K"],
    "s--",
    linewidth=2,
    markersize=8,
    label="Samples",
    color="#F77F00",
    alpha=0.7,
    markeredgewidth=1.5,
    markeredgecolor="white",
)

ax.set_xlabel("Number of Partitions (Z)", fontsize=12, fontweight="bold")
ax.set_ylabel(
    "Training Energy per Client (kJ)", fontsize=12, fontweight="bold", color="#06A77D"
)
ax2.set_ylabel("Samples per Client", fontsize=12, fontweight="bold", color="#F77F00")
ax.set_title(
    "Training Cost",
    fontsize=13,
    fontweight="bold",
)
ax.tick_params(axis="y", labelcolor="#06A77D")
ax2.tick_params(axis="y", labelcolor="#F77F00")

# Combine legends
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc="best", fontsize=10)

ax.grid(True, alpha=0.3)
ax.spines["top"].set_visible(False)
ax2.spines["top"].set_visible(False)

plt.tight_layout()
plt.savefig("analysis_output/overhead_and_samples", dpi=300, bbox_inches="tight")
