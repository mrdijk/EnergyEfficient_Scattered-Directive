import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Data from your results (corrected categorization)
data = {
    "Z": [15, 30, 60, 90, 120, 150, 190, 230, 260, 300, 400],
    "Infrastructure_kJ": [
        36.84,
        35.26,
        34.85,
        34.01,
        33.89,
        34.06,
        33.93,
        33.80,
        33.72,
        33.69,
        33.75,
    ],
    "Coordination_kJ": [
        1.42,
        1.38,
        1.34,
        1.34,
        1.32,
        1.35,
        1.34,
        1.35,
        1.34,
        1.33,
        1.34,
    ],
    "Training_kJ": [11.03, 9.45, 8.53, 8.29, 8.17, 8.04, 8.10, 8.04, 7.93, 7.97, 7.98],
    "Client_apps_kJ": [
        15.33,
        14.79,
        14.51,
        14.29,
        14.18,
        14.12,
        14.24,
        14.15,
        14.08,
        14.08,
        14.08,
    ],
    "Total_kJ": [
        64.38,
        60.64,
        59.01,
        57.69,
        57.33,
        57.34,
        57.38,
        57.10,
        56.85,
        56.84,
        56.92,
    ],
}

df = pd.DataFrame(data)

# CORRECTED: Client apps are infrastructure, not training
df["Fixed_infrastructure_kJ"] = (
    df["Infrastructure_kJ"] + df["Coordination_kJ"] + df["Client_apps_kJ"]
)
df["Variable_training_kJ"] = df["Training_kJ"]  # Only hfl-train and hfl-train-model

df["Fixed_pct"] = (df["Fixed_infrastructure_kJ"] / df["Total_kJ"]) * 100
df["Variable_pct"] = (df["Variable_training_kJ"] / df["Total_kJ"]) * 100

# Calculate samples per round
TOTAL_SAMPLES = 531130
K = 5
df["Samples_per_partition"] = TOTAL_SAMPLES / df["Z"]
df["Samples_per_round"] = df["Samples_per_partition"] * K

# Energy per sample
df["Energy_per_sample_J"] = (df["Total_kJ"] * 1000) / df["Samples_per_round"]
df["Fixed_per_sample_J"] = (df["Fixed_infrastructure_kJ"] * 1000) / df[
    "Samples_per_round"
]
df["Variable_per_sample_J"] = (df["Variable_training_kJ"] * 1000) / df[
    "Samples_per_round"
]

df["Energy_per_Client"] = (df["Total_kJ"] * 1000) / 5
df["Fixed_energy_per_Client"] = (df["Fixed_infrastructure_kJ"] * 1000) / 5
df["Var_Energy_per_Client"] = (df["Variable_training_kJ"] * 1000) / 5

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
    f"\n{'Z':<6} {'Samples/Round':<15} {'Infrastructure (kJ)':<20} {'Training (kJ)':<15} "
    f"{'Total (kJ)':<12} {'Infra %':<10}"
)
print("-" * 100)

for idx, row in df.iterrows():
    print(
        f"{int(row['Z']):<6} {row['Samples_per_round']:<15,.0f} "
        f"{row['Fixed_infrastructure_kJ']:<20.2f} {row['Variable_training_kJ']:<15.2f} "
        f"{row['Total_kJ']:<12.2f} {row['Fixed_pct']:<10.1f}%"
    )

print("\n" + "=" * 100)
print("ENERGY PER SAMPLE BREAKDOWN (CORRECTED)")
print("=" * 100)

print(
    f"\n{'Z':<6} {'Infrastructure/Sample (J)':<28} {'Training/Sample (J)':<24} "
    f"{'Total/Sample (J)':<20} {'Infra %':<12}"
)
print("-" * 110)

for idx, row in df.iterrows():
    overhead_pct = (row["Fixed_per_sample_J"] / row["Energy_per_sample_J"]) * 100
    print(
        f"{int(row['Z']):<6} {row['Fixed_per_sample_J']:<28.4f} "
        f"{row['Variable_per_sample_J']:<24.4f} "
        f"{row['Energy_per_sample_J']:<20.4f} {overhead_pct:<12.1f}%"
    )

print("\n" + "=" * 100)
print("CORRECTED KEY INSIGHTS")
print("=" * 100)

avg_fixed = df["Fixed_infrastructure_kJ"].mean()
std_fixed = df["Fixed_infrastructure_kJ"].std()
avg_training = df["Variable_training_kJ"].mean()
std_training = df["Variable_training_kJ"].std()

print(f"\nFixed Infrastructure per Round:")
print(f"  Average: {avg_fixed:.2f} kJ")
print(f"  Std Dev: {std_fixed:.2f} kJ ({std_fixed / avg_fixed * 100:.1f}% variation)")
print(
    f"  Range: {df['Fixed_infrastructure_kJ'].min():.2f} - {df['Fixed_infrastructure_kJ'].max():.2f} kJ"
)
print(f"  → Infrastructure is EXTREMELY constant (~{avg_fixed:.0f} kJ/round)")

print(f"\nActual Training Work per Round:")
print(f"  Average: {avg_training:.2f} kJ")
print(
    f"  Std Dev: {std_training:.2f} kJ ({std_training / avg_training * 100:.1f}% variation)"
)
print(
    f"  Range: {df['Variable_training_kJ'].min():.2f} - {df['Variable_training_kJ'].max():.2f} kJ"
)
print(
    f"  Decrease from Z=15 to Z=400: {((df.loc[0, 'Variable_training_kJ'] - df.loc[10, 'Variable_training_kJ']) / df.loc[0, 'Variable_training_kJ'] * 100):.1f}%"
)

print(f"\nInfrastructure Dominance:")
print(
    f"  Infrastructure: {avg_fixed:.2f} kJ/round ({avg_fixed / (avg_fixed + avg_training) * 100:.1f}% of total)"
)
print(
    f"  Training: {avg_training:.2f} kJ/round ({avg_training / (avg_fixed + avg_training) * 100:.1f}% of total)"
)
print(
    f"  → Infrastructure uses {avg_fixed / avg_training:.1f}× more energy than actual training!"
)

print(f"\nWhy Energy/Sample Explodes with High Z:")
print(
    f"  Z=15:  {avg_fixed:.0f} kJ infrastructure ÷ {df.loc[0, 'Samples_per_round']:,.0f} samples = {df.loc[0, 'Fixed_per_sample_J']:.2f} J/sample overhead"
)
print(
    f"  Z=400: {avg_fixed:.0f} kJ infrastructure ÷ {df.loc[10, 'Samples_per_round']:,.0f} samples = {df.loc[10, 'Fixed_per_sample_J']:.2f} J/sample overhead"
)
print(
    f"  → {df.loc[10, 'Fixed_per_sample_J'] / df.loc[0, 'Fixed_per_sample_J']:.1f}× more overhead per sample at Z=400!"
)

print(f"\nPure Training Cost (per sample):")
best_z15_cost = df.loc[0, "Variable_per_sample_J"]
print(f"  At Z=15 (most samples): {best_z15_cost:.6f} J/sample")
print(f"  At Z=400 (least samples): {df.loc[10, 'Variable_per_sample_J']:.6f} J/sample")
print(f"  Average across all Z: {df['Variable_per_sample_J'].mean():.6f} J/sample")
print(
    f"  → True cost of training one sample: ~{df['Variable_per_sample_J'].mean():.4f} J"
)

print("\n" + "=" * 100)
print("THE SHOCKING TRUTH")
print("=" * 100)

print(f"\nAt Z=400 (worst case):")
print(f"  Total energy per sample: {df.loc[10, 'Energy_per_sample_J']:.2f} J")
print(
    f"  Infrastructure overhead: {df.loc[10, 'Fixed_per_sample_J']:.2f} J ({df.loc[10, 'Fixed_per_sample_J'] / df.loc[10, 'Energy_per_sample_J'] * 100:.1f}%)"
)
print(
    f"  Actual training: {df.loc[10, 'Variable_per_sample_J']:.2f} J ({df.loc[10, 'Variable_per_sample_J'] / df.loc[10, 'Energy_per_sample_J'] * 100:.1f}%)"
)
print(
    f"\n  → You spend {df.loc[10, 'Fixed_per_sample_J'] / df.loc[10, 'Variable_per_sample_J']:.1f}× more energy on infrastructure than on actual training!"
)

print(f"\nAt Z=15 (best case):")
print(f"  Total energy per sample: {df.loc[0, 'Energy_per_sample_J']:.2f} J")
print(
    f"  Infrastructure overhead: {df.loc[0, 'Fixed_per_sample_J']:.2f} J ({df.loc[0, 'Fixed_per_sample_J'] / df.loc[0, 'Energy_per_sample_J'] * 100:.1f}%)"
)
print(
    f"  Actual training: {df.loc[0, 'Variable_per_sample_J']:.2f} J ({df.loc[0, 'Variable_per_sample_J'] / df.loc[0, 'Energy_per_sample_J'] * 100:.1f}%)"
)
print(
    f"\n  → Much better ratio: {df.loc[0, 'Fixed_per_sample_J'] / df.loc[0, 'Variable_per_sample_J']:.1f}× infrastructure vs training"
)

# Create corrected visualizations
fig, ax = plt.subplots()

# Plot 1: Stacked bar showing infrastructure vs training
# ax = axes[0, 0]
# x = np.arange(len(df))
# width = 0.6

# ax.bar(
#     x,
#     df["Fixed_infrastructure_kJ"],
#     width,
#     label="Infrastructure Overhead",
#     color="#E63946",
#     alpha=0.8,
# )
# ax.bar(
#     x,
#     df["Variable_training_kJ"],
#     width,
#     bottom=df["Fixed_infrastructure_kJ"],
#     label="Actual Training",
#     color="#06A77D",
#     alpha=0.8,
# )

# ax.set_xticks(x)
# ax.set_xticklabels([f"Z={int(z)}" for z in df["Z"]], rotation=45)
# ax.set_ylabel("Energy per Round (kJ)", fontsize=12, fontweight="bold")
# ax.set_title(
#     "Energy Breakdown: Infrastructure vs Training\n(Infrastructure dominates at ~86% of total)",
#     fontsize=13,
#     fontweight="bold",
# )
# ax.legend(fontsize=11)
# ax.grid(True, alpha=0.3, axis="y")
# ax.spines["top"].set_visible(False)
# ax.spines["right"].set_visible(False)

# # Add percentage labels
# for i, (idx, row) in enumerate(df.iterrows()):
#     total = row["Total_kJ"]
#     ax.text(
#         i,
#         total + 1,
#         f"{row['Fixed_pct']:.0f}%\ninfra",
#         ha="center",
#         fontsize=8,
#         fontweight="bold",
#     )

# Plot 2: Energy per sample breakdown
# ax = axes[0, 1]

ax.plot(
    df["Z"],
    df["Fixed_energy_per_Client"],
    "o-",
    linewidth=2.5,
    markersize=9,
    label="DYNAMOS overhead/client",
    color="#2875E2",
    markeredgewidth=1.5,
    markeredgecolor="white",
)
ax.plot(
    df["Z"],
    df["Var_Energy_per_Client"],
    "s-",
    linewidth=2.5,
    markersize=9,
    label="Training cost/client",
    color="#06A77D",
    markeredgewidth=1.5,
    markeredgecolor="white",
)
ax.plot(
    df["Z"],
    df["Energy_per_Client"],
    "^-",
    linewidth=3,
    markersize=10,
    label="Total energy/client",
    color="#E63946",
    markeredgewidth=1.5,
    markeredgecolor="white",
    alpha=0.7,
)

ax.set_xlabel("Number of Partitions (Z)", fontsize=12, fontweight="bold")
ax.set_ylabel("Energy per Client (kJ)", fontsize=12, fontweight="bold")
ax.set_title(
    "Energy overhead per Client",
    fontsize=13,
    fontweight="bold",
)
ax.legend(fontsize=10, loc="upper right")
ax.grid(True, alpha=0.3)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Plot 3: Infrastructure percentage
# ax = axes[1, 0]

# colors = ["green" if x < 83 else "orange" if x < 85 else "red" for x in df["Fixed_pct"]]
# bars = ax.bar(
#     range(len(df)),
#     df["Fixed_pct"],
#     color=colors,
#     alpha=0.7,
#     edgecolor="white",
#     linewidth=1.5,
# )

# # Add value labels on bars
# for i, (bar, pct) in enumerate(zip(bars, df["Fixed_pct"])):
#     height = bar.get_height()
#     ax.text(
#         bar.get_x() + bar.get_width() / 2.0,
#         height + 0.3,
#         f"{pct:.1f}%",
#         ha="center",
#         va="bottom",
#         fontsize=9,
#         fontweight="bold",
#     )

# ax.set_xticks(range(len(df)))
# ax.set_xticklabels([f"Z={int(z)}" for z in df["Z"]], rotation=45)
# ax.set_ylabel("Infrastructure Overhead (%)", fontsize=12, fontweight="bold")
# ax.set_title(
#     "Infrastructure as Percentage of Total Energy\n(~86% of energy wasted on infrastructure!)",
#     fontsize=13,
#     fontweight="bold",
# )
# ax.axhline(
#     y=85, color="red", linestyle="--", alpha=0.5, linewidth=2, label="85% threshold"
# )
# ax.set_ylim([80, 88])
# ax.legend(fontsize=10)
# ax.grid(True, alpha=0.3, axis="y")
# ax.spines["top"].set_visible(False)
# ax.spines["right"].set_visible(False)

# # Plot 4: Training energy detail
# ax = axes[1, 1]

# ax.plot(
#     df["Z"],
#     df["Variable_training_kJ"],
#     "o-",
#     linewidth=2.5,
#     markersize=9,
#     label="Training energy per round",
#     color="#06A77D",
#     markeredgewidth=1.5,
#     markeredgecolor="white",
# )

# ax2 = ax.twinx()
# ax2.plot(
#     df["Z"],
#     df["Samples_per_round"],
#     "s--",
#     linewidth=2,
#     markersize=8,
#     label="Samples per round",
#     color="#F77F00",
#     alpha=0.7,
#     markeredgewidth=1.5,
#     markeredgecolor="white",
# )

# ax.set_xlabel("Number of Partitions (Z)", fontsize=12, fontweight="bold")
# ax.set_ylabel("Training Energy (kJ)", fontsize=12, fontweight="bold", color="#06A77D")
# ax2.set_ylabel("Samples per Round", fontsize=12, fontweight="bold", color="#F77F00")
# ax.set_title(
#     "Training Energy Decreases as Partition Size Shrinks\n(Less data to process per round)",
#     fontsize=13,
#     fontweight="bold",
# )
# ax.tick_params(axis="y", labelcolor="#06A77D")
# ax2.tick_params(axis="y", labelcolor="#F77F00")

# # Combine legends
# lines1, labels1 = ax.get_legend_handles_labels()
# lines2, labels2 = ax2.get_legend_handles_labels()
# ax.legend(lines1 + lines2, labels1 + labels2, loc="best", fontsize=10)

# ax.grid(True, alpha=0.3)
# ax.spines["top"].set_visible(False)
# ax2.spines["top"].set_visible(False)

plt.tight_layout()
plt.savefig("analysis_output/per_client_overhead.png", dpi=300, bbox_inches="tight")
print("\nSaved: overhead_corrected_analysis.png")

# Export corrected data
# df.to_csv("analysis_output/overhead_corrected_breakdown.csv", index=False)
# print("Saved: overhead_corrected_breakdown.csv")

# Summary statistics table
print("\n" + "=" * 100)
print("SUMMARY TABLE FOR PAPER")
print("=" * 100)

summary_df = df[
    [
        "Z",
        "Samples_per_round",
        "Fixed_infrastructure_kJ",
        "Variable_training_kJ",
        "Total_kJ",
        "Fixed_pct",
        "Energy_per_sample_J",
    ]
].copy()
summary_df.columns = [
    "Z",
    "Samples/Round",
    "Infrastructure (kJ)",
    "Training (kJ)",
    "Total (kJ)",
    "Infra %",
    "J/Sample",
]

print(summary_df.to_string(index=False))
