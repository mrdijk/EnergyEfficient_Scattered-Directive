import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Data from your results
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

# Calculate fixed and variable
df["Fixed_kJ"] = df["Infrastructure_kJ"] + df["Coordination_kJ"]
df["Variable_kJ"] = df["Training_kJ"] + df["Client_apps_kJ"]
df["Fixed_pct"] = (df["Fixed_kJ"] / df["Total_kJ"]) * 100
df["Variable_pct"] = (df["Variable_kJ"] / df["Total_kJ"]) * 100

# Calculate samples per round
TOTAL_SAMPLES = 531130
K = 5
df["Samples_per_partition"] = TOTAL_SAMPLES / df["Z"]
df["Samples_per_round"] = df["Samples_per_partition"] * K

# Energy per sample
df["Energy_per_sample_J"] = (df["Total_kJ"] * 1000) / df["Samples_per_round"]
df["Fixed_per_sample_J"] = (df["Fixed_kJ"] * 1000) / df["Samples_per_round"]
df["Variable_per_sample_J"] = (df["Variable_kJ"] * 1000) / df["Samples_per_round"]

print("=" * 100)
print("COMPLETE OVERHEAD ANALYSIS")
print("=" * 100)

print(
    f"\n{'Z':<6} {'Samples/Round':<15} {'Fixed (kJ)':<12} {'Variable (kJ)':<15} "
    f"{'Total (kJ)':<12} {'Fixed %':<10}"
)
print("-" * 100)

for idx, row in df.iterrows():
    print(
        f"{int(row['Z']):<6} {row['Samples_per_round']:<15,.0f} "
        f"{row['Fixed_kJ']:<12.2f} {row['Variable_kJ']:<15.2f} "
        f"{row['Total_kJ']:<12.2f} {row['Fixed_pct']:<10.1f}%"
    )

print("\n" + "=" * 100)
print("ENERGY PER SAMPLE BREAKDOWN")
print("=" * 100)

print(
    f"\n{'Z':<6} {'Fixed/Sample (J)':<20} {'Variable/Sample (J)':<22} "
    f"{'Total/Sample (J)':<20} {'Overhead %':<12}"
)
print("-" * 100)

for idx, row in df.iterrows():
    overhead_pct = (row["Fixed_per_sample_J"] / row["Energy_per_sample_J"]) * 100
    print(
        f"{int(row['Z']):<6} {row['Fixed_per_sample_J']:<20.4f} "
        f"{row['Variable_per_sample_J']:<22.4f} "
        f"{row['Energy_per_sample_J']:<20.4f} {overhead_pct:<12.1f}%"
    )

print("\n" + "=" * 100)
print("KEY INSIGHTS")
print("=" * 100)

avg_fixed = df["Fixed_kJ"].mean()
std_fixed = df["Fixed_kJ"].std()
avg_variable = df["Variable_kJ"].mean()
std_variable = df["Variable_kJ"].std()

print(f"\nFixed Overhead per Round:")
print(f"  Average: {avg_fixed:.2f} kJ")
print(f"  Std Dev: {std_fixed:.2f} kJ ({std_fixed / avg_fixed * 100:.1f}% variation)")
print(f"  Range: {df['Fixed_kJ'].min():.2f} - {df['Fixed_kJ'].max():.2f} kJ")
print(f"  → CONCLUSION: Fixed overhead is nearly constant (~{avg_fixed:.0f} kJ/round)")

print(f"\nVariable Work per Round:")
print(f"  Average: {avg_variable:.2f} kJ")
print(
    f"  Std Dev: {std_variable:.2f} kJ ({std_variable / avg_variable * 100:.1f}% variation)"
)
print(f"  Range: {df['Variable_kJ'].min():.2f} - {df['Variable_kJ'].max():.2f} kJ")
print(
    f"  Decrease from Z=15 to Z=400: {((df.loc[0, 'Variable_kJ'] - df.loc[10, 'Variable_kJ']) / df.loc[0, 'Variable_kJ'] * 100):.1f}%"
)

print(f"\nWhy Energy/Sample Increases with Z:")
print(
    f"  Z=15:  {avg_fixed:.0f} kJ fixed ÷ {df.loc[0, 'Samples_per_round']:,.0f} samples = {df.loc[0, 'Fixed_per_sample_J']:.2f} J/sample overhead"
)
print(
    f"  Z=400: {avg_fixed:.0f} kJ fixed ÷ {df.loc[10, 'Samples_per_round']:,.0f} samples = {df.loc[10, 'Fixed_per_sample_J']:.2f} J/sample overhead"
)
print(
    f"  → {df.loc[10, 'Fixed_per_sample_J'] / df.loc[0, 'Fixed_per_sample_J']:.1f}× more overhead per sample at Z=400!"
)

# Calculate the "pure" variable cost (what training actually costs)
print(f"\n" + "=" * 100)
print("PURE TRAINING COST (removing fixed overhead)")
print("=" * 100)

# If we remove fixed overhead, what's the actual cost of processing one sample?
print(f"\n{'Z':<6} {'Variable/Sample (J)':<25} {'Interpretation':<50}")
print("-" * 90)

for idx, row in df.iterrows():
    interpretation = ""
    if idx == 0:
        interpretation = "← Most samples = most accurate variable cost estimate"
    elif idx == len(df) - 1:
        interpretation = "← Fewest samples = least accurate estimate"

    print(
        f"{int(row['Z']):<6} {row['Variable_per_sample_J']:<25.6f} {interpretation:<50}"
    )

# Best estimate of pure variable cost
best_estimate_z = df.loc[0, "Z"]
best_estimate_cost = df.loc[0, "Variable_per_sample_J"]

print(f"\nBest estimate of PURE variable cost: {best_estimate_cost:.6f} J/sample")
print(f"  (from Z={int(best_estimate_z)}, which has the most samples per round)")

# Verify with theoretical calculation
print(f"\nVerification:")
print(f"  If variable cost is {best_estimate_cost:.6f} J/sample,")
print(f"  then at Z=400 with {df.loc[10, 'Samples_per_round']:,.0f} samples:")
print(
    f"    Expected variable energy = {best_estimate_cost * df.loc[10, 'Samples_per_round'] / 1000:.2f} kJ"
)
print(f"    Actual variable energy = {df.loc[10, 'Variable_kJ']:.2f} kJ")
print(
    f"    Difference: {abs(best_estimate_cost * df.loc[10, 'Samples_per_round'] / 1000 - df.loc[10, 'Variable_kJ']):.2f} kJ"
)

# Create visualizations
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Stacked bar showing fixed vs variable
ax = axes[0, 0]
x = np.arange(len(df))
width = 0.6

ax.bar(x, df["Fixed_kJ"], width, label="Fixed Overhead", color="coral", alpha=0.8)
ax.bar(
    x,
    df["Variable_kJ"],
    width,
    bottom=df["Fixed_kJ"],
    label="Variable Work",
    color="steelblue",
    alpha=0.8,
)

ax.set_xticks(x)
ax.set_xticklabels([f"Z={int(z)}" for z in df["Z"]], rotation=45)
ax.set_ylabel("Energy per Round (kJ)", fontsize=12, fontweight="bold")
ax.set_title(
    "Energy Breakdown: Fixed vs Variable\n(Fixed overhead is nearly constant)",
    fontsize=13,
    fontweight="bold",
)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis="y")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Add percentage labels
for i, (idx, row) in enumerate(df.iterrows()):
    total = row["Total_kJ"]
    ax.text(
        i,
        total + 1,
        f"{row['Fixed_pct']:.0f}%",
        ha="center",
        fontsize=9,
        fontweight="bold",
    )

# Plot 2: Energy per sample breakdown
ax = axes[0, 1]

ax.plot(
    df["Z"],
    df["Fixed_per_sample_J"],
    "o-",
    linewidth=2.5,
    markersize=9,
    label="Fixed overhead/sample",
    color="coral",
    markeredgewidth=1.5,
    markeredgecolor="white",
)
ax.plot(
    df["Z"],
    df["Variable_per_sample_J"],
    "s-",
    linewidth=2.5,
    markersize=9,
    label="Variable cost/sample",
    color="steelblue",
    markeredgewidth=1.5,
    markeredgecolor="white",
)
ax.plot(
    df["Z"],
    df["Energy_per_sample_J"],
    "^-",
    linewidth=3,
    markersize=10,
    label="Total energy/sample",
    color="green",
    markeredgewidth=1.5,
    markeredgecolor="white",
    alpha=0.7,
)

ax.set_xlabel("Number of Partitions (Z)", fontsize=12, fontweight="bold")
ax.set_ylabel("Energy per Sample (J)", fontsize=12, fontweight="bold")
ax.set_title(
    "Why Smaller Partitions Are Less Efficient\n(Fixed overhead dominates as samples decrease)",
    fontsize=13,
    fontweight="bold",
)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Plot 3: Fixed overhead percentage
ax = axes[1, 0]

colors = ["green" if x < 60 else "orange" if x < 62 else "red" for x in df["Fixed_pct"]]
ax.bar(range(len(df)), df["Fixed_pct"], color=colors, alpha=0.7)
ax.set_xticks(range(len(df)))
ax.set_xticklabels([f"Z={int(z)}" for z in df["Z"]], rotation=45)
ax.set_ylabel("Fixed Overhead (%)", fontsize=12, fontweight="bold")
ax.set_title(
    "Fixed Overhead as Percentage of Total Energy\n(Higher Z = more overhead)",
    fontsize=13,
    fontweight="bold",
)
ax.axhline(
    y=60, color="orange", linestyle="--", alpha=0.5, linewidth=2, label="60% threshold"
)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis="y")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Plot 4: Component breakdown
ax = axes[1, 1]

ax.plot(
    df["Z"],
    df["Infrastructure_kJ"],
    "o-",
    linewidth=2.5,
    markersize=8,
    label="Infrastructure (sidecar/linkerd)",
    color="#E63946",
    markeredgewidth=1.5,
    markeredgecolor="white",
)
ax.plot(
    df["Z"],
    df["Coordination_kJ"],
    "s-",
    linewidth=2.5,
    markersize=8,
    label="Coordination (api-gateway/orchestrator)",
    color="#F77F00",
    markeredgewidth=1.5,
    markeredgecolor="white",
)
ax.plot(
    df["Z"],
    df["Training_kJ"],
    "^-",
    linewidth=2.5,
    markersize=8,
    label="Training (hfl-train)",
    color="#06A77D",
    markeredgewidth=1.5,
    markeredgecolor="white",
)
ax.plot(
    df["Z"],
    df["Client_apps_kJ"],
    "d-",
    linewidth=2.5,
    markersize=8,
    label="Client apps",
    color="#4EA8DE",
    markeredgewidth=1.5,
    markeredgecolor="white",
)

ax.set_xlabel("Number of Partitions (Z)", fontsize=12, fontweight="bold")
ax.set_ylabel("Energy per Round (kJ)", fontsize=12, fontweight="bold")
ax.set_title(
    "Energy by Component Type\n(Infrastructure dominates)",
    fontsize=13,
    fontweight="bold",
)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig(
    "analysis_output/overhead_component_analysis.png", dpi=300, bbox_inches="tight"
)
print("\nSaved: overhead_component_analysis.png")
