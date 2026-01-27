import constants
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# from matplotlib import cm

# Load your experiment data
OUTPUT_DIR = "analysis_output"
global_data = pd.read_csv(f"{OUTPUT_DIR}/experiment_summary.csv")
client_data = pd.read_csv(f"{OUTPUT_DIR}/all_client_stats.csv")
energy_data = pd.read_csv(f"{OUTPUT_DIR}/all_energy_stats.csv")
SORTED_CLIENTS = sorted(constants.DATA_PROVIDERS.items(), key=lambda x: x[1])

# Filter out bad experiments if needed
bad_experiments = ["13-01-26-1165414"]
for bad_exp in bad_experiments:
    if "experiment" in global_data.columns:
        global_data = global_data[
            ~global_data["experiment"].str.contains(bad_exp, na=False)
        ].copy()


# Filter experiments with low energy
energy_threshold = 100
if "total_energy_joules" in global_data.columns:
    global_data = global_data[
        global_data["total_energy_joules"] > energy_threshold
    ].copy()

# Prepare data for grouped bar chart
# Group by clients and rounds
grouped = (
    global_data.groupby(["clients", "rounds"])["total_energy_joules"]
    .mean()
    .reset_index()
)
grouped["energy_kj"] = grouped["total_energy_joules"] / 1000

# Get unique values
unique_rounds = sorted(grouped["rounds"].unique())
unique_clients = sorted(grouped["clients"].unique())

print("=" * 80)
print("GROUPED BAR CHART DATA")
print("=" * 80)
print(f"Client configurations: {unique_clients}")
print(f"Round configurations: {unique_rounds}")
print("\nData overview:")
print(grouped[["clients", "rounds", "energy_kj"]].to_string(index=False))

# Create the grouped bar chart
fig, ax = plt.subplots(figsize=(12, 7))

# Set the width of bars and positions
x = np.arange(len(unique_rounds))
width = 0.8 / len(unique_clients)  # Width of each bar

# Colors for different client counts
colors = ["#3b82f6", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6", "#ec4899"]

# Plot bars for each client configuration
for i, n_clients in enumerate(unique_clients):
    # Get energy values for this client count across all rounds
    values = []
    for n_rounds in unique_rounds:
        data_point = grouped[
            (grouped["clients"] == n_clients) & (grouped["rounds"] == n_rounds)
        ]
        if not data_point.empty:
            values.append(data_point["energy_kj"].values[0])
        else:
            values.append(0)

        errors = np.std(values)
    # Calculate position for this group of bars
    position = x + (i - len(unique_clients) / 2 + 0.5) * width
    # Create bars
    bars = ax.bar(
        position,
        values,
        width,
        yerr=errors,
        capsize=8,
        label=f"{int(n_clients)} Clients",
        color=colors[i % len(colors)],
        alpha=0.85,
        edgecolor="white",
        linewidth=1.5,
    )

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 4.0,
                height,
                f"{height:.0f}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

# Customize the chart
ax.set_xlabel("Number of Rounds", fontsize=14, fontweight="bold", labelpad=10)
ax.set_ylabel("Energy Consumption (kJ)", fontsize=14, fontweight="bold", labelpad=10)
ax.set_title("Total Energy Consumption\n", fontsize=16, fontweight="bold", pad=20)

# Set x-axis
ax.set_xticks(x)
ax.set_xticklabels([f"{int(r)} Rounds" for r in unique_rounds], fontsize=12)

# Customize y-axis
ax.tick_params(axis="y", labelsize=11)

# Add grid
ax.grid(axis="y", alpha=0.3, linestyle="--", linewidth=0.8)
ax.set_axisbelow(True)

# Add legend
ax.legend(loc="upper left", fontsize=11, framealpha=0.95, edgecolor="gray")

# Add background color
ax.set_facecolor("#f9fafb")
fig.set_facecolor("white")

plt.tight_layout()
plt.savefig(
    f"{OUTPUT_DIR}/energy_grouped_bar_chart.png",
    dpi=300,
    bbox_inches="tight",
    facecolor="white",
)
plt.show()

# Create the grouped bar chart
fig, ax = plt.subplots(figsize=(12, 7))

print("=" * 80)
print("GROUPED BAR CHART PER ROW")
print("=" * 80)
print(f"Client configurations: {unique_clients}")
print(f"Round configurations: {unique_rounds}")
print("\nData overview:")
for i, n_clients in enumerate(unique_clients):
    n_rows = np.sum([size for _, size in SORTED_CLIENTS[:n_clients]])
    grouped["energy_per_row"] = grouped["total_energy_joules"] / n_rows

print(grouped[["clients", "rounds", "energy_per_row"]].to_string(index=False))

# Create the grouped bar chart
fig, ax = plt.subplots(figsize=(12, 7))

# Set the width of bars and positions
x = np.arange(len(unique_rounds))
width = 0.8 / len(unique_clients)  # Width of each bar

# Colors for different client counts
colors = ["#3b82f6", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6", "#ec4899"]

# Plot bars for each client configuration
for i, n_clients in enumerate(unique_clients):
    n_rows = np.sum([size for _, size in SORTED_CLIENTS[:n_clients]])
    # Get energy values for this client count across all rounds
    # number_of_clients = len(unique_clients)
    values = []
    for n_rounds in unique_rounds:
        data_point = grouped[
            (grouped["clients"] == n_clients) & (grouped["rounds"] == n_rounds)
        ]
        if not data_point.empty:
            values.append((data_point["total_energy_joules"].values[0] / n_rows) * 1000)
            # values.append(data_point["energy_per_row"].values[0] * 1000)
        else:
            values.append(0)
        errors = np.std(values)
    # Calculate position for this group of bars
    position = x + (i - len(unique_clients) / 2 + 0.5) * width

    # Create bars
    bars = ax.bar(
        position,
        values,
        width,
        yerr=errors,
        capsize=8,
        label=f"{int(n_clients)} Clients",
        color=colors[i % len(colors)],
        alpha=0.85,
        edgecolor="white",
        linewidth=1.5,
    )

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 4.0,
                height,
                f"{height:.0f}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

# Customize the chart
ax.set_xlabel("Number of Rounds", fontsize=14, fontweight="bold", labelpad=10)
ax.set_ylabel("Energy Consumption (kJ)", fontsize=14, fontweight="bold", labelpad=10)
ax.set_title("Energy Consumption per 1k rows\n", fontsize=16, fontweight="bold", pad=20)

# Set x-axis
ax.set_xticks(x)
ax.set_xticklabels([f"{int(r)} Rounds" for r in unique_rounds], fontsize=12)

# Customize y-axis
ax.tick_params(axis="y", labelsize=11)

# Add grid
ax.grid(axis="y", alpha=0.3, linestyle="--", linewidth=0.8)
ax.set_axisbelow(True)

# Add legend
ax.legend(loc="upper left", fontsize=11, framealpha=0.95, edgecolor="gray")

# Add background color
ax.set_facecolor("#f9fafb")
fig.set_facecolor("white")

plt.tight_layout()
plt.savefig(
    f"{OUTPUT_DIR}/grouped_bar_chart_per_row.png",
    dpi=300,
    bbox_inches="tight",
    facecolor="white",
)
plt.show()

# Print detailed statistics
print("\n" + "=" * 80)
print("DETAILED STATISTICS")
print("=" * 80)

for n_rounds in unique_rounds:
    print(f"\n{int(n_rounds)} Rounds:")
    round_data = grouped[grouped["rounds"] == n_rounds]
    for _, row in round_data.iterrows():
        print(f"  {int(row['clients'])} Clients: {row['energy_kj']:.2f} kJ")

print("\n" + "=" * 80)
print("ENERGY SCALING ANALYSIS")
print("=" * 80)

# Analyze scaling with rounds (keeping clients constant)
print("\nEnergy scaling with ROUNDS (for each client config):")
for n_clients in unique_clients:
    client_data = grouped[grouped["clients"] == n_clients].sort_values("rounds")
    if len(client_data) > 1:
        energies = client_data["energy_kj"].values
        rounds = client_data["rounds"].values
        energy_per_round = energies / rounds
        print(f"\n{int(n_clients)} Clients:")
        for i, (r, e, epr) in enumerate(zip(rounds, energies, energy_per_round)):
            print(f"  {int(r)} rounds: {e:.2f} kJ total, {epr:.2f} kJ/round")

        if len(energies) > 1:
            avg_energy_per_round = np.mean(energy_per_round)
            std_energy_per_round = np.std(energy_per_round)
            print(
                f"  → Avg energy/round: {avg_energy_per_round:.2f} ± {std_energy_per_round:.2f} kJ"
            )
            print(
                f"  → Linearity: {'Good (±{:.1f}%)'.format(std_energy_per_round / avg_energy_per_round * 100) if std_energy_per_round / avg_energy_per_round < 0.1 else 'Variable (±{:.1f}%)'.format(std_energy_per_round / avg_energy_per_round * 100)}"
            )

# Analyze scaling with clients (keeping rounds constant)
print("\nEnergy scaling with CLIENTS (for each round config):")
for n_rounds in unique_rounds:
    round_data = grouped[grouped["rounds"] == n_rounds].sort_values("clients")
    if len(round_data) > 1:
        energies = round_data["energy_kj"].values
        clients = round_data["clients"].values
        energy_per_client = energies / clients
        print(f"\n{int(n_rounds)} Rounds:")
        for i, (c, e, epc) in enumerate(zip(clients, energies, energy_per_client)):
            print(f"  {int(c)} clients: {e:.2f} kJ total, {epc:.2f} kJ/client")

        if len(energies) > 1:
            avg_energy_per_client = np.mean(energy_per_client)
            std_energy_per_client = np.std(energy_per_client)
            print(
                f"  → Avg energy/client: {avg_energy_per_client:.2f} ± {std_energy_per_client:.2f} kJ"
            )
            print(
                f"  → Linearity: {'Good (±{:.1f}%)'.format(std_energy_per_client / avg_energy_per_client * 100) if std_energy_per_client / avg_energy_per_client < 0.1 else 'Variable (±{:.1f}%)'.format(std_energy_per_client / avg_energy_per_client * 100)}"
            )
