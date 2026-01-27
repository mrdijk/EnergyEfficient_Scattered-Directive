import constants
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Load your experiment data
OUTPUT_DIR = "analysis_output"
master = pd.read_csv(f"{OUTPUT_DIR}/experiment_summary.csv")
global_data = pd.read_csv(f"{OUTPUT_DIR}/all_global_stats.csv")
client_data = pd.read_csv(f"{OUTPUT_DIR}/all_client_stats.csv")
energy_data = pd.read_csv(f"{OUTPUT_DIR}/all_energy_stats.csv")
SORTED_CLIENTS = sorted(constants.DATA_PROVIDERS.items(), key=lambda x: x[1])

# Filter out bad experiments if needed
bad_experiments = ["13-01-26-1165414"]
for bad_exp in bad_experiments:
    if "experiment" in master.columns:
        master = master[~master["experiment"].str.contains(bad_exp, na=False)].copy()

# Filter experiments with low energy
energy_threshold = 100
if "total_energy_joules" in master.columns:
    master = master[master["total_energy_joules"] > energy_threshold].copy()

# Add experiment size column if not present
if "exp_size" not in master.columns:
    master["exp_size"] = master["experiment"].str.extract(r"_(small|large)")

# Prepare data for grouped bar chart - now including exp_size
grouped = (
    master.groupby(["clients", "rounds", "exp_size"])["total_energy_joules"]
    .mean()
    .reset_index()
)

grouped["energy_kj"] = grouped["total_energy_joules"] / 1000

# Get unique values
unique_rounds = sorted(grouped["rounds"].unique())
unique_clients = sorted(grouped["clients"].unique())
unique_sizes = sorted(grouped["exp_size"].unique())

print("=" * 80)
print("GROUPED BAR CHART DATA")
print("=" * 80)
print(f"Client configurations: {unique_clients}")
print(f"Round configurations: {unique_rounds}")
print(f"Experiment sizes: {unique_sizes}")
print("\nData overview:")
print(grouped[["clients", "rounds", "exp_size", "energy_kj"]].to_string(index=False))

# Create the grouped bar chart
fig, ax = plt.subplots(figsize=(14, 7))

# Set the width of bars and positions
x = np.arange(len(unique_rounds))
width = 0.8 / (len(unique_clients) * len(unique_sizes))  # Width of each bar

# Colors for different client counts
colors = ["#3b82f6", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6", "#ec4899"]

# Plot bars for each client configuration and size
bar_index = 0
for i, n_clients in enumerate(unique_clients):
    for j, exp_size in enumerate(unique_sizes):
        # Get energy values for this client count and size across all rounds
        values = []
        errors = []
        for n_rounds in unique_rounds:
            data_point = grouped[
                (grouped["clients"] == n_clients)
                & (grouped["rounds"] == n_rounds)
                & (grouped["exp_size"] == exp_size)
            ]
            if not data_point.empty:
                values.append(data_point["energy_kj"].values[0])
                # Calculate std from raw data if available
                raw_data = master[
                    (master["clients"] == n_clients)
                    & (master["rounds"] == n_rounds)
                    & (master["exp_size"] == exp_size)
                ]["total_energy_joules"]
                errors.append((raw_data.std() / 1000) if len(raw_data) > 1 else 0)
            else:
                values.append(0)
                errors.append(0)

        # Calculate position for this group of bars
        position = (
            x
            + (bar_index - (len(unique_clients) * len(unique_sizes)) / 2 + 0.5) * width
        )

        # Create bars with different alpha for small vs large
        alpha = 0.85 if exp_size == "large" else 0.5
        label = f"{int(n_clients)} Clients - {exp_size}"

        bars = ax.bar(
            position,
            values,
            width,
            yerr=errors,
            capsize=3,
            label=label,
            color=colors[i % len(colors)],
            alpha=alpha,
            edgecolor="white",
            linewidth=1.5,
        )

        # Add value labels on bars
        for bar, err in zip(bars, errors):
            height = bar.get_height()
            if height > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + err + (max(values) * 0.02) if max(values) > 0 else 0,
                    f"{height:.0f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    fontweight="bold",
                )

        bar_index += 1

# Customize the chart
ax.set_xlabel("Number of Rounds", fontsize=14, fontweight="bold", labelpad=10)
ax.set_ylabel("Energy Consumption (kJ)", fontsize=14, fontweight="bold", labelpad=10)
ax.set_title(
    "Total Energy Consumption by Size\n", fontsize=16, fontweight="bold", pad=20
)

# Set x-axis
ax.set_xticks(x)
ax.set_xticklabels([f"{int(r)} Rounds" for r in unique_rounds], fontsize=12)

# Customize y-axis
ax.tick_params(axis="y", labelsize=11)

# Add grid
ax.grid(axis="y", alpha=0.3, linestyle="--", linewidth=0.8)
ax.set_axisbelow(True)

# Add legend
ax.legend(loc="upper left", fontsize=9, framealpha=0.95, edgecolor="gray", ncol=2)

# Add background color
ax.set_facecolor("#f9fafb")
fig.set_facecolor("white")

plt.tight_layout()
plt.savefig(
    f"{OUTPUT_DIR}/energy_grouped_bar_chart_by_size.png",
    dpi=300,
    bbox_inches="tight",
    facecolor="white",
)
plt.show()

# Create the grouped bar chart for energy per 1000 rows
fig, ax = plt.subplots(figsize=(14, 7))

print("=" * 80)
print("GROUPED BAR CHART PER 1000 ROWS")
print("=" * 80)
print(f"Client configurations: {unique_clients}")
print(f"Round configurations: {unique_rounds}")
print(f"Experiment sizes: {unique_sizes}")
print("\nData overview:")

# Calculate energy per 1000 rows for each group
grouped_per_row = grouped.copy()
for i, row in grouped_per_row.iterrows():
    n_clients = int(row["clients"])
    n_rows = np.sum([size for _, size in SORTED_CLIENTS[:n_clients]])
    grouped_per_row.at[i, "energy_per_1k_rows"] = (row["energy_kj"] / n_rows) * 1000

print(
    grouped_per_row[["clients", "rounds", "exp_size", "energy_per_1k_rows"]].to_string(
        index=False
    )
)

# Set the width of bars and positions
x = np.arange(len(unique_rounds))
width = 0.8 / (len(unique_clients) * len(unique_sizes))  # Width of each bar

# Colors for different client counts
colors = ["#3b82f6", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6", "#ec4899"]

# Plot bars for each client configuration and size
bar_index = 0
for i, n_clients in enumerate(unique_clients):
    n_rows = np.sum([size for _, size in SORTED_CLIENTS[:n_clients]])

    for j, exp_size in enumerate(unique_sizes):
        # Get energy values for this client count and size across all rounds
        values = []
        errors = []
        for n_rounds in unique_rounds:
            data_point = grouped_per_row[
                (grouped_per_row["clients"] == n_clients)
                & (grouped_per_row["rounds"] == n_rounds)
                & (grouped_per_row["exp_size"] == exp_size)
            ]
            if not data_point.empty:
                values.append(data_point["energy_per_1k_rows"].values[0])
                # Calculate std from raw data if available
                raw_data = master[
                    (master["clients"] == n_clients)
                    & (master["rounds"] == n_rounds)
                    & (master["exp_size"] == exp_size)
                ]["total_energy_joules"]
                # Convert std to per 1k rows
                errors.append(
                    ((raw_data.std() / 1000) / n_rows * 1000)
                    if len(raw_data) > 1
                    else 0
                )
            else:
                values.append(0)
                errors.append(0)

        # Calculate position for this group of bars
        position = (
            x
            + (bar_index - (len(unique_clients) * len(unique_sizes)) / 2 + 0.5) * width
        )

        # Create bars with different alpha for small vs large
        alpha = 0.85 if exp_size == "large" else 0.5
        label = f"{int(n_clients)} Clients - {exp_size}"

        bars = ax.bar(
            position,
            values,
            width,
            yerr=errors,
            capsize=3,
            label=label,
            color=colors[i % len(colors)],
            alpha=alpha,
            edgecolor="white",
            linewidth=1.5,
        )

        # Add value labels on bars
        for bar, err in zip(bars, errors):
            height = bar.get_height()
            if height > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + err + (max(values) * 0.02) if max(values) > 0 else 0,
                    f"{height:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    fontweight="bold",
                )

        bar_index += 1

# Customize the chart
ax.set_xlabel("Number of Rounds", fontsize=14, fontweight="bold", labelpad=10)
ax.set_ylabel(
    "Energy Consumption (kJ per 1k rows)", fontsize=14, fontweight="bold", labelpad=10
)
ax.set_title(
    "Energy Consumption per 1000 Data Points\n", fontsize=16, fontweight="bold", pad=20
)

# Set x-axis
ax.set_xticks(x)
ax.set_xticklabels([f"{int(r)} Rounds" for r in unique_rounds], fontsize=12)

# Customize y-axis
ax.tick_params(axis="y", labelsize=11)

# Add grid
ax.grid(axis="y", alpha=0.3, linestyle="--", linewidth=0.8)
ax.set_axisbelow(True)

# Add legend
ax.legend(loc="upper left", fontsize=9, framealpha=0.95, edgecolor="gray", ncol=2)

# Add background color
ax.set_facecolor("#f9fafb")
fig.set_facecolor("white")

plt.tight_layout()
plt.savefig(
    f"{OUTPUT_DIR}/energy_per_1k_rows_by_size.png",
    dpi=300,
    bbox_inches="tight",
    facecolor="white",
)
plt.show()

# Prepare data for training time grouped bar chart
training_time_grouped = (
    global_data.groupby(["clients", "rounds", "size"])["TotalTrainingTime"]
    .mean()
    .reset_index()
)

# Add experiment size column if not present
if "exp_size" not in training_time_grouped.columns:
    training_time_grouped["exp_size"] = training_time_grouped["size"]

# Convert to seconds for readability
training_time_grouped["training_time_s"] = training_time_grouped["TotalTrainingTime"]

# Get unique values
unique_rounds_tt = sorted(training_time_grouped["rounds"].unique())
unique_clients_tt = sorted(training_time_grouped["clients"].unique())
unique_sizes_tt = sorted(training_time_grouped["exp_size"].unique())

print("=" * 80)
print("TRAINING TIME GROUPED BAR CHART DATA")
print("=" * 80)
print(f"Client configurations: {unique_clients_tt}")
print(f"Round configurations: {unique_rounds_tt}")
print(f"Experiment sizes: {unique_sizes_tt}")
print("\nData overview:")
print(
    training_time_grouped[
        ["clients", "rounds", "exp_size", "TotalTrainingTime"]
    ].to_string(index=False)
)

# Create the grouped bar chart for training time
fig, ax = plt.subplots(figsize=(14, 7))

# Set the width of bars and positions
x = np.arange(len(unique_rounds_tt))
width = 0.8 / (len(unique_clients_tt) * len(unique_sizes_tt))  # Width of each bar

# Colors for different client counts
colors = ["#3b82f6", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6", "#ec4899"]

# Plot bars for each client configuration and size
bar_index = 0
for i, n_clients in enumerate(unique_clients_tt):
    for j, exp_size in enumerate(unique_sizes_tt):
        # Get training time values for this client count and size across all rounds
        values = []
        errors = []
        for n_rounds in unique_rounds_tt:
            data_point = training_time_grouped[
                (training_time_grouped["clients"] == n_clients)
                & (training_time_grouped["rounds"] == n_rounds)
                & (training_time_grouped["exp_size"] == exp_size)
            ]
            if not data_point.empty:
                values.append(data_point["TotalTrainingTime"].values[0])
                # Calculate std from raw data if available
                raw_data = global_data[
                    (global_data["clients"] == n_clients)
                    & (global_data["rounds"] == n_rounds)
                    & (global_data["size"] == exp_size)
                ]["TotalTrainingTime"]
                errors.append(raw_data.std() if len(raw_data) > 1 else 0)
            else:
                values.append(0)
                errors.append(0)

        # Calculate position for this group of bars
        position = (
            x
            + (bar_index - (len(unique_clients_tt) * len(unique_sizes_tt)) / 2 + 0.5)
            * width
        )

        # Create bars with different alpha for small vs large
        alpha = 0.85 if exp_size == "large" else 0.5
        label = f"{int(n_clients)} Clients - {exp_size}"

        bars = ax.bar(
            position,
            values,
            width,
            yerr=errors,
            capsize=3,
            label=label,
            color=colors[i % len(colors)],
            alpha=alpha,
            edgecolor="white",
            linewidth=1.5,
        )

        # Add value labels on bars
        for bar, err in zip(bars, errors):
            height = bar.get_height()
            if height > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + err + (max(values) * 0.02) if max(values) > 0 else 0,
                    f"{height:.0f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    fontweight="bold",
                )

        bar_index += 1

# Customize the chart
ax.set_xlabel("Number of Rounds", fontsize=14, fontweight="bold", labelpad=10)
ax.set_ylabel("Training Time (ms)", fontsize=14, fontweight="bold", labelpad=10)
ax.set_title("Total Training Time per Round\n", fontsize=16, fontweight="bold", pad=20)

# Set x-axis
ax.set_xticks(x)
ax.set_xticklabels([f"{int(r)} Rounds" for r in unique_rounds_tt], fontsize=12)

# Customize y-axis
ax.tick_params(axis="y", labelsize=11)

# Add grid
ax.grid(axis="y", alpha=0.3, linestyle="--", linewidth=0.8)
ax.set_axisbelow(True)

# Add legend
ax.legend(loc="upper left", fontsize=9, framealpha=0.95, edgecolor="gray", ncol=2)

# Add background color
ax.set_facecolor("#f9fafb")
fig.set_facecolor("white")

plt.tight_layout()
plt.savefig(
    f"{OUTPUT_DIR}/training_time_grouped_bar_chart.png",
    dpi=300,
    bbox_inches="tight",
    facecolor="white",
)
plt.show()
