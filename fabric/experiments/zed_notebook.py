# %% Cell 1
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# %% Cell 2
DATA_PROVIDERS = {
    "client1": 3799,
    "client2": 10570,
    "client3": 4725,
    "client4": 2182,
    "client5": 17938,
    "client6": 2447,
    "client7": 1681,
    "client8": 1729,
    "client9": 6896,
    "client10": 14812,
    "client11": 2778,
    "client12": 3746,
    "client13": 4337,
    "client14": 2146,
    "client15": 2665,
    "client16": 1711,
    "client17": 2094,
    "client18": 3188,
    "client19": 2265,
    "client20": 8281,
}

# %% Cell 3
data = {
    "client": [
        # 5 largest
        "client9",
        "client20",
        "client2",
        "client10",
        "client5",
        # 5 smallest
        "client7",
        "client16",
        "client8",
        "client17",
        "client14",
        # 10 largest
        "client18",
        "client12",
        "client1",
        "client13",
        "client3",
        "client9",
        "client20",
        "client2",
        "client10",
        "client5",
        # 10 smallest
        "client7",
        "client16",
        "client8",
        "client17",
        "client14",
        "client4",
        "client19",
        "client6",
        "client15",
        "client11",
        # 15 largest
        "client4",
        "client19",
        "client6",
        "client15",
        "client11",
        "client18",
        "client12",
        "client1",
        "client13",
        "client3",
        "client9",
        "client20",
        "client2",
        "client10",
        "client5",
        # 15 smallest
        "client7",
        "client16",
        "client8",
        "client17",
        "client14",
        "client4",
        "client19",
        "client6",
        "client15",
        "client11",
        "client18",
        "client12",
        "client1",
        "client13",
        "client3",
    ],
    "rows": [
        # 5 largest
        6896,
        8281,
        10570,
        14812,
        17938,
        # 5 smallest
        1681,
        1711,
        1729,
        2094,
        2146,
        # 10 largest
        3188,
        3746,
        3799,
        4337,
        4725,
        6896,
        8281,
        10570,
        14812,
        17938,
        # 10 smallest
        1681,
        1711,
        1729,
        2094,
        2146,
        2182,
        2265,
        2447,
        2665,
        2778,
        # 15 largest
        2182,
        2265,
        2447,
        2665,
        2778,
        3188,
        3746,
        3799,
        4337,
        4725,
        6896,
        8281,
        10570,
        14812,
        17938,
        # 15 smallest
        1681,
        1711,
        1729,
        2094,
        2146,
        2182,
        2265,
        2447,
        2665,
        2778,
        3188,
        3746,
        3799,
        4337,
        4725,
    ],
    "group": (
        ["5 largest"] * 5
        + ["5 smallest"] * 5
        + ["10 largest"] * 10
        + ["10 smallest"] * 10
        + ["15 largest"] * 15
        + ["15 smallest"] * 15
    ),
}

df = pd.DataFrame(data)
print(df)

# %% Cell 4
print(df[df["group"] == "5 smallest"])
# Group statistics
print(df.groupby("group")["rows"].agg(["mean", "min", "max", "sum"]))

#      client  rows       group
# 5   client7  1681  5 smallest
# 6  client16  1711  5 smallest
# 7   client8  1729  5 smallest
# 8  client17  2094  5 smallest
# 9  client14  2146  5 smallest
#
# group         mean   min    max    sum
# 15 largest    6041.93  2182  17938  90629
# 15 smallest   2766.20  1681   4725  41493
# 10 largest    7829.20  3188  17938  78292
# 10 smallest   2169.80  1681   2778  21698
# 5 largest    11699.40  6896  17938  58497
# 5 smallest    1872.20  1681   2146   9361

#  %% Cell 5
# Melt the dataframe to long format for easier plotting
df_melted = grouped_df.melt(
    id_vars=["n_clients", "type"],
    value_vars=["mean", "min", "max", "sum"],
    var_name="statistic",
    value_name="value",
)

# Create the plot
fig, ax = plt.subplots(figsize=(10, 5))

# Get unique values
n_clients_unique = sorted(grouped_df["n_clients"].unique())
stats = ["mean", "min", "max", "sum"]
types = ["largest", "smallest"]

# Set up bar positions
x = np.arange(len(n_clients_unique))
n_stats = len(stats)
width = 0.4 / n_stats  # Width of each bar

# Colors for each statistic
colors = {"mean": "#3498db", "min": "#2ecc71", "max": "#e74c3c", "sum": "#f39c12"}

# Plot bars
for i, stat in enumerate(stats):
    for j, type_ in enumerate(types):
        data = (
            grouped_df[(grouped_df["type"] == type_)]
            .sort_values("n_clients")[stat]
            .values
        )
        offset = (i * len(types) + j - (n_stats * len(types) - 1) / 2) * width

        label = f"{stat} - {type_}"
        bars = ax.bar(
            x + offset,
            data,
            width,
            label=label,
            color=colors[stat],
            alpha=0.8 if type_ == "largest" else 0.5,
        )
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.0f}",
                ha="center",
                va="bottom",
                fontsize=7,
                # rotation=90,
            )

ax.set_xlabel("Number of Clients", fontsize=12)
ax.set_ylabel("Number of Rows", fontsize=12)
ax.set_title(
    "Row Statistics by Number of Clients and Experiment", fontsize=14, fontweight="bold"
)
ax.set_xticks(x)
ax.set_xticklabels(n_clients_unique)
ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.show()
# %% Cell 6
OUTPUT_DIR = "analysis_output"
global_data = pd.read_csv(f"{OUTPUT_DIR}/experiment_summary.csv")
client_data = pd.read_csv(f"{OUTPUT_DIR}/all_client_stats.csv")
energy_data = pd.read_csv(f"{OUTPUT_DIR}/all_energy_stats.csv")

# %% Cell 7
experiment_stats = pd.DataFrame()
experiment_stats["total_time (s)"] = (
    global_data.groupby("experiment")["RoundDuration"].sum() / 1000000
)
# print(Total_time)

#   %% Cell 8
experiment_stats["total_joules (kj)"] = (
    energy_data.groupby("experiment")["joules"].sum() / 1000
)

# %% Cell 9
grouped = (
    global_data.groupby(["clients", "rounds"])["total_energy_joules"]
    .mean()
    .reset_index()
)
grouped["energy_kj"] = grouped["total_energy_joules"] / 1000
unique_rounds = sorted(grouped["rounds"].unique())
unique_clients = sorted(grouped["clients"].unique())

for i, n_clients in enumerate(unique_clients):
    # Get energy values for this client count across all rounds
    number_of_clients = len(unique_clients)
    values = []
    for n_rounds in unique_rounds:
        data_point = grouped[
            (grouped["clients"] == n_clients) & (grouped["rounds"] == n_rounds)
        ]
        if not data_point.empty:
            values.append(data_point["energy_kj"].values[0] / number_of_clients)
        else:
            values.append(0)

    yerr = np.std(values)
    print(yerr)
# %% Cell 10
import constants

SORTED_CLIENTS = sorted(constants.DATA_PROVIDERS.items(), key=lambda x: x[1])
size = np.sum([size for _, size in SORTED_CLIENTS[:5]])
print(size)
