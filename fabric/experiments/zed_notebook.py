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
OUTPUT_DIR = "analysis_output"
master = pd.read_csv(f"{OUTPUT_DIR}/experiment_summary.csv")
global_data = pd.read_csv(
    "/home/maurits/EnergyEfficient_Scattered-Directive/fabric/experiments/analysis_output/all_global_stats.csv"
)
energy_data = pd.read_csv(
    "/home/maurits/EnergyEfficient_Scattered-Directive/fabric/experiments/analysis_output/all_energy_stats.csv"
)

# %% Cell 7
global_data["experiment_duration_m"] = global_data.groupby("experiment")[
    "RoundDuration"
].transform(lambda x: x.sum() / 1e9 / 60)
# global_data.groupby("experiment").sum()
print(global_data[global_data["experiment_duration_m"] < 1])

# %% Cell 8
duration_sum = global_data.groupby("experiment")["experiment_duration_m"].mean()
master["experiment_duration_m"] = master["experiment"].map(duration_sum)

# Identify short experiments (duration < 1 minute)
mask = master["experiment_duration_m"] < 1

# # Update joules column for short experiments only
master.loc[mask, "total_energy_joules"] = (
    master.loc[mask, "total_energy_joules"] * master.loc[mask, "experiment_duration_m"]
)

print(master[master["experiment_duration_m"] < 1])

# %% Cell 9
global_data["experiment_duration_m"] = global_data.groupby("experiment")[
    "RoundDuration"
].transform(lambda x: x.sum() / 1e9 / 60)

duration_sum = global_data.groupby("experiment")["experiment_duration_m"].mean()
master["experiment_duration_m"] = master["experiment"].map(duration_sum)
# Identify short experiments (duration < 1 minute)
mask = master["experiment_duration_m"] < 1

# # Update joules column for short experiments only
master.loc[mask, "total_energy_joules"] = (
    master.loc[mask, "total_energy_joules"] * master.loc[mask, "experiment_duration_m"]
)
