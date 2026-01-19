import pandas as pd
from glob import glob
from pathlib import Path
import matplotlib.pyplot as plt

DATA_DIR = "/home/maurits/EnergyEfficient_Scattered-Directive/fabric/data"
OUTPUT_DIR = "analysis_output"
Path(OUTPUT_DIR).mkdir(exist_ok=True)

# Dataset sizes per client
DATA_PROVIDERS = {
    'client1': 3799, 'client2': 10570, 'client3': 4725, 'client4': 2182, 'client5': 17938, 
    'client6': 2447, 'client7': 1681, 'client8': 1729, 'client9': 6896, 'client10': 14812, 
    'client12': 3746, 'client13': 4337, 'client14': 2146, 
    'client16': 1711, 'client17': 2094, 'client18': 3188, 'client20': 8281
}

def extract_meta(file_path):
    p = Path(file_path)

    timestamp = p.parent.name
    rounds = int(p.parent.parent.name)
    clients = int(p.parent.parent.parent.name)

    experiment = f"C{clients}_R{rounds}_{timestamp}"

    return clients, rounds, timestamp, experiment

def load_csv_flexible(file_path):
    """
    Load CSV handling both indexed and non-indexed files.
    Detects if first column is an unnamed index and handles accordingly.
    """
    # First, peek at the file to check structure
    df_peek = pd.read_csv(file_path, nrows=5)
    
    # Check if first column looks like an index (unnamed or numeric sequence)
    first_col = df_peek.columns[0]
    
    # If first column is 'Unnamed: 0' or empty, it's likely an index column
    if first_col.startswith('Unnamed') or first_col == '':
        df = pd.read_csv(file_path, index_col=0)
    else:
        # Otherwise, read normally
        df = pd.read_csv(file_path)
    
    return df

print("Loading global stats...")

global_files = glob(f"{DATA_DIR}/**/global_stats.csv", recursive=True)

global_all = []

for f in global_files:
    try:
        df = load_csv_flexible(f)
        
        clients, rounds, ts, exp = extract_meta(f)

        df["clients"] = clients
        df["rounds"] = rounds
        df["timestamp"] = ts
        df["experiment"] = exp

        global_all.append(df)
    except Exception as e:
        print(f"Error loading {f}: {e}")

global_df = pd.concat(global_all, ignore_index=True)

print("Global experiments loaded:", len(global_files))

print("Loading client stats...")

client_files = glob(f"{DATA_DIR}/**/client_stats.csv", recursive=True)

client_all = []

for f in client_files:
    try:
        df = load_csv_flexible(f)
        
        clients, rounds, ts, exp = extract_meta(f)

        df["clients"] = clients
        df["rounds"] = rounds
        df["timestamp"] = ts
        df["experiment"] = exp

        client_all.append(df)
    except Exception as e:
        print(f"Error loading {f}: {e}")

client_df = pd.concat(client_all, ignore_index=True)

print("Client experiments loaded:", len(client_files))

print("Loading energy stats...")

energy_files = glob(f"{DATA_DIR}/**/energy_consumption.csv", recursive=True)

energy_all = []

for f in energy_files:
    try:
        df = load_csv_flexible(f)
        # Remove any remaining unnamed columns
        df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

        clients, rounds, ts, exp = extract_meta(f)

        df["clients"] = clients
        df["rounds"] = rounds
        df["timestamp"] = ts
        df["experiment"] = exp

        energy_all.append(df)

    except Exception as e:
        print("Skipping:", f, "->", e)

energy_df = pd.concat(energy_all, ignore_index=True)

print("Energy experiments loaded:", len(energy_files))


# -----------------------------------
# GLOBAL SUMMARY (final round)
# -----------------------------------

print("Computing global summaries...")

# Check if we have a Round column (might be capitalized differently)
round_col = None
for col in global_df.columns:
    if col.lower() == 'round':
        round_col = col
        break

if round_col:
    final_global = (
        global_df
        .sort_values(round_col)
        .groupby("experiment")
        .tail(1)
    )
else:
    # If no Round column, just take unique experiments
    print("Warning: No Round column found in global_df")
    final_global = global_df.drop_duplicates(subset=['experiment'], keep='last')

final_global = final_global[[
    "experiment",
    "clients",
    "rounds",
    "GlobalAccuracy",
    "TotalTrainingTime",
    "AggregationTime",
    "RoundDuration"
]]

print("Computing client fairness...")

# Check for Round column in client_df
round_col_client = None
for col in client_df.columns:
    if col.lower() == 'round':
        round_col_client = col
        break

if round_col_client:
    fairness = (
        client_df
        .groupby(["experiment", round_col_client])["ClientAccuracy"]
        .std()
        .groupby("experiment")
        .mean()
        .reset_index(name="avg_client_accuracy_std")
    )
else:
    print("Warning: No Round column found in client_df")
    fairness = (
        client_df
        .groupby("experiment")["ClientAccuracy"]
        .std()
        .reset_index(name="avg_client_accuracy_std")
    )


# -----------------------------------
# ENERGY SUMMARY
# -----------------------------------

print("Computing energy usage...")

energy_summary = (
    energy_df
    .groupby("experiment")["joules"]
    .sum()
    .reset_index(name="total_energy_joules")
)


# -----------------------------------
# MERGE MASTER TABLE
# -----------------------------------

print("Merging experiment summary...")

master = final_global.merge(fairness, on="experiment", how="left")
master = master.merge(energy_summary, on="experiment", how="left")

# Sort by clients and rounds for easier reading
master = master.sort_values(["clients", "rounds"]).reset_index(drop=True)


# -----------------------------------
# Save Outputs
# -----------------------------------

print("Saving CSV outputs...")

global_df.to_csv(f"{OUTPUT_DIR}/all_global_stats.csv", index=False)
client_df.to_csv(f"{OUTPUT_DIR}/all_client_stats.csv", index=False)
energy_df.to_csv(f"{OUTPUT_DIR}/all_energy_stats.csv", index=False)
master.to_csv(f"{OUTPUT_DIR}/experiment_summary.csv", index=False)

print("\n" + "=" * 80)
print("SUMMARY STATISTICS")
print("=" * 80)
print(f"\nTotal experiments: {len(master)}")
print(f"Client configurations: {sorted(master['clients'].unique())}")
print(f"Round configurations: {sorted(master['rounds'].unique())}")

print("\n" + "=" * 80)
print("OUTPUT FILES SAVED")
print("=" * 80)
print(f"{OUTPUT_DIR}/all_global_stats.csv - {len(global_df)} rows")
print(f"{OUTPUT_DIR}/all_client_stats.csv - {len(client_df)} rows")
print(f"{OUTPUT_DIR}/all_energy_stats.csv - {len(energy_df)} rows")
print(f"{OUTPUT_DIR}/experiment_summary.csv - {len(master)} experiments")

# Display sample of master table
print("\n" + "=" * 80)
print("EXPERIMENT SUMMARY (first 10 rows)")
print("=" * 80)
print(master.head(10).to_string(index=False))