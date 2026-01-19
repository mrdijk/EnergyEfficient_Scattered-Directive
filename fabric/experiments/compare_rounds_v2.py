import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import matplotlib.cm as cm

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (18, 12)

# Load the data
# Dataset sizes per client
DATA_PROVIDERS = {
    'client1': 3799, 'client2': 10570, 'client3': 4725, 'client4': 2182, 'client5': 17938, 
    'client6': 2447, 'client7': 1681, 'client8': 1729, 'client9': 6896, 'client10': 14812, 
    'client12': 3746, 'client13': 4337, 'client14': 2146, 
    'client16': 1711, 'client17': 2094, 'client18': 3188, 'client20': 8281
}

OUTPUT_DIR = "analysis_output"
global_df = pd.read_csv(f"{OUTPUT_DIR}/all_global_stats.csv")
client_df = pd.read_csv(f"{OUTPUT_DIR}/all_client_stats.csv")
energy_df = pd.read_csv(f"{OUTPUT_DIR}/all_energy_stats.csv")
master = pd.read_csv(f"{OUTPUT_DIR}/experiment_summary.csv")

# Filter out bad experiments from ALL dataframes
bad_experiments = ['13-01-26-1165414'] 

# Filter detailed dataframes by timestamp (they have it)
if 'timestamp' in global_df.columns:
    global_df = global_df[~global_df['timestamp'].isin(bad_experiments)].copy()
if 'timestamp' in client_df.columns:
    client_df = client_df[~client_df['timestamp'].isin(bad_experiments)].copy()
if 'timestamp' in energy_df.columns:
    energy_df = energy_df[~energy_df['timestamp'].isin(bad_experiments)].copy()

# Filter all dataframes by experiment ID if it contains bad experiment patterns
for bad_exp in bad_experiments:
    if 'experiment' in global_df.columns:
        global_df = global_df[~global_df['experiment'].str.contains(bad_exp, na=False)].copy()
    if 'experiment' in client_df.columns:
        client_df = client_df[~client_df['experiment'].str.contains(bad_exp, na=False)].copy()
    if 'experiment' in energy_df.columns:
        energy_df = energy_df[~energy_df['experiment'].str.contains(bad_exp, na=False)].copy()
    if 'experiment' in master.columns:
        master = master[~master['experiment'].str.contains(bad_exp, na=False)].copy()

# Also filter out experiments with suspiciously low energy (likely failed)
# Calculate total energy per experiment from detailed energy data
energy_threshold = 1000  # Joules - adjust as needed
if 'joules' in energy_df.columns:
    energy_totals = energy_df.groupby('experiment')['joules'].sum()
    bad_exp_ids_low_energy = energy_totals[energy_totals <= energy_threshold].index.tolist()
else:
    bad_exp_ids_low_energy = []

# Also check master for low energy
if 'total_energy_joules' in master.columns:
    bad_exp_ids_from_master = master[master['total_energy_joules'] <= energy_threshold]['experiment'].tolist()
else:
    bad_exp_ids_from_master = []

# Combine all bad experiment IDs
bad_exp_ids = list(set(bad_exp_ids_low_energy + bad_exp_ids_from_master))

# Filter all dataframes by bad experiment IDs
if bad_exp_ids:
    if 'experiment' in global_df.columns:
        global_df = global_df[~global_df['experiment'].isin(bad_exp_ids)].copy()
    if 'experiment' in client_df.columns:
        client_df = client_df[~client_df['experiment'].isin(bad_exp_ids)].copy()
    if 'experiment' in energy_df.columns:
        energy_df = energy_df[~energy_df['experiment'].isin(bad_exp_ids)].copy()
    if 'experiment' in master.columns:
        master = master[~master['experiment'].isin(bad_exp_ids)].copy()

print("=" * 80)
print("DATA FILTERING")
print("=" * 80)
if bad_exp_ids:
    print(f"Filtered out {len(bad_exp_ids)} bad experiments (low energy < {energy_threshold}J):")
    for bad_exp in bad_exp_ids[:10]:  # Show first 10
        print(f"  - {bad_exp}")
    if len(bad_exp_ids) > 10:
        print(f"  ... and {len(bad_exp_ids) - 10} more")
if bad_experiments:
    print(f"\nManually excluded experiments: {', '.join(bad_experiments)}")

# Filter for specific rounds to compare
rounds_to_compare = 40
experiments_10r = master[master['rounds'] == rounds_to_compare].copy()

print("\n" + "=" * 80)
print(f"COMPARING EXPERIMENTS WITH {rounds_to_compare} ROUNDS")
print("=" * 80)
print(f"\nFound {len(experiments_10r)} experiments with {rounds_to_compare} rounds (after filtering)")
print(f"Client configurations: {sorted(experiments_10r['clients'].unique())}")

if len(experiments_10r) == 0:
    print(f"\n⚠ No experiments found with {rounds_to_compare} rounds!")
    print(f"Available round configurations: {sorted(master['rounds'].unique())}")
    exit()

# Calculate dataset metrics for each experiment
def calculate_dataset_metrics(exp_id, client_df_full):
    """Calculate total dataset size and distribution for an experiment."""
    exp_clients = client_df_full[client_df_full['experiment'] == exp_id]['ClientID'].unique()
    
    total_data = 0
    client_data_sizes = []
    
    for client in exp_clients:
        if client in DATA_PROVIDERS:
            size = DATA_PROVIDERS[client]
            total_data += size
            client_data_sizes.append(size)
    
    return {
        'total_data_points': total_data,
        'avg_data_per_client': np.mean(client_data_sizes) if client_data_sizes else 0,
        'std_data_per_client': np.std(client_data_sizes) if client_data_sizes else 0,
        'min_data_per_client': np.min(client_data_sizes) if client_data_sizes else 0,
        'max_data_per_client': np.max(client_data_sizes) if client_data_sizes else 0,
        'data_imbalance_ratio': np.max(client_data_sizes) / np.min(client_data_sizes) if client_data_sizes and np.min(client_data_sizes) > 0 else 1
    }

# Add dataset metrics to experiments_10r
dataset_metrics_list = []
for _, row in experiments_10r.iterrows():
    metrics = calculate_dataset_metrics(row['experiment'], client_df)
    dataset_metrics_list.append(metrics)

dataset_metrics_df = pd.DataFrame(dataset_metrics_list)
experiments_10r = pd.concat([experiments_10r.reset_index(drop=True), dataset_metrics_df], axis=1)

# Sort by number of clients
experiments_10r = experiments_10r.sort_values('clients')

# Display summary table
print("\n" + "=" * 80)
print("EXPERIMENT SUMMARY WITH DATASET METRICS")
print("=" * 80)
display_cols = ['experiment', 'clients', 'total_data_points', 'data_imbalance_ratio', 
                'GlobalAccuracy', 'total_energy_joules', 'TotalTrainingTime']
print(experiments_10r[display_cols].to_string(index=False))

print("\nDataset Statistics:")
print(f"  Total data points range: {experiments_10r['total_data_points'].min():.0f} - {experiments_10r['total_data_points'].max():.0f}")
print(f"  Data imbalance ratio range: {experiments_10r['data_imbalance_ratio'].min():.2f}x - {experiments_10r['data_imbalance_ratio'].max():.2f}x")

# Calculate additional metrics
experiments_10r['energy_kj'] = experiments_10r['total_energy_joules'] / 1000
experiments_10r['energy_per_accuracy'] = experiments_10r['total_energy_joules'] / experiments_10r['GlobalAccuracy']
experiments_10r['time_per_accuracy'] = experiments_10r['TotalTrainingTime'] / experiments_10r['GlobalAccuracy']
experiments_10r['energy_per_client'] = experiments_10r['total_energy_joules'] / experiments_10r['clients']
experiments_10r['time_per_client'] = experiments_10r['TotalTrainingTime'] / experiments_10r['clients']
experiments_10r['energy_per_data_point'] = experiments_10r['total_energy_joules'] / experiments_10r['total_data_points']
experiments_10r['time_per_data_point'] = experiments_10r['TotalTrainingTime'] / experiments_10r['total_data_points']
experiments_10r['accuracy_per_data_point'] = experiments_10r['GlobalAccuracy'] / experiments_10r['total_data_points']

# Get detailed data for these experiments
exp_ids = experiments_10r['experiment'].tolist()
global_10r = global_df[global_df['experiment'].isin(exp_ids)].copy()
client_10r = client_df[client_df['experiment'].isin(exp_ids)].copy()
energy_10r = energy_df[energy_df['experiment'].isin(exp_ids)].copy()

# Find Round column
round_col = None
for col in global_10r.columns:
    if col.lower() == 'round':
        round_col = col
        break

# Create comprehensive visualizations
fig = plt.figure(figsize=(22, 18))

# 1. Final Global Accuracy Comparison with Dataset Size
ax1 = plt.subplot(4, 4, 1)
bars = ax1.bar(range(len(experiments_10r)), experiments_10r['GlobalAccuracy'], 
               color='steelblue', alpha=0.7)
ax1.set_xticks(range(len(experiments_10r)))
ax1.set_xticklabels([f"{int(c)}C\n{int(d/1000)}K" for c, d in 
                      zip(experiments_10r['clients'], experiments_10r['total_data_points'])], 
                     rotation=0, ha='center', fontsize=8)
ax1.set_ylabel('Final Global Accuracy', fontweight='bold')
ax1.set_title(f'Final Accuracy\n(Clients / Total Data)', fontweight='bold', fontsize=10)
ax1.grid(axis='y', alpha=0.3)

# Add value labels on bars
for i, (bar, val) in enumerate(zip(bars, experiments_10r['GlobalAccuracy'])):
    ax1.text(bar.get_x() + bar.get_width()/2, val, f'{val:.3f}', 
            ha='center', va='bottom', fontsize=7, fontweight='bold')

# 2. Total Energy with Data Imbalance indicator
ax2 = plt.subplot(4, 4, 2)
colors_energy = plt.cm.RdYlGn_r(experiments_10r['data_imbalance_ratio'] / experiments_10r['data_imbalance_ratio'].max())
bars = ax2.bar(range(len(experiments_10r)), experiments_10r['energy_kj'], 
               color=colors_energy, alpha=0.8)
ax2.set_xticks(range(len(experiments_10r)))
ax2.set_xticklabels([f"{int(c)}C" for c in experiments_10r['clients']], fontsize=8)
ax2.set_ylabel('Total Energy (kJ)', fontweight='bold')
ax2.set_title('Energy Consumption\n(color=data imbalance)', fontweight='bold', fontsize=10)
ax2.grid(axis='y', alpha=0.3)

for i, (bar, val) in enumerate(zip(bars, experiments_10r['energy_kj'])):
    ax2.text(bar.get_x() + bar.get_width()/2, val, f'{val:.3f}', 
            ha='center', va='bottom', fontsize=7, fontweight='bold')

# Add colorbar
sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn_r, 
                           norm=plt.Normalize(vmin=experiments_10r['data_imbalance_ratio'].min(), 
                                            vmax=experiments_10r['data_imbalance_ratio'].max()))
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax2, pad=0.01)
cbar.set_label('Imbalance Ratio', fontsize=8)

# 3. Dataset Size Distribution
ax3 = plt.subplot(4, 4, 3)
bars = ax3.bar(range(len(experiments_10r)), experiments_10r['total_data_points']/1000, 
               color='green', alpha=0.7)
ax3.set_xticks(range(len(experiments_10r)))
ax3.set_xticklabels([f"{int(c)}C" for c in experiments_10r['clients']], fontsize=8)
ax3.set_ylabel('Total Data Points (K)', fontweight='bold')
ax3.set_title('Dataset Size', fontweight='bold', fontsize=10)
ax3.grid(axis='y', alpha=0.3)

for i, (bar, val) in enumerate(zip(bars, experiments_10r['total_data_points']/1000)):
    ax3.text(bar.get_x() + bar.get_width()/2, val, f'{val:.0f}K', 
            ha='center', va='bottom', fontsize=7, fontweight='bold')

# 4. Data Imbalance Ratio
ax4 = plt.subplot(4, 4, 4)
bars = ax4.bar(range(len(experiments_10r)), experiments_10r['data_imbalance_ratio'], 
               color='orange', alpha=0.7)
ax4.set_xticks(range(len(experiments_10r)))
ax4.set_xticklabels([f"{int(c)}C" for c in experiments_10r['clients']], fontsize=8)
ax4.set_ylabel('Max/Min Data Ratio', fontweight='bold')
ax4.set_title('Data Imbalance\n(lower=more balanced)', fontweight='bold', fontsize=10)
ax4.grid(axis='y', alpha=0.3)
ax4.axhline(y=1, color='green', linestyle='--', alpha=0.5, linewidth=2, label='Perfect balance')
ax4.legend(fontsize=8)

for i, (bar, val) in enumerate(zip(bars, experiments_10r['data_imbalance_ratio'])):
    ax4.text(bar.get_x() + bar.get_width()/2, val, f'{val:.1f}x', 
            ha='center', va='bottom', fontsize=7, fontweight='bold')

# 5. Global Accuracy Over Rounds
ax5 = plt.subplot(4, 4, 5)
if not global_10r.empty:
    for exp_id in exp_ids:
        exp_data = global_10r[global_10r['experiment'] == exp_id].copy()
        
        # If there's a Round column, use it; otherwise create one based on row order
        if round_col:
            exp_data = exp_data.sort_values(round_col)
            rounds_x = exp_data[round_col]
        else:
            # Create implicit round numbers (0, 1, 2, ...)
            exp_data = exp_data.reset_index(drop=True)
            rounds_x = range(len(exp_data))
        
        n_clients = experiments_10r[experiments_10r['experiment'] == exp_id]['clients'].iloc[0]
        total_data = experiments_10r[experiments_10r['experiment'] == exp_id]['total_data_points'].iloc[0]
        ax5.plot(rounds_x, exp_data['GlobalAccuracy'], 
                marker='o', linewidth=2, label=f"{int(n_clients)}C ({int(total_data/1000)}K)")
    ax5.set_xlabel('Round', fontweight='bold')
    ax5.set_ylabel('Global Accuracy', fontweight='bold')
    ax5.set_title('Accuracy Progression', fontweight='bold', fontsize=10)
    ax5.legend(fontsize=7)
    ax5.grid(alpha=0.3)
else:
    ax5.text(0.5, 0.5, 'Round data not available', ha='center', va='center', transform=ax5.transAxes)

# 6. Energy per Data Point
ax6 = plt.subplot(4, 4, 6)
bars = ax6.bar(range(len(experiments_10r)), experiments_10r['energy_per_data_point']*1000, 
               color='darkred', alpha=0.7)
ax6.set_xticks(range(len(experiments_10r)))
ax6.set_xticklabels([f"{int(c)}C" for c in experiments_10r['clients']], fontsize=8)
ax6.set_ylabel('Energy per 1K Data Points (J)', fontweight='bold')
ax6.set_title('Energy Efficiency per Data', fontweight='bold', fontsize=10)
ax6.grid(axis='y', alpha=0.3)

for i, (bar, val) in enumerate(zip(bars, experiments_10r['energy_per_data_point']*1000)):
    ax6.text(bar.get_x() + bar.get_width()/2, val, f'{val:.0f}', 
            ha='center', va='bottom', fontsize=7, fontweight='bold')

# 7. Accuracy vs Total Dataset Size
ax7 = plt.subplot(4, 4, 7)
scatter = ax7.scatter(experiments_10r['total_data_points']/1000, experiments_10r['GlobalAccuracy'], 
                     s=experiments_10r['clients']*30, alpha=0.6, 
                     c=experiments_10r['data_imbalance_ratio'], cmap='RdYlGn_r')
for _, row in experiments_10r.iterrows():
    ax7.annotate(f"{int(row['clients'])}C", 
                (row['total_data_points']/1000, row['GlobalAccuracy']), 
                fontsize=8, ha='center', fontweight='bold')
ax7.set_xlabel('Total Data Points (K)', fontweight='bold')
ax7.set_ylabel('Final Accuracy', fontweight='bold')
ax7.set_title('Accuracy vs Dataset Size\n(size=# clients, color=imbalance)', fontweight='bold', fontsize=9)
ax7.grid(alpha=0.3)
cbar = plt.colorbar(scatter, ax=ax7)
cbar.set_label('Imbalance', fontsize=8)

# 8. Energy vs Dataset Size
ax8 = plt.subplot(4, 4, 8)
scatter = ax8.scatter(experiments_10r['total_data_points']/1000, experiments_10r['energy_kj'], 
                     s=experiments_10r['clients']*30, alpha=0.6, 
                     c=experiments_10r['data_imbalance_ratio'], cmap='RdYlGn_r')
for _, row in experiments_10r.iterrows():
    ax8.annotate(f"{int(row['clients'])}C", 
                (row['total_data_points']/1000, row['energy_kj']), 
                fontsize=8, ha='center', fontweight='bold')
ax8.set_xlabel('Total Data Points (K)', fontweight='bold')
ax8.set_ylabel('Total Energy (kJ)', fontweight='bold')
ax8.set_title('Energy vs Dataset Size', fontweight='bold', fontsize=10)
ax8.grid(alpha=0.3)



# 9-16. Keep remaining plots but update subplot grid
# 9. Accuracy vs Energy Trade-off
ax9 = plt.subplot(4, 4, 9)
scatter = ax9.scatter(experiments_10r['energy_kj'], experiments_10r['GlobalAccuracy'], 
                     s=experiments_10r['clients']*50, alpha=0.6, c=range(len(experiments_10r)), 
                     cmap='viridis')
for i, row in experiments_10r.iterrows():
    ax9.annotate(f"{int(row['clients'])}C", 
                (row['energy_kj'], row['GlobalAccuracy']), 
                fontsize=10, ha='center', fontweight='bold')
ax9.set_xlabel('Total Energy (kJ)', fontweight='bold')
ax9.set_ylabel('Final Global Accuracy', fontweight='bold')
ax9.set_title('Accuracy vs Energy Trade-off\n(bubble size = # clients)', fontweight='bold')
ax9.grid(alpha=0.3)

# 10. Energy Breakdown by Component
ax10 = plt.subplot(4, 4, 10)
if not energy_10r.empty:
    # Clean energy data
    energy_clean = energy_10r[energy_10r['container_name'] != 'linkerd-init'].copy()
    
    # Group by experiment and namespace
    energy_by_exp_ns = energy_clean.groupby(['experiment', 'namespace'])['joules'].sum().reset_index()
    
    # Create stacked bar chart
    namespaces = []
    for exp_id in exp_ids:
        exp_energy = energy_by_exp_ns[energy_by_exp_ns['experiment'] == exp_id]
        for ns in exp_energy['namespace'].unique():
            if ns not in namespaces:
                namespaces.append(ns)
    
    # Prepare data for stacking
    bottom = np.zeros(len(exp_ids))
    colors_stack = plt.cm.Set3(np.linspace(0, 1, len(namespaces)))
    
    for i, ns in enumerate(namespaces):
        values = []
        for exp_id in exp_ids:
            exp_ns_energy = energy_by_exp_ns[(energy_by_exp_ns['experiment'] == exp_id) & 
                                             (energy_by_exp_ns['namespace'] == ns)]
            values.append(exp_ns_energy['joules'].sum() / 1000 if not exp_ns_energy.empty else 0)
        
        ax10.bar(range(len(exp_ids)), values, bottom=bottom, label=ns, color=colors_stack[i])
        bottom += values
    
    ax10.set_xticks(range(len(experiments_10r)))
    ax10.set_xticklabels([f"{int(c)}C" for c in experiments_10r['clients']], fontsize=8)
    ax10.set_ylabel('Energy (kJ)', fontweight='bold')
    ax10.set_title('Energy Distribution', fontweight='bold', fontsize=10)
    ax10.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7)
    ax10.grid(axis='y', alpha=0.3)
else:
    ax10.text(0.5, 0.5, 'Energy data not available', ha='center', va='center', transform=ax10.transAxes)

# 11. Scalability: Metrics vs Number of Clients (with dataset size)
ax11 = plt.subplot(4, 4, 11)
ax11_twin = ax11.twinx()
ax11_triple = ax11.twinx()
ax11_triple.spines['right'].set_position(('outward', 60))

ln1 = ax11.plot(experiments_10r['clients'], experiments_10r['GlobalAccuracy'], 
               'o-', color='green', label='Accuracy', linewidth=2, markersize=10)
ln2 = ax11_twin.plot(experiments_10r['clients'], experiments_10r['energy_kj'], 
                    's-', color='red', label='Energy (kJ)', linewidth=2, markersize=10)
ln3 = ax11_triple.plot(experiments_10r['clients'], experiments_10r['total_data_points']/1000, 
                      '^-', color='blue', label='Data (K)', linewidth=2, markersize=10)

ax11.set_xlabel('Number of Clients', fontweight='bold')
ax11.set_ylabel('Final Accuracy', color='green', fontweight='bold')
ax11_twin.set_ylabel('Energy (kJ)', color='red', fontweight='bold')
ax11_triple.set_ylabel('Data (K)', color='blue', fontweight='bold')
ax11.set_title('Scalability Analysis', fontweight='bold', fontsize=10)
ax11.tick_params(axis='y', labelcolor='green')
ax11_twin.tick_params(axis='y', labelcolor='red')
ax11_triple.tick_params(axis='y', labelcolor='blue')
ax11.grid(alpha=0.3)

lns = ln1 + ln2 + ln3
labs = [l.get_label() for l in lns]
ax11.legend(lns, labs, loc='best', fontsize=8)

# 12. Data Imbalance vs Fairness
ax12 = plt.subplot(4, 4, 12)
scatter = ax12.scatter(experiments_10r['data_imbalance_ratio'], 
                      experiments_10r['avg_client_accuracy_std'], 
                      s=experiments_10r['clients']*30, alpha=0.6, 
                      c=experiments_10r['GlobalAccuracy'], cmap='viridis')
for _, row in experiments_10r.iterrows():
    ax12.annotate(f"{int(row['clients'])}C", 
                (row['data_imbalance_ratio'], row['avg_client_accuracy_std']), 
                fontsize=8, ha='center')
ax12.set_xlabel('Data Imbalance Ratio', fontweight='bold')
ax12.set_ylabel('Client Accuracy Std Dev', fontweight='bold')
ax12.set_title('Imbalance vs Fairness\n(color=accuracy)', fontweight='bold', fontsize=9)
ax12.grid(alpha=0.3)
cbar = plt.colorbar(scatter, ax=ax12)
cbar.set_label('Accuracy', fontsize=8)

# 13. Energy Efficiency Comparison
ax13 = plt.subplot(4, 4, 13)
x = np.arange(len(experiments_10r))
width = 0.35
bars1 = ax13.bar(x - width/2, experiments_10r['energy_per_client'], width, 
                label='Per Client', alpha=0.7, color='steelblue')
bars2 = ax13.bar(x + width/2, experiments_10r['energy_per_data_point']*1000, width, 
                label='Per 1K Data', alpha=0.7, color='coral')
ax13.set_xticks(x)
ax13.set_xticklabels([f"{int(c)}C" for c in experiments_10r['clients']], fontsize=8)
ax13.set_ylabel('Energy (J)', fontweight='bold')
ax13.set_title('Energy Efficiency', fontweight='bold', fontsize=10)
ax13.legend(fontsize=8)
ax13.grid(axis='y', alpha=0.3)

for i, (bar1, val) in enumerate(zip(bars, experiments_10r['energy_per_client'])):
    ax13.text(bar1.get_x() + bar1.get_width()/2, val, f'{val:.0f}', 
            ha='right', va='bottom', fontsize=7, fontweight='bold')

for i, (bar2, val) in enumerate(zip(bars, experiments_10r['energy_per_data_point']*1000)):
    ax13.text(bar2.get_x() + bar2.get_width()/2, val, f'{val:.0f}J', 
            ha='left', va='bottom', fontsize=7, fontweight='bold')
    
# 14. Time per Data Point
ax14 = plt.subplot(4, 4, 14)
bars = ax14.bar(range(len(experiments_10r)), experiments_10r['time_per_data_point']*1000, 
               color='purple', alpha=0.7)
ax14.set_xticks(range(len(experiments_10r)))
ax14.set_xticklabels([f"{int(c)}C" for c in experiments_10r['clients']], fontsize=8)
ax14.set_ylabel('Time per 1K Data Points (s)', fontweight='bold')
ax14.set_title('Time Efficiency per Data', fontweight='bold', fontsize=10)
ax14.grid(axis='y', alpha=0.3)

for i, (bar, val) in enumerate(zip(bars, experiments_10r['time_per_data_point']*1000)):
    ax14.text(bar.get_x() + bar.get_width()/2, val, f'{val:.0f}', 
            ha='center', va='bottom', fontsize=7, fontweight='bold')

# 15. Client Fairness
ax15 = plt.subplot(4, 4, 15)
bars = ax15.bar(range(len(experiments_10r)), experiments_10r['avg_client_accuracy_std'], 
               color='teal', alpha=0.7)
ax15.set_xticks(range(len(experiments_10r)))
ax15.set_xticklabels([f"{int(c)}C" for c in experiments_10r['clients']], fontsize=8)
ax15.set_ylabel('Accuracy Std Dev', fontweight='bold')
ax15.set_title('Client Fairness\n(lower=better)', fontweight='bold', fontsize=10)
ax15.grid(axis='y', alpha=0.3)

# 16. Summary Table
ax16 = plt.subplot(4, 4, 16)
ax16.axis('tight')
ax16.axis('off')

summary_stats = [
    ['Metric', 'Mean', 'Min', 'Max'],
    ['Accuracy', 
     f"{experiments_10r['GlobalAccuracy'].mean():.4f}",
     f"{experiments_10r['GlobalAccuracy'].min():.4f}",
     f"{experiments_10r['GlobalAccuracy'].max():.4f}"],
    ['Energy (kJ)', 
     f"{experiments_10r['energy_kj'].mean():.1f}",
     f"{experiments_10r['energy_kj'].min():.1f}",
     f"{experiments_10r['energy_kj'].max():.1f}"],
    ['Data (K)', 
     f"{experiments_10r['total_data_points'].mean()/1000:.1f}",
     f"{experiments_10r['total_data_points'].min()/1000:.1f}",
     f"{experiments_10r['total_data_points'].max()/1000:.1f}"],
    ['Imbalance', 
     f"{experiments_10r['data_imbalance_ratio'].mean():.2f}",
     f"{experiments_10r['data_imbalance_ratio'].min():.2f}",
     f"{experiments_10r['data_imbalance_ratio'].max():.2f}"],
    ['Fairness (σ)', 
     f"{experiments_10r['avg_client_accuracy_std'].mean():.4f}",
     f"{experiments_10r['avg_client_accuracy_std'].min():.4f}",
     f"{experiments_10r['avg_client_accuracy_std'].max():.4f}"],
]

table = ax16.table(cellText=summary_stats, cellLoc='center', loc='center',
                  colWidths=[0.35, 0.22, 0.22, 0.22])
table.auto_set_font_size(False)
table.set_fontsize(8)
table.scale(1, 2.5)

# Style header row
for i in range(4):
    table[(0, i)].set_facecolor('#3498db')
    table[(0, i)].set_text_props(weight='bold', color='white')

ax16.set_title('Summary Statistics', fontweight='bold', pad=20, fontsize=10)

plt.suptitle(f'Federated Learning: {rounds_to_compare} Rounds Comparison', 
             fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/comparison_{rounds_to_compare}rounds_v2.png', dpi=300, bbox_inches='tight')
plt.show()

# Print detailed analysis
print("\n" + "=" * 80)
print("DETAILED ANALYSIS")
print("=" * 80)

print(f"\n📊 ACCURACY:")
best_acc = experiments_10r.loc[experiments_10r['GlobalAccuracy'].idxmax()]
print(f"  Best: {int(best_acc['clients'])} clients → {best_acc['GlobalAccuracy']:.4f}")
print(f"  Range: {experiments_10r['GlobalAccuracy'].min():.4f} to {experiments_10r['GlobalAccuracy'].max():.4f}")
print(f"  Improvement: {((experiments_10r['GlobalAccuracy'].max() - experiments_10r['GlobalAccuracy'].min()) / experiments_10r['GlobalAccuracy'].min() * 100):.2f}%")

print(f"\n⚡ ENERGY:")
best_energy = experiments_10r.loc[experiments_10r['total_energy_joules'].idxmin()]
print(f"  Most Efficient: {int(best_energy['clients'])} clients → {best_energy['energy_kj']:.1f} kJ")
print(f"  Range: {experiments_10r['energy_kj'].min():.1f} to {experiments_10r['energy_kj'].max():.1f} kJ")
print(f"  Scaling: {((experiments_10r['energy_kj'].max() - experiments_10r['energy_kj'].min()) / experiments_10r['energy_kj'].min() * 100):.1f}% increase")

print(f"\n⏱️  TIME:")
best_time = experiments_10r.loc[experiments_10r['TotalTrainingTime'].idxmin()]
print(f"  Fastest: {int(best_time['clients'])} clients → {best_time['TotalTrainingTime']:.0f} s")
print(f"  Range: {experiments_10r['TotalTrainingTime'].min():.0f} to {experiments_10r['TotalTrainingTime'].max():.0f} s")

print(f"\n🎯 EFFICIENCY:")
best_energy_eff = experiments_10r.loc[experiments_10r['energy_per_accuracy'].idxmin()]
best_time_eff = experiments_10r.loc[experiments_10r['time_per_accuracy'].idxmin()]
print(f"  Best Energy Efficiency: {int(best_energy_eff['clients'])} clients")
print(f"  Best Time Efficiency: {int(best_time_eff['clients'])} clients")

print(f"\n⚖️  FAIRNESS:")
best_fairness = experiments_10r.loc[experiments_10r['avg_client_accuracy_std'].idxmin()]
print(f"  Most Fair: {int(best_fairness['clients'])} clients → σ = {best_fairness['avg_client_accuracy_std']:.4f}")

print("\n" + "=" * 80)
print(f"✓ Visualization saved: {OUTPUT_DIR}/comparison_{rounds_to_compare}rounds_v2.png")
print("=" * 80)