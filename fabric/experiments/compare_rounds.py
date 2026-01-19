import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path


# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (18, 12)

# Load the data
OUTPUT_DIR = "analysis_output"
global_df = pd.read_csv(f"{OUTPUT_DIR}/all_global_stats.csv")
client_df = pd.read_csv(f"{OUTPUT_DIR}/all_client_stats.csv")
energy_df = pd.read_csv(f"{OUTPUT_DIR}/all_energy_stats.csv")
master = pd.read_csv(f"{OUTPUT_DIR}/experiment_summary.csv")

# Filter out bad experiments from ALL dataframes
bad_experiments = ['13-01-26-1165414']  # Add bad timestamp/experiment IDs here

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

# Sort by number of clients
experiments_10r = experiments_10r.sort_values('clients')

# Display summary table
print("\n" + "=" * 80)
print("EXPERIMENT SUMMARY")
print("=" * 80)
print(experiments_10r[['experiment', 'clients', 'GlobalAccuracy', 'total_energy_joules', 
                       'TotalTrainingTime', 'avg_client_accuracy_std']].to_string(index=False))

# Calculate additional metrics
experiments_10r['energy_kj'] = experiments_10r['total_energy_joules'] / 1000
experiments_10r['energy_per_accuracy'] = experiments_10r['total_energy_joules'] / experiments_10r['GlobalAccuracy']
experiments_10r['time_per_accuracy'] = experiments_10r['TotalTrainingTime'] / experiments_10r['GlobalAccuracy']
experiments_10r['energy_per_client'] = experiments_10r['total_energy_joules'] / experiments_10r['clients']
experiments_10r['time_per_client'] = experiments_10r['TotalTrainingTime'] / experiments_10r['clients']

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
fig = plt.figure(figsize=(20, 16))

# 1. Final Global Accuracy Comparison
ax1 = plt.subplot(4, 3, 1)
bars = ax1.bar(range(len(experiments_10r)), experiments_10r['GlobalAccuracy'], 
               color='steelblue', alpha=0.7)
ax1.set_xticks(range(len(experiments_10r)))
ax1.set_xticklabels([f"{int(c)} clients" for c in experiments_10r['clients']], rotation=45, ha='right')
ax1.set_ylabel('Final Global Accuracy', fontweight='bold')
ax1.set_title(f'Final Accuracy Comparison ({rounds_to_compare} Rounds)', fontweight='bold')
ax1.grid(axis='y', alpha=0.3)

# Add value labels on bars
for i, (bar, val) in enumerate(zip(bars, experiments_10r['GlobalAccuracy'])):
    ax1.text(bar.get_x() + bar.get_width()/2, val, f'{val:.4f}', 
            ha='center', va='bottom', fontsize=9, fontweight='bold')

# 2. Total Energy Consumption
ax2 = plt.subplot(4, 3, 2)
bars = ax2.bar(range(len(experiments_10r)), experiments_10r['energy_kj'], 
               color='coral', alpha=0.7)
ax2.set_xticks(range(len(experiments_10r)))
ax2.set_xticklabels([f"{int(c)} clients" for c in experiments_10r['clients']], rotation=45, ha='right')
ax2.set_ylabel('Total Energy (kJ)', fontweight='bold')
ax2.set_title(f'Energy Consumption ({rounds_to_compare} Rounds)', fontweight='bold')
ax2.grid(axis='y', alpha=0.3)

for i, (bar, val) in enumerate(zip(bars, experiments_10r['energy_kj'])):
    ax2.text(bar.get_x() + bar.get_width()/2, val, f'{val:.1f}', 
            ha='center', va='bottom', fontsize=9, fontweight='bold')

# 3. Total Training Time
ax3 = plt.subplot(4, 3, 3)
bars = ax3.bar(range(len(experiments_10r)), experiments_10r['TotalTrainingTime'], 
               color='green', alpha=0.7)
ax3.set_xticks(range(len(experiments_10r)))
ax3.set_xticklabels([f"{int(c)} clients" for c in experiments_10r['clients']], rotation=45, ha='right')
ax3.set_ylabel('Total Training Time (s)', fontweight='bold')
ax3.set_title(f'Training Time ({rounds_to_compare} Rounds)', fontweight='bold')
ax3.grid(axis='y', alpha=0.3)

for i, (bar, val) in enumerate(zip(bars, experiments_10r['TotalTrainingTime'])):
    ax3.text(bar.get_x() + bar.get_width()/2, val, f'{val:.0f}', 
            ha='center', va='bottom', fontsize=9, fontweight='bold')

# 4. Global Accuracy Over Rounds
ax4 = plt.subplot(4, 3, 4)
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
        ax4.plot(rounds_x, exp_data['GlobalAccuracy'], 
                marker='o', linewidth=2, label=f"{int(n_clients)} clients")
    ax4.set_xlabel('Round', fontweight='bold')
    ax4.set_ylabel('Global Accuracy', fontweight='bold')
    ax4.set_title('Accuracy Progression Over Rounds', fontweight='bold')
    ax4.legend()
    ax4.grid(alpha=0.3)
else:
    ax4.text(0.5, 0.5, 'Round data not available', ha='center', va='center', transform=ax4.transAxes)

# 5. Energy Efficiency (Energy per Accuracy Point)
ax5 = plt.subplot(4, 3, 5)
bars = ax5.bar(range(len(experiments_10r)), experiments_10r['energy_per_accuracy'], 
               color='purple', alpha=0.7)
ax5.set_xticks(range(len(experiments_10r)))
ax5.set_xticklabels([f"{int(c)} clients" for c in experiments_10r['clients']], rotation=45, ha='right')
ax5.set_ylabel('Energy / Accuracy (J)', fontweight='bold')
ax5.set_title('Energy Efficiency (Lower is Better)', fontweight='bold')
ax5.grid(axis='y', alpha=0.3)

# 6. Time Efficiency
ax6 = plt.subplot(4, 3, 6)
bars = ax6.bar(range(len(experiments_10r)), experiments_10r['time_per_accuracy'], 
               color='orange', alpha=0.7)
ax6.set_xticks(range(len(experiments_10r)))
ax6.set_xticklabels([f"{int(c)} clients" for c in experiments_10r['clients']], rotation=45, ha='right')
ax6.set_ylabel('Time / Accuracy (s)', fontweight='bold')
ax6.set_title('Time Efficiency (Lower is Better)', fontweight='bold')
ax6.grid(axis='y', alpha=0.3)

# 7. Client Accuracy Fairness (Std Dev)
ax7 = plt.subplot(4, 3, 7)
bars = ax7.bar(range(len(experiments_10r)), experiments_10r['avg_client_accuracy_std'], 
               color='teal', alpha=0.7)
ax7.set_xticks(range(len(experiments_10r)))
ax7.set_xticklabels([f"{int(c)} clients" for c in experiments_10r['clients']], rotation=45, ha='right')
ax7.set_ylabel('Avg Client Accuracy Std Dev', fontweight='bold')
ax7.set_title('Client Fairness (Lower is Better)', fontweight='bold')
ax7.grid(axis='y', alpha=0.3)

# 8. Energy per Client
ax8 = plt.subplot(4, 3, 8)
bars = ax8.bar(range(len(experiments_10r)), experiments_10r['energy_per_client'], 
               color='darkred', alpha=0.7)
ax8.set_xticks(range(len(experiments_10r)))
ax8.set_xticklabels([f"{int(c)} clients" for c in experiments_10r['clients']], rotation=45, ha='right')
ax8.set_ylabel('Energy per Client (J)', fontweight='bold')
ax8.set_title('Average Energy per Client', fontweight='bold')
ax8.grid(axis='y', alpha=0.3)

# 9. Accuracy vs Energy Trade-off
ax9 = plt.subplot(4, 3, 9)
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
ax10 = plt.subplot(4, 3, 10)
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
    ax10.set_xticklabels([f"{int(c)} clients" for c in experiments_10r['clients']], rotation=45, ha='right')
    ax10.set_ylabel('Energy (kJ)', fontweight='bold')
    ax10.set_title('Energy Distribution by Component', fontweight='bold')
    ax10.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax10.grid(axis='y', alpha=0.3)
else:
    ax10.text(0.5, 0.5, 'Energy data not available', ha='center', va='center', transform=ax10.transAxes)

# 11. Scalability: Metrics vs Number of Clients
ax11 = plt.subplot(4, 3, 11)
ax11_twin = ax11.twinx()

ln1 = ax11.plot(experiments_10r['clients'], experiments_10r['GlobalAccuracy'], 
               'o-', color='green', label='Accuracy', linewidth=2, markersize=10)
ln2 = ax11_twin.plot(experiments_10r['clients'], experiments_10r['energy_kj'], 
                    's-', color='red', label='Energy (kJ)', linewidth=2, markersize=10)

ax11.set_xlabel('Number of Clients', fontweight='bold')
ax11.set_ylabel('Final Global Accuracy', color='green', fontweight='bold')
ax11_twin.set_ylabel('Total Energy (kJ)', color='red', fontweight='bold')
ax11.set_title('Scalability: Accuracy & Energy vs # Clients', fontweight='bold')
ax11.tick_params(axis='y', labelcolor='green')
ax11_twin.tick_params(axis='y', labelcolor='red')
ax11.grid(alpha=0.3)

lns = ln1 + ln2
labs = [l.get_label() for l in lns]
ax11.legend(lns, labs, loc='best')

# 12. Summary Table
ax12 = plt.subplot(4, 3, 12)
ax12.axis('tight')
ax12.axis('off')

summary_stats = [
    ['Metric', 'Mean', 'Min', 'Max', 'Range'],
    ['Accuracy', 
     f"{experiments_10r['GlobalAccuracy'].mean():.4f}",
     f"{experiments_10r['GlobalAccuracy'].min():.4f}",
     f"{experiments_10r['GlobalAccuracy'].max():.4f}",
     f"{experiments_10r['GlobalAccuracy'].max() - experiments_10r['GlobalAccuracy'].min():.4f}"],
    ['Energy (kJ)', 
     f"{experiments_10r['energy_kj'].mean():.1f}",
     f"{experiments_10r['energy_kj'].min():.1f}",
     f"{experiments_10r['energy_kj'].max():.1f}",
     f"{experiments_10r['energy_kj'].max() - experiments_10r['energy_kj'].min():.1f}"],
    ['Time (s)', 
     f"{experiments_10r['TotalTrainingTime'].mean():.0f}",
     f"{experiments_10r['TotalTrainingTime'].min():.0f}",
     f"{experiments_10r['TotalTrainingTime'].max():.0f}",
     f"{experiments_10r['TotalTrainingTime'].max() - experiments_10r['TotalTrainingTime'].min():.0f}"],
    ['Fairness (σ)', 
     f"{experiments_10r['avg_client_accuracy_std'].mean():.4f}",
     f"{experiments_10r['avg_client_accuracy_std'].min():.4f}",
     f"{experiments_10r['avg_client_accuracy_std'].max():.4f}",
     f"{experiments_10r['avg_client_accuracy_std'].max() - experiments_10r['avg_client_accuracy_std'].min():.4f}"],
]

table = ax12.table(cellText=summary_stats, cellLoc='center', loc='center',
                  colWidths=[0.25, 0.18, 0.18, 0.18, 0.18])
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2.5)

# Style header row
for i in range(5):
    table[(0, i)].set_facecolor('#3498db')
    table[(0, i)].set_text_props(weight='bold', color='white')

ax12.set_title('Summary Statistics', fontweight='bold', pad=20, fontsize=12)

plt.suptitle(f'Federated Learning: {rounds_to_compare} Rounds Comparison', 
             fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/comparison_{rounds_to_compare}rounds.png', dpi=300, bbox_inches='tight')
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
print(f"✓ Visualization saved: {OUTPUT_DIR}/comparison_{rounds_to_compare}rounds.png")
print("=" * 80)