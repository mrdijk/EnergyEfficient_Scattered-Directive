import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)

# Read the data files
energy_df = pd.read_csv(
    '/home/maurits/EnergyEfficient_Scattered-Directive/fabric/experiments/data/combined_energy_stats.csv',
    index_col=0)
client_metrics_df = pd.read_csv(
    '/home/maurits/EnergyEfficient_Scattered-Directive/fabric/experiments/data/combined_client_stats.csv')
global_metrics_df = pd.read_csv(
    '/home/maurits/EnergyEfficient_Scattered-Directive/fabric/experiments/data/combined_global_stats.csv',
    index_col=0)

energy_df = energy_df[energy_df['exp'] == 'exp1']
client_metrics_df = client_metrics_df[client_metrics_df['exp'] == 'exp1']
global_metrics_df = global_metrics_df[global_metrics_df['exp'] == 'exp1']

# ============= ENERGY ANALYSIS =============
print("=" * 80)
print("ENERGY CONSUMPTION ANALYSIS")
print("=" * 80)

# Remove linkerd-init (0 energy)
energy_clean = energy_df[energy_df['container_name'] != 'linkerd-init'].copy()

# Total energy by namespace
energy_by_namespace = energy_clean.groupby('namespace')['joules'].sum().sort_values(ascending=False)
print("\nTotal Energy by Namespace:")
for ns, energy in energy_by_namespace.items():
    print(f"  {ns:20s}: {energy:10.2f} J ({energy/1000:.3f} kJ)")

# Energy by container type
energy_by_container = energy_clean.groupby('container_name')['joules'].sum().sort_values(ascending=False)
print("\nTotal Energy by Container Type:")
for container, energy in energy_by_container.items():
    print(f"  {container:20s}: {energy:10.2f} J")

# Client-specific energy (main containers only)
client_energy = energy_clean[energy_clean['container_name'].str.startswith('client')].groupby('namespace')['joules'].sum().sort_values(ascending=False)
print("\nClient Main Container Energy:")
for client, energy in client_energy.items():
    print(f"  {client:20s}: {energy:10.2f} J")

print(f"\nTotal System Energy: {energy_clean['joules'].sum():.2f} J ({energy_clean['joules'].sum()/1000:.3f} kJ)")

# ============= TRAINING PERFORMANCE ANALYSIS =============
print("\n" + "=" * 80)
print("TRAINING PERFORMANCE ANALYSIS")
print("=" * 80)

# Client performance over rounds
print("\nClient Performance Summary:")
for client_id in client_metrics_df['ClientID'].unique():
    client_data = client_metrics_df[client_metrics_df['ClientID'] == client_id]
    avg_acc = client_data['ClientAccuracy'].mean()
    final_acc = client_data['ClientAccuracy'].iloc[-1]
    avg_time = client_data['ClientTrainingTime'].mean()

    print(f"\n{client_id}:")
    print(f"  Average Accuracy: {avg_acc:.4f}")
    print(f"  Final Accuracy:   {final_acc:.4f}")
    print(f"  Avg Training Time: {avg_time:.2f} seconds")
    print(f"  Improvement: {((final_acc - client_data['ClientAccuracy'].iloc[0])/client_data['ClientAccuracy'].iloc[0])*100:+.2f}%")

# Global model performance
print("\nGlobal Model Performance:")
print(f"  Initial Accuracy: {global_metrics_df['GlobalAccuracy'].iloc[0]:.5f}")
print(f"  Final Accuracy:   {global_metrics_df['GlobalAccuracy'].iloc[-1]:.5f}")
print(f"  Best Accuracy:    {global_metrics_df['GlobalAccuracy'].max():.5f} (Round {global_metrics_df['GlobalAccuracy'].idxmax()})")
print(f"  Improvement:      {((global_metrics_df['GlobalAccuracy'].iloc[-1] - global_metrics_df['GlobalAccuracy'].iloc[0])/global_metrics_df['GlobalAccuracy'].iloc[0])*100:+.2f}%")

print(f"\n  Avg Aggregation Time: {global_metrics_df['AggregationTime'].mean():.2f} seconds")
print(f"  Avg Round Duration:   {global_metrics_df['RoundDuration'].mean()/1e9:.2f} seconds")

# ============= ENERGY EFFICIENCY ANALYSIS =============
print("\n" + "=" * 80)
print("ENERGY EFFICIENCY ANALYSIS")
print("=" * 80)

# Calculate energy per training instance
for client_id in client_metrics_df['ClientID'].unique():
    client_data = client_metrics_df[client_metrics_df['ClientID'] == client_id]
    total_training_time = client_data['ClientTrainingTime'].sum()

    # Get energy for this client's main container
    namespace = client_id
    if namespace in energy_by_namespace.index:
        client_total_energy = energy_by_namespace[namespace]
        energy_per_second = client_total_energy / total_training_time if total_training_time > 0 else 0
        final_acc = client_data['ClientAccuracy'].iloc[-1]

        print(f"\n{client_id}:")
        print(f"  Total Energy: {client_total_energy:.2f} J")
        print(f"  Total Training Time: {total_training_time:.2f} seconds")
        print(f"  Energy per Second: {energy_per_second:.4f} J/s")
        print(f"  Energy per Accuracy Point: {client_total_energy/final_acc:.2f} J")

# ============= VISUALIZATIONS =============
fig = plt.figure(figsize=(20, 12))

# ROW 1: ACCURACY METRICS
# 1. Client accuracy over rounds
ax1 = plt.subplot(3, 3, 1)
for client_id in client_metrics_df['ClientID'].unique():
    client_data = client_metrics_df[client_metrics_df['ClientID'] == client_id]
    ax1.plot(client_data['Round'], client_data['ClientAccuracy'], marker='o', label=client_id, linewidth=2)
ax1.set_xlabel('Round')
ax1.set_ylabel('Accuracy')
ax1.set_title('Client Accuracy Over Training Rounds')
ax1.legend()
ax1.grid(alpha=0.3)

# 2. Global accuracy over rounds
ax2 = plt.subplot(3, 3, 2)
ax2.plot(global_metrics_df.index, global_metrics_df['GlobalAccuracy'], marker='o', linewidth=2, color='green')
ax2.set_xlabel('Round')
ax2.set_ylabel('Global Accuracy')
ax2.set_title('Global Model Accuracy Over Rounds')
ax2.grid(alpha=0.3)

# 3. Client accuracy comparison (box plot)
ax3 = plt.subplot(3, 3, 3)
client_metrics_df.boxplot(column='ClientAccuracy', by='ClientID', ax=ax3)
ax3.set_xlabel('Client')
ax3.set_ylabel('Accuracy')
ax3.set_title('Client Accuracy Distribution')
plt.suptitle('')  # Remove default title

# ROW 2: TRAINING TIME METRICS
# 4. Training time per client per round
ax4 = plt.subplot(3, 3, 4)
for client_id in client_metrics_df['ClientID'].unique():
    client_data = client_metrics_df[client_metrics_df['ClientID'] == client_id]
    ax4.plot(client_data['Round'], client_data['ClientTrainingTime'], marker='o', label=client_id, linewidth=2)
ax4.set_xlabel('Round')
ax4.set_ylabel('Training Time (seconds)')
ax4.set_title('Client Training Time Per Round')
ax4.legend()
ax4.grid(alpha=0.3)

# 5. Total training time per round
ax5 = plt.subplot(3, 3, 5)
ax5.plot(global_metrics_df.index, global_metrics_df['TotalTrainingTime'], marker='o', linewidth=2, color='orange')
ax5.set_xlabel('Round')
ax5.set_ylabel('Total Training Time (seconds)')
ax5.set_title('Total Training Time Per Round')
ax5.grid(alpha=0.3)

# 6. Aggregation time over rounds
ax6 = plt.subplot(3, 3, 6)
ax6.plot(global_metrics_df.index, global_metrics_df['AggregationTime'], marker='o', linewidth=2, color='purple')
ax6.set_xlabel('Round')
ax6.set_ylabel('Aggregation Time (seconds)')
ax6.set_title('Aggregation Time Per Round')
ax6.grid(alpha=0.3)

# ROW 3: ENERGY METRICS
# 7. Energy consumption by namespace
ax7 = plt.subplot(3, 3, 7)
energy_by_namespace.plot(kind='barh', ax=ax7, color='steelblue')
ax7.set_xlabel('Energy (Joules)')
ax7.set_title('Total Energy by Namespace')
ax7.grid(axis='x', alpha=0.3)

# 8. Energy by container type
ax8 = plt.subplot(3, 3, 8)
energy_by_container.head(10).plot(kind='barh', ax=ax8, color='coral')
ax8.set_xlabel('Energy (Joules)')
ax8.set_title('Energy by Container Type (Top 10)')
ax8.grid(axis='x', alpha=0.3)

# 9. Energy efficiency scatter
ax9 = plt.subplot(3, 3, 9)
efficiency_data = []
for client_id in client_metrics_df['ClientID'].unique():
    client_data = client_metrics_df[client_metrics_df['ClientID'] == client_id]
    final_acc = client_data['ClientAccuracy'].iloc[-1]
    if client_id in energy_by_namespace.index:
        energy = energy_by_namespace[client_id]
        efficiency_data.append({'Client': client_id, 'Energy': energy, 'Accuracy': final_acc})

if efficiency_data:
    eff_df = pd.DataFrame(efficiency_data)
    ax9.scatter(eff_df['Energy'], eff_df['Accuracy'], s=200, alpha=0.6)
    for _, row in eff_df.iterrows():
        ax9.annotate(row['Client'], (row['Energy'], row['Accuracy']), fontsize=9)
    ax9.set_xlabel('Total Energy (Joules)')
    ax9.set_ylabel('Final Accuracy')
    ax9.set_title('Energy vs Accuracy Trade-off')
    ax9.grid(alpha=0.3)

file_name = 'large_clients_30_r_analysis.png'
plt.tight_layout()
plt.savefig(file_name, dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "=" * 80)
print(f"Analysis complete! Figure saved as {file_name}")
print("=" * 80)
