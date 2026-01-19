import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import seaborn as sns

# Set style
sns.set_style("whitegrid")

# Read the data
energy_df = pd.read_csv('/home/maurits/EnergyEfficient_Scattered-Directive/fabric/data/3/30/12-01-26-153806/energy_consumption.csv')
client_metrics_df = pd.read_csv('/home/maurits/EnergyEfficient_Scattered-Directive/fabric/data/3/30/12-01-26-153806/client_stats.csv')

# Dataset sizes
dataset_sizes = {
    'client2': 10570,
    'client5': 17938,
    'client10': 14812
}

# Geographical locations (server is in Amsterdam)
locations = {
    'client2': 'Los Angeles',
    'client5': 'Amsterdam',
    'client10': 'Tokyo'
}

# Approximate network distances and latencies to Amsterdam
network_info = {
    'client2': {'distance_km': 8800, 'approx_latency_ms': 150},
    'client5': {'distance_km': 0, 'approx_latency_ms': 1},
    'client10': {'distance_km': 9300, 'approx_latency_ms': 240}
}

# Calculate mean training times (in milliseconds)
mean_training_times = {}
for client_id in client_metrics_df['ClientID'].unique():
    client_data = client_metrics_df[client_metrics_df['ClientID'] == client_id]
    mean_training_times[client_id] = client_data['ClientTrainingTime'].mean() * 1000

# Remove linkerd-init containers
energy_clean = energy_df[energy_df['container_name'] != 'linkerd-init'].copy()

# Calculate total energy by namespace and by container type
energy_by_namespace = energy_clean.groupby('namespace')['joules'].sum()

# Break down energy by container type for each client
print("=" * 80)
print("DETAILED ENERGY BREAKDOWN BY CONTAINER TYPE")
print("=" * 80)

energy_breakdown = {}
for client in ['client2', 'client5', 'client10']:
    client_energy = energy_clean[energy_clean['namespace'] == client]
    breakdown = client_energy.groupby('container_name')['joules'].sum()
    energy_breakdown[client] = breakdown
    
    print(f"\n{client.upper()} Energy Breakdown:")
    print(f"  Dataset Size: {dataset_sizes[client]} rows")
    print(f"  Mean Training Time: {mean_training_times[client]:.2f} ms")
    print(f"  Total Energy: {energy_by_namespace[client]:.2f} J")
    print(f"\n  By Container:")
    for container, energy in breakdown.items():
        percentage = (energy / energy_by_namespace[client]) * 100
        print(f"    {container:20s}: {energy:8.2f} J ({percentage:5.1f}%)")

# Calculate ratios
print("\n" + "=" * 80)
print("RATIO ANALYSIS")
print("=" * 80)

comparisons = [
    ('client2', 'client5', '1'),
    ('client2', 'client10', '2'),
    ('client10', 'client5', '3')
]

for client1, client2, label in comparisons:
    data_ratio = dataset_sizes[client1] / dataset_sizes[client2]
    time_ratio = mean_training_times[client1] / mean_training_times[client2]
    energy_ratio = energy_by_namespace[client1] / energy_by_namespace[client2]
    
    print(f"\n{label}. {client1.upper()} vs {client2.upper()}:")
    print(f"   Data Ratio:          {data_ratio:.3f} ({dataset_sizes[client1]}/{dataset_sizes[client2]})")
    print(f"   Training Time Ratio: {time_ratio:.3f} ({mean_training_times[client1]:.0f}/{mean_training_times[client2]:.0f} ms)")
    print(f"   Energy Ratio:        {energy_ratio:.3f} ({energy_by_namespace[client1]:.0f}/{energy_by_namespace[client2]:.0f} J)")
    print(f"   Time/Data Alignment: {(time_ratio/data_ratio):.3f} (close to 1.0 = proportional)")
    print(f"   Energy/Data Alignment: {(energy_ratio/data_ratio):.3f} (close to 1.0 = proportional)")
    print(f"   Energy/Time Alignment: {(energy_ratio/time_ratio):.3f} (close to 1.0 = proportional)")

# Analyze energy components
print("\n" + "=" * 80)
print("ENERGY COMPONENT ANALYSIS")
print("=" * 80)

for client in ['client2', 'client5', 'client10']:
    client_energy = energy_clean[energy_clean['namespace'] == client]
    
    # Separate training containers from infrastructure
    training_containers = client_energy[client_energy['container_name'].str.contains('hfl-train|' + client)]
    infrastructure_containers = client_energy[~client_energy['container_name'].str.contains('hfl-train|' + client)]
    
    training_energy = training_containers['joules'].sum()
    infrastructure_energy = infrastructure_containers['joules'].sum()
    total_energy = energy_by_namespace[client]
    
    print(f"\n{client.upper()}:")
    print(f"  Training-related Energy:      {training_energy:8.2f} J ({training_energy/total_energy*100:5.1f}%)")
    print(f"  Infrastructure Energy:        {infrastructure_energy:8.2f} J ({infrastructure_energy/total_energy*100:5.1f}%)")
    print(f"  Energy per Training Second:   {training_energy/(mean_training_times[client]/1000):.2f} J/s")
    print(f"  Energy per Data Point:        {total_energy/dataset_sizes[client]:.4f} J/row")

# Visualizations
fig = plt.figure(figsize=(18, 12))

# 1. Dataset Size vs Mean Training Time
ax1 = plt.subplot(2, 3, 1)
clients = list(dataset_sizes.keys())
sizes = [dataset_sizes[c] for c in clients]
times = [mean_training_times[c] for c in clients]

ax1.scatter(sizes, times, s=200, alpha=0.6, color='steelblue')
for i, client in enumerate(clients):
    ax1.annotate(client, (sizes[i], times[i]), fontsize=11, ha='right')

# Add trend line
z = np.polyfit(sizes, times, 1)
p = np.poly1d(z)
x_line = np.linspace(min(sizes), max(sizes), 100)
ax1.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2, label=f'Linear fit: R²={np.corrcoef(sizes, times)[0,1]**2:.3f}')

ax1.set_xlabel('Dataset Size (rows)', fontweight='bold')
ax1.set_ylabel('Mean Training Time (ms)', fontweight='bold')
ax1.set_title('Dataset Size vs Training Time\n(Strong Linear Relationship)')
ax1.legend()
ax1.grid(alpha=0.3)

# 2. Dataset Size vs Total Energy
ax2 = plt.subplot(2, 3, 2)
energies = [energy_by_namespace[c] for c in clients]

ax2.scatter(sizes, energies, s=200, alpha=0.6, color='coral')
for i, client in enumerate(clients):
    ax2.annotate(client, (sizes[i], energies[i]), fontsize=11, ha='right')

# Add trend line
z = np.polyfit(sizes, energies, 1)
p = np.poly1d(z)
ax2.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2, label=f'Linear fit: R²={np.corrcoef(sizes, energies)[0,1]**2:.3f}')

ax2.set_xlabel('Dataset Size (rows)', fontweight='bold')
ax2.set_ylabel('Total Energy (J)', fontweight='bold')
ax2.set_title('Dataset Size vs Total Energy\n(Weaker Relationship)')
ax2.legend()
ax2.grid(alpha=0.3)

# 3. Training Time vs Total Energy
ax3 = plt.subplot(2, 3, 3)
ax3.scatter(times, energies, s=200, alpha=0.6, color='green')
for i, client in enumerate(clients):
    ax3.annotate(client, (times[i], energies[i]), fontsize=11, ha='right')

# Add trend line
z = np.polyfit(times, energies, 1)
p = np.poly1d(z)
x_line_time = np.linspace(min(times), max(times), 100)
ax3.plot(x_line_time, p(x_line_time), "r--", alpha=0.8, linewidth=2, label=f'Linear fit: R²={np.corrcoef(times, energies)[0,1]**2:.3f}')

ax3.set_xlabel('Mean Training Time (ms)', fontweight='bold')
ax3.set_ylabel('Total Energy (J)', fontweight='bold')
ax3.set_title('Training Time vs Total Energy\n(Moderate Relationship)')
ax3.legend()
ax3.grid(alpha=0.3)

# 4. Energy breakdown stacked bar
ax4 = plt.subplot(2, 3, 4)
container_types = set()
for breakdown in energy_breakdown.values():
    container_types.update(breakdown.index)
container_types = sorted(list(container_types))

bottom = np.zeros(len(clients))
colors = cm.get_cmap('Set3')(np.linspace(0, 1, len(container_types)))

for i, container in enumerate(container_types):
    values = [energy_breakdown[c].get(container, 0) for c in clients]
    ax4.bar(clients, values, bottom=bottom, label=container, color=colors[i])
    bottom += values

ax4.set_ylabel('Energy (J)', fontweight='bold')
ax4.set_title('Energy Breakdown by Container Type')
ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
ax4.grid(axis='y', alpha=0.3)

# 5. Ratio comparison
ax5 = plt.subplot(2, 3, 5)
comparison_data = []
for client1, client2, label in comparisons:
    data_ratio = dataset_sizes[client1] / dataset_sizes[client2]
    time_ratio = mean_training_times[client1] / mean_training_times[client2]
    energy_ratio = energy_by_namespace[client1] / energy_by_namespace[client2]
    comparison_data.append({
        'Comparison': f'{client1}\nvs\n{client2}',
        'Data Ratio': data_ratio,
        'Time Ratio': time_ratio,
        'Energy Ratio': energy_ratio
    })

comp_df = pd.DataFrame(comparison_data)
x = np.arange(len(comp_df))
width = 0.25

ax5.bar(x - width, comp_df['Data Ratio'], width, label='Data Ratio', color='steelblue')
ax5.bar(x, comp_df['Time Ratio'], width, label='Time Ratio', color='orange')
ax5.bar(x + width, comp_df['Energy Ratio'], width, label='Energy Ratio', color='coral')

ax5.set_ylabel('Ratio', fontweight='bold')
ax5.set_title('Comparison of Ratios\n(Time tracks Data, Energy diverges)')
ax5.set_xticks(x)
ax5.set_xticklabels(comp_df['Comparison'], fontsize=9)
ax5.legend()
ax5.grid(axis='y', alpha=0.3)
ax5.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, linewidth=1)

# 6. Energy efficiency (Energy per data point)
ax6 = plt.subplot(2, 3, 6)
energy_per_datapoint = [energy_by_namespace[c] / dataset_sizes[c] for c in clients]

bars = ax6.bar(clients, energy_per_datapoint, color=['steelblue', 'coral', 'green'], alpha=0.7)
ax6.set_ylabel('Energy per Data Point (J/row)', fontweight='bold')
ax6.set_title('Energy Efficiency by Client\n(Shows Non-Uniform Overhead)')
ax6.grid(axis='y', alpha=0.3)

# Add values on bars
for i, (client, val) in enumerate(zip(clients, energy_per_datapoint)):
    ax6.text(i, val, f'{val:.4f}', ha='center', va='bottom', fontweight='bold')

file_name = 'energy_time_analysis_L_30.png'
plt.tight_layout()
plt.savefig(file_name, dpi=300, bbox_inches='tight')
# plt.show()

# Add network overhead analysis
print("\n" + "=" * 80)
print("NETWORK OVERHEAD ESTIMATION")
print("=" * 80)

for client in ['client2', 'client5', 'client10']:
    # Estimate network-related energy (linkerd-proxy + part of sidecar)
    client_energy = energy_clean[energy_clean['namespace'] == client]
    network_containers = client_energy[client_energy['container_name'].isin(['linkerd-proxy', 'sidecar'])]
    network_energy = network_containers['joules'].sum()
    
    total_energy = energy_by_namespace[client]
    computation_energy = total_energy - network_energy
    
    print(f"\n{client.upper()} - {locations[client]}:")
    print(f"  Estimated Network Energy:     {network_energy:8.2f} J ({network_energy/total_energy*100:5.1f}%)")
    print(f"  Estimated Computation Energy: {computation_energy:8.2f} J ({computation_energy/total_energy*100:5.1f}%)")
    print(f"  Network Energy per Round:     {network_energy/30:8.2f} J (30 rounds total)")
    print(f"  Computation per Data Point:   {computation_energy/dataset_sizes[client]:.4f} J/row")

print(f"\nAnalysis complete! Figure saved as {file_name}")
print("=" * 80)