import pandas as pd

# Read both CSV files
df_precise = pd.read_csv('fabric/experiments/09-01-26-115008/energy_consumption_precise.csv')
df_regular = pd.read_csv('fabric/experiments/09-01-26-115008/energy_consumption.csv')

# Convert joules column to float (in case it was read as string)
df_precise['joules'] = pd.to_numeric(df_precise['joules'], errors='coerce')
df_regular['joules'] = pd.to_numeric(df_regular['joules'], errors='coerce')

print("=== DATA TYPES ===")
print(f"Precise dtypes:\n{df_precise.dtypes}\n")
print(f"Regular dtypes:\n{df_regular.dtypes}\n")

print("=== SAMPLE DATA ===")
print("\nPrecise method (first 10 rows):")
print(df_precise.head(10))

print("\nRegular method (first 10 rows):")
print(df_regular.head(10))

print("\n=== OVERALL COMPARISON ===")
print(f"Total energy (precise): {df_precise['joules'].sum():.2f} J")
print(f"Total energy (regular): {df_regular['joules'].sum():.2f} J")
print(f"Difference: {abs(df_precise['joules'].sum() - df_regular['joules'].sum()):.2f} J")
print(f"Percent difference: {abs(df_precise['joules'].sum() - df_regular['joules'].sum()) / df_regular['joules'].sum() * 100:.2f}%")

# Filter for client1, client2, client3 only
clients_of_interest = ['client1', 'client2', 'client3']
df_precise_clients = df_precise[df_precise['namespace'].isin(clients_of_interest)]
df_regular_clients = df_regular[df_regular['namespace'].isin(clients_of_interest)]

print("\n=== CLIENT1, CLIENT2, CLIENT3 ONLY ===")
print(f"Precise - Client energy: {df_precise_clients['joules'].sum():.2f} J")
print(f"Regular - Client energy: {df_regular_clients['joules'].sum():.2f} J")
print(f"Difference: {abs(df_precise_clients['joules'].sum() - df_regular_clients['joules'].sum()):.2f} J")

# Group by namespace
print("\n=== PER CLIENT TOTALS ===")
precise_by_client = df_precise_clients.groupby('namespace')['joules'].sum().sort_index()
regular_by_client = df_regular_clients.groupby('namespace')['joules'].sum().sort_index()

comparison = pd.DataFrame({
    'Precise (J)': precise_by_client,
    'Regular (J)': regular_by_client,
    'Difference (J)': precise_by_client - regular_by_client,
    'Percent_Diff (%)': ((precise_by_client - regular_by_client) / regular_by_client * 100).round(2)
})
print(comparison)

# Merge dataframes to compare row by row
merged = df_precise.merge(
    df_regular, 
    on=['namespace', 'pod_name', 'container_name'],
    suffixes=('_precise', '_regular')
)

# Add difference column
merged['difference'] = merged['joules_precise'] - merged['joules_regular']
merged['percent_diff'] = (merged['difference'] / merged['joules_regular'] * 100).round(2)

# Show containers with biggest differences
print("\n=== TOP 10 CONTAINERS WITH LARGEST DIFFERENCES ===")
print(merged.nlargest(10, 'difference')[['namespace', 'container_name', 'joules_precise', 'joules_regular', 'difference', 'percent_diff']])

# Check for any containers with zero or negative values
print("\n=== CHECKING FOR ANOMALIES ===")
print(f"Precise - Zero values: {(df_precise['joules'] == 0).sum()}")
print(f"Precise - Negative values: {(df_precise['joules'] < 0).sum()}")
print(f"Regular - Zero values: {(df_regular['joules'] == 0).sum()}")
print(f"Regular - Negative values: {(df_regular['joules'] < 0).sum()}")

# Statistical summary
print("\n=== STATISTICAL SUMMARY OF DIFFERENCES ===")
print(merged['difference'].describe())