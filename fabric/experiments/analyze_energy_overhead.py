import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Set style
sns.set_style("whitegrid")
sns.set_palette("deep")

# Load data
df = pd.read_csv(
    "/home/maurits/EnergyEfficient_Scattered-Directive/fabric/experiments/analysis_output/exp1/combined_energy_stats.csv",
    index_col=[0],
)
df = df[df["K"] == 5]
active_clients = ["client1", "client5", "client9", "client13", "client17"]
entities = active_clients + ["server"]
orch_containers = ["orchestrator", "policy-enforcer", "api-gateway"]
infra_base = ["sidecar", "linkerd-proxy", "linkerd-init"]

results = []

# Process Clients and Server
for entity in entities:
    entity_df = df[df["namespace"] == entity]

    if entity == "server":
        train_container = "hfl-train-model"
    else:
        train_container = "hfl-train"

    training_energy_kj = (
        entity_df[entity_df["container_name"] == train_container]["joules"].sum()
        / 1000.0
    )

    # Infra components in this namespace
    agent_energy_kj = (
        entity_df[entity_df["container_name"] == entity]["joules"].sum() / 1000.0
    )
    sidecar_energy_kj = (
        entity_df[entity_df["container_name"] == "sidecar"]["joules"].sum() / 1000.0
    )
    linkerd_energy_kj = (
        entity_df[entity_df["container_name"].isin(["linkerd-proxy", "linkerd-init"])][
            "joules"
        ].sum()
        / 1000.0
    )

    infra_total_kj = agent_energy_kj + sidecar_energy_kj + linkerd_energy_kj
    infra_no_linkerd_kj = agent_energy_kj + sidecar_energy_kj

    results.append(
        {
            "Entity": entity,
            "Training Cost (kJ)": training_energy_kj,
            "Total Infra Overhead (kJ)": infra_total_kj,
            "Infra Without Linkerd (kJ)": infra_no_linkerd_kj,
        }
    )

# Process Orchestration (aggregated)
# Orchestration containers might be in different namespaces (like 'orchestrator', 'api-gateway', etc.)
# We treat the trio as one 'Orchestration' entity.
orch_mask = df["container_name"].isin(orch_containers)
orch_pods = df[orch_mask]["pod_name"].unique()
orch_df = df[df["pod_name"].isin(orch_pods)]

orch_agent_energy_kj = (
    orch_df[orch_df["container_name"].isin(orch_containers)]["joules"].sum() / 1000.0
)
orch_sidecar_energy_kj = (
    orch_df[orch_df["container_name"] == "sidecar"]["joules"].sum() / 1000.0
)
orch_linkerd_energy_kj = (
    orch_df[orch_df["container_name"].isin(["linkerd-proxy", "linkerd-init"])][
        "joules"
    ].sum()
    / 1000.0
)

orch_total_infra_kj = (
    orch_agent_energy_kj + orch_sidecar_energy_kj + orch_linkerd_energy_kj
)
orch_no_linkerd_kj = orch_agent_energy_kj + orch_sidecar_energy_kj

results.append(
    {
        "Entity": "Orchestration",
        "Training Cost (kJ)": 0.0,
        "Total Infra Overhead (kJ)": orch_total_infra_kj,
        "Infra Without Linkerd (kJ)": orch_no_linkerd_kj,
    }
)

plot_df = pd.DataFrame(results)
all_labels = plot_df["Entity"].tolist()

# Create the plot
x = np.arange(len(all_labels))
width = 0.25  # the width of the bars

fig, ax = plt.subplots(figsize=(14, 8))

# Define colors
# colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

rects1 = ax.bar(
    x - width,
    plot_df["Training Cost (kJ)"],
    width,
    label="Training Cost",
    color="#06A77D",
    # edgecolor="black",
)
rects2 = ax.bar(
    x,
    plot_df["Total Infra Overhead (kJ)"],
    width,
    label="Total Infra Overhead",
    color="#E63946",
    # edgecolor="black",
)
rects3 = ax.bar(
    x + width,
    plot_df["Infra Without Linkerd (kJ)"],
    width,
    label="Infra Without Linkerd",
    # color=colors[2],
    # edgecolor="black",
)

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel("Energy (kJ)")
ax.set_title("Energy Breakdown per Client and Server (in kJ)")
ax.set_xticks(x)
ax.set_xticklabels(all_labels)
ax.legend()
ax.grid(True, alpha=0.3, axis="y")


# Function to add labels on top of bars
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(
            f"{height:.1f}",
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )


autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

plt.tight_layout()
plt.savefig("analysis_output/plots/combined_energy_breakdown.png")

# Output a summary table for the user
print(plot_df.to_string(index=False))
