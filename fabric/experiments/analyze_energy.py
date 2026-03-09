import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_style("white")
sns.set_palette("deep")
plt.rcParams["font.family"] = "sans-serif"

TOTAL_DATASET_SIZE = 531130
ROUNDS = 25

df_energy = pd.read_csv("analysis_output/combined_energy_stats.csv")

df_filtered = df_energy[
    (df_energy["K"].isin([5, 10])) & (df_energy["Z"].isin([90, 120]))
]


def categorize_container(container_name):
    name = str(container_name).lower()
    if "sidecar" in name:
        return "sidecar"
    elif "linkerd-proxy" in name:
        return "linkerd-proxy"
    elif "hfl-train-model" in name:
        return "hfl-train-model"
    elif "hfl-train" in name:
        return "hfl-train"
    elif name.startswith("client"):
        return "client-apps"
    elif "server" in name:
        return "server"
    elif "api-gateway" in name:
        return "api-gateway"
    elif "orchestrator" in name:
        return "orchestrator"
    else:
        return "other"


df_filtered["category"] = df_filtered["container_name"].apply(categorize_container)

# Compute metrics per (K, Z), then average over Z
metrics_data = []
for k in [5, 10]:
    for z in [90, 120]:
        kz_data = df_filtered[(df_filtered["K"] == k) & (df_filtered["Z"] == z)]
        total_joules = kz_data["joules"].sum()
        total_kJ = total_joules / 1000
        samples_per_partition = TOTAL_DATASET_SIZE / z
        total_samples = samples_per_partition * k
        metrics_data.append(
            {
                "K": k,
                "Z": z,
                "total_kJ": total_kJ,
                "energy_per_round_kJ": total_kJ / ROUNDS,
                "energy_per_client_kJ": total_kJ / k,
                "energy_per_sample_J": total_joules / total_samples,
            }
        )

metrics_df = pd.DataFrame(metrics_data)

# Average over Z for each K
avg_metrics = metrics_df.groupby("K").mean(numeric_only=True).reset_index()

# Average container breakdown over Z for each K
container_breakdown = (
    df_filtered.groupby(["K", "Z", "category"])["joules"].sum().reset_index()
)
container_breakdown["kJ"] = container_breakdown["joules"] / 1000
avg_container = (
    container_breakdown.groupby(["K", "category"])["kJ"].mean().reset_index()
)

container_colors = {
    "sidecar": "#E63946",
    "linkerd-proxy": "#F77F00",
    "hfl-train": "#118AB2",
    "hfl-train-model": "#06A77D",
    "client-apps": "#FFB703",
    "server": "#8338EC",
    "api-gateway": "#3A86FF",
    "orchestrator": "#FB5607",
    "other": "#6C757D",
}

k_values = [5, 10]
x = np.arange(len(k_values))
width = 0.5
colors_k = sns.color_palette("deep", 2)

fig = plt.figure(figsize=(15, 7))
gs = fig.add_gridspec(1, 3, hspace=0.4, wspace=0.35)


def plot_metric(ax, metric, xlabel, ylabel, title, fmt=".0f", offset=None):
    values = [avg_metrics[avg_metrics["K"] == k][metric].values[0] for k in k_values]
    bars = ax.bar(
        x, values, width, color=colors_k, alpha=0.85, edgecolor="white", linewidth=1.5
    )
    y_max = max(values) * 1.15
    ax.set_ylim(0, y_max)
    for i, val in enumerate(values):
        ax.text(
            i,
            val + y_max * 0.02,
            f"{val:{fmt}}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
    ax.set_xticks(x)
    ax.set_xticklabels([f"{k}" for k in k_values])
    ax.set_ylabel(ylabel, fontsize=11, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=11, fontweight="bold")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


plot_metric(
    fig.add_subplot(gs[0, 0]), "total_kJ", "", "Total Energy (kJ)", "Total Energy"
)
plot_metric(
    fig.add_subplot(gs[0, 1]),
    "energy_per_round_kJ",
    "Number of Clients",
    "Energy per Round (kJ)",
    "Energy per Round",
    fmt=".1f",
)
plot_metric(
    fig.add_subplot(gs[0, 2]),
    "energy_per_client_kJ",
    "",
    "Energy per Client (kJ)",
    "Energy per Client",
)
# plot_metric(
#     fig.add_subplot(gs[1, 0]),
#     "energy_per_sample_J",
#     "Energy per Sample (J)",
#     "Energy per Sample",
#     fmt=".2f",
# )

# # Stacked bar: container breakdown averaged over Z, grouped by K
# ax5 = fig.add_subplot(gs[1, 1:])
# categories = sorted(avg_container["category"].unique())

# for k_idx, k in enumerate(k_values):
#     bottom = 0
#     group = avg_container[avg_container["K"] == k]
#     for cat in categories:
#         cat_data = group[group["category"] == cat]
#         value = cat_data["kJ"].values[0] if len(cat_data) > 0 else 0
#         ax5.bar(
#             k_idx,
#             value,
#             width,
#             bottom=bottom,
#             label=cat if k_idx == 0 else None,
#             color=container_colors.get(cat, "#6C757D"),
#             alpha=0.85,
#             edgecolor="white",
#             linewidth=0.5,
#         )
#         bottom += value
#     ax5.text(
#         k_idx,
#         bottom + 10,
#         f"{bottom:.0f}",
#         ha="center",
#         va="bottom",
#         fontsize=10,
#         fontweight="bold",
#     )

# ax5.set_xticks(x)
# ax5.set_xticklabels([f"K={k}" for k in k_values])
# ax5.set_ylabel("Energy (kJ)", fontsize=11, fontweight="bold")
# ax5.set_title(
#     "Energy Breakdown by Container Type\n(averaged over Z=90 and Z=120)",
#     fontsize=12,
#     fontweight="bold",
# )
# ax5.legend(
#     title="Container Type", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=9
# )
# ax5.grid(True, alpha=0.3, axis="y")
# ax5.spines["top"].set_visible(False)
# ax5.spines["right"].set_visible(False)

plt.savefig(
    "analysis_output/energy_comparison_avg_z.png",
    dpi=300,
    bbox_inches="tight",
    facecolor="white",
)
plt.show()
