import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Set style
sns.set_style("white")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 11

TOTAL_DATASET_SIZE = 531130
MODEL_SIZE_MB = 6.1


def load_data():
    """Load energy and global stats data."""
    base_path = "analysis_output"

    df_energy = pd.read_csv(f"{base_path}/combined_energy_stats.csv")
    df_global = pd.read_csv(f"{base_path}/combined_global_stats.csv", index_col=0)

    return df_energy, df_global


def calculate_metrics(df_energy, df_global):
    """Calculate energy and dataset metrics."""

    # Energy metrics per K and Z
    energy_summary = (
        df_energy.groupby(["K", "Z", "timestamp"])
        .agg({"joules": "sum", "namespace": "nunique"})
        .reset_index()
    )
    energy_summary.columns = ["K", "Z", "timestamp", "total_energy_J", "num_services"]

    # Average across experiments
    avg_energy = (
        energy_summary.groupby(["K", "Z"])["total_energy_J"].mean().reset_index()
    )
    avg_energy["total_energy_kJ"] = avg_energy["total_energy_J"] / 1000

    # Calculate dataset metrics
    bytes_per_sample = 32 * 32 * 3  # SVHN image size
    avg_energy["samples_per_partition"] = TOTAL_DATASET_SIZE / avg_energy["Z"]
    avg_energy["partition_size_mb"] = (
        avg_energy["samples_per_partition"] * bytes_per_sample
    ) / (1024 * 1024)
    avg_energy["data_to_model_ratio"] = avg_energy["partition_size_mb"] / MODEL_SIZE_MB
    avg_energy["total_samples_processed"] = (
        avg_energy["samples_per_partition"] * avg_energy["K"]
    )
    avg_energy["energy_per_sample_J"] = (
        avg_energy["total_energy_J"] / avg_energy["total_samples_processed"]
    )
    avg_energy["energy_per_1k_samples_kJ"] = (
        avg_energy["energy_per_sample_J"] * 1000
    ) / 1000

    # Get final accuracy if available
    if not df_global.empty:
        final_acc = (
            df_global.groupby(["K", "Z", "timestamp"])["GlobalAccuracy"]
            .last()
            .reset_index()
        )
        avg_acc = final_acc.groupby(["K", "Z"])["GlobalAccuracy"].mean().reset_index()
        avg_energy = pd.merge(avg_energy, avg_acc, on=["K", "Z"], how="left")

    return avg_energy.sort_values(["K", "Z"])


def create_comprehensive_plots(avg_energy):
    """Create comprehensive energy vs dataset size analysis."""

    k_values = sorted(avg_energy["K"].unique())
    colors = ["#2E86AB", "#A23B72", "#F18F01"]
    markers = ["o", "s", "^"]

    # Create 3x2 figure
    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3)

    # Get reference data for secondary x-axes
    reference_data = avg_energy[avg_energy["K"] == k_values[0]].sort_values("Z")
    z_positions = reference_data["data_to_model_ratio"].values
    z_labels = reference_data["Z"].astype(int).values

    # Plot 1: Total Energy vs Data-to-Model Ratio
    ax1 = fig.add_subplot(gs[0, 0])
    ax1_top = ax1.twiny()

    for i, k in enumerate(k_values):
        k_data = avg_energy[avg_energy["K"] == k].sort_values("Z")
        ax1.plot(
            k_data["data_to_model_ratio"],
            k_data["total_energy_kJ"],
            marker=markers[i],
            linewidth=2.5,
            markersize=9,
            label=f"K={k}",
            color=colors[i],
            markeredgewidth=1.5,
            markeredgecolor="white",
            alpha=0.9,
        )

    ax1.set_xlabel("Data-to-Model Ratio", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Total Energy Consumption (kJ)", fontsize=12, fontweight="bold")
    ax1.set_title(
        "Total Energy vs Data-to-Model Ratio", fontsize=13, fontweight="bold", pad=30
    )
    ax1.legend(loc="best", fontsize=10, frameon=True, shadow=True)
    ax1.grid(False)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.spines["left"].set_linewidth(1.5)
    ax1.spines["bottom"].set_linewidth(1.5)

    ax1_top.set_xlim(ax1.get_xlim())
    ax1_top.set_xticks(z_positions)
    ax1_top.set_xticklabels(z_labels, fontsize=9)
    ax1_top.set_xlabel("Number of Partitions (Z)", fontsize=12, fontweight="bold")
    ax1_top.spines["top"].set_linewidth(1.5)
    ax1_top.spines["bottom"].set_visible(False)
    ax1_top.spines["left"].set_visible(False)
    ax1_top.spines["right"].set_visible(False)

    # Plot 2: Energy per Sample vs Data-to-Model Ratio
    ax2 = fig.add_subplot(gs[0, 1])
    ax2_top = ax2.twiny()

    for i, k in enumerate(k_values):
        k_data = avg_energy[avg_energy["K"] == k].sort_values("Z")
        ax2.plot(
            k_data["data_to_model_ratio"],
            k_data["energy_per_sample_J"],
            marker=markers[i],
            linewidth=2.5,
            markersize=9,
            label=f"K={k}",
            color=colors[i],
            markeredgewidth=1.5,
            markeredgecolor="white",
            alpha=0.9,
        )

    ax2.set_xlabel("Data-to-Model Ratio", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Energy per Sample (J)", fontsize=12, fontweight="bold")
    ax2.set_title(
        "Energy Efficiency vs Data-to-Model Ratio",
        fontsize=13,
        fontweight="bold",
        pad=30,
    )
    ax2.legend(loc="best", fontsize=10, frameon=True, shadow=True)
    ax2.grid(False)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.spines["left"].set_linewidth(1.5)
    ax2.spines["bottom"].set_linewidth(1.5)

    ax2_top.set_xlim(ax2.get_xlim())
    ax2_top.set_xticks(z_positions)
    ax2_top.set_xticklabels(z_labels, fontsize=9)
    ax2_top.set_xlabel("Number of Partitions (Z)", fontsize=12, fontweight="bold")
    ax2_top.spines["top"].set_linewidth(1.5)
    ax2_top.spines["bottom"].set_visible(False)
    ax2_top.spines["left"].set_visible(False)
    ax2_top.spines["right"].set_visible(False)

    # Plot 3: Total Energy vs Samples per Partition
    ax3 = fig.add_subplot(gs[1, 0])

    for i, k in enumerate(k_values):
        k_data = avg_energy[avg_energy["K"] == k].sort_values("Z")
        ax3.scatter(
            k_data["samples_per_partition"],
            k_data["total_energy_kJ"],
            s=150,
            alpha=0.7,
            label=f"K={k}",
            color=colors[i],
            marker=markers[i],
            edgecolors="white",
            linewidths=1.5,
        )

        # Add trend line
        z_fit = np.polyfit(
            k_data["samples_per_partition"], k_data["total_energy_kJ"], 2
        )
        p = np.poly1d(z_fit)
        x_smooth = np.linspace(
            k_data["samples_per_partition"].min(),
            k_data["samples_per_partition"].max(),
            100,
        )
        ax3.plot(
            x_smooth,
            p(x_smooth),
            color=colors[i],
            linestyle="--",
            alpha=0.5,
            linewidth=2,
        )

    ax3.set_xlabel("Samples per Partition", fontsize=12, fontweight="bold")
    ax3.set_ylabel("Total Energy (kJ)", fontsize=12, fontweight="bold")
    ax3.set_title("Energy vs Dataset Partition Size", fontsize=13, fontweight="bold")
    ax3.legend(loc="best", fontsize=10, frameon=True, shadow=True)
    ax3.grid(False)
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)
    ax3.spines["left"].set_linewidth(1.5)
    ax3.spines["bottom"].set_linewidth(1.5)

    # Plot 4: Energy per 1k Samples vs Partition Size
    ax4 = fig.add_subplot(gs[1, 1])

    for i, k in enumerate(k_values):
        k_data = avg_energy[avg_energy["K"] == k].sort_values("Z")
        ax4.plot(
            k_data["samples_per_partition"],
            k_data["energy_per_1k_samples_kJ"],
            marker=markers[i],
            linewidth=2.5,
            markersize=9,
            label=f"K={k}",
            color=colors[i],
            markeredgewidth=1.5,
            markeredgecolor="white",
            alpha=0.9,
        )

    ax4.set_xlabel("Samples per Partition", fontsize=12, fontweight="bold")
    ax4.set_ylabel("Energy per 1000 Samples (kJ)", fontsize=12, fontweight="bold")
    ax4.set_title("Energy Efficiency vs Partition Size", fontsize=13, fontweight="bold")
    ax4.legend(loc="best", fontsize=10, frameon=True, shadow=True)
    ax4.grid(False)
    ax4.spines["top"].set_visible(False)
    ax4.spines["right"].set_visible(False)
    ax4.spines["left"].set_linewidth(1.5)
    ax4.spines["bottom"].set_linewidth(1.5)

    # Plot 5: Partition Size (MB) vs Energy
    ax5 = fig.add_subplot(gs[2, 0])
    ax5_top = ax5.twiny()

    for i, k in enumerate(k_values):
        k_data = avg_energy[avg_energy["K"] == k].sort_values("Z")
        ax5.plot(
            k_data["partition_size_mb"],
            k_data["total_energy_kJ"],
            marker=markers[i],
            linewidth=2.5,
            markersize=9,
            label=f"K={k}",
            color=colors[i],
            markeredgewidth=1.5,
            markeredgecolor="white",
            alpha=0.9,
        )

    # Add model size reference line
    ax5.axvline(
        x=MODEL_SIZE_MB,
        color="red",
        linestyle="--",
        linewidth=2,
        alpha=0.5,
        label=f"Model Size ({MODEL_SIZE_MB} MB)",
    )

    ax5.set_xlabel("Partition Size (MB)", fontsize=12, fontweight="bold")
    ax5.set_ylabel("Total Energy (kJ)", fontsize=12, fontweight="bold")
    ax5.set_title(
        "Energy vs Partition Size (MB)", fontsize=13, fontweight="bold", pad=30
    )
    ax5.legend(loc="best", fontsize=10, frameon=True, shadow=True)
    ax5.grid(False)
    ax5.spines["top"].set_visible(False)
    ax5.spines["right"].set_visible(False)
    ax5.spines["left"].set_linewidth(1.5)
    ax5.spines["bottom"].set_linewidth(1.5)

    # Secondary axis showing Z values
    ref_sorted = reference_data.sort_values("partition_size_mb")
    z_pos_mb = ref_sorted["partition_size_mb"].values
    z_lab_mb = ref_sorted["Z"].astype(int).values
    ax5_top.set_xlim(ax5.get_xlim())
    ax5_top.set_xticks(z_pos_mb)
    ax5_top.set_xticklabels(z_lab_mb, fontsize=9)
    ax5_top.set_xlabel("Number of Partitions (Z)", fontsize=12, fontweight="bold")
    ax5_top.spines["top"].set_linewidth(1.5)
    ax5_top.spines["bottom"].set_visible(False)
    ax5_top.spines["left"].set_visible(False)
    ax5_top.spines["right"].set_visible(False)

    # Plot 6: Accuracy vs Energy (if available)
    ax6 = fig.add_subplot(gs[2, 1])

    if "GlobalAccuracy" in avg_energy.columns:
        for i, k in enumerate(k_values):
            k_data = avg_energy[avg_energy["K"] == k].sort_values("Z")

            # Create scatter with size based on data-to-model ratio
            sizes = (k_data["data_to_model_ratio"] * 20).values
            scatter = ax6.scatter(
                k_data["total_energy_kJ"],
                k_data["GlobalAccuracy"],
                s=sizes,
                alpha=0.6,
                label=f"K={k}",
                color=colors[i],
                edgecolors="white",
                linewidths=1.5,
            )

            # Add Z labels
            for idx, row in k_data.iterrows():
                ax6.annotate(
                    f"Z={int(row['Z'])}",
                    (row["total_energy_kJ"], row["GlobalAccuracy"]),
                    fontsize=7,
                    alpha=0.6,
                    xytext=(3, 3),
                    textcoords="offset points",
                )

        ax6.set_xlabel("Total Energy (kJ)", fontsize=12, fontweight="bold")
        ax6.set_ylabel("Final Accuracy", fontsize=12, fontweight="bold")
        ax6.set_title(
            "Accuracy vs Energy Trade-off\n(Size = Data-to-Model Ratio)",
            fontsize=13,
            fontweight="bold",
        )
        ax6.legend(loc="best", fontsize=10, frameon=True, shadow=True)
        ax6.grid(False)
        ax6.spines["top"].set_visible(False)
        ax6.spines["right"].set_visible(False)
        ax6.spines["left"].set_linewidth(1.5)
        ax6.spines["bottom"].set_linewidth(1.5)
    else:
        ax6.text(
            0.5,
            0.5,
            "Accuracy data not available",
            ha="center",
            va="center",
            transform=ax6.transAxes,
            fontsize=14,
        )
        ax6.set_title("Accuracy vs Energy Trade-off", fontsize=13, fontweight="bold")

    plt.savefig(
        "analysis_output/energy_dataset_comprehensive.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    print("Saved: energy_dataset_comprehensive.png")


def create_focused_plots(avg_energy):
    """Create focused single plots for publication."""

    k_values = sorted(avg_energy["K"].unique())
    colors = ["#2E86AB", "#A23B72", "#F18F01"]
    markers = ["o", "s", "^"]

    # Focus Plot 1: Energy per Sample vs Data-to-Model Ratio (MAIN RESULT)
    fig, ax = plt.subplots(figsize=(12, 7))
    ax_top = ax.twiny()

    for i, k in enumerate(k_values):
        k_data = avg_energy[avg_energy["K"] == k].sort_values("Z")
        ax.plot(
            k_data["data_to_model_ratio"],
            k_data["energy_per_sample_J"],
            marker=markers[i],
            linewidth=3,
            markersize=10,
            label=f"K={k} clients",
            color=colors[i],
            markeredgewidth=1.5,
            markeredgecolor="white",
            alpha=0.9,
        )

    ax.set_xlabel("Data-to-Model Size Ratio", fontsize=13, fontweight="bold")
    ax.set_ylabel("Energy per Sample (J)", fontsize=13, fontweight="bold")
    ax.set_title(
        "Energy Efficiency vs Data-to-Model Ratio",
        fontsize=15,
        fontweight="bold",
        pad=25,
    )
    ax.legend(loc="best", fontsize=12, frameon=True, shadow=True)
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.5)
    ax.spines["bottom"].set_linewidth(1.5)

    # Secondary x-axis
    reference_data = avg_energy[avg_energy["K"] == k_values[0]].sort_values("Z")
    z_positions = reference_data["data_to_model_ratio"].values
    z_labels = reference_data["Z"].astype(int).values

    ax_top.set_xlim(ax.get_xlim())
    ax_top.set_xticks(z_positions)
    ax_top.set_xticklabels(z_labels, fontsize=10)
    ax_top.set_xlabel("Number of Partitions (Z)", fontsize=13, fontweight="bold")
    ax_top.spines["top"].set_linewidth(1.5)
    ax_top.spines["bottom"].set_visible(False)
    ax_top.spines["left"].set_visible(False)
    ax_top.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(
        "analysis_output/energy_vs_data_model_ratio_main.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    print("Saved: energy_vs_data_model_ratio_main.png")

    # Focus Plot 2: Energy vs Partition Size with Model Reference
    fig, ax = plt.subplots(figsize=(12, 7))

    for i, k in enumerate(k_values):
        k_data = avg_energy[avg_energy["K"] == k].sort_values("Z")
        ax.scatter(
            k_data["partition_size_mb"],
            k_data["total_energy_kJ"],
            s=150,
            alpha=0.7,
            label=f"K={k}",
            color=colors[i],
            marker=markers[i],
            edgecolors="white",
            linewidths=2,
        )

        # Fit and plot trend line
        z_fit = np.polyfit(k_data["partition_size_mb"], k_data["total_energy_kJ"], 2)
        p = np.poly1d(z_fit)
        x_smooth = np.linspace(
            k_data["partition_size_mb"].min(), k_data["partition_size_mb"].max(), 100
        )
        ax.plot(
            x_smooth,
            p(x_smooth),
            color=colors[i],
            linestyle="--",
            alpha=0.5,
            linewidth=2.5,
            label=f"K={k} trend",
        )

    # Add model size reference
    ax.axvline(
        x=MODEL_SIZE_MB,
        color="red",
        linestyle="--",
        linewidth=2.5,
        alpha=0.7,
        label=f"Model Size ({MODEL_SIZE_MB} MB)",
    )

    ax.set_xlabel("Dataset Partition Size (MB)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Total Energy Consumption (kJ)", fontsize=13, fontweight="bold")
    ax.set_title(
        "Energy Consumption vs Dataset Partition Size", fontsize=15, fontweight="bold"
    )
    ax.legend(loc="best", fontsize=11, frameon=True, shadow=True, ncol=2)
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.5)
    ax.spines["bottom"].set_linewidth(1.5)

    plt.tight_layout()
    plt.savefig(
        "analysis_output/energy_vs_partition_size.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    print("Saved: energy_vs_partition_size.png")


def print_analysis_summary(avg_energy):
    """Print detailed analysis summary."""

    print("\n" + "=" * 80)
    print("ENERGY AND DATASET SIZE ANALYSIS")
    print("=" * 80)

    print(f"\nDataset Information:")
    print(f"  Total dataset size: {TOTAL_DATASET_SIZE:,} samples")
    print(f"  Model size: {MODEL_SIZE_MB} MB")
    print(f"  Sample size: {32 * 32 * 3:,} bytes ({(32 * 32 * 3) / 1024:.2f} KB)")

    print(
        f"\n{'K':<5} {'Z':<6} {'Samples/Part':<15} {'Part Size (MB)':<16} {'Data/Model':<12} "
        f"{'Total Energy (kJ)':<18} {'Energy/Sample (J)':<18}"
    )
    print("-" * 100)

    for idx, row in avg_energy.iterrows():
        print(
            f"{int(row['K']):<5} {int(row['Z']):<6} {row['samples_per_partition']:<15,.0f} "
            f"{row['partition_size_mb']:<16.2f} {row['data_to_model_ratio']:<12.2f} "
            f"{row['total_energy_kJ']:<18.2f} {row['energy_per_sample_J']:<18.4f}"
        )

    print("\n" + "=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)

    for k in sorted(avg_energy["K"].unique()):
        k_data = avg_energy[avg_energy["K"] == k]

        print(f"\nK={k} Clients:")
        print(
            f"  Data-to-model ratio range: {k_data['data_to_model_ratio'].min():.2f} - "
            f"{k_data['data_to_model_ratio'].max():.2f}"
        )
        print(
            f"  Partition size range: {k_data['partition_size_mb'].min():.2f} - "
            f"{k_data['partition_size_mb'].max():.2f} MB"
        )
        print(
            f"  Energy per sample range: {k_data['energy_per_sample_J'].min():.4f} - "
            f"{k_data['energy_per_sample_J'].max():.4f} J"
        )

        # Find most efficient
        most_efficient = k_data.loc[k_data["energy_per_sample_J"].idxmin()]
        print(
            f"  Most efficient: Z={int(most_efficient['Z'])}, "
            f"Ratio={most_efficient['data_to_model_ratio']:.2f}, "
            f"{most_efficient['energy_per_sample_J']:.4f} J/sample"
        )

        # Find least efficient
        least_efficient = k_data.loc[k_data["energy_per_sample_J"].idxmax()]
        print(
            f"  Least efficient: Z={int(least_efficient['Z'])}, "
            f"Ratio={least_efficient['data_to_model_ratio']:.2f}, "
            f"{least_efficient['energy_per_sample_J']:.4f} J/sample"
        )

    print("\n" + "=" * 80)
    print("GENERAL OBSERVATIONS")
    print("=" * 80)

    # Calculate correlation
    from scipy.stats import pearsonr

    for k in sorted(avg_energy["K"].unique()):
        k_data = avg_energy[avg_energy["K"] == k]

        # Exclude Z=330 if it's an anomaly
        k_data_clean = k_data[k_data["Z"] != 330]

        if len(k_data_clean) > 2:
            corr_ratio, p_ratio = pearsonr(
                k_data_clean["data_to_model_ratio"], k_data_clean["energy_per_sample_J"]
            )
            corr_size, p_size = pearsonr(
                k_data_clean["partition_size_mb"], k_data_clean["total_energy_kJ"]
            )

            print(f"\nK={k}:")
            print(
                f"  Correlation (data-to-model ratio vs energy/sample): r={corr_ratio:.3f} (p={p_ratio:.4f})"
            )
            print(
                f"  Correlation (partition size vs total energy): r={corr_size:.3f} (p={p_size:.4f})"
            )


def main():
    """Main analysis function."""

    print("Loading data...")
    df_energy, df_global = load_data()

    print("Calculating metrics...")
    avg_energy = calculate_metrics(df_energy, df_global)

    print("\nCreating comprehensive plots...")
    create_comprehensive_plots(avg_energy)

    print("\nCreating focused plots...")
    create_focused_plots(avg_energy)

    print_analysis_summary(avg_energy)

    # Export data
    # avg_energy.to_csv("analysis_output/energy_dataset_analysis.csv", index=False)
    # print("\nSaved: energy_dataset_analysis.csv")

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print("\nGenerated files:")
    print("  - analysis_output/energy_dataset_comprehensive.png (6 subplots)")
    print("  - analysis_output/energy_vs_data_model_ratio_main.png (main result)")
    print("  - analysis_output/energy_vs_partition_size.png (with model reference)")
    print("  - analysis_output/energy_dataset_analysis.csv (data export)")


if __name__ == "__main__":
    main()
