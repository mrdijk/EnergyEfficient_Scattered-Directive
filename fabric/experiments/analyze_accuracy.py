import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Set style
sns.set_style("whitegrid")
sns.set_palette("deep")
plt.rcParams["figure.figsize"] = (18, 16)

TOTAL_DATASET_SIZE = 531130
MODEL_SIZE_MB = 6.1


def load_data():
    """Load all necessary data."""
    base_path = "analysis_output"

    df_global = pd.read_csv(f"{base_path}/combined_global_stats.csv", index_col=0)
    df_client = pd.read_csv(f"{base_path}/combined_client_stats.csv", index_col=0)

    return df_global, df_client


def analyze_accuracy_dataset_metrics(df_global, df_client):
    """Analyze accuracy and dataset metrics."""

    k_values = sorted(df_global["K"].unique()) if "K" in df_global.columns else [5]

    fig, axes = plt.subplots(3, 2, figsize=(18, 16))

    # 1. Final global accuracy
    ax = axes[0, 0]
    final_acc = (
        df_global.groupby(["K", "Z", "timestamp"])["GlobalAccuracy"]
        .last()
        .reset_index()
    )
    avg_final_acc = final_acc.groupby(["K", "Z"])["GlobalAccuracy"].mean().reset_index()
    std_final_acc = final_acc.groupby(["K", "Z"])["GlobalAccuracy"].std().reset_index()

    z_values = sorted(avg_final_acc["Z"].unique())
    x = np.arange(len(z_values))
    width = 0.8 / len(k_values)

    for i, k in enumerate(k_values):
        k_data = avg_final_acc[avg_final_acc["K"] == k].set_index("Z").reindex(z_values)
        k_std = std_final_acc[std_final_acc["K"] == k].set_index("Z").reindex(z_values)
        offset = (i - len(k_values) / 2 + 0.5) * width
        ax.bar(
            x + offset,
            k_data["GlobalAccuracy"],
            width,
            yerr=k_std["GlobalAccuracy"],
            capsize=3,
            label=f"K={k}",
            alpha=0.8,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(z_values)
    ax.set_title("Final Global Accuracy", fontsize=12, fontweight="bold")
    ax.set_xlabel("Number of Partitions (Z)")
    ax.set_ylabel("Accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # 2. Average local (client) accuracy
    ax = axes[0, 1]
    if not df_client.empty:
        final_client_acc = (
            df_client.groupby(["K", "Z", "ClientID", "Round"])["ClientAccuracy"]
            .last()
            .reset_index()
        )
        avg_client_acc = (
            final_client_acc.groupby(["K", "Z"])["ClientAccuracy"].mean().reset_index()
        )
        std_client_acc = (
            final_client_acc.groupby(["K", "Z"])["ClientAccuracy"].std().reset_index()
        )

        z_values = sorted(avg_client_acc["Z"].unique())
        x = np.arange(len(z_values))
        width = 0.8 / len(k_values)

        for i, k in enumerate(k_values):
            k_data = (
                avg_client_acc[avg_client_acc["K"] == k]
                .set_index("Z")
                .reindex(z_values)
            )
            k_std = (
                std_client_acc[std_client_acc["K"] == k]
                .set_index("Z")
                .reindex(z_values)
            )
            offset = (i - len(k_values) / 2 + 0.5) * width
            ax.bar(
                x + offset,
                k_data["ClientAccuracy"],
                width,
                yerr=k_std["ClientAccuracy"],
                capsize=3,
                label=f"K={k}",
                alpha=0.8,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(z_values)
        ax.set_title("Average Local (Client) Accuracy", fontsize=12, fontweight="bold")
        ax.set_xlabel("Number of Partitions (Z)")
        ax.set_ylabel("Accuracy")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

    # 3. Samples per partition
    ax = axes[1, 0]
    dataset_metrics = pd.DataFrame(
        {"K": [], "Z": [], "samples_per_partition": [], "data_percentage": []}
    )

    for k in k_values:
        for z in sorted(df_global[df_global["K"] == k]["Z"].unique()):
            samples = TOTAL_DATASET_SIZE / z
            pct = (samples / TOTAL_DATASET_SIZE) * 100
            dataset_metrics = pd.concat(
                [
                    dataset_metrics,
                    pd.DataFrame(
                        {
                            "K": [k],
                            "Z": [z],
                            "samples_per_partition": [samples],
                            "data_percentage": [pct],
                        }
                    ),
                ],
                ignore_index=True,
            )

    z_values = sorted(dataset_metrics["Z"].unique())
    x = np.arange(len(z_values))
    width = 0.8 / len(k_values)

    for i, k in enumerate(k_values):
        k_data = (
            dataset_metrics[dataset_metrics["K"] == k].set_index("Z").reindex(z_values)
        )
        offset = (i - len(k_values) / 2 + 0.5) * width
        ax.bar(
            x + offset,
            k_data["samples_per_partition"],
            width,
            label=f"K={k}",
            alpha=0.8,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(z_values)
    ax.set_title("Dataset Size per Partition", fontsize=12, fontweight="bold")
    ax.set_xlabel("Number of Partitions (Z)")
    ax.set_ylabel("Samples per Partition")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # 4. Data to model ratio
    ax = axes[1, 1]
    bytes_per_sample = 32 * 32 * 3  # SVHN image size
    dataset_metrics["partition_size_mb"] = (
        dataset_metrics["samples_per_partition"] * bytes_per_sample
    ) / (1024 * 1024)
    dataset_metrics["data_to_model_ratio"] = (
        dataset_metrics["partition_size_mb"] / MODEL_SIZE_MB
    )

    for k in k_values:
        k_data = dataset_metrics[dataset_metrics["K"] == k].sort_values("Z")
        ax.plot(
            k_data["Z"],
            k_data["data_to_model_ratio"],
            marker="o",
            label=f"K={k}",
            linewidth=2,
        )

    ax.set_title("Data-to-Model Size Ratio", fontsize=12, fontweight="bold")
    ax.set_xlabel("Number of Partitions (Z)")
    ax.set_ylabel("Ratio (Partition Size / Model Size)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 5. Accuracy vs dataset size
    ax = axes[2, 0]
    merged = pd.merge(avg_final_acc, dataset_metrics, on=["K", "Z"])

    for k in k_values:
        k_data = merged[merged["K"] == k]
        ax.scatter(
            k_data["samples_per_partition"],
            k_data["GlobalAccuracy"],
            s=100,
            alpha=0.6,
            label=f"K={k}",
        )

        # Add Z labels
        for idx, row in k_data.iterrows():
            ax.annotate(
                f"Z={int(row['Z'])}",
                (row["samples_per_partition"], row["GlobalAccuracy"]),
                fontsize=7,
                alpha=0.7,
            )

    ax.set_title(
        "Accuracy vs Dataset Size per Partition", fontsize=12, fontweight="bold"
    )
    ax.set_xlabel("Samples per Partition")
    ax.set_ylabel("Final Global Accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 6. Global vs Local accuracy comparison
    ax = axes[2, 1]
    if not df_client.empty:
        comparison = pd.merge(
            avg_final_acc[["K", "Z", "GlobalAccuracy"]],
            avg_client_acc[["K", "Z", "ClientAccuracy"]],
            on=["K", "Z"],
        )

        for k in k_values:
            k_data = comparison[comparison["K"] == k].sort_values("Z")
            ax.plot(
                k_data["Z"],
                k_data["GlobalAccuracy"],
                marker="o",
                label=f"K={k} Global",
                linewidth=2,
            )
            ax.plot(
                k_data["Z"],
                k_data["ClientAccuracy"],
                marker="s",
                linestyle="--",
                label=f"K={k} Local",
                linewidth=2,
                alpha=0.7,
            )

        ax.set_title("Global vs Local Accuracy", fontsize=12, fontweight="bold")
        ax.set_xlabel("Number of Partitions (Z)")
        ax.set_ylabel("Accuracy")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        "analysis_output/accuracy_dataset_analysis.png", dpi=300, bbox_inches="tight"
    )
    print("Saved: accuracy_dataset_analysis.png")

    return avg_final_acc, avg_client_acc, dataset_metrics


def print_accuracy_dataset_statistics(avg_final_acc, avg_client_acc, dataset_metrics):
    """Print detailed accuracy and dataset statistics."""

    print("\n" + "=" * 80)
    print("ACCURACY AND DATASET METRICS ANALYSIS")
    print("=" * 80)

    print("\nFinal Global Accuracy:")
    print(avg_final_acc.to_string(index=False))

    if avg_client_acc is not None and not avg_client_acc.empty:
        print("\nAverage Local Accuracy:")
        print(avg_client_acc.to_string(index=False))

    print("\nDataset Metrics:")
    print(
        dataset_metrics[
            [
                "K",
                "Z",
                "samples_per_partition",
                "data_percentage",
                "data_to_model_ratio",
            ]
        ].to_string(index=False)
    )

    print("\nBest Configurations by Accuracy:")
    best_global = avg_final_acc.loc[avg_final_acc["GlobalAccuracy"].idxmax()]
    print(
        f"  Best global accuracy: K={best_global['K']}, Z={best_global['Z']}, "
        f"Accuracy={best_global['GlobalAccuracy']:.4f}"
    )

    if avg_client_acc is not None and not avg_client_acc.empty:
        best_local = avg_client_acc.loc[avg_client_acc["ClientAccuracy"].idxmax()]
        print(
            f"  Best local accuracy: K={best_local['K']}, Z={best_local['Z']}, "
            f"Accuracy={best_local['ClientAccuracy']:.4f}"
        )


def export_accuracy_dataset_metrics(avg_final_acc, avg_client_acc, dataset_metrics):
    """Export accuracy and dataset metrics to CSV."""

    combined = pd.merge(avg_final_acc, dataset_metrics, on=["K", "Z"])

    if avg_client_acc is not None and not avg_client_acc.empty:
        combined = pd.merge(
            combined,
            avg_client_acc[["K", "Z", "ClientAccuracy"]],
            on=["K", "Z"],
            how="left",
        )

    combined.to_csv("analysis_output/accuracy_dataset_metrics_summary.csv", index=False)
    print("Saved: accuracy_dataset_metrics_summary.csv")


def main():
    """Main analysis function."""

    print("Loading data...")
    df_global, df_client = load_data()

    print("Analyzing accuracy and dataset metrics...")
    avg_final_acc, avg_client_acc, dataset_metrics = analyze_accuracy_dataset_metrics(
        df_global, df_client
    )

    print_accuracy_dataset_statistics(avg_final_acc, avg_client_acc, dataset_metrics)

    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
