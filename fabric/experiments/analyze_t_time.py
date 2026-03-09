import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Set style
sns.set_style("whitegrid")
sns.set_palette("deep")
plt.rcParams["figure.figsize"] = (18, 16)


def load_data():
    """Load all necessary data."""
    base_path = "analysis_output"

    df_global = pd.read_csv(f"{base_path}/combined_global_stats.csv", index_col=0)
    df_client = pd.read_csv(f"{base_path}/combined_client_stats.csv", index_col=0)

    return df_global, df_client


def analyze_time_metrics(df_global, df_client):
    """Analyze timing metrics."""

    k_values = sorted(df_global["K"].unique()) if "K" in df_global.columns else [5]

    fig, axes = plt.subplots(3, 2, figsize=(18, 16))

    # 1. Average training duration per round
    ax = axes[0, 0]
    train_time = df_global.groupby(["K", "Z"])["TotalTrainingTime"].mean().reset_index()
    train_time["TotalTrainingTime_sec"] = train_time["TotalTrainingTime"] / 1000

    z_values = sorted(train_time["Z"].unique())
    x = np.arange(len(z_values))
    width = 0.8 / len(k_values)

    for i, k in enumerate(k_values):
        k_data = train_time[train_time["K"] == k].set_index("Z").reindex(z_values)
        offset = (i - len(k_values) / 2 + 0.5) * width
        ax.bar(
            x + offset,
            k_data["TotalTrainingTime_sec"],
            width,
            label=f"K={k}",
            alpha=0.8,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(z_values)
    ax.set_title("Average Training Duration per Round", fontsize=12, fontweight="bold")
    ax.set_xlabel("Number of Partitions (Z)")
    ax.set_ylabel("Training Time (seconds)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # 2. Average aggregation duration per round
    ax = axes[0, 1]
    agg_time = df_global.groupby(["K", "Z"])["AggregationTime"].mean().reset_index()

    z_values = sorted(agg_time["Z"].unique())
    x = np.arange(len(z_values))
    width = 0.8 / len(k_values)

    for i, k in enumerate(k_values):
        k_data = agg_time[agg_time["K"] == k].set_index("Z").reindex(z_values)
        offset = (i - len(k_values) / 2 + 0.5) * width
        ax.bar(x + offset, k_data["AggregationTime"], width, label=f"K={k}", alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(z_values)
    ax.set_title(
        "Average Aggregation Duration per Round", fontsize=12, fontweight="bold"
    )
    ax.set_xlabel("Number of Partitions (Z)")
    ax.set_ylabel("Aggregation Time (ms)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # 3. Total duration per experiment
    ax = axes[1, 0]
    # Sum all rounds per experiment
    total_duration = (
        df_global.groupby(["K", "Z", "timestamp"])
        .agg({"TotalTrainingTime": "sum", "AggregationTime": "sum"})
        .reset_index()
    )
    total_duration["total_time_sec"] = (
        total_duration["TotalTrainingTime"] + total_duration["AggregationTime"]
    ) / 1000

    # Average across experiments
    avg_total = (
        total_duration.groupby(["K", "Z"])["total_time_sec"].mean().reset_index()
    )

    z_values = sorted(avg_total["Z"].unique())
    x = np.arange(len(z_values))
    width = 0.8 / len(k_values)

    for i, k in enumerate(k_values):
        k_data = avg_total[avg_total["K"] == k].set_index("Z").reindex(z_values)
        offset = (i - len(k_values) / 2 + 0.5) * width
        ax.bar(x + offset, k_data["total_time_sec"], width, label=f"K={k}", alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(z_values)
    ax.set_title("Total Duration per Experiment", fontsize=12, fontweight="bold")
    ax.set_xlabel("Number of Partitions (Z)")
    ax.set_ylabel("Total Time (seconds)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # 4. Training vs Aggregation time breakdown
    ax = axes[1, 1]
    for k in k_values:
        k_data = train_time[train_time["K"] == k].sort_values("Z")
        ax.plot(
            k_data["Z"],
            k_data["TotalTrainingTime_sec"],
            marker="o",
            label=f"K={k} Training",
            linewidth=2,
        )

        k_agg = agg_time[agg_time["K"] == k].sort_values("Z")
        ax.plot(
            k_agg["Z"],
            k_agg["AggregationTime"] / 1000,
            marker="s",
            linestyle="--",
            label=f"K={k} Aggregation",
            linewidth=2,
            alpha=0.7,
        )

    ax.set_title("Training vs Aggregation Time", fontsize=12, fontweight="bold")
    ax.set_xlabel("Number of Partitions (Z)")
    ax.set_ylabel("Time (seconds)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 5. Client training time variance
    ax = axes[2, 0]
    if not df_client.empty:
        client_time_var = (
            df_client.groupby(["K", "Z", "Round"])["ClientTrainingTime"]
            .std()
            .reset_index()
        )
        avg_var = (
            client_time_var.groupby(["K", "Z"])["ClientTrainingTime"]
            .mean()
            .reset_index()
        )

        z_values = sorted(avg_var["Z"].unique())
        x = np.arange(len(z_values))
        width = 0.8 / len(k_values)

        for i, k in enumerate(k_values):
            k_data = avg_var[avg_var["K"] == k].set_index("Z").reindex(z_values)
            offset = (i - len(k_values) / 2 + 0.5) * width
            ax.bar(
                x + offset,
                k_data["ClientTrainingTime"],
                width,
                label=f"K={k}",
                alpha=0.8,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(z_values)
        ax.set_title("Client Training Time Std Dev", fontsize=12, fontweight="bold")
        ax.set_xlabel("Number of Partitions (Z)")
        ax.set_ylabel("Std Dev (ms)")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

    # 6. Time efficiency (time per sample)
    ax = axes[2, 1]
    TOTAL_DATASET_SIZE = 531130

    train_time["samples_per_partition"] = TOTAL_DATASET_SIZE / train_time["Z"]
    train_time["total_samples"] = train_time["samples_per_partition"] * train_time["K"]
    train_time["ms_per_sample"] = (
        train_time["TotalTrainingTime"] / train_time["total_samples"]
    )

    for k in k_values:
        k_data = train_time[train_time["K"] == k].sort_values("Z")
        ax.plot(
            k_data["Z"],
            k_data["ms_per_sample"],
            marker="o",
            label=f"K={k}",
            linewidth=2,
        )

    ax.set_title("Training Time per Sample", fontsize=12, fontweight="bold")
    ax.set_xlabel("Number of Partitions (Z)")
    ax.set_ylabel("Time per Sample (ms)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        "analysis_output/time_metrics_analysis.png", dpi=300, bbox_inches="tight"
    )
    print("Saved: time_metrics_analysis.png")

    return train_time, agg_time, avg_total


def print_time_statistics(train_time, agg_time, avg_total):
    """Print detailed time statistics."""

    print("\n" + "=" * 80)
    print("TIME METRICS ANALYSIS")
    print("=" * 80)

    print("\nAverage Training Time per Round (seconds):")
    print(train_time[["K", "Z", "TotalTrainingTime_sec"]].to_string(index=False))

    print("\nAverage Aggregation Time per Round (ms):")
    print(agg_time[["K", "Z", "AggregationTime"]].to_string(index=False))

    print("\nTotal Experiment Duration (seconds):")
    print(avg_total[["K", "Z", "total_time_sec"]].to_string(index=False))

    print("\nTime per Sample (ms):")
    print(train_time[["K", "Z", "ms_per_sample"]].to_string(index=False))


def export_time_metrics(train_time, agg_time, avg_total):
    """Export time metrics to CSV."""

    combined = train_time[["K", "Z", "TotalTrainingTime_sec", "ms_per_sample"]]
    combined = pd.merge(
        combined, agg_time[["K", "Z", "AggregationTime"]], on=["K", "Z"]
    )
    combined = pd.merge(
        combined, avg_total[["K", "Z", "total_time_sec"]], on=["K", "Z"]
    )

    combined.to_csv("analysis_output/time_metrics_summary.csv", index=False)
    print("Saved: time_metrics_summary.csv")


def main():
    """Main analysis function."""

    print("Loading data...")
    df_global, df_client = load_data()

    print("Analyzing time metrics...")
    train_time, agg_time, avg_total = analyze_time_metrics(df_global, df_client)

    print_time_statistics(train_time, agg_time, avg_total)

    print("\nExporting metrics...")
    export_time_metrics(train_time, agg_time, avg_total)

    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
