import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Set style
sns.set_style("whitegrid")
sns.set_palette("deep")
plt.rcParams["figure.figsize"] = (15, 10)


def analyze_experiments(df_client, df_global, df_energy):
    """Comprehensive analysis of experimental results."""
    print("\nLoaded data:")
    print(f"  - Client stats: {len(df_client)} rows")
    print(f"  - Global stats: {len(df_global)} rows")
    print(f"  - Energy data: {len(df_energy)} rows")

    # Get unique K values for grouping
    k_values = sorted(df_global["K"].unique()) if "K" in df_global.columns else [5]
    print(f"\nNumber of clients (K): {k_values}")

    # ========== Analysis 1: Accuracy vs Number of Partitions ==========
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1.1 Final accuracy by Z (grouped by K)
    ax = axes[0, 0]
    if not df_global.empty:
        # Get final round accuracy for each experiment
        final_accuracy = (
            df_global.groupby(["K", "Z", "timestamp"])["GlobalAccuracy"]
            .last()
            .reset_index()
        )
        summary = (
            final_accuracy.groupby(["K", "Z"])["GlobalAccuracy"]
            .agg(["mean", "std"])
            .reset_index()
        )

        # Prepare for grouped bar chart
        z_values = sorted(summary["Z"].unique())
        x = np.arange(len(z_values))
        width = 0.8 / len(k_values)  # Width of bars

        for i, k in enumerate(k_values):
            k_data = summary[summary["K"] == k].set_index("Z").reindex(z_values)
            offset = (i - len(k_values) / 2 + 0.5) * width
            ax.bar(
                x + offset,
                k_data["mean"],
                width,
                yerr=k_data["std"],
                capsize=3,
                label=f"K={k}",
                alpha=0.8,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(z_values)
        ax.set_title(
            "Final Global Accuracy by Number of Partitions (Z) and Clients (K)",
            fontsize=12,
            fontweight="bold",
        )
        ax.set_xlabel("Number of Partitions (Z)")
        ax.set_ylabel("Accuracy")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

    # 1.2 Accuracy convergence over rounds
    ax = axes[0, 1]
    if not df_global.empty:
        for k in k_values:
            k_data = df_global[df_global["K"] == k]
            for z in sorted(k_data["Z"].unique()):
                z_data = k_data[k_data["Z"] == z]
                # Average across experiments with same K and Z
                z_avg = z_data.groupby(z_data.index)["GlobalAccuracy"].mean()
                ax.plot(
                    z_avg.index,
                    z_avg.values,
                    marker="o",
                    label=f"K={k}, Z={z}",
                    linewidth=2,
                    alpha=0.7,
                )

        ax.set_title("Accuracy Convergence Over Rounds", fontsize=12, fontweight="bold")
        ax.set_xlabel("Round")
        ax.set_ylabel("Global Accuracy")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
        ax.grid(True, alpha=0.3)

    # 1.3 Client accuracy variance (grouped by K)
    ax = axes[1, 0]
    if not df_client.empty:
        # Calculate variance of client accuracies per round
        client_variance = (
            df_client.groupby(["K", "Z", "Round"])["ClientAccuracy"].std().reset_index()
        )
        z_variance = (
            client_variance.groupby(["K", "Z"])["ClientAccuracy"].mean().reset_index()
        )

        # Grouped bar chart
        z_values = sorted(z_variance["Z"].unique())
        x = np.arange(len(z_values))
        width = 0.8 / len(k_values)

        for i, k in enumerate(k_values):
            k_data = z_variance[z_variance["K"] == k].set_index("Z").reindex(z_values)
            offset = (i - len(k_values) / 2 + 0.5) * width
            ax.bar(
                x + offset, k_data["ClientAccuracy"], width, label=f"K={k}", alpha=0.8
            )

        ax.set_xticks(x)
        ax.set_xticklabels(z_values)
        ax.set_title(
            "Average Client Accuracy Std Dev by Z and K", fontsize=12, fontweight="bold"
        )
        ax.set_xlabel("Number of Partitions (Z)")
        ax.set_ylabel("Std Dev of Client Accuracies")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

    # 1.4 Training time by Z (grouped by K)
    ax = axes[1, 1]
    if not df_global.empty:
        time_summary = (
            df_global.groupby(["K", "Z"])["TotalTrainingTime"].mean() / 1000
        ).reset_index()  # Convert to seconds

        # Grouped bar chart
        z_values = sorted(time_summary["Z"].unique())
        x = np.arange(len(z_values))
        width = 0.8 / len(k_values)

        for i, k in enumerate(k_values):
            k_data = (
                time_summary[time_summary["K"] == k].set_index("Z").reindex(z_values)
            )
            offset = (i - len(k_values) / 2 + 0.5) * width
            ax.bar(
                x + offset,
                k_data["TotalTrainingTime"],
                width,
                label=f"K={k}",
                alpha=0.8,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(z_values)
        ax.set_title(
            "Average Total Training Time by Z and K", fontsize=12, fontweight="bold"
        )
        ax.set_xlabel("Number of Partitions (Z)")
        ax.set_ylabel("Time (seconds)")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig("analysis_output/analysis_accuracy.png", dpi=300, bbox_inches="tight")
    print("\nSaved: analysis_accuracy.png")

    # ========== Analysis 2: Energy Consumption ==========
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 2.1 Total energy by container type
    ax = axes[0, 0]
    energy_by_container = (
        df_energy.groupby("container_name")["joules"].sum().sort_values(ascending=False)
    )
    (energy_by_container / 1000).plot(kind="barh", ax=ax, color="green")
    ax.set_title(
        "Total Energy Consumption by Container Type", fontsize=12, fontweight="bold"
    )
    ax.set_xlabel("Energy (kJ)")
    ax.set_ylabel("Container")
    ax.grid(True, alpha=0.3, axis="x")

    # 2.2 Energy by Z (grouped by K)
    ax = axes[0, 1]
    if "Z" in df_energy.columns and "K" in df_energy.columns:
        energy_summary = (
            df_energy.groupby(["K", "Z"])["joules"].sum() / 1000
        ).reset_index()  # kJ

        z_values = sorted(energy_summary["Z"].unique())
        x = np.arange(len(z_values))
        width = 0.8 / len(k_values)

        for i, k in enumerate(k_values):
            k_data = (
                energy_summary[energy_summary["K"] == k]
                .set_index("Z")
                .reindex(z_values)
            )
            offset = (i - len(k_values) / 2 + 0.5) * width
            ax.bar(x + offset, k_data["joules"], width, label=f"K={k}", alpha=0.8)

        ax.set_xticks(x)
        ax.set_xticklabels(z_values)
        ax.set_title(
            "Total Energy Consumption by Z and K", fontsize=12, fontweight="bold"
        )
        ax.set_xlabel("Number of Partitions (Z)")
        ax.set_ylabel("Total Energy (kJ)")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

    # 2.3 Energy by namespace (service)
    ax = axes[1, 0]
    energy_by_namespace = (
        df_energy.groupby("namespace")["joules"]
        .sum()
        .sort_values(ascending=False)
        .head(15)
    )
    (energy_by_namespace / 1000).plot(kind="barh", ax=ax, color="purple")
    ax.set_title(
        "Top 15 Services by Energy Consumption", fontsize=12, fontweight="bold"
    )
    ax.set_xlabel("Energy (kJ)")
    ax.set_ylabel("Service")
    ax.grid(True, alpha=0.3, axis="x")

    # 2.4 Energy distribution by container type and Z
    ax = axes[1, 1]
    if "Z" in df_energy.columns:
        main_containers = df_energy[~df_energy["container_name"].isin(["linkerd-init"])]

        pivot = (
            main_containers.pivot_table(
                values="joules", index="Z", columns="container_name", aggfunc="sum"
            )
            / 1000
        )

        pivot.plot(kind="bar", ax=ax, stacked=True)
        ax.set_title(
            "Energy Distribution by Container Type and Z",
            fontsize=12,
            fontweight="bold",
        )
        ax.set_xlabel("Number of Partitions (Z)")
        ax.set_ylabel("Total Energy (kJ)")
        ax.legend(title="Container", bbox_to_anchor=(1.05, 1), loc="upper left")
        ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig("analysis_output/analysis_energy.png", dpi=300, bbox_inches="tight")
    print("Saved: analysis_output/analysis_energy.png.png")

    # Print energy summary
    total_joules = df_energy["joules"].sum()
    print("\nEnergy Consumption Summary:")
    print(f"  Total energy: {total_joules:,.0f} J ({total_joules / 1000:.2f} kJ)")

    if "Z" in df_energy.columns:
        print("\n  Energy by Z (kilojoules):")
        energy_z = df_energy.groupby("Z")["joules"].sum().sort_index()
        for z, energy in energy_z.items():
            print(f"    Z={z:3d}: {energy / 1000:8.2f} kJ")

    # ========== Analysis 3: Client Performance ==========
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    plt.rcParams["figure.figsize"] = (12, 6)
    if not df_client.empty:
        # 3.1 Average client training time by Z with error bars (grouped by K)
        ax = axes[0]
        time_stats = (
            df_client.groupby(["K", "Z"])["ClientTrainingTime"]
            .agg(["mean", "std"])
            .reset_index()
        )

        z_values = sorted(time_stats["Z"].unique())
        x = np.arange(len(z_values))
        width = 0.8 / len(k_values)

        for i, k in enumerate(k_values):
            k_data = time_stats[time_stats["K"] == k].set_index("Z").reindex(z_values)
            offset = (i - len(k_values) / 2 + 0.5) * width
            ax.bar(
                x + offset,
                k_data["mean"],
                width,
                yerr=k_data["std"],
                capsize=3,
                label=f"K={k}",
                alpha=0.8,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(z_values)
        ax.set_title(
            "Average Client Training Time by Z and K (mean ± std)",
            fontsize=12,
            fontweight="bold",
        )
        ax.set_xlabel("Number of Partitions (Z)")
        ax.set_ylabel("Training Time (ms)")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

        # 3.2 Client accuracy distribution
        ax = axes[1]
        df_client.boxplot(column="ClientAccuracy", by="Z", ax=ax)
        ax.set_title(
            "Client Accuracy Distribution by Z", fontsize=12, fontweight="bold"
        )
        ax.set_xlabel("Number of Partitions (Z)")
        ax.set_ylabel("Client Accuracy")
        plt.sca(ax)
        plt.xticks(rotation=0)
        ax.get_figure().suptitle("")

        # # 3.3 Training time over rounds
        # ax = axes[1, 0]
        # for k in k_values:
        #     k_data = df_client[df_client["K"] == k]
        #     for z in sorted(k_data["Z"].unique())[:3]:  # Show first 3 Z values per K
        #         z_data = k_data[k_data["Z"] == z]
        #         z_avg = z_data.groupby("Round")["ClientTrainingTime"].mean()
        #         ax.plot(
        #             z_avg.index,
        #             z_avg.values,
        #             marker="o",
        #             label=f"K={k}, Z={z}",
        #             linewidth=2,
        #             alpha=0.7,
        #         )

        # ax.set_title("Client Training Time Over Rounds", fontsize=12, fontweight="bold")
        # ax.set_xlabel("Round")
        # ax.set_ylabel("Avg Training Time (ms)")
        # ax.legend(fontsize=8)
        # ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        "analysis_output/analysis_client_performance.png", dpi=300, bbox_inches="tight"
    )
    print("Saved: analysis_client_performance.png")

    # ========== Summary Statistics ==========
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)

    if not df_global.empty:
        print("\nGlobal Performance:")
        summary = (
            df_global.groupby(["K", "Z"])
            .agg(
                {
                    "GlobalAccuracy": ["mean", "std", "max"],
                    "TotalTrainingTime": ["mean", "std"],
                    "AggregationTime": ["mean", "std"],
                }
            )
            .round(4)
        )
        print(summary)

        print("\nBest Configurations:")
        final_acc = df_global.groupby(["K", "Z", "timestamp"])["GlobalAccuracy"].last()
        best_idx = final_acc.idxmax()
        print(
            f"  Highest accuracy: K={best_idx[0]}, Z={best_idx[1]}, Accuracy={final_acc[best_idx]:.4f}"
        )

        avg_time = df_global.groupby(["K", "Z"])["TotalTrainingTime"].mean()
        fastest_idx = avg_time.idxmin()
        print(
            f"  Fastest training: K={fastest_idx[0]}, Z={fastest_idx[1]}, Time={avg_time[fastest_idx]:.0f}ms"
        )


# Run the analysis
if __name__ == "__main__":
    data_path = "/home/maurits/EnergyEfficient_Scattered-Directive/fabric/experiments/analysis_output"
    df_client = pd.read_csv(
        f"{data_path}/combined_client_stats.csv",
        index_col=[0],
    )
    df_global = pd.read_csv(
        f"{data_path}/combined_global_stats.csv",
        index_col=[0],
    )
    df_energy = pd.read_csv(
        f"{data_path}/combined_energy_stats.csv",
        index_col=[0],
    )

    analyze_experiments(df_client, df_global, df_energy)

    print("\nAnalysis complete!")
