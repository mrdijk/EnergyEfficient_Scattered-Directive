import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns  # pyright: ignore[reportMissingModuleSource]

# Set style
sns.set_style("whitegrid")
sns.set_palette("deep")
# plt.rcParams["figure.figsize"] = (18, 16)

TOTAL_DATASET_SIZE = 531130
MODEL_SIZE_MB = 6.1

ED = [1000, 1.7]
IID = [10, 6, 3]


def plot_accuracy(df_global, df_client):
    fig, axes = plt.subplots(1, 1)
    ax = axes[0]
    metrics = df_global.columns.tolist()
    # Compute consistent y-max per metric (max value + 15% headroom)
    y_max = {metric: df_global["GlobalAccuracy"].max() * 1.15 for metric in metrics}

    exp_data = df_global = (
        df_global.groupby(["sigma_ed", "sigma_iid"])["GlobalAccuracy"]
        .mean()
        .reset_index()
    )

    iid_values = IID
    ed_values = ED
    x = np.arange(len(iid_values))
    width = 0.8 / len(ED)

    for i, ed in enumerate(ed_values):
        ed_data = (
            exp_data[exp_data["sigma_ed"] == ed]
            .set_index("sigma_iid")
            .reindex(iid_values)
        )
        offset = (i - len(ED) / 2 + 0.5) * width
        bars = ax.bar(
            x + offset,
            ed_data["GlobalAccuracy"],
            width,
            label=f"ed={ed}",
            alpha=0.8,
        )
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + y_max["GlobalAccuracy"] * 0,
                f"{bar.get_height():.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([f"{i}" for i in iid_values], fontsize=10)
    # ax.set_xticklabels(ed_values)
    ax.set_title(
        "Effect e.d. and i.i.d. on Global Accuracy", fontsize=12, fontweight="bold"
    )
    ax.set_xlabel("IID (number of classes)")
    ax.set_ylabel("Accuracy (%)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig("analysis_output/exp3/data_dist.png", dpi=300, bbox_inches="tight")
    print("Saved: data_dist.png")


def main():
    """Main analysis function."""
    base_path = "analysis_output/exp3"
    print("Loading data...")
    df_global = pd.read_csv(f"{base_path}/combined_global_stats.csv", index_col=0)
    df_client = pd.read_csv(f"{base_path}/combined_client_stats.csv", index_col=0)

    print("Effect of client data size (e.d.) and content (i.i.d.) variations.")
    # ed = 1000, 1.7
    # iid = 3,6,10

    global_acc = df_global.groupby(["sigma_ed", "sigma_iid"])["GlobalAccuracy"].mean()
    print(global_acc)
    client_acc = df_client.groupby(["ClientID", "sigma_ed", "sigma_iid"])[
        "ClientAccuracy"
    ].mean()
    print(client_acc)

    plot_accuracy(df_global, df_client)


if __name__ == "__main__":
    main()
