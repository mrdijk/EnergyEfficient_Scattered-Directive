import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns  # pyright: ignore[reportMissingModuleSource]

# Set style
sns.set_style("whitegrid")
sns.set_palette("deep")
# plt.rcParams["figure.figsize"] = (18, 16)

TOTAL_DATASET_SIZE = 531130
MODEL_SIZE_MB = 8.3

ED = [1000, 1.7]
IID = [10, 6, 3]
CLIENTS = ["client1", "client5", "client9", "client13", "client17"]


def plot_global_accuracy(df_global):
    fig, ax = plt.subplots(1, 1)
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
    plt.savefig("ED_IID_global.png", dpi=300, bbox_inches="tight")
    print("Saved: ED_IID_global.png")


def plot_client_accuracy(df_client):
    fig, ax = plt.subplots(1, 1)

    data = df_client.groupby(["ClientID", "sigma_ed", "sigma_iid"]).last().reset_index()

    # Show all configurations for each client
    x_pos = np.arange(6)  # 6 configurations
    width = 0.15

    for i, client_id in enumerate(CLIENTS):
        client_data = data[data["ClientID"] == client_id].sort_values(
            ["sigma_ed", "sigma_iid"], ascending=False
        )

        offset = (i - 2) * width
        bars = ax.bar(
            x_pos + offset,
            client_data["ClientAccuracy"].values,
            width,
            label=client_id,
            alpha=0.85,
            edgecolor="white",
            linewidth=1,
        )

    ax.set_ylabel("Client Accuracy", fontsize=11, fontweight="bold")
    # ax.set_title(
    #     "All Clients Across All Configurations", fontsize=12, fontweight="bold"
    # )
    ax.set_xticks(x_pos)
    ax.set_xticklabels(
        [
            "σ_ed=1.7\nσ_iid=3",
            "σ_ed=1.7\nσ_iid=6",
            "σ_ed=1.7\nσ_iid=10",
            "σ_ed=1000\nσ_iid=3",
            "σ_ed=1000\nσ_iid=6",
            "σ_ed=1000\nσ_iid=10",
        ],
        fontsize=9,
    )
    ax.legend(fontsize=9, ncol=5, loc="upper right")
    ax.grid(True, alpha=0.3, axis="y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Overall title
    # fig.suptitle(
    #     "Client Accuracy Analysis: Effect of σ_iid (Classes) and σ_ed (Data Distribution)",
    #     fontsize=16,
    #     fontweight="bold",
    #     y=0.995,
    # )

    plt.tight_layout()
    plt.savefig("ED_IID_clients.png", dpi=300, bbox_inches="tight")
    print("Saved: ED_IID_clients.png")


def main():
    """Main analysis function."""
    base_path = "data"
    print("Loading data...")
    df_global = pd.read_csv(f"{base_path}/combined_global_stats.csv", index_col=0)
    df_client = pd.read_csv(f"{base_path}/combined_client_stats.csv", index_col=0)

    df_global = df_global[df_global["exp"] == "exp3"]
    df_client = df_client[df_client["exp"] == "exp3"]

    print("Effect of client data size (e.d.) and content (i.i.d.) variations.")

    global_acc = df_global.groupby(["sigma_ed", "sigma_iid"])["GlobalAccuracy"].mean()
    print(global_acc)
    client_acc = df_client.groupby(["ClientID", "sigma_ed", "sigma_iid"])[
        "ClientAccuracy"
    ].mean()
    print(client_acc)

    plot_global_accuracy(df_global)
    plot_client_accuracy(df_client)


if __name__ == "__main__":
    main()
