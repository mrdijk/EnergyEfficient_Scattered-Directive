import ast
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Set style
sns.set_style("whitegrid")
sns.set_palette("deep")
# plt.rcParams["figure.figsize"] = (18, 16)


def parse_bandwidth_data(bandwidth_str):
    """Parse bandwidth_data string to extract values."""
    try:
        # Convert string representation to dict
        data = ast.literal_eval(bandwidth_str)
        return data
    except:
        return None


def load_bandwidth_data(csv_path):
    """Load and parse bandwidth CSV data."""
    df = pd.read_csv(csv_path)

    # Parse bandwidth_data column
    bandwidth_parsed = []
    for idx, row in df.iterrows():
        parsed = parse_bandwidth_data(row["bandwidth_data"])
        if parsed:
            bandwidth_parsed.append(
                {
                    "K": row["K"],
                    "Z": row["Z"],
                    "sigma_ed": row["sigma_ed"],
                    "sigma_iid": row["sigma_iid"],
                    "timestamp": row["timestamp"],
                    "service": row["service"],
                    "rx_mb": parsed.get("rx_mb", 0),
                    "tx_mb": parsed.get("tx_mb", 0),
                    "rx_bytes": parsed.get("rx_bytes", 0),
                    "tx_bytes": parsed.get("tx_bytes", 0),
                    "total_mb": parsed.get("rx_mb", 0) + parsed.get("tx_mb", 0),
                    "total_bytes": parsed.get("rx_bytes", 0)
                    + parsed.get("tx_bytes", 0),
                }
            )

    return pd.DataFrame(bandwidth_parsed)


def analyze_bandwidth(df_bandwidth):
    """Analyze bandwidth consumption."""

    # Get unique K values
    k_values = sorted(df_bandwidth["K"].unique())
    print(f"\nNumber of clients (K): {k_values}")

    # Create comprehensive plots
    fig, ax = plt.subplots()
    # 3. Bandwidth by service type (grouped by K)
    # ax = axes[1, 0]
    # df_bandwidth = df_bandwidth.copy()
    df_bandwidth.loc[df_bandwidth["service"].str.contains("client"), "service"] = (
        "client"
    )
    service_summary = (
        df_bandwidth.groupby(["K", "service"])["total_mb"].sum().reset_index()
    )

    # Get top services
    top_services = (
        service_summary.groupby("service")["total_mb"].sum().nlargest(8).index
    )

    services = sorted(top_services)
    x = np.arange(len(services))
    width = 0.8 / len(k_values)

    for i, k in enumerate(k_values):
        k_data = (
            service_summary[service_summary["K"] == k]
            .set_index("service")
            .reindex(services)
        )
        offset = (i - len(k_values) / 2 + 0.5) * width
        ax.barh(x + offset, k_data["total_mb"], width, label=f"K={k}", alpha=0.8)

    ax.set_yticks(x)
    ax.set_yticklabels(services)
    ax.set_title("Bandwidth by Service Type and K", fontweight="bold")
    ax.set_xlabel("Total Bandwidth (MB)")
    ax.set_ylabel("Service")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    plt.savefig(
        "analysis_output/plots/bandwidth_analysis.png", dpi=300, bbox_inches="tight"
    )
    print("Saved: bandwidth_analysis.png")


def main():
    """Main analysis function."""

    print("Loading bandwidth data...")

    # Update this path to your bandwidth CSV file
    csv_path = "analysis_output/combined_bandwidth_stats.csv"

    df_bandwidth = load_bandwidth_data(csv_path)

    if df_bandwidth.empty:
        print("ERROR: No bandwidth data found")
        return

    print(f"Loaded {len(df_bandwidth)} bandwidth measurements")
    print(f"  Unique services: {df_bandwidth['service'].nunique()}")
    print(f"  K values: {sorted(df_bandwidth['K'].unique())}")
    print(f"  Z values: {sorted(df_bandwidth['Z'].unique())}")

    # Analyze
    analyze_bandwidth(df_bandwidth)

    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
