import pandas as pd
from scipy.stats import shapiro


def load_global_stats(filepath: str) -> dict[tuple, pd.DataFrame]:
    df = pd.read_csv(filepath, index_col=0)
    df = df[df["exp"] == "exp1"]
    df = df.rename(columns={"Unnamed: 0": "round"}) if "Unnamed: 0" in df.columns else df

    config_cols = ["exp", "K", "Z", "sigma_ed", "sigma_iid"]
    grouped = {}
    for config_key, group in df.groupby(config_cols):
        grouped[config_key] = group.reset_index(drop=True)
    return grouped

def run_shapiro(df: pd.DataFrame, label: str = ""):
    columns_to_test = ["AggregationTime", "TotalTrainingTime", "RoundDuration"]
    print(f"\n--- Shapiro-Wilk normality tests {label} ---")
    for col in columns_to_test:
        if col not in df.columns:
            print(f"  {col}: column not found")
            continue
        data = df[col].dropna().values
        if len(data) < 3:
            print(f"  {col}: not enough data points ({len(data)})")
            continue
        stat, p = shapiro(data)
        verdict = "NOT normal" if p < 0.05 else "normal"
        print(f"  {col}: W={stat:.4f}, p={p:.4f}  → {verdict} (n={len(data)})")

if __name__ == "__main__":
    configs = load_global_stats(
        "/home/maurits/EnergyEfficient_Scattered-Directive/fabric/experiments/data/combined_global_stats.csv"
    )

    for (exp, K, Z, sigma_ed, sigma_iid), config_df in configs.items():
        config_label = f"exp={exp} K={K} Z={Z} σ_ed={sigma_ed} σ_iid={sigma_iid}"
        print(f"\n{'='*60}\n{config_label}")

        for timestamp, run_df in config_df.groupby("timestamp", sort=True):
            n_rows = len(run_df)
            run_shapiro(run_df, label=f"run={timestamp} ({n_rows} rows)")
        # n_runs = config_df["timestamp"].nunique()
        # n_rows = len(config_df)
        # label = f"exp={exp} K={K} Z={Z} σ_ed={sigma_ed} σ_iid={sigma_iid} ({n_runs} runs, {n_rows} rows)"
        # run_shapiro(config_df, label=label)
