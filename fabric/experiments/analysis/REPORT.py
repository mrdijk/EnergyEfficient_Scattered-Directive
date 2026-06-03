from pathlib import Path

import pandas as pd
from utils import (
    CLIENT_CSV,
    ENERGY_CSV,
    GLOBAL_CSV,
    OUT_DIR,
    REPORT_CSV,
    _categorise,
    _is_infra,
    load_client,
    load_energy,
    load_global,
    savefig,
)


# ─────────────────────────────────────────────────────────────────────────────
# General experiment analysis – summary report printed + saved to CSV
# Covers: energy per category, round duration, client training time,
#         aggregation time, global accuracy — all per experiment config.
# ─────────────────────────────────────────────────────────────────────────────
def experiment_analysis(energy: pd.DataFrame,
                        global_df: pd.DataFrame,
                        client_df: pd.DataFrame):

    cfg_cols = ["exp", "K", "Z", "sigma_ed", "sigma_iid"]

    # ── 1. Energy per category per config ─────────────────────────────────────
    # Mean joules per round, per category, averaged across timestamps/rounds
    energy_agg = (
        energy.groupby(cfg_cols + ["timestamp", "round", "category"])["joules"]
        .sum().reset_index()
        .groupby(cfg_cols + ["category"])["joules"]
        .agg(mean="mean", std="std", min="min", max="max")
        .reset_index()
    )
    energy_agg["cv"] = energy_agg["std"] / energy_agg["mean"]
    energy_agg.rename(columns={"mean": "energy_mean_J", "std": "energy_std_J",
                                "min": "energy_min_J",  "max": "energy_max_J",
                                "cv":  "energy_cv"}, inplace=True)

    # Total energy per config (sum across categories)
    total_energy = (
        energy.groupby(cfg_cols + ["timestamp", "round"])["joules"]
        .sum().reset_index()
        .groupby(cfg_cols)["joules"]
        .agg(total_mean="mean", total_std="std",
             total_min="min",  total_max="max")
        .reset_index()
    )
    total_energy["total_cv"] = total_energy["total_std"] / total_energy["total_mean"]

    # ── 2. Time metrics per config ─────────────────────────────────────────────
    time_agg = (
        global_df.groupby(cfg_cols)
        .agg(
            round_dur_mean =("round_duration_s",  "mean"),
            round_dur_std  =("round_duration_s",  "std"),
            round_dur_min  =("round_duration_s",  "min"),
            round_dur_max  =("round_duration_s",  "max"),
            agg_time_mean  =("AggregationTime",   "mean"),
            agg_time_std   =("AggregationTime",   "std"),
        )
        .reset_index()
    )

    client_time_agg = (
        client_df.groupby(cfg_cols)
        .agg(
            train_time_mean=("max_client_training_ms", "mean"),
            train_time_std =("max_client_training_ms", "std"),
            train_time_min =("max_client_training_ms", "min"),
            train_time_max =("max_client_training_ms", "max"),
        )
        .reset_index()
    )

    # ── 3. Accuracy metrics per config ────────────────────────────────────────
    acc_agg = (
        global_df.groupby(cfg_cols)
        .agg(
            acc_final_mean =("GlobalAccuracy", lambda x:
                             global_df.loc[x.index[global_df.loc[x.index, "round"]
                             == global_df.loc[x.index, "round"].max()],
                             "GlobalAccuracy"].mean()),
            acc_mean       =("GlobalAccuracy", "mean"),
            acc_max        =("GlobalAccuracy", "max"),
            acc_std        =("GlobalAccuracy", "std"),
            n_rounds       =("round",          "max"),
        )
        .reset_index()
    )

    # ── 4. Merge everything ───────────────────────────────────────────────────
    summary = (total_energy
               .merge(time_agg,        on=cfg_cols, how="left")
               .merge(client_time_agg, on=cfg_cols, how="left")
               .merge(acc_agg,         on=cfg_cols, how="left"))

    # ── 5. Print report ───────────────────────────────────────────────────────
    sep = "─" * 80
    print(f"\n{'═'*80}")
    print("  EXPERIMENT ANALYSIS REPORT")
    print(f"{'═'*80}")

    for _, row in summary.sort_values(cfg_cols).iterrows():
        print(f"\n{sep}")
        print(f"  Experiment : {row['exp']}  |  K={int(row['K'])}  "
              f"Z={int(row['Z'])}  σ_ed={row['sigma_ed']}  σ_iid={int(row['sigma_iid'])}")
        print(sep)

        print("  ENERGY (per round, J)")
        print(f"    Total      mean={row['total_mean']:.1f}  std={row['total_std']:.1f}  "
              f"min={row['total_min']:.1f}  max={row['total_max']:.1f}  "
              f"CV={row['total_cv']:.3f}")

        # Per-category breakdown for this config
        e_sub = energy_agg[
            (energy_agg["exp"]      == row["exp"]) &
            (energy_agg["K"]        == row["K"]) &
            (energy_agg["Z"]        == row["Z"]) &
            (energy_agg["sigma_ed"] == row["sigma_ed"]) &
            (energy_agg["sigma_iid"]== row["sigma_iid"])
        ].sort_values("energy_mean_J", ascending=False)

        for _, er in e_sub.iterrows():
            pct = 100 * er["energy_mean_J"] / row["total_mean"] if row["total_mean"] > 0 else 0
            print(f"    {er['category']:<18} mean={er['energy_mean_J']:>8.1f} J  "
                  f"std={er['energy_std_J']:>7.1f}  CV={er['energy_cv']:.3f}  "
                  f"({pct:.1f}% of total)")

        print("\n  TIME")
        print(f"    Round duration     mean={row['round_dur_mean']:.2f}s  "
              f"std={row['round_dur_std']:.2f}s  "
              f"min={row['round_dur_min']:.2f}s  max={row['round_dur_max']:.2f}s")
        if pd.notna(row.get("train_time_mean")):
            print(f"    Client train time  mean={row['train_time_mean']:.0f}ms  "
                  f"std={row['train_time_std']:.0f}ms  "
                  f"min={row['train_time_min']:.0f}ms  max={row['train_time_max']:.0f}ms")
        print(f"    Aggregation time   mean={row['agg_time_mean']:.0f}ms  "
              f"std={row['agg_time_std']:.0f}ms")

        print(f"\n  ACCURACY  (over {int(row['n_rounds'])} rounds)")
        print(f"    Mean={row['acc_mean']:.4f}  Max={row['acc_max']:.4f}  "
              f"Std={row['acc_std']:.4f}  "
              f"Final(mean)={row['acc_final_mean']:.4f}")

    print(f"\n{'═'*80}\n")

    # ── 6. Save flat CSV ──────────────────────────────────────────────────────
    summary.to_csv(REPORT_CSV, index=False)
    print(f"  Summary saved to {REPORT_CSV}")

    # Also save per-category energy detail
    cat_csv = OUT_DIR / "experiment_analysis_by_category.csv"
    energy_agg.to_csv(cat_csv, index=False)
    print(f"  Category detail saved to {cat_csv}")

def main():
    print("Loading data...")
    energy    = load_energy(ENERGY_CSV)
    global_df = load_global(GLOBAL_CSV)
    client_df = load_client(CLIENT_CSV)

    print("General experiment analysis...")
    experiment_analysis(energy, global_df, client_df)

if __name__ == "__main__":
    main()
