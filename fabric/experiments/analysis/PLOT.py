"""
Thesis plotting script
Figures:
  RQ1 : Time-series energy rate by container group (watts per round)
  RQ2a: Line plot – total / infra / training energy vs. K
  RQ2b: Stacked bar per K (absolute joules)
  RQ2c: Grouped bar – per-client normalised energy vs. K
  RQ3a: Grouped bar – total energy vs. σ_iid, grouped by σ_ed
  RQ3b: Heatmap – 2×3 grid (σ_ed × σ_iid) of mean total energy
  RQ3c: Line – global accuracy vs. round, coloured by σ_iid
"""

# from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from utils import (
    CAT_COLORS,
    CAT_COLORS_MID,
    CAT_ORDER,
    CLIENT_CSV,
    ENERGY_CSV,
    GLOBAL_CSV,
    OUT_DIR,
    _categorise,
    _is_infra,
    load_client,
    load_energy,
    load_global,
    savefig,
)

# ── style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "sans-serif",
    "font.size":        9,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.grid":        True,
    "grid.alpha":       0.35,
    "grid.linewidth":   0.5,
    "figure.dpi":       150,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
})

# ─────────────────────────────────────────────────────────────────────────────
# RQ1 – Time-series energy rate (watts) by container group
# ─────────────────────────────────────────────────────────────────────────────

def _rq1_watts(energy: pd.DataFrame, global_df: pd.DataFrame,
               z_val: int, fixed_cols: list, fixed_vals) -> pd.DataFrame:
    """Return per-(round, category) watts for a single Z value."""
    e = energy[(energy["exp"] == "exp1") & (energy["Z"] == z_val)].copy()
    for col, val in zip(fixed_cols, fixed_vals):
        e = e[e[col] == val]
    g = global_df[(global_df["exp"] == "exp1") & (global_df["Z"] == z_val)].copy()
    for col, val in zip(fixed_cols, fixed_vals):
        g = g[g[col] == val]
    if e.empty or g.empty:
        return pd.DataFrame()
    dur = g.groupby("round")["round_duration_s"].mean()
    grp = (
        e.groupby(["round", "category", "timestamp"])["joules"]
        .sum().reset_index()
        .groupby(["round", "category"])["joules"]
        .agg(mean="mean", std="std").reset_index()
    )
    grp = grp.merge(dur.rename("dur_s"), on="round", how="left")
    grp["watts"]     = grp["mean"] / grp["dur_s"]
    grp["watts_std"] = grp["std"]  / grp["dur_s"]
    grp["Z"] = z_val
    return grp


def plot_rq1_energy_rate(energy: pd.DataFrame, global_df: pd.DataFrame):
    e1_all = energy[energy["exp"] == "exp1"].copy()
    if e1_all.empty:
        print("  [skip] no exp1 rows found")
        return

    # Fix K, sigma_ed, sigma_iid to the dominant config
    fixed_cols = ["K", "sigma_ed", "sigma_iid"]
    fixed_vals = e1_all.groupby(fixed_cols)["round"].nunique().idxmax()

    z_specs = [
        (15,  CAT_COLORS_MID, "-",  "Z=15"),
        (400, CAT_COLORS,     "--", "Z=400"),
    ]

    # Gather all categories present across both Z values
    all_grps = []
    for z_val, _, _, _ in z_specs:
        g = _rq1_watts(energy, global_df, z_val, fixed_cols, fixed_vals)
        if not g.empty:
            all_grps.append(g)
    if not all_grps:
        print("  [skip] no data for Z=15 or Z=400")
        return

    cats = [c for c in CAT_ORDER
            if any(c in g["category"].values for g in all_grps)]

    fig, ax = plt.subplots(figsize=(7, 4))

    legend_handles = []
    for z_val, palette, ls, z_label in z_specs:
        grp = next((g for g in all_grps if g["Z"].iloc[0] == z_val), None)
        if grp is None:
            continue
        for cat in cats:
            sub = grp[grp["category"] == cat].sort_values("round")
            if sub.empty:
                continue
            line, = ax.plot(sub["round"], sub["watts"],
                            color=palette[cat], linewidth=1.6,
                            linestyle=ls, marker="o", markersize=2.5,
                            label=f"{cat} ({z_label})")
            legend_handles.append(line)
            if sub["watts_std"].notna().any():
                ax.fill_between(sub["round"],
                                sub["watts"] - sub["watts_std"],
                                sub["watts"] + sub["watts_std"],
                                color=palette[cat], alpha=0.10)

    # Two-column legend: group by category, then Z shade
    ax.legend(handles=legend_handles, ncol=2, framealpha=0.7,
              fontsize=7.5, columnspacing=1.0, loc="lower right")

    cfg_str = (f"K={int(fixed_vals[0])}, "
               f"sigma_ed={fixed_vals[1]}, sigma_iid={int(fixed_vals[2])}")
    ax.set_xlabel("Round")
    ax.set_ylabel("Mean power (W)")
    ax.set_title(f"RQ1 – Energy rate by container group\n{cfg_str}", fontsize=9)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    fig.tight_layout()
    savefig(fig, "rq1_energy_rate_timeseries")


# ─────────────────────────────────────────────────────────────────────────────
# RQ2 helpers
# ─────────────────────────────────────────────────────────────────────────────

def rq2_aggregates(energy: pd.DataFrame) -> pd.DataFrame:
    e2 = energy[energy["exp"] == "exp2"].copy()
    if e2.empty:
        e2 = energy.copy()

    e2["is_training"] = e2["category"] == "Training"
    agg = (
        e2.groupby(["K", "timestamp", "is_training"])["joules"]
        .sum().unstack("is_training", fill_value=0)
        .rename(columns={True: "training_J", False: "infra_J"})
        .reset_index()
    )
    agg["total_J"] = agg["training_J"] + agg["infra_J"]
    return agg


# ─────────────────────────────────────────────────────────────────────────────
# RQ2a – Line: total / infra / training energy vs. K
# ─────────────────────────────────────────────────────────────────────────────

def plot_rq2a_line(agg: pd.DataFrame):
    summary = (agg.groupby("K")[["total_J", "infra_J", "training_J"]]
               .agg(["mean", "std"]))
    summary.columns = ["_".join(c) for c in summary.columns]
    summary = summary.reset_index().sort_values("K")

    metrics = [
        ("total_J",    "Total",           "#2C2C2A"),
        ("infra_J",    "Infrastructure",  "#D85A30"),
        ("training_J", "Training",        "#378ADD"),
    ]

    fig, ax = plt.subplots(figsize=(5.5, 4))
    for col, label, color in metrics:
        ax.plot(summary["K"], summary[f"{col}_mean"],
                label=label, color=color, linewidth=2, marker="o", markersize=5)
        ax.fill_between(summary["K"],
                        summary[f"{col}_mean"] - summary[f"{col}_std"],
                        summary[f"{col}_mean"] + summary[f"{col}_std"],
                        color=color, alpha=0.12)

    ax.set_xlabel("Number of clients (K)")
    ax.set_ylabel("Energy per experiment (J)")
    ax.set_title("RQ2a – Energy vs. number of clients", fontsize=9)
    ax.xaxis.set_major_locator(mticker.FixedLocator(sorted(summary["K"].unique())))
    ax.legend(framealpha=0.7, fontsize=8)
    fig.tight_layout()
    savefig(fig, "rq2a_energy_vs_k_line")


# ─────────────────────────────────────────────────────────────────────────────
# RQ2b – Stacked bar: infra + training per K
# ─────────────────────────────────────────────────────────────────────────────

def plot_rq2b_stacked(agg: pd.DataFrame):
    summary = agg.groupby("K")[["infra_J", "training_J"]].mean().reset_index().sort_values("K")
    std     = agg.groupby("K")[["infra_J", "training_J"]].std().reset_index().sort_values("K")

    x     = np.arange(len(summary))
    width = 0.5

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(x, summary["infra_J"], width,
           label="Infrastructure", color="#D85A30", alpha=0.85)
    ax.bar(x, summary["training_J"], width,
           bottom=summary["infra_J"],
           label="Training", color="#378ADD", alpha=0.85,
           yerr=std["training_J"], capsize=3, error_kw={"linewidth": 0.8})

    for i, (inf, tot) in enumerate(zip(summary["infra_J"],
                                       summary["infra_J"] + summary["training_J"])):
        pct = 100 * inf / tot if tot > 0 else 0
        ax.text(x[i], tot * 1.01, f"{pct:.0f}% infra",
                ha="center", va="bottom", fontsize=7.5, color="#5F5E5A")

    ax.set_xticks(x)
    ax.set_xticklabels([f"K={int(k)}" for k in summary["K"]])
    ax.set_ylabel("Energy per experiment (J)")
    ax.set_title("RQ2b – Energy breakdown per K", fontsize=9)
    ax.legend(framealpha=0.7, fontsize=8)
    fig.tight_layout()
    savefig(fig, "rq2b_stacked_bar_k")


# ─────────────────────────────────────────────────────────────────────────────
# RQ2c – Grouped bar: per-client normalised energy vs. K
# ─────────────────────────────────────────────────────────────────────────────

def plot_rq2c_per_client(agg: pd.DataFrame):
    df = agg.copy()
    df["total_per_client_J"]    = df["total_J"]    / df["K"]
    df["infra_per_client_J"]    = df["infra_J"]    / df["K"]
    df["training_per_client_J"] = df["training_J"] / df["K"]

    summary = (df.groupby("K")[["total_per_client_J", "infra_per_client_J",
                                 "training_per_client_J"]]
               .agg(["mean", "std"]))
    summary.columns = ["_".join(c) for c in summary.columns]
    summary = summary.reset_index().sort_values("K")

    x     = np.arange(len(summary))
    width = 0.22

    cols = [
        ("total_per_client_J",    "Total",          "#2C2C2A"),
        ("infra_per_client_J",    "Infrastructure", "#D85A30"),
        ("training_per_client_J", "Training",       "#378ADD"),
    ]

    fig, ax = plt.subplots(figsize=(5.5, 4))
    offsets = [-width, 0, width]
    for (col, label, color), off in zip(cols, offsets):
        ax.bar(x + off, summary[f"{col}_mean"], width,
               label=label, color=color, alpha=0.85,
               yerr=summary[f"{col}_std"], capsize=3,
               error_kw={"linewidth": 0.8})

    ax.set_xticks(x)
    ax.set_xticklabels([f"K={int(k)}" for k in summary["K"]])
    ax.set_ylabel("Energy per client per experiment (J)")
    ax.set_title("RQ2c – Per-client normalised energy vs. K", fontsize=9)
    ax.legend(framealpha=0.7, fontsize=8)
    fig.tight_layout()
    savefig(fig, "rq2c_per_client_energy_k")


# ─────────────────────────────────────────────────────────────────────────────
# RQ3 helpers
# ─────────────────────────────────────────────────────────────────────────────

def rq3_aggregates(energy: pd.DataFrame) -> pd.DataFrame:
    e3 = energy[energy["exp"] == "exp3"].copy()
    if e3.empty:
        e3 = energy[(energy["sigma_ed"].isin([1000.0, 1.7])) &
                    (energy["sigma_iid"].isin([3, 6, 10]))].copy()
    agg = (
        e3.groupby(["sigma_ed", "sigma_iid", "round", "timestamp"])["joules"]
        .sum().reset_index()
        .rename(columns={"joules": "total_J"})
    )
    return agg


# ─────────────────────────────────────────────────────────────────────────────
# RQ3a – Grouped bar: total energy vs. sigma_iid, grouped by sigma_ed
# ─────────────────────────────────────────────────────────────────────────────

def plot_rq3a_grouped_bar(agg: pd.DataFrame):
    summary = (
        agg.groupby(["sigma_ed", "sigma_iid", "timestamp", "round"])["total_J"]
        .sum()
        .reset_index()
        .groupby(["sigma_ed", "sigma_iid"])["total_J"]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary["mean_kJ"] = summary["mean"] / 1000
    summary["std_kJ"]  = summary["std"]  / 1000

    iid_vals = sorted(summary["sigma_iid"].unique())
    ed_vals  = sorted(summary["sigma_ed"].unique())
    x        = np.arange(len(iid_vals))
    n        = len(ed_vals)
    width    = 0.35
    offsets  = np.linspace(-(n-1)*width/2, (n-1)*width/2, n)
    ed_colors = {ed_vals[0]: "#1D9E75", ed_vals[-1]: "#EF9F27"}
    ed_labels = {v: ("Uniform (σ_ed=1000)" if v >= 100 else f"Skewed (σ_ed={v})")
                 for v in ed_vals}

    fig, ax = plt.subplots(figsize=(6, 4))
    for ed, off in zip(ed_vals, offsets):
        sub   = summary[summary["sigma_ed"] == ed].sort_values("sigma_iid")
        means = [sub[sub["sigma_iid"] == i]["mean_kJ"].values[0]
                 if i in sub["sigma_iid"].values else 0 for i in iid_vals]
        stds  = [sub[sub["sigma_iid"] == i]["std_kJ"].values[0]
                 if i in sub["sigma_iid"].values else 0 for i in iid_vals]
        ax.bar(x + off, means, width, label=ed_labels[ed],
               color=ed_colors[ed], alpha=0.85,
               yerr=stds, capsize=3, error_kw={"linewidth": 0.8})

    ax.set_xticks(x)
    ax.set_xticklabels([f"σ_iid={i}" for i in iid_vals])
    ax.set_ylabel("Energy per round (kJ)")
    ax.legend(framealpha=0.7, fontsize=8)
    fig.tight_layout()
    savefig(fig, "rq3a_grouped_bar_iid_ed")


# ─────────────────────────────────────────────────────────────────────────────
# RQ3b – Heatmap: 2x3 grid sigma_ed x sigma_iid of mean total energy
# ─────────────────────────────────────────────────────────────────────────────

def plot_rq3b_heatmap(agg: pd.DataFrame):
    pivot = agg.groupby(["sigma_ed", "sigma_iid"])["total_J"].mean().unstack("sigma_iid")

    fig, ax = plt.subplots(figsize=(5, 2.8))
    im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrBr")

    ed_labels  = [("Uniform\n(σ_ed=1000)" if v >= 100 else f"Skewed\n(σ_ed={v})")
                  for v in pivot.index]
    iid_labels = [f"σ_iid={c}" for c in pivot.columns]

    ax.set_yticks(range(len(pivot.index)));   ax.set_yticklabels(ed_labels, fontsize=8)
    ax.set_xticks(range(len(pivot.columns))); ax.set_xticklabels(iid_labels, fontsize=8)
    # ax.set_title("RQ3b – Mean total energy (J) across data distribution conditions", fontsize=9)

    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Mean total energy (J)", fontsize=8)

    vmin, vmax = np.nanmin(pivot.values), np.nanmax(pivot.values)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            if np.isnan(val):
                continue
            text_color = "white" if val > vmin + 0.6 * (vmax - vmin) else "black"
            ax.text(j, i, f"{val/1000:.1f}kJ",
                    ha="center", va="center", fontsize=7.5, color=text_color)

    fig.tight_layout()
    savefig(fig, "rq3b_heatmap_ed_iid")


# ─────────────────────────────────────────────────────────────────────────────
# RQ3c – Line: global accuracy vs. round, coloured by sigma_iid
# ─────────────────────────────────────────────────────────────────────────────

def plot_rq3c_accuracy_lines(global_df: pd.DataFrame):
    g3 = global_df[global_df["exp"] == "exp3"].copy()
    if g3.empty:
        g3 = global_df[(global_df["sigma_ed"].isin([1000.0, 1.7])) &
                       (global_df["sigma_iid"].isin([3, 6, 10]))].copy()

    ed_vals  = sorted(g3["sigma_ed"].unique())
    iid_vals = sorted(g3["sigma_iid"].unique())
    iid_colors = {3: "#D85A30", 6: "#EF9F27", 10: "#378ADD"}

    n_ed = len(ed_vals)
    fig, axes = plt.subplots(1, n_ed, figsize=(4.5 * n_ed, 4), sharey=True)
    if n_ed == 1:
        axes = [axes]

    for ax, ed in zip(axes, ed_vals):
        sub_ed = g3[g3["sigma_ed"] == ed]
        for iid in iid_vals:
            sub = sub_ed[sub_ed["sigma_iid"] == iid]
            if sub.empty:
                continue
            mean_acc = sub.groupby("round")["GlobalAccuracy"].agg(["mean", "std"])
            rounds   = mean_acc.index.values
            color    = iid_colors.get(iid, "#888780")
            ax.plot(rounds, mean_acc["mean"], label=f"σ_iid={iid}",
                    color=color, linewidth=1.8, marker="o", markersize=2.5)
            ax.fill_between(rounds,
                            mean_acc["mean"] - mean_acc["std"],
                            mean_acc["mean"] + mean_acc["std"],
                            color=color, alpha=0.12)

        ed_str = "Uniform (σ_ed=1000)" if ed >= 100 else f"Skewed (σ_ed={ed})"
        ax.set_title(ed_str, fontsize=9)
        ax.set_xlabel("Round")
        if ax is axes[0]:
            ax.set_ylabel("Global accuracy")
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        ax.legend(framealpha=0.7, fontsize=8)

    # fig.suptitle("RQ3c – Accuracy vs. round by class heterogeneity", fontsize=9, y=1.02)
    fig.tight_layout()
    savefig(fig, "rq3c_accuracy_vs_round")

# ─────────────────────────────────────────────────────────────────────────────
# RQ1 extra – Heatmap: mean joules per container × Z (exp1)
# Rows = containers, columns = Z values, cell = mean joules across rounds/runs.
# ─────────────────────────────────────────────────────────────────────────────

def plot_rq1_container_heatmap(energy: pd.DataFrame):
    df = energy[energy["exp"] == "exp1"].copy()
    if df.empty:
        print("  [skip] no exp1 rows")
        return

    # Mean joules per (container_name, Z) across all rounds and timestamps
    pivot = (
        df.groupby(["container_name", "Z"])["joules"]
        .sum()
        .unstack("Z")
    )
    # Sort containers by total mean energy descending so dominant ones are on top
    pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).index]

    z_vals = sorted(pivot.columns)
    pivot  = pivot[z_vals]

    fig_h = max(3, len(pivot) * 0.45)
    fig_w = max(5, len(z_vals) * 1.2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrBr")

    ax.set_xticks(range(len(z_vals)))
    ax.set_xticklabels([f"Z={z}" for z in z_vals], fontsize=8)
    ax.set_yticks(range(len(pivot)))
    ax.set_yticklabels(pivot.index, fontsize=8)
    ax.set_xlabel("Dataset size (Z)")
    ax.set_title("RQ1 – Mean energy per container by Z (Exp 1)", fontsize=9)

    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.03)
    cb.set_label("Mean joules per round", fontsize=8)

    # Annotate cells
    vmin, vmax = np.nanmin(pivot.values), np.nanmax(pivot.values)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            if np.isnan(val):
                continue
            text_color = "white" if val > vmin + 0.6 * (vmax - vmin) else "black"
            ax.text(j, i, f"{val:.0f}",
                    ha="center", va="center", fontsize=7, color=text_color)

    fig.tight_layout()
    savefig(fig, "rq1_container_heatmap_z")

# ─────────────────────────────────────────────────────────────────────────────
# Energy–time correlation scatter (all experiments)
# Three subplots:
#   1. Client training time  vs. Training category energy
#   2. Aggregation time      vs. Orchestration category energy
#   3. Round duration        vs. Total energy
# Each point = one round; coloured by experiment.
# ─────────────────────────────────────────────────────────────────────────────

EXP_COLORS = {
    "exp1": "#378ADD",
    "exp2": "#D85A30",
    "exp3": "#534AB7",
}

def plot_energy_time_scatter(energy: pd.DataFrame,
                              global_df: pd.DataFrame,
                              client_df: pd.DataFrame):
    run_cols = ["exp", "K", "Z", "sigma_ed", "sigma_iid", "timestamp", "round"]

    # ── per-round category energy ─────────────────────────────────────────────
    cat_energy = (
        energy.groupby(run_cols + ["category"])["joules"]
        .sum().unstack("category", fill_value=0).reset_index()
    )
    # Total energy per round
    energy_cols = [c for c in cat_energy.columns if c in CAT_COLORS]
    cat_energy["total_J"] = cat_energy[energy_cols].sum(axis=1)
    # Training energy = Training category
    cat_energy["training_J"]      = cat_energy.get("Training",      0)
    # Orchestration energy = Orchestration category
    cat_energy["orchestration_J"] = cat_energy.get("Orchestration", 0)

    # ── merge global stats (round duration, aggregation time) ────────────────
    global_slim = global_df[
        run_cols + ["round_duration_s", "AggregationTime"]
    ].copy()
    merged = cat_energy.merge(global_slim, on=run_cols, how="inner")

    # ── merge client training time (max per round across active clients) ───────
    merged = merged.merge(client_df, on=run_cols, how="left")

    if merged.empty:
        print("  [skip] merged DataFrame is empty")
        return

    # ── plot ──────────────────────────────────────────────────────────────────
    panels = [
        ("max_client_training_ms", "training_J",
         "Client training time (ms)", "Training energy (J)",
         "Training time vs. training energy"),
        ("AggregationTime", "orchestration_J",
         "Aggregation time (ms)", "Orchestration energy (J)",
         "Aggregation time vs. orchestration energy"),
        ("round_duration_s", "total_J",
         "Round duration (s)", "Total energy (J)",
         "Round duration vs. total energy"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    for ax, (x_col, y_col, x_label, y_label, title) in zip(axes, panels):
        sub = merged[[x_col, y_col, "exp"]].dropna()
        if sub.empty:
            ax.set_visible(False)
            continue

        for exp, grp in sub.groupby("exp"):
            color = EXP_COLORS.get(exp, "#888780")
            ax.scatter(grp[x_col], grp[y_col],
                       label=exp, color=color,
                       alpha=0.45, s=18, linewidths=0)

            # Regression line
            x_vals = grp[x_col].values
            y_vals = grp[y_col].values
            if len(x_vals) > 2 and x_vals.std() > 0:
                m, b = np.polyfit(x_vals, y_vals, 1)
                x_line = np.linspace(x_vals.min(), x_vals.max(), 100)
                ax.plot(x_line, m * x_line + b,
                        color=color, linewidth=1.4, alpha=0.85)

        ax.set_xlabel(x_label, fontsize=8)
        ax.set_ylabel(y_label, fontsize=8)
        ax.set_title(title, fontsize=9)
        ax.legend(fontsize=7.5, framealpha=0.7)

    fig.suptitle("Energy–time correlation by experiment", fontsize=10, y=1.02)
    fig.tight_layout()
    savefig(fig, "energy_time_scatter")

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("Loading data...")
    energy    = load_energy(ENERGY_CSV)
    global_df = load_global(GLOBAL_CSV)
    client_df = load_client(CLIENT_CSV)

    print(f"  Energy rows  : {len(energy)}")
    print(f"  Categories   : {sorted(energy['category'].unique())}")
    print(f"  Experiments  : {sorted(energy['exp'].unique())}")
    print(f"  K values     : {sorted(energy['K'].unique())}")
    print(f"  sigma_ed     : {sorted(energy['sigma_ed'].unique())}")
    print(f"  sigma_iid    : {sorted(energy['sigma_iid'].unique())}")
    print()

    # print("RQ1 – energy rate time-series...")
    # plot_rq1_energy_rate(energy, global_df)

    # print("RQ1 – container heatmap by Z...")
    # plot_rq1_container_heatmap(energy)

    # print("Energy–time correlation scatter...")
    # plot_energy_time_scatter(energy, global_df, client_df)

    # print("RQ2 – scalability...")
    # agg2 = rq2_aggregates(energy)
    # plot_rq2a_line(agg2)
    # plot_rq2b_stacked(agg2)
    # plot_rq2c_per_client(agg2)

    print("RQ3 – data distribution...")
    agg3 = rq3_aggregates(energy)
    plot_rq3a_grouped_bar(agg3)
    plot_rq3b_heatmap(agg3)
    plot_rq3c_accuracy_lines(global_df)

    print(f"\nDone. All figures saved to {OUT_DIR}/")

if __name__ == "__main__":
    main()
