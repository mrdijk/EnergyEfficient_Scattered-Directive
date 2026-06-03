"""
Plot infrastructure vs. training energy per active client vs. number of
partitions (Z), with samples-per-client on a secondary y-axis.

Data source: combined_energy_stats.csv

Active client  = a client namespace that has a job pod in that round
                 (pod name starts with "maurits-dijk-ab40b6ad...")

Training energy = hfl-train + hfl-train-model containers (in job pod)
Infra energy    = everything else in the client namespace
                  (idle pod: linkerd-proxy, sidecar, clientN;
                   job pod:  sidecar)

Values are averaged over all rounds and all active clients.
"""

import re

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

# ── Config ────────────────────────────────────────────────────────────────────

CSV_PATH      = "/home/maurits/EnergyEfficient_Scattered-Directive/fabric/experiments/data/combined_energy_stats.csv"
TOTAL_SAMPLES = 531_131
Z_ORDER       = [15, 30, 60, 90, 120, 150, 190, 230, 260, 300, 330, 360, 400]

ACTIVE_CLIENTS = ["client1", "client5", "client9", "client13", "client17"]
ENTITIES       = ACTIVE_CLIENTS + ["server"]

# ── Load ──────────────────────────────────────────────────────────────────────

df = pd.read_csv(CSV_PATH)
df = df[df["exp"] == "exp1"]
# df["joules"] = pd.to_numeric(df["joules"], errors="coerce").fillna(0.0)

# ── Aggregate per (Z, entity) using the caller's exact logic ─────────────────

records = []
for z in sorted(df["Z"].unique()):
    zdf = df[df["Z"] == z]
    for entity in ENTITIES:
        edf = zdf[zdf["namespace"] == entity]
        records.append({
            "Z":                   z,
            "entity":              entity,
            "hfl_train_kj":        edf[edf["container_name"] == "hfl-train"]["joules"].sum() / 1e3,
            "hfl_train_model_kj":  edf[edf["container_name"] == "hfl-train-model"]["joules"].sum() / 1e3,
            "agent_kj":            edf[edf["container_name"] == entity]["joules"].sum() / 1e3,
            "sidecar_kj":          edf[edf["container_name"] == "sidecar"]["joules"].sum() / 1e3,
            "linkerd_kj":          edf[edf["container_name"].isin(["linkerd-proxy", "linkerd-init"])]["joules"].sum() / 1e3,
        })

edf_all = pd.DataFrame(records)
edf_all["training_kj"] = edf_all["hfl_train_kj"] + edf_all["hfl_train_model_kj"]
edf_all["infra_kj"]    = edf_all["agent_kj"] + edf_all["sidecar_kj"] + edf_all["linkerd_kj"]

# Keep only Z values present in data, in order
z_present = [z for z in Z_ORDER if z in edf_all["Z"].unique()]
edf_all   = edf_all[edf_all["Z"].isin(z_present)].sort_values("Z")

samples_per_z = {z: TOTAL_SAMPLES / z for z in z_present}

print("Z values in plot:", z_present)
print(edf_all[["Z", "entity", "infra_kj", "training_kj"]].to_string(index=False))

# ── Separate clients vs server, average clients ───────────────────────────────

clients_df = edf_all[edf_all["entity"].isin(ACTIVE_CLIENTS)]
server_df  = edf_all[edf_all["entity"] == "server"]

# Mean over the K active clients for each Z
mean_clients = (
    clients_df
    .groupby("Z")[["infra_kj", "training_kj"]]
    .mean()
    .reset_index()
)

server_mean = (
    server_df
    .groupby("Z")[["infra_kj", "training_kj"]]
    .mean()
    .reset_index()
)

# Keep only Z values present in data, ordered
z_present = sorted(mean_clients["Z"].unique())
z_plot    = [z for z in Z_ORDER if z in z_present] + \
            [z for z in z_present if z not in Z_ORDER]
z_plot    = sorted(set(z_plot))

mean_clients = mean_clients[mean_clients["Z"].isin(z_plot)].sort_values("Z")
server_mean  = server_mean [server_mean ["Z"].isin(z_plot)].sort_values("Z")

samples_per_client = TOTAL_SAMPLES / mean_clients["Z"].astype(float)

print("Z values in plot:", mean_clients["Z"].tolist())
print(mean_clients[["Z", "infra_kj", "training_kj"]].to_string(index=False))

# ── Plot ──────────────────────────────────────────────────────────────────────

COLOR_ORCH  = "#1f77b4"   # blue  – active clients (mean)
COLOR_INFRA  = "#d62728"   # red   – server
COLOR_TRAINING = "#2ca02c"   # green – samples
COLOR_SAMPLES = '#ffa02c'

z_vals = mean_clients["Z"].astype(int).tolist()

fig, ax = plt.subplots(
    # 1, 2,
    # figsize=(14, 5.5),
    # sharey=False,
)

def add_samples_axis(ax, z_vals, samples, annotate=True):
    ax2 = ax.twinx()
    ax2.plot(z_vals, samples, marker="^", linewidth=1.8, markersize=5,
             linestyle="--", color=COLOR_SAMPLES, label="Samples", zorder=2)
    ax2.set_ylabel("Samples per client", fontsize=11, color=COLOR_SAMPLES)
    ax2.tick_params(axis="y", labelcolor=COLOR_SAMPLES)
    ax2.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    if annotate:
        for z, s in zip(z_vals, samples):
            ax2.annotate(
                f"{s/1_000:.1f}k",
                xy=(z, s / 1_000),
                xytext=(0, 7), textcoords="offset points",
                ha="center", fontsize=6.5, color=COLOR_SAMPLES, alpha=0.9,
            )
    return ax2

def style_ax(ax, title):
    ax.set_xlabel("Number of partitions (Z)", fontsize=11)
    ax.set_xticks(z_vals)
    ax.tick_params(axis="x", rotation=45)
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.grid(axis="x", linestyle=":", alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # ax.set_title(title, fontsize=12, fontweight="bold", pad=8)

# ── Left: Infrastructure ──────────────────────────────────────────────────────

# lns_c1 = ax_infra.plot(z_vals, mean_clients["infra_kj"].values,
#                         marker="o", linewidth=2, markersize=6,
#                         color=COLOR_INFRA, label="Active clients (mean)")
# lns_s1 = ax_infra.plot(z_vals, server_mean["infra_kj"].values,
#                         marker="D", linewidth=2, markersize=6,
#                         color=COLOR_TRAINING,  label="Server")

# ax_infra.set_ylabel("Infrastructure energy (kJ)", fontsize=11)
# style_ax(ax_infra, "Infrastructure energy\n(agent + sidecar + linkerd)")

# ax2_infra = add_samples_axis(ax_infra, z_vals, samples_per_client.values, annotate=False)

# lines  = lns_c1 + lns_s1 + [ax2_infra.lines[0]]
# labels = [l.get_label() for l in lines]
# ax_infra.legend(lines, labels, fontsize=9, framealpha=0.92, loc="upper right")

# ── Right: Training ───────────────────────────────────────────────────────────
lns_c1 = ax.plot(z_vals, mean_clients["infra_kj"].values,
                        marker="o", linewidth=2, markersize=6,
                        color=COLOR_INFRA, label="Infrastructure")
lns_c2 = ax.plot(z_vals, mean_clients["training_kj"].values,
                        marker="o", linewidth=2, markersize=6,
                        color=COLOR_TRAINING, label="Training")
# lns_s2 = ax.plot(z_vals, server_mean["training_kj"].values,
#                         marker="D", linewidth=2, markersize=6,
#                         color=COLOR_TRAINING,  label="Server")

ax.set_ylabel("Energy/Client (kJ)", fontsize=11)
style_ax(ax, " ")

ax2_train = add_samples_axis(ax, z_vals, samples_per_client.values, annotate=False)

lines  = lns_c1 + lns_c2 + [ax2_train.lines[0]]
labels = [l.get_label() for l in lines]
ax.legend(lines, labels, fontsize=9, framealpha=0.92, loc="upper right")

# ── Overall title & layout ────────────────────────────────────────────────────

# fig.suptitle(
#     f"Energy per entity vs. number of partitions (Z)  ·  "
#     f"Total dataset: {TOTAL_SAMPLES:,} samples  ·  1 partition per client",
#     fontsize=11, y=1.02,
# )
fig.tight_layout()
out = "figures/energy_vs_partitions.png"
plt.savefig(out, dpi=300, bbox_inches="tight")
print(f"\nSaved: {out}")
# plt.show()
