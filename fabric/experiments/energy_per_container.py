"""
Horizontal stacked bar chart: energy per container name per round.

Active clients : client1, client5, client9, client13, client17
Idle clients   : all other clientN containers
Server         : server container
Everything else: sidecar, linkerd-proxy, linkerd-init, hfl-train, hfl-train-model, other
"""

import re

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd

# ── Config ────────────────────────────────────────────────────────────────────

CSV_PATH = "/home/maurits/EnergyEfficient_Scattered-Directive/fabric/experiments/data/combined_energy_stats.csv"

ACTIVE_CLIENTS = {"client1", "client5", "client9", "client13", "client17", "server"}

# Category order (bottom → top in horizontal bars = left → right)
CATEGORIES = [
    "api-gateway",
    "policy-enforcer",
    "orchestrator",
    "hfl-train",
    "hfl-train-model",
    "server",
    "agent (active)",
    "agent (idle)",
    "sidecar",
    "linkerd-proxy",
]


COLORS = {
    "agent (active)":  "#d62728",
    "agent (idle)":    "#d62728",
    "hfl-train":       "#2ca02c",
    "hfl-train-model": "#2ca02c",
    "sidecar":         "#d62728",
    "linkerd-proxy":   "#d62728",
    "server":          "#d62728",
    "api-gateway":     "#1f77b4",
    "policy-enforcer": "#1f77b4",
    "orchestrator":    "#1f77b4",
}

df = pd.read_csv(CSV_PATH)
# df["joules"] = pd.to_numeric(df["joules"], errors="coerce").fillna(0.0)
# df["Z"] = df["Z"].astype(int)
df = df[df["exp"] == "exp1"]

def classify(row):
    cname = row["container_name"]
    if cname in ACTIVE_CLIENTS:
        return "agent (active)"
    if re.match(r"^client\d+$", cname):       # clientN but not in active set
        return "agent (idle)"
    if cname in CATEGORIES:                    # exact match for named categories
        return cname
    return "other"

df["category"] = df.apply(classify, axis=1)
# print(df)

# ── Average energy per container instance per round ───────────────────────────
# Step 1: sum energy per (round, pod, container_name) — one value per container per round
per_container_round = (
    df.groupby(["round", "timestamp", "category"])["joules"]
    .sum()
    .reset_index()
)
print(per_container_round[per_container_round['category'] == "hfl-train"]['joules'].sum())

# Step 2: average over all containers of that category and all rounds
avg_per_category = (
    per_container_round
    .groupby("category")["joules"]
    .mean()
    .div(1e3)   # J → kJ
    # .div(3)     # Per timestamp
    .reindex(CATEGORIES)
    # .fillna(0.0)
    .reset_index()

).sort_values(by="joules", ascending=True)
avg_per_category.columns = ["category", "avg_kj"]

print(avg_per_category.to_string(index=False))

# ── Plot ──────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(10, 5))

bars = ax.barh(
    avg_per_category["category"],
    avg_per_category["avg_kj"],
    color=[COLORS[c] for c in avg_per_category["category"]],
    height=0.6,
    edgecolor="white",
    linewidth=0.5,
)

# Value labels
for bar, val in zip(bars, avg_per_category["avg_kj"]):
    if val > 0:
        ax.text(val + avg_per_category["avg_kj"].max() * 0.008,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.2f} kJ",
                va="center", fontsize=9)

labels = [item.get_text() for item in ax.get_yticklabels()]
labels[3] = "agent (server)"
ax.set_yticklabels(labels)

# ax.set_xlabel("Average energy per container per round (kJ)", fontsize=11)
ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
ax.grid(axis="x", linestyle="--", alpha=0.35)
ax.set_xlim(0, avg_per_category["avg_kj"].max() * 1.18)
ax.invert_yaxis()   # largest bar at top
ax.spines["right"].set_visible(False)
# fig.suptitle(
#     "Average energy consumption per container type per round",
#     fontsize=12, fontweight="bold",
# )
orch_patch =  mpatches.Patch(color='#1f77b4', label='Orchestration')
train_patch =  mpatches.Patch(color='#2ca02c', label='Training')
infra_patch =  mpatches.Patch(color='#d62728', label='Infrastructure')
fig.legend(handles=[orch_patch, train_patch, infra_patch])
fig.tight_layout()
ax.spines["top"].set_visible(False)
out = "energy_per_container_type.png"
plt.savefig(f"figures/{out}", dpi=150)
print(f"\nSaved: {out}")
# plt.show()
