import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_style("whitegrid")

df = pd.read_csv(
    "/home/maurits/EnergyEfficient_Scattered-Directive/fabric/experiments/data/exp1/combined_energy_stats.csv",
    index_col=[0],
)
df = df[df["K"] == 5]

active_clients = ["client1", "client5", "client9", "client13", "client17"]
entities = active_clients + ["server"]


def darken(hex_color, factor=0.6):
    """Return a darkened version of hex_color (factor < 1 = darker)."""
    r, g, b = mcolors.to_rgb(hex_color)
    return mcolors.to_hex((r * factor, g * factor, b * factor))


# Base colors
C_TRAIN = "#06A77D"
C_INFRA = "#E63946"
C_ORCH  = "#1f77b4"

# Training sub-components: hfl-train (lighter), hfl-train-model (darker)
COLOR_HFL_TRAIN       = C_TRAIN
COLOR_HFL_TRAIN_MODEL = darken(C_TRAIN, 0.6)

# Infra sub-components: agent (lighter), sidecar (mid), linkerd (darker)
COLOR_AGENT   = C_INFRA
COLOR_SIDECAR = darken(C_INFRA, 0.72)
COLOR_LINKERD = darken(C_INFRA, 0.50)

# Orchestration sub-components: api-gateway (lighter), orchestrator (mid), policy-enforcer (darker)
# Orchestration infra: sidecar + linkerd use same shades as entity infra
COLOR_API_GW   = C_ORCH
COLOR_ORCH_CTR = darken(C_ORCH, 0.72)
COLOR_POLICY   = darken(C_ORCH, 0.50)

# --- Collect entity data ---
results = []
for entity in entities:
    edf = df[df["namespace"] == entity]

    hfl_train_kj = (
        edf[edf["container_name"] == "hfl-train"]["joules"].sum() / 1000.0
    )
    hfl_train_model_kj = (
        edf[edf["container_name"] == "hfl-train-model"]["joules"].sum() / 1000.0
    )
    agent_kj = (
        edf[edf["container_name"] == entity]["joules"].sum() / 1000.0
    )
    sidecar_kj = (
        edf[edf["container_name"] == "sidecar"]["joules"].sum() / 1000.0
    )
    linkerd_kj = (
        edf[edf["container_name"].isin(["linkerd-proxy", "linkerd-init"])]["joules"].sum()
        / 1000.0
    )

    results.append({
        "entity":              entity,
        "hfl_train_kj":        hfl_train_kj,
        "hfl_train_model_kj":  hfl_train_model_kj,
        "agent_kj":            agent_kj,
        "sidecar_kj":          sidecar_kj,
        "linkerd_kj":          linkerd_kj,
    })

edf_plot = pd.DataFrame(results)

# --- Collect orchestration data ---
orch_containers = ["orchestrator", "policy-enforcer", "api-gateway"]
orch_pods = df[df["container_name"].isin(orch_containers)]["pod_name"].unique()
odf = df[df["pod_name"].isin(orch_pods)]

api_gw_kj  = odf[odf["container_name"] == "api-gateway"]["joules"].sum()       / 1000.0
orch_ctr_kj = odf[odf["container_name"] == "orchestrator"]["joules"].sum()     / 1000.0
policy_kj  = odf[odf["container_name"] == "policy-enforcer"]["joules"].sum()   / 1000.0
orch_sidecar_kj = odf[odf["container_name"] == "sidecar"]["joules"].sum()      / 1000.0
orch_linkerd_kj = (
    odf[odf["container_name"].isin(["linkerd-proxy", "linkerd-init"])]["joules"].sum()
    / 1000.0
)

# --- Layout ---
n_entities  = len(edf_plot)   # 6
n_groups    = n_entities + 1  # +1 for Orchestration
bar_width   = 0.35
group_gap   = 0.3

group_centers = np.arange(n_groups) * (2 * bar_width + group_gap)
left_bar      = group_centers - bar_width / 2
right_bar     = group_centers + bar_width / 2

MIN_LABEL_H = 0.03  # kJ — skip segment label if too thin


def draw_stacked(ax, positions, segments, bar_width):
    """segments: list of (values, color, label). Returns (drawn, final_bottoms)."""
    bottoms = np.zeros(len(np.atleast_1d(positions)))
    drawn = []
    for values, color, label in segments:
        vals = np.atleast_1d(np.asarray(values, dtype=float))
        rects = ax.bar(positions, vals, bar_width, bottom=bottoms,
                       color=color, label=label)
        drawn.append((rects, vals.copy(), bottoms.copy()))
        bottoms += vals
    return drawn, bottoms


def label_segments(ax, drawn):
    for rects, vals, bots in drawn:
        for rect, val, bot in zip(rects, vals, bots):
            if val >= MIN_LABEL_H:
                ax.text(
                    rect.get_x() + rect.get_width() / 2,
                    bot + val / 2,
                    f"{val:.2f}",
                    ha="center", va="center",
                    fontsize=7, color="white", fontweight="bold",
                )


def label_totals(ax, positions, totals):
    for pos, total in zip(np.atleast_1d(positions), np.atleast_1d(totals)):
        ax.text(pos, total + 0.01, f"{total:.2f}",
                ha="center", va="bottom", fontsize=8, fontweight="bold")


fig, ax = plt.subplots(figsize=(16, 7))

# --- Entity bars (clients + server) ---

# Left: Training — hfl-train (lighter) + hfl-train-model (darker)
drawn_train, totals_train = draw_stacked(
    ax, left_bar[:n_entities],
    [
        (edf_plot["hfl_train_kj"].values,       COLOR_HFL_TRAIN,       "hfl-train"),
        (edf_plot["hfl_train_model_kj"].values,  COLOR_HFL_TRAIN_MODEL, "hfl-train-model"),
    ],
    bar_width,
)

# Right: Infra — agent (lighter) + sidecar (mid) + linkerd (darker)
drawn_infra, totals_infra = draw_stacked(
    ax, right_bar[:n_entities],
    [
        (edf_plot["agent_kj"].values,   COLOR_AGENT,   "Agent"),
        (edf_plot["sidecar_kj"].values,  COLOR_SIDECAR, "Sidecar"),
        (edf_plot["linkerd_kj"].values,  COLOR_LINKERD, "Linkerd"),
    ],
    bar_width,
)

label_segments(ax, drawn_train)
label_segments(ax, drawn_infra)
label_totals(ax, left_bar[:n_entities],  totals_train)
label_totals(ax, right_bar[:n_entities], totals_infra)

# --- Orchestration bars ---
orch_idx = n_entities

# Left: Orchestration containers — api-gateway / orchestrator / policy-enforcer
drawn_orch, totals_orch = draw_stacked(
    ax, [left_bar[orch_idx]],
    [
        (np.array([api_gw_kj]),   COLOR_API_GW,   "api-gateway"),
        (np.array([orch_ctr_kj]), COLOR_ORCH_CTR, "orchestrator"),
        (np.array([policy_kj]),   COLOR_POLICY,   "policy-enforcer"),
    ],
    bar_width,
)

# Right: Orchestration infra — sidecar + linkerd (reuse same infra shades)
drawn_orch_infra, totals_orch_infra = draw_stacked(
    ax, [right_bar[orch_idx]],
    [
        (np.array([orch_sidecar_kj]), COLOR_SIDECAR, "_Sidecar"),
        (np.array([orch_linkerd_kj]), COLOR_LINKERD, "_Linkerd"),
    ],
    bar_width,
)

label_segments(ax, drawn_orch)
label_segments(ax, drawn_orch_infra)
label_totals(ax, [left_bar[orch_idx]],  totals_orch)
label_totals(ax, [right_bar[orch_idx]], totals_orch_infra)

# --- X-axis group labels ---
group_labels = edf_plot["entity"].tolist() + ["Orchestration"]
ax.set_xticks(group_centers)
ax.set_xticklabels(group_labels, rotation=15, ha="right")

# Sub-labels below each bar pair
for i, (lp, rp) in enumerate(zip(left_bar, right_bar)):
    is_orch = (i == n_entities)
    ax.text(lp, -0.055, "orch" if is_orch else "train",
            ha="center", va="top", fontsize=7, color="gray",
            transform=ax.get_xaxis_transform())
    ax.text(rp, -0.055, "infra",
            ha="center", va="top", fontsize=7, color="gray",
            transform=ax.get_xaxis_transform())

# --- Vertical separator before Orchestration ---
sep_x = (group_centers[n_entities - 1] + group_centers[n_entities]) / 2
ax.axvline(sep_x, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)

# --- Legend (skip _ prefix duplicates) ---
handles, labels_leg = ax.get_legend_handles_labels()
seen = {}
for h, l in zip(handles, labels_leg):
    if not l.startswith("_") and l not in seen:
        seen[l] = h
ax.legend(seen.values(), seen.keys(), loc="upper right", framealpha=0.9, fontsize=9)

ax.set_ylabel("Energy (kJ)")
ax.set_title("Energy Breakdown per Entity — Training vs Infrastructure (K=5)")
ax.grid(True, alpha=0.3, axis="y")
ax.set_xlim(left_bar[0] - bar_width, right_bar[-1] + bar_width)

plt.tight_layout()
plt.savefig(
    "/home/maurits/EnergyEfficient_Scattered-Directive/fabric/experiments/data/exp1/combined_energy_breakdown.png",
    dpi=150,
)

print(edf_plot.to_string(index=False))
print(f"\nOrchestration containers: api-gateway={api_gw_kj:.3f}, orchestrator={orch_ctr_kj:.3f}, policy-enforcer={policy_kj:.3f} kJ")
print(f"Orchestration infra:      sidecar={orch_sidecar_kj:.3f}, linkerd={orch_linkerd_kj:.3f} kJ")
