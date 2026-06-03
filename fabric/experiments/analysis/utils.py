from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# ── paths ────────────────────────────────────────────────────────────────────
DATA_DIR  = Path("/home/maurits/EnergyEfficient_Scattered-Directive/fabric/experiments/data")          # adjust if CSVs live elsewhere
OUT_DIR   = Path("figures")
OUT_DIR.mkdir(exist_ok=True)

ENERGY_CSV  = DATA_DIR / "combined_energy_stats.csv"
GLOBAL_CSV  = DATA_DIR / "combined_global_stats.csv"
CLIENT_CSV  = DATA_DIR / "combined_client_stats.csv"
REPORT_CSV = OUT_DIR / "experiment_analysis.csv"

FIG_EXT = "png"

# ── active client sets per K ──────────────────────────────────────────────────
ACTIVE_CLIENTS = {
    5:  {"client1", "client5", "client9", "client13", "client17"},
    10: {"client1", "client2", "client5", "client6", "client9",
         "client10", "client13", "client14", "client17", "client18"},
}

# ── colour palettes ──────────────────────────────────────────────────────────
# Mid-tone (400-stop) used for Z=15 lines
CAT_COLORS_MID = {
    "Training":      "#378ADD",   # blue-400
    "Agent":         "#1D9E75",   # teal-400
    "Linkerd proxy": "#EF9F27",   # amber-400
    "Sidecar":       "#D85A30",   # coral-400
    "Orchestration": "#534AB7",   # purple-400
    "Other":         "#888780",   # gray-400
}
# Dark (800-stop) used for Z=400 lines
CAT_COLORS = {
    "Training":      "#0C447C",   # blue-800
    "Agent":         "#085041",   # teal-800
    "Linkerd proxy": "#633806",   # amber-800
    "Sidecar":       "#712B13",   # coral-800
    "Orchestration": "#3C3489",   # purple-800
    "Other":         "#444441",   # gray-800
}

# Stack order for stacked bars / time-series (infra on bottom, training on top)
CAT_ORDER = ["Orchestration", "Linkerd proxy", "Sidecar", "Agent", "Training"]

def load_energy(ENERGY_CSV) -> pd.DataFrame:
    df = pd.read_csv(ENERGY_CSV)
    df = df[df["joules"] > 0].copy()   # drop linkerd-init (always 0)
    df["category"] = df.apply(_categorise, axis=1)
    return df


def load_global(GLOBAL_CSV) -> pd.DataFrame:
    df = pd.read_csv(GLOBAL_CSV)
    df["round_duration_s"] = df["RoundDuration"] / 1e9
    # Derive round index per run (no explicit round column in global stats)
    id_cols = ["exp", "K", "Z", "sigma_ed", "sigma_iid", "timestamp"]
    df["round"] = df.groupby(id_cols).cumcount()
    return df


def load_client(CLIENT_CSV) -> pd.DataFrame:
    df = pd.read_csv(CLIENT_CSV)
    # Aggregate per-client training time to per-round max (slowest client = bottleneck)
    id_cols = ["exp", "K", "Z", "sigma_ed", "sigma_iid", "timestamp"]
    agg = (
        df.rename(columns={"Round": "round"})
        .groupby(id_cols + ["round"])["ClientTrainingTime"]
        .max()
        .reset_index()
        .rename(columns={"ClientTrainingTime": "max_client_training_ms"})
    )
    return agg

def _categorise(row) -> str:
    ns  = str(row["namespace"]).lower()
    ctr = str(row["container_name"]).lower()

    # Training containers (live inside a client pod)
    if ctr in ("hfl-train", "hfl-train-model"):
        return "Training"

    # Linkerd proxy (anywhere)
    if ctr == "linkerd-proxy":
        return "Linkerd proxy"

    # Sidecar (inside a client pod)
    if ctr == "sidecar":
        return "Sidecar"

    # Agent: container_name matches the namespace, e.g. client1/client1
    if ns.startswith("client") and ctr == ns:
        return "Agent"

    # Orchestration layer
    if ctr in ("api-gateway", "policy-enforcer", "orchestrator"):
        return "Orchestration"

    return "Other"

def _is_infra(cat: str) -> bool:
    return cat in ("Agent", "Linkerd proxy", "Sidecar", "Orchestration")

def savefig(fig, name: str):
    out = OUT_DIR / f"{name}.{FIG_EXT}"
    fig.savefig(out)
    print(f"  Saved {out}")
    plt.close(fig)
