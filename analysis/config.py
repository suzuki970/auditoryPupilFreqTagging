from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler


ANALYSIS_DIR = Path(__file__).resolve().parent
PROJECT_DIR = ANALYSIS_DIR.parent
OUTPUT_DIR = ANALYSIS_DIR / "output"

# Raw in-repo inputs for 00_build_dataset.py.
RAW_TRIAL_DIR = PROJECT_DIR / "preprocessing" / "data" / "20230718"
STIM_COMPARISON_DIR = PROJECT_DIR / "stim" / "comparison"

# Preprocessed tables (trial_table.csv, pupil_long.parquet) consumed by the
# figure scripts. 00_build_dataset.py regenerates them into OUTPUT_DIR from
# the raw in-repo inputs; run it first. Override the location with the
# PFT_SOURCE_DIR env var. Self-contained: no dependency on any sibling tree.
if os.environ.get("PFT_SOURCE_DIR"):
    SOURCE_OUTPUT_DIR = Path(os.environ["PFT_SOURCE_DIR"])
else:
    SOURCE_OUTPUT_DIR = OUTPUT_DIR
SOURCE_DEBUG_DIR = SOURCE_OUTPUT_DIR

HIGH_FREQ_CONDITIONS = {17, 18, 19}

T_START = 3.22
T_END = 23.22
PHASE_ORIGIN = 1.0
PAIR_CYCLE_FREQ = 0.45
TONE_RATE_FREQ = 0.90

BLUE = "#738CD9"
ORANGE = "#E69F00"
RED = "#d97168"
GRAY = "#bebebe"
INK = "#0b0b0b"
SEC_INK = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"


def ensure_output_dir() -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=2)


def _source(name: str) -> Path:
    path = SOURCE_OUTPUT_DIR / name
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found -- run `python 00_build_dataset.py` first "
            "(or set PFT_SOURCE_DIR)."
        )
    return path


def load_trial_table(main_only: bool = True) -> pd.DataFrame:
    df = pd.read_csv(_source("trial_table.csv"))
    if main_only:
        df = df[~df["condition_spl"].isin(HIGH_FREQ_CONDITIONS)].copy()
    return df


def load_pupil_long() -> pd.DataFrame:
    return pd.read_parquet(_source("pupil_long.parquet"))


def relative_skill(trial_table: pd.DataFrame) -> pd.Series:
    task_trials = trial_table[trial_table["n_task"] > 0].copy()
    rows = []
    for (sub, cond), g in task_trials.groupby(["sub", "condition_spl"]):
        rows.append(
            {
                "sub": sub,
                "condition_spl": cond,
                "accuracy": g["n_hit"].sum() / g["n_task"].sum(),
            }
        )
    acc = pd.DataFrame(rows)
    acc["residual"] = acc.groupby("condition_spl")["accuracy"].transform(
        lambda s: s - s.mean()
    )
    return acc.groupby("sub")["residual"].mean().sort_index()


def condition_accuracy_from_trials(trial_table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    task_trials = trial_table[trial_table["n_task"] > 0].copy()
    for (sub, cond), g in task_trials.groupby(["sub", "condition_spl"]):
        rows.append(
            {
                "sub": sub,
                "condition_spl": cond,
                "freq_hz": g["freq_hz"].iloc[0],
                "spl_db": g["spl_db"].iloc[0],
                "freq_group": g["freq_group"].iloc[0],
                "n_task": int(g["n_task"].sum()),
                "n_hit": int(g["n_hit"].sum()),
                "n_trials": int(len(g)),
                "accuracy": g["n_hit"].sum() / g["n_task"].sum(),
            }
        )
    return pd.DataFrame(rows).sort_values(["sub", "condition_spl"]).reset_index(drop=True)


def sanitize_frame(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    q = out[cols].quantile([0.02, 0.98])
    for col in cols:
        out[col] = out[col].clip(q.loc[0.02, col], q.loc[0.98, col])
    std = out[cols].std().replace(0, 1.0)
    out[cols] = ((out[cols] - out[cols].mean()) / std).clip(-8, 8)
    return out


def fisher_z_ci(r: float, n: int) -> tuple[float, float]:
    z = np.arctanh(r)
    se = 1.0 / np.sqrt(n - 3)
    z_crit = 1.959963984540054
    return float(np.tanh(z - z_crit * se)), float(np.tanh(z + z_crit * se))


def loso_predict_1d(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float).reshape(-1, 1)
    y = np.asarray(y, dtype=float)
    preds = np.zeros(len(y), dtype=float)
    for train, test in LeaveOneOut().split(x):
        scaler = StandardScaler().fit(x[train])
        model = LinearRegression().fit(scaler.transform(x[train]), y[train])
        preds[test] = model.predict(scaler.transform(x[test]))
    return preds


def compute_trial_cos_features(
    trial_ids: pd.Series | np.ndarray,
    pupil_long: pd.DataFrame,
    freq: float = PAIR_CYCLE_FREQ,
) -> pd.DataFrame:
    keep = set(pd.Series(trial_ids).astype(str))
    seg = pupil_long[(pupil_long["time"] >= T_START) & (pupil_long["time"] <= T_END)]
    seg = seg[seg["id"].isin(keep)]

    rows = []
    for trial_id, g in seg.groupby("id"):
        t = g["time"].to_numpy() - PHASE_ORIGIN
        x = g["value"].to_numpy()
        coef = np.sum(x * np.exp(-2j * np.pi * freq * t))
        rows.append(
            {
                "trial_id": trial_id,
                "cos_045": np.cos(np.angle(coef)),
                "real_045": np.real(coef),
                "imag_045": np.imag(coef),
                "abs_045": np.abs(coef),
            }
        )
    return pd.DataFrame(rows).set_index("trial_id").sort_index()
