"""
00_build_dataset.py -- regenerate the preprocessed tables the figure
scripts consume, writing them to analysis/output/.

Inputs (raw, in-repo):
  preprocessing/data/20230718/s*_trial.json  per-subject preprocessed pupil
      epochs (full presentation window, resampled to cfg.RESAMPLING_RATE Hz)
  stim/comparison/*.wav                       stimulus filenames encode
      (freq_hz, spl_db) for stimulus conditions 1..19, in sorted order

Outputs (analysis/output/):
  trial_table.csv     one row per trial
      (trial_id, sub, run, condition_spl, freq_hz, spl_db, freq_group,
       n_task, n_hit)
  pupil_long.parquet  long-format (id, time, value) pupil trace per trial;
      trials whose trace is entirely NaN contribute no rows

Run this before 01/02/03. config.SOURCE_OUTPUT_DIR then resolves to
analysis/output/ automatically.
"""
from __future__ import annotations

import glob
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    OUTPUT_DIR,
    RAW_TRIAL_DIR,
    STIM_COMPARISON_DIR,
    ensure_output_dir,
)


def build_condition_lookup() -> dict[int, dict[str, float]]:
    """Stimulus condition index (1..19) -> {freq_hz, spl_db}, parsed from the
    sorted stimulus wav filenames. This ordering matches how MATLAB's dir()
    listed ./stim/comparison, which is what cfg.condition_frame_spl indexes."""
    files = sorted(p.name for p in STIM_COMPARISON_DIR.glob("*.wav"))
    lookup: dict[int, dict[str, float]] = {}
    for i, fname in enumerate(files, start=1):
        m = re.match(r"\d+_f_(\d+)SPL(-?[\d.]+)\.wav", fname)
        if m is None:
            raise ValueError(f"unexpected stimulus filename: {fname}")
        lookup[i] = {"freq_hz": int(m.group(1)), "spl_db": float(m.group(2))}
    return lookup


def main() -> None:
    ensure_output_dir()

    cond_lookup = build_condition_lookup()
    assert len(cond_lookup) == 19, f"expected 19 stimulus conditions, got {len(cond_lookup)}"

    files = sorted(glob.glob(str(RAW_TRIAL_DIR / "s*_trial.json")))
    if not files:
        raise FileNotFoundError(f"no s*_trial.json found under {RAW_TRIAL_DIR}")
    print(f"found {len(files)} subject files")

    trial_rows: list[dict] = []
    long_frames: list[pd.DataFrame] = []

    for path in files:
        sub_tag = Path(path).name.split("_")[0]  # e.g. "s01"
        with open(path) as f:
            d = json.load(f)

        pdr = np.asarray(d["PDR"], dtype=float)
        n_trials = pdr.shape[0]
        fs = d["cfg"]["RESAMPLING_RATE"]

        for i in range(n_trials):
            cond = int(d["condition_frame_spl"][i])
            task = d["task"][i]  # list of 1 (hit) / [] (miss) per task onset
            n_task = len(task)
            n_hit = sum(1 for t in task if t == 1)

            trial_id = f"{sub_tag}_r{int(d['Run'][i])}_t{i:03d}"
            trial_rows.append(
                {
                    "trial_id": trial_id,
                    "sub": sub_tag,
                    "run": int(d["Run"][i]),
                    "condition_spl": cond,
                    "freq_hz": cond_lookup[cond]["freq_hz"],
                    "spl_db": cond_lookup[cond]["spl_db"],
                    "freq_group": int(d["condition_frame_freq"][i]),
                    "n_task": n_task,
                    "n_hit": n_hit,
                }
            )

            trace = pdr[i]
            valid = ~np.isnan(trace)
            long_frames.append(
                pd.DataFrame(
                    {
                        "id": trial_id,
                        "time": np.arange(trace.shape[0])[valid] / fs,
                        "value": trace[valid],
                    }
                )
            )

        print(f"{sub_tag}: {n_trials} trials, fs={fs} Hz, trace_len={pdr.shape[1]}")

    trial_table = pd.DataFrame(trial_rows)
    pupil_long = pd.concat(long_frames, ignore_index=True)

    trial_path = OUTPUT_DIR / "trial_table.csv"
    pupil_path = OUTPUT_DIR / "pupil_long.parquet"
    trial_table.to_csv(trial_path, index=False)
    pupil_long.to_parquet(pupil_path, index=False)

    print(f"\ntrial_table: {tuple(trial_table.shape)} -> {trial_path}")
    print(f"pupil_long : {tuple(pupil_long.shape)} -> {pupil_path}")
    print(f"  trials with no valid samples: {trial_table.shape[0] - pupil_long['id'].nunique()}")


if __name__ == "__main__":
    main()
