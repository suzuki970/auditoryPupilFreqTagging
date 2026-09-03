from __future__ import annotations

import matplotlib.pyplot as plt

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["font.family"] = "Arial"

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

from config import (
    BLUE,
    GRID,
    INK,
    OUTPUT_DIR,
    SURFACE,
    compute_trial_cos_features,
    ensure_output_dir,
    fisher_z_ci,
    load_pupil_long,
    load_trial_table,
    loso_predict_1d,
    relative_skill,
    write_json,
)

N_PERM = 2000
N_BOOT = 5000


def main() -> None:
    ensure_output_dir()

    trial_table = load_trial_table(main_only=False)
    pupil_long = load_pupil_long()
    trial_features = compute_trial_cos_features(trial_table["trial_id"], pupil_long)

    skill = relative_skill(trial_table)
    merged = trial_table.set_index("trial_id").join(trial_features, how="inner").reset_index()
    subject_features = merged.groupby("sub")["cos_045"].mean().sort_index()
    common = skill.index.intersection(subject_features.index)

    y = skill.loc[common].to_numpy()
    x = subject_features.loc[common].to_numpy()

    scaler = StandardScaler().fit(x.reshape(-1, 1))
    fitted = LinearRegression().fit(scaler.transform(x.reshape(-1, 1)), y).predict(
        scaler.transform(x.reshape(-1, 1))
    )
    r_in, p_in = pearsonr(y, fitted)

    pred = loso_predict_1d(x, y)
    r_loso, p_parametric = pearsonr(y, pred)
    r_ci_lo, r_ci_hi = fisher_z_ci(r_loso, len(y))
    r2_loso = r2_score(y, pred)

    rng = np.random.default_rng(0)
    null = np.zeros(N_PERM)
    for i in range(N_PERM):
        y_perm = rng.permutation(y)
        pred_perm = loso_predict_1d(x, y_perm)
        null[i], _ = pearsonr(y_perm, pred_perm)
    p_perm_one = (np.sum(null >= r_loso) + 1) / (N_PERM + 1)
    p_perm_two = (np.sum(np.abs(null) >= abs(r_loso)) + 1) / (N_PERM + 1)

    table = pd.DataFrame(
        {
            "sub": list(common),
            "relative_skill": y,
            "cos_045": x,
            "fitted_main": fitted,
            "loso_pred_main": pred,
        }
    )
    table.to_csv(OUTPUT_DIR / "main_045hz_cos_loso_subject_table_19conditions.csv", index=False)
    np.save(OUTPUT_DIR / "main_045hz_cos_loso_null_19conditions.npy", null)
    write_json(
        OUTPUT_DIR / "main_045hz_cos_loso_stats_19conditions.json",
        {
            "main_analysis_conditions": 19,
            "n_subjects": int(len(y)),
            "in_sample": {"r": float(r_in), "p": float(p_in), "R2": float(r2_score(y, fitted))},
            "loso": {
                "r": float(r_loso),
                "r_ci95_fisher_z": [float(r_ci_lo), float(r_ci_hi)],
                "p_parametric": float(p_parametric),
                "R2": float(r2_loso),
                "p_perm_one_sided": float(p_perm_one),
                "p_perm_two_sided_reference": float(p_perm_two),
                "n_perm": N_PERM,
            },
        },
    )

    x_actual = table["relative_skill"].to_numpy()
    y_pred = table["loso_pred_main"].to_numpy()
    x_grid = np.linspace(x_actual.min(), x_actual.max(), 100)
    boot_lines = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, len(x_actual), len(x_actual))
        if len(np.unique(x_actual[idx])) < 2:
            continue
        coef = np.polyfit(x_actual[idx], y_pred[idx], 1)
        boot_lines.append(np.polyval(coef, x_grid))
    boot_lines = np.asarray(boot_lines)
    band_lo = np.percentile(boot_lines, 2.5, axis=0)
    band_hi = np.percentile(boot_lines, 97.5, axis=0)
    fit_line = np.polyval(np.polyfit(x_actual, y_pred, 1), x_grid)

    fig, ax = plt.subplots(figsize=(7.2, 6.6), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.fill_between(x_grid, band_lo, band_hi, color=BLUE, alpha=0.18, zorder=1)
    ax.plot(x_grid, fit_line, color=BLUE, linewidth=1.8, zorder=2)
    ax.scatter(x_actual, y_pred, s=95, color=BLUE, edgecolors="white", linewidths=1.0, zorder=3)

    lim = 0.2
    ticks = np.arange(-lim, lim + 1e-9, 0.05)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xlabel("Actual relative detection performance", fontsize=14, color=INK)
    ax.set_ylabel("LOSO predicted relative detection performance", fontsize=14, color=INK)
    ax.set_title(
        "0.45Hz pupil phase predicts relative detection performance\n"
        f"All 19 conditions, n={len(y)}, "
        f"r={r_loso:+.3f} [{r_ci_lo:+.3f}, {r_ci_hi:+.3f}], R2={r2_loso:.3f}, "
        f"permutation p={p_perm_one:.4f}",
        fontsize=12,
        color=INK,
    )

    ax.tick_params(colors=INK, direction="out", length=12, width=1.0, labelsize=11)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color("black")
        ax.spines[spine].set_position(("outward", 8))
    ax.spines["bottom"].set_bounds(ticks[0], ticks[-1])
    ax.spines["left"].set_bounds(ticks[0], ticks[-1])
    ax.grid(True, color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)

    fig.savefig(OUTPUT_DIR / "figure4_main_045hz_loso_prediction_19conditions.pdf", bbox_inches="tight", facecolor=SURFACE)
    print(f"saved {OUTPUT_DIR / 'figure4_main_045hz_loso_prediction_19conditions.pdf'}")
    print(
        f"19-condition LOSO r={r_loso:+.3f}, 95% CI [{r_ci_lo:+.3f}, {r_ci_hi:+.3f}], "
        f"R2={r2_loso:.3f}, one-sided permutation p={p_perm_one:.4f}"
    )


if __name__ == "__main__":
    main()
