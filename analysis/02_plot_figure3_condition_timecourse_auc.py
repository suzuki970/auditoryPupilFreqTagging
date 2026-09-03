from __future__ import annotations

import matplotlib.pyplot as plt

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["font.family"] = "Arial"

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.preprocessing import OneHotEncoder

from config import (
    BLUE,
    GRAY,
    GRID,
    INK,
    MUTED,
    ORANGE,
    OUTPUT_DIR,
    RED,
    SEC_INK,
    SURFACE,
    TONE_RATE_FREQ,
    compute_trial_cos_features,
    condition_accuracy_from_trials,
    ensure_output_dir,
    load_pupil_long,
    load_trial_table,
    sanitize_frame,
    write_json,
)

N_BOOT = 5000
STIM_START = 1.0


def one_hot_freq(freq_group: pd.Series) -> np.ndarray:
    try:
        enc = OneHotEncoder(sparse_output=False)
    except TypeError:
        enc = OneHotEncoder(sparse=False)
    return enc.fit_transform(freq_group.to_frame())


def style_axis(ax, xlo, xhi, ylo, yhi):
    ax.tick_params(colors=INK, direction="out", length=14, width=1.0, labelsize=12)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color("black")
        ax.spines[spine].set_linewidth(1.0)
        ax.spines[spine].set_position(("outward", 8))
    ax.spines["bottom"].set_bounds(xlo, xhi)
    ax.spines["left"].set_bounds(ylo, yhi)


def timecourse_by_detection(trial_table: pd.DataFrame, cond: pd.DataFrame, pupil_long: pd.DataFrame) -> tuple:
    labels = cond.assign(detected=cond["accuracy"] >= 0.5)[["sub", "condition_spl", "detected"]]
    trial_labels = trial_table.merge(labels, on=["sub", "condition_spl"], how="inner")

    lengths = pupil_long.groupby("id").size()
    standard_ids = set(lengths[lengths == lengths.mode()[0]].index)
    trial_labels = trial_labels[trial_labels["trial_id"].isin(standard_ids)]

    pupil = pupil_long[pupil_long["id"].isin(set(trial_labels["trial_id"]))]
    pupil = pupil.merge(
        trial_labels[["trial_id", "sub", "detected"]],
        left_on="id",
        right_on="trial_id",
        how="inner",
    )
    sub_avg = pupil.groupby(["sub", "detected", "time"])["value"].mean().reset_index()
    avg = sub_avg.groupby(["detected", "time"])["value"].mean().reset_index()
    det = avg[avg["detected"]]
    miss = avg[~avg["detected"]]
    return (
        det["time"].to_numpy(),
        det["value"].to_numpy(),
        miss["time"].to_numpy(),
        miss["value"].to_numpy(),
        int(sub_avg.loc[sub_avg["detected"], "sub"].nunique()),
        int(sub_avg.loc[~sub_avg["detected"], "sub"].nunique()),
    )


def cos_by_detection(cond: pd.DataFrame, trial_table: pd.DataFrame, trial_features: pd.DataFrame) -> tuple:
    cond_feat = trial_features.merge(
        trial_table[["trial_id", "sub", "condition_spl"]],
        on="trial_id",
        how="inner",
    )
    cond_feat = cond_feat.groupby(["sub", "condition_spl"])["cos_045"].mean().reset_index()
    df = cond.merge(cond_feat, on=["sub", "condition_spl"], how="inner")
    df = sanitize_frame(df, ["cos_045"])
    df["detected"] = df["accuracy"] >= 0.5
    sub_means = df.groupby(["sub", "detected"])["cos_045"].mean().unstack("detected")
    sub_means = sub_means.dropna(subset=[False, True])
    g_det = sub_means[True].to_numpy()
    g_not = sub_means[False].to_numpy()
    t_stat, p_t = ttest_rel(g_det, g_not)
    w_stat, p_w = wilcoxon(g_det - g_not)
    return g_det, g_not, float(t_stat), float(p_t), float(w_stat), float(p_w), int(len(sub_means))


def bootstrap_auc_ci(y, proba, groups, rng):
    subs = np.unique(groups)
    aucs = []
    for _ in range(N_BOOT):
        sampled = rng.choice(subs, size=len(subs), replace=True)
        idx = np.concatenate([np.where(groups == sub)[0] for sub in sampled])
        if len(np.unique(y[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y[idx], proba[idx]))
    return np.percentile(aucs, [2.5, 97.5])


def condition_level_auc(cond: pd.DataFrame, trial_table: pd.DataFrame, trial_features: pd.DataFrame) -> dict:
    cond_feat = trial_features.merge(
        trial_table[["trial_id", "sub", "condition_spl"]],
        on="trial_id",
        how="inner",
    )
    cond_feat = cond_feat.groupby(["sub", "condition_spl"])["cos_045"].mean().reset_index()
    df = cond.merge(cond_feat, on=["sub", "condition_spl"], how="inner")
    df = sanitize_frame(df, ["spl_db", "cos_045"]).reset_index(drop=True)

    y = (df["accuracy"] >= 0.5).astype(int).to_numpy()
    groups = df["sub"].to_numpy()
    freq = one_hot_freq(df["freq_group"])
    baseline_x = np.column_stack([df[["spl_db"]].to_numpy(), freq])
    pupil_x = df[["cos_045"]].to_numpy()
    combined_x = np.column_stack([baseline_x, pupil_x])

    models = {
        "SPL + frequency": baseline_x,
        "SPL + frequency + 0.45Hz cos": combined_x,
        "0.45Hz cos only": pupil_x,
    }
    gkf = GroupKFold(n_splits=df["sub"].nunique())
    rng = np.random.default_rng(0)
    out = {}
    for label, x in models.items():
        proba = cross_val_predict(
            LogisticRegression(max_iter=1000),
            x,
            y,
            groups=groups,
            cv=gkf,
            method="predict_proba",
        )[:, 1]
        ci = bootstrap_auc_ci(y, proba, groups, rng)
        out[label] = {
            "auc": float(roc_auc_score(y, proba)),
            "ci95": [float(ci[0]), float(ci[1])],
        }
    return out


def main() -> None:
    ensure_output_dir()
    trial_table = load_trial_table(main_only=False)
    pupil_long = load_pupil_long()
    cond = condition_accuracy_from_trials(trial_table)
    trial_features = compute_trial_cos_features(trial_table["trial_id"], pupil_long).reset_index()
    trial_features.to_csv(OUTPUT_DIR / "trial_045hz_cos_features_19conditions.csv", index=False)

    t_det, y_det, t_not, y_not, n_det, n_not = timecourse_by_detection(trial_table, cond, pupil_long)
    g_det, g_not, t_stat, p_t, w_stat, p_w, n_paired = cos_by_detection(cond, trial_table, trial_features)
    aucs = condition_level_auc(cond, trial_table, trial_features)
    write_json(
        OUTPUT_DIR / "figure3_condition_timecourse_auc_19conditions_stats.json",
        {
            "conditions": 19,
            "cos_detected_vs_not_detected": {
                "paired_t": t_stat,
                "p": p_t,
                "wilcoxon": w_stat,
                "wilcoxon_p": p_w,
                "n_paired_subjects": n_paired,
            },
            "condition_level_auc": aucs,
        },
    )

    fig = plt.figure(figsize=(16, 5.8), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.65, 0.85, 1.0], wspace=0.35)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])

    interval = 1 / TONE_RATE_FREQ
    onsets = STIM_START + np.arange(20) * interval
    onsets = onsets[onsets <= t_det.max()]
    for i, onset in enumerate(onsets):
        linestyle = "-" if i % 2 == 0 else (0, (4, 2))
        ax_a.axvline(onset, color=SEC_INK, linewidth=0.7, linestyle=linestyle, alpha=0.35, zorder=1)
    ax_a.plot(t_det, y_det, color=RED, linewidth=2.0, label=f"Detected (n={n_det})")
    ax_a.plot(t_not, y_not, color=GRAY, linewidth=2.0, label=f"Not detected (n={n_not})")
    ax_a.set_xlim(0, t_det.max())
    ax_a.set_ylim(-0.4, max(0.8, float(np.nanmax([y_det.max(), y_not.max()]))))
    ax_a.set_xlabel("Time within trial (s)", color=INK)
    ax_a.set_ylabel("Pupil diameter (a.u.)", color=INK)
    ax_a.set_title("(a) Pupil time course", color=INK)
    style_axis(ax_a, 0, t_det.max(), -0.4, ax_a.get_ylim()[1])
    ax_a.grid(True, axis="y", color=GRID, linewidth=0.6)
    ax_a.legend(frameon=False, fontsize=9)

    bp = ax_b.boxplot([g_det, g_not], positions=[0, 1], widths=0.42, patch_artist=True, showfliers=False)
    for patch, color in zip(bp["boxes"], [RED, GRAY]):
        patch.set_facecolor(color)
        patch.set_alpha(0.3)
        patch.set_edgecolor(color)
    for a, b in zip(g_det, g_not):
        ax_b.plot([0, 1], [a, b], color=SEC_INK, alpha=0.35, linewidth=1.0)
    ax_b.scatter(np.zeros(len(g_det)), g_det, color=RED, edgecolors="white", zorder=3)
    ax_b.scatter(np.ones(len(g_not)), g_not, color=GRAY, edgecolors="white", zorder=3)
    y_top = max(g_det.max(), g_not.max()) + 0.18
    ax_b.plot([0, 0, 1, 1], [y_top - 0.05, y_top, y_top, y_top - 0.05], color=INK, linewidth=1.2)
    ax_b.text(0.5, y_top + 0.04, "*", ha="center", va="bottom", fontsize=16, color=INK)
    ax_b.set_xticks([0, 1])
    ax_b.set_xticklabels(["Detected", "Not detected"], rotation=20, ha="right")
    ax_b.set_ylabel("0.45Hz cos component (z)", color=INK)
    ax_b.set_title(f"(b) 0.45Hz cos\npaired n={n_paired}, p={p_t:.4f}", color=INK)
    ymin = float(np.floor(min(g_det.min(), g_not.min()) * 2) / 2)
    ymax = float(np.ceil((y_top + 0.25) * 2) / 2)
    ax_b.set_ylim(ymin, ymax)
    style_axis(ax_b, 0, 1, ymin, ymax)
    ax_b.grid(True, axis="y", color=GRID, linewidth=0.6)

    labels = list(aucs)
    vals = [aucs[k]["auc"] for k in labels]
    err = [
        [vals[i] - aucs[k]["ci95"][0] for i, k in enumerate(labels)],
        [aucs[k]["ci95"][1] - vals[i] for i, k in enumerate(labels)],
    ]
    x = np.arange(len(labels))
    ax_c.bar(x, vals, color=[MUTED, ORANGE, BLUE], width=0.6, zorder=3)
    ax_c.errorbar(x, vals, yerr=err, fmt="none", ecolor="black", capsize=4, zorder=4)
    ax_c.axhline(0.5, color="black", linestyle=(0, (3, 2)), linewidth=1.0)
    ax_c.set_xticks(x)
    ax_c.set_xticklabels(["SPL +\nfreq", "SPL + freq\n+ cos", "cos\nonly"], fontsize=10)
    ax_c.set_ylim(0.5, 1.08)
    ax_c.set_ylabel("AUC (LOSO logistic regression)", color=INK)
    ax_c.set_title("(c) Condition-level classification\n19 conditions", color=INK)
    style_axis(ax_c, 0, len(labels) - 1, 0.5, 1.0)
    ax_c.grid(True, axis="y", color=GRID, linewidth=0.6)
    for i, val in enumerate(vals):
        ax_c.text(i, val + 0.025, f"{val:.3f}", ha="center", va="bottom", fontsize=10)

    out = OUTPUT_DIR / "figure3_condition_timecourse_auc_19conditions.pdf"
    fig.savefig(out, bbox_inches="tight", facecolor=SURFACE)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
