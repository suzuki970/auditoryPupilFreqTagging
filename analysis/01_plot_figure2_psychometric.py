from __future__ import annotations

import matplotlib.pyplot as plt

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["font.family"] = "Arial"

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from config import (
    BLUE,
    GRAY,
    GRID,
    HIGH_FREQ_CONDITIONS,
    INK,
    MUTED,
    ORANGE,
    OUTPUT_DIR,
    RED,
    SEC_INK,
    SURFACE,
    condition_accuracy_from_trials,
    ensure_output_dir,
    load_trial_table,
    write_json,
)

SPL_CAL_OFFSET = 65.0


def expand_hits(g: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    spl = np.repeat(g["spl_db_cal"].to_numpy(), g["n_task"].to_numpy().astype(int))
    hit = np.concatenate(
        [
            np.r_[np.ones(int(h)), np.zeros(int(n) - int(h))]
            for n, h in zip(g["n_task"].to_numpy(), g["n_hit"].to_numpy())
        ]
    )
    return spl, hit


def fit_logistic_curve(g: pd.DataFrame, x_grid: np.ndarray) -> tuple[np.ndarray, float]:
    spl, hit = expand_hits(g)
    clf = LogisticRegression(C=np.inf)
    clf.fit(spl.reshape(-1, 1), hit)
    slope = clf.coef_[0, 0]
    intercept = clf.intercept_[0]
    y_grid = 1 / (1 + np.exp(-(slope * x_grid + intercept)))
    pse = -intercept / slope
    return y_grid, float(pse)


def style_axis(ax, xlo, xhi, ylo=0.0, yhi=1.0):
    ax.tick_params(colors=INK, direction="out", length=12, width=1.0, labelsize=12)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color("black")
        ax.spines[spine].set_linewidth(1.0)
        ax.spines[spine].set_position(("outward", 8))
    ax.spines["bottom"].set_bounds(xlo, xhi)
    ax.spines["left"].set_bounds(ylo, yhi)


def plot_psychometric(include_high_frequency: bool, out_name: str) -> dict:
    trial_table = load_trial_table(main_only=False)
    cond = condition_accuracy_from_trials(trial_table)
    cond["spl_db_cal"] = cond["spl_db"] + SPL_CAL_OFFSET

    main = cond[~cond["condition_spl"].isin(HIGH_FREQ_CONDITIONS)].copy()
    high = cond[cond["condition_spl"].isin(HIGH_FREQ_CONDITIONS)].copy()

    fig, ax = plt.subplots(figsize=(7.2, 6.2), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    colors = {1000: BLUE, 4000: ORANGE}
    labels = {1000: "1,000 Hz", 4000: "4,000 Hz"}
    x_grid = np.linspace(0, 80, 300)
    stats = {"include_high_frequency": include_high_frequency, "main_conditions": 16}

    for freq in [1000, 4000]:
        g = main[main["freq_hz"] == freq].copy()
        group = g.groupby("spl_db_cal")["accuracy"].agg(["mean", "std", "count"]).reset_index()
        group["sem"] = group["std"] / np.sqrt(group["count"])
        ax.errorbar(
            group["spl_db_cal"],
            group["mean"],
            yerr=group["sem"],
            fmt="o",
            markersize=7,
            capsize=4,
            color=colors[freq],
            ecolor="black",
            elinewidth=1.0,
            linewidth=1.6,
            zorder=3,
            label=labels[freq],
        )
        y_grid, pse = fit_logistic_curve(g, x_grid)
        ax.plot(x_grid, y_grid, color=colors[freq], linewidth=2.0, zorder=2)
        ax.axvline(pse, color=colors[freq], linestyle=(0, (3, 2)), linewidth=1.4, alpha=0.75)
        ax.text(pse, 0.93 if freq == 1000 else 0.84, f"PSE={pse:.1f} dB SPL", ha="center", fontsize=10, color=colors[freq])
        stats[str(freq)] = {"pse_db_spl": pse}

    if include_high_frequency:
        hf_group = high.groupby("freq_hz").agg(
            spl_db_cal=("spl_db_cal", "mean"),
            accuracy=("accuracy", "mean"),
            sem=("accuracy", lambda s: s.std() / np.sqrt(s.count())),
            n_subjects=("sub", "nunique"),
        ).reset_index()
        x_offsets = {16000: -2.2, 18000: 0.0, 20000: 2.2}
        markers = {16000: "D", 18000: "s", 20000: "^"}
        for _, row in hf_group.iterrows():
            freq = int(row["freq_hz"])
            x_plot = row["spl_db_cal"] + x_offsets[freq]
            ax.errorbar(
                x_plot,
                row["accuracy"],
                yerr=row["sem"] if np.isfinite(row["sem"]) else 0,
                fmt=markers[freq],
                markersize=6,
                capsize=3,
                color=RED,
                ecolor="black",
                elinewidth=1.3,
                capthick=1.3,
                zorder=5,
            )
            ax.annotate(
                f"{freq // 1000} kHz",
                (x_plot, row["accuracy"]),
                textcoords="offset points",
                xytext=(6, 5),
                fontsize=9,
                color=SEC_INK,
            )
        for freq in [16000, 18000, 20000]:
            ax.scatter([], [], marker=markers[freq], color=RED, label=f"{freq // 1000} kHz")
        stats["high_frequency_conditions"] = hf_group.to_dict(orient="records")

    ax.set_xlim(0, 80)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(np.arange(0, 81, 10))
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_xlabel("SPL (dB SPL, approx.)", fontsize=14, color=INK, labelpad=10)
    ax.set_ylabel("Detection accuracy", fontsize=14, color=INK, labelpad=10)
    title_suffix = "all 19 conditions" if include_high_frequency else "16 SPL-varying conditions"
    ax.set_title(f"Detection performance across SPL\n{title_suffix}", fontsize=13, color=INK, pad=12)
    style_axis(ax, 0, 80)
    ax.grid(True, axis="y", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=11, loc="upper right", bbox_to_anchor=(0.98, 0.78))

    fig.savefig(OUTPUT_DIR / out_name, bbox_inches="tight", facecolor=SURFACE)
    print(f"saved {OUTPUT_DIR / out_name}")
    return stats


def main() -> None:
    ensure_output_dir()
    stats19 = plot_psychometric(True, "figure2_psychometric_19conditions.pdf")
    write_json(
        OUTPUT_DIR / "figure2_psychometric_19conditions_stats.json",
        {"nineteen_conditions": stats19},
    )


if __name__ == "__main__":
    main()
