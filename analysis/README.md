Reproducible analysis scripts for the auditory pupil frequency-tagging
manuscript. Each script regenerates one manuscript figure and its
accompanying statistics.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Tested with CPython 3.14.6. See `requirements.txt` for pinned versions.

## Running

Run from this directory. `00_build_dataset.py` regenerates the
preprocessed tables into `output/` and must run first; the three figure
scripts are independent of one another afterwards.

```bash
python 00_build_dataset.py
python 01_plot_figure2_psychometric.py
python 02_plot_figure3_condition_timecourse_auc.py
python 03_plot_figure4_main_prediction.py
```

The figure scripts read `output/trial_table.csv` and
`output/pupil_long.parquet`, which `00_build_dataset.py` produces, so run
step 00 first. To read them from elsewhere, set the `PFT_SOURCE_DIR`
environment variable to a directory containing both files.

## Scripts and outputs

| Script | Manuscript figure | Figure PDF (in `output/`) | Other outputs |
|---|---|---|---|
| `00_build_dataset.py` | — (preprocessing) | — | `trial_table.csv`, `pupil_long.parquet` |
| `01_plot_figure2_psychometric.py` | Figure 2 — detection accuracy vs. SPL, logistic fits, PSE | `figure2_psychometric_19conditions.pdf` | `figure2_psychometric_19conditions_stats.json` |
| `02_plot_figure3_condition_timecourse_auc.py` | Figure 3 — pupil time course (detected vs. undetected) and condition-level AUC | `figure3_condition_timecourse_auc_19conditions.pdf` | `figure3_condition_timecourse_auc_19conditions_stats.json`, `trial_045hz_cos_features_19conditions.csv` |
| `03_plot_figure4_main_prediction.py` | Figure 4 — leave-one-subject-out prediction of relative detection performance | `figure4_main_045hz_loso_prediction_19conditions.pdf` | `main_045hz_cos_loso_stats_19conditions.json`, `main_045hz_cos_loso_subject_table_19conditions.csv`, `main_045hz_cos_loso_null_19conditions.npy` |

Figure 1 (auditory stimulus timing schematic) is drawn by hand and has no
generating script.

The manuscript figure files live in
`.../P12_auditoryPFT/draft/figure/Fig{1..4}.pdf`; copy the regenerated
PDFs there manually.

## Data source

The project is self-contained; no sibling project tree is required.

`00_build_dataset.py` reads raw in-repo inputs, resolved relative to the
project root:

```
<project root>/preprocessing/data/20230718/s*_trial.json   per-subject pupil epochs
<project root>/stim/comparison/*.wav                        stimulus condition table
```

and writes `trial_table.csv` + `pupil_long.parquet` to `analysis/output/`,
which the figure scripts then consume (via `config.py`).

`preprocessing/data/20230718/s*_trial.json` is the only input required
under `preprocessing/`. `preprocessing/rawData/` (per-subject `.asc`
EyeLink recordings) and the legacy `preprocessing/*.py` preprocessing
scripts are upstream provenance and are not used by this pipeline.

To read the preprocessed tables from elsewhere, set the `PFT_SOURCE_DIR`
environment variable to a directory containing them. All regenerated
tables/figures are written to `analysis/output/`.

## Conditions

These figures use all 19 stimulus conditions (the 16 SPL-varying
conditions from the 1,000- and 4,000-Hz bands plus the three
high-frequency conditions `condition_spl = 17, 18, 19` at
16,000/18,000/20,000 Hz).
