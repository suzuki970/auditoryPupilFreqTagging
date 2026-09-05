## Experimental data for *"Predicting individual differences in near-threshold auditory detection from pupil frequency tagging to periodic sound stimulation"*
Copyright 2026 Yuta Suzuki

### Article information
Yuta Suzuki.

```
auditoryPupilFreqTagging/
├── stim/                   stimulus wav files (control/comparison/task tones)
├── preprocessing/          raw EyeLink data -> per-subject trial JSON
│   ├── rawData/s01..s17/   raw .asc EyeLink recordings (6 sessions each)
│   ├── data/20260903/      re-run of the same pipeline (see Reproducibility notes)
│   ├── parseData.py        preprocessing entry point
│   ├── asc2array_cls.py    .asc parsing, blink interpolation, low-pass filtering
│   ├── pre_processing_cls.py  resampling / outlier-rejection helpers
│   ├── makeFixation.py     fixation detection (imported but unused by parseData.py)
│   └── requirements.txt
└── analysis/               manuscript figures and statistics
    ├── config.py           shared paths, constants, feature/stat helpers
    ├── 00_build_dataset.py builds trial_table.csv + pupil_long.parquet
    ├── 01_plot_figure2_psychometric.py            -> Figure 2
    ├── 02_plot_figure3_condition_timecourse_auc.py -> Figure 3
    ├── 03_plot_figure4_main_prediction.py          -> Figure 4
    ├── output/              generated tables, stats (JSON), figure PDFs
    ├── requirements.txt
    └── README.md            analysis pipeline details
```
### Reproducibility notes
## Pre-processing

`parseData.py` reads each subject's six `.asc` EyeLink sessions from `rawData/sNN/`, and for every trial:

If you only need the manuscript figures, `preprocessing/data/20260903/` already contains the frozen per-subject trial data, so the raw `.asc`
files and the `preprocessing/` scripts are not required:

## make figures

```bash
python 00_build_dataset.py                          # -> output/trial_table.csv, pupil_long.parquet
python 01_plot_figure2_psychometric.py               # -> output/figure2_psychometric_19conditions.pdf
python 02_plot_figure3_condition_timecourse_auc.py    # -> output/figure3_condition_timecourse_auc_19conditions.pdf
python 03_plot_figure4_main_prediction.py             # -> output/figure4_main_045hz_loso_prediction_19conditions.pdf
```


