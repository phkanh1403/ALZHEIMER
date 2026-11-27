# MCI Dementia Risk

Clinical decision-support app that estimates the risk that a patient with mild cognitive impairment (MCI)
will receive a dementia diagnosis within 1, 2 or 3 years, and the research code that builds and checks the models
on ADNI data. The accompanying paper (Vietnamese, IEEE format) is in `docs/`.

## Run the app (Windows, Python 3.11+)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Fitted models and aggregated results (`runs/`) are included, so the app runs without the ADNI data and without
retraining. ADNI data and every participant-level file are not distributed (see **Data** below); without them the
*Study cohort* page and the *ADNI study cohort* patient picker are hidden, and everything else works. Saved patients
are kept in `instance/profiles.sqlite` (set `MCI_RISK_DB` to use another file). The app has no login: run it on a
trusted machine, and do not store directly identifying names.

## What the app offers

| Page | Purpose |
| --- | --- |
| Patient risk | A one-paragraph overview written from this patient's numbers (risk, 95% range, band, position in the study cohort, main drivers); risk card and gauge; comparison with the study cohort; the factors driving the estimate; values and their dates; side-by-side comparison of all four models with an explanation of how they differ for this patient. Every chart has a short caption |
| Add patient | Sliders and number fields for one visit, plus tables for earlier cognitive tests and MRI scans; trends are computed automatically. Entries the models were not trained on (history after the visit, MRI older than 365 days or after the visit, earlier scans at another field strength, out-of-range values) are rejected with an explanation |
| Patient list | Search, star and delete patients; import an ADNI export (ZIP) at each participant's most recent MCI visit |
| Model performance | Accuracy, calibration, decision curve, **confusion matrix** (IPCW-weighted, at the 15% and 30% band cut-offs, cross-validation or test, with sensitivity, specificity, PPV, NPV and F1), model trade-off radar chart and the analysis of whether repeat MRI adds value. Every chart has a short caption |
| Study cohort | Diagnosis mix, outcomes, age and MMSE distributions, hippocampal volume by outcome, timing of diagnoses and data-quality checks (needs the local ADNI files) |

Models, from simplest to most complete: **Clinical** (age, sex, education, MMSE, FAQ, ADAS-Cog13, delayed recall,
time since MCI diagnosis), **+ MRI** (hippocampal volume and entorhinal thickness), **+ cognitive trend** and
**+ atrophy trend** (yearly change from repeat MRI). The app uses the most complete model the patient's data allow.

## Reproduce the results

| Step | Command | Time (2 CPU cores) | Main outputs |
| --- | --- | --- | --- |
| 0 | `python -m pytest` | ~30 s | data pipeline, models, metrics, confusion matrix, patient narratives, manual entry, store, import, release builder, every app page (tests needing ADNI data are skipped without it) |
| 1 | `python train.py` | ~1 min | `preprocessing.json`, `models.json` (paired set), `secondary_models.json` (available-data set), `cohort_risk.json` (aggregated), `data_audit.json`; local only: `train_inputs.csv`, `train_landmarks.csv`, `conflicts.csv` |
| 2 | `python validation.py` | ~40–60 min | `runs/cv/*`: main cross-validation, leakage audit, sensitivity analyses, per-model results |
| 3 | `python uncertainty.py` | ~5 min | bootstrap draws for individual risk ranges |
| 4 | `python evaluate.py <test_folder>` | ~3 min | `test_*.csv`, `test_info.json` |
| 5 | `python report.py` | seconds | `summary.json`, `sensitivity_summary.csv` |
| 6 | `python build_report.py` | ~1 min | `docs/CoSoLyThuyet_v8.pdf` and `.docx`: every result number, table and figure generated from `runs/` |
| 7 | `python make_release.py` | seconds | `dist/MCI_Dementia_Risk.zip` without ADNI or participant-level data |

`python split_dataset.py <original> data/train data/test` recreates the frozen 80/20 split from `data/partition.csv`.
Step 6 needs XeLaTeX (with TeX Gyre Termes) and pandoc.
The analysis plan, thresholds and decision rule are fixed in `config/protocol.json`; the code reads time frames,
follow-up window, cross-validation design and decision rule from that file.

## Code layout

```
app.py                  Streamlit entry point (page routing only)
train.py  validation.py  evaluate.py  uncertainty.py  report.py  split_dataset.py    command-line steps
build_report.py  make_release.py                                                    report and distribution
config/protocol.json    locked analysis protocol
mci_risk/               library (never imports the scripts)
  config.py             paths, protocol values and every fixed constant
  data.py               ADNI sources, landmarks, interval labels, as-of-visit features, preprocessing parameters
  model.py              interval-censored proportional hazards model, ridge Cox model, prediction
  metrics.py            outcome status, IPCW, Turnbull estimator, AUC/IPA/net benefit, decision rule
  cv.py                 participant-level folds, nested lambda choice, repeated CV, paired bootstrap deltas
  analyses.py           declarative specs of the main, L1 and S1–S10 analyses
  risk.py               patient-level logic: applicable models, estimate with range, risk drivers, cohort position
  narrative.py          plain-language captions and summaries for one patient (pure functions, tested)
  release.py            what may be distributed; refuses any table with participant identifiers
  manual_entry.py       hand-entered patients, validated against the pipeline's rules
  adni_import.py        ZIP import of ADNI exports
  store.py              SQLite patient store
  ui/                   one module per app page, plus shared helpers
reportgen/              report numbers, tables, figures and LaTeX-to-DOCX conversion
docs/                   report source (CoSoLyThuyet_v8.tex), generated PDF/DOCX
tests/                  pytest suite
```

Development: `pip install -r requirements-dev.txt`, then `ruff check .` and `python -m pytest`.

## Intended use

For adults with confirmed MCI and no previous dementia diagnosis who are not receiving anti-amyloid antibodies.
Atrophy trends need at least two MRI scans on the same FreeSurfer version and field strength, 300 or more days apart.
The tool supports follow-up planning; it does not diagnose. It was developed and checked on the ADNI research cohort,
and its performance in other populations is not yet known.

## Data

ADNI data cannot be redistributed under the ADNI Data Use Agreement, so this package contains no ADNI tables and no
file with one row per participant or visit. To retrain or rerun the analyses, request access at
https://adni.loni.usc.edu, download the tables listed in `mci_risk/config.py` (`REQUIRED_TABLES`, `OPTIONAL_TABLES`)
plus `DATADIC`, and split them with `split_dataset.py` into `data/train` and `data/test`.

`make_release.py` builds the distributable ZIP. It leaves out `data/train`, `data/test`, `train_inputs.csv`,
`train_landmarks.csv`, `conflicts.csv` and out-of-fold prediction files, then scans every remaining CSV header and JSON
key and refuses to build if any participant identifier (`RID`, `PTID`, image ids) is left. The only exception is
`data/partition.csv` (participant numbers and split side, no measurements), needed to reproduce the split. Do not
share local copies that contain `data/train` or the participant-level files in `runs/`.
