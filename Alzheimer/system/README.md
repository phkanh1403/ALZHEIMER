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

Training data (80% of participants, `data/train`), fitted models and all results (`runs/`) are included, so the app
runs without retraining. Saved patients are kept in `instance/profiles.sqlite` (set `MCI_RISK_DB` to use another
file). The app has no login: run it on a trusted machine, and do not store directly identifying names.

## What the app offers

| Page | Purpose |
| --- | --- |
| Patient risk | Risk card, gauge and 3-year risk curve with a 95% range; how the patient compares with the study cohort; the factors driving the estimate; values and their dates; optional side-by-side comparison of all four models |
| Add patient | Sliders and number fields for one visit, plus tables for earlier cognitive tests and MRI scans; trends are computed automatically. Entries the models were not trained on (history after the visit, MRI older than 365 days or after the visit, earlier scans at another field strength, out-of-range values) are rejected with an explanation |
| Patient list | Search, star and delete patients; import an ADNI export (ZIP) at each participant's most recent MCI visit |
| Model performance | Accuracy, calibration, decision curve, model trade-off radar chart and the analysis of whether repeat MRI adds value |
| Study cohort | Diagnosis mix, outcomes, age and MMSE distributions, hippocampal volume by outcome, timing of diagnoses and data-quality checks |

Models, from simplest to most complete: **Clinical** (age, sex, education, MMSE, FAQ, ADAS-Cog13, delayed recall,
time since MCI diagnosis), **+ MRI** (hippocampal volume and entorhinal thickness), **+ cognitive trend** and
**+ atrophy trend** (yearly change from repeat MRI). The app uses the most complete model the patient's data allow.

## Reproduce the results

| Step | Command | Time (2 CPU cores) | Main outputs |
| --- | --- | --- | --- |
| 0 | `python -m pytest` | ~20 s | 58 tests (data pipeline, models, metrics, manual entry, store, import, every app page) |
| 1 | `python train.py` | ~1.5 min | `preprocessing.json`, `models.json` (paired set), `secondary_models.json` (available-data set), `train_inputs.csv`, `conflicts.csv`, `data_audit.json` |
| 2 | `python validation.py` | ~1–1.5 h | `runs/cv/*`: main cross-validation, leakage audit, sensitivity analyses, per-model results |
| 3 | `python uncertainty.py` | ~10 min | bootstrap draws for individual risk ranges |
| 4 | `python evaluate.py <test_folder>` | ~4 min | `test_*.csv`, `test_info.json` |
| 5 | `python report.py` | seconds | `summary.json`, `sensitivity_summary.csv` |

`python split_dataset.py <original> <train> <test>` recreates the frozen 80/20 split from `data/partition.csv`.
The analysis plan, thresholds and decision rule are fixed in `config/protocol.json`; the code reads time frames,
follow-up window, cross-validation design and decision rule from that file.

## Code layout

```
app.py                  Streamlit entry point (page routing only)
train.py  validation.py  evaluate.py  uncertainty.py  report.py  split_dataset.py    command-line steps
config/protocol.json    locked analysis protocol
mci_risk/               library (never imports the scripts)
  config.py             paths, protocol values and every fixed constant
  data.py               ADNI sources, landmarks, interval labels, as-of-visit features, preprocessing parameters
  model.py              interval-censored proportional hazards model, ridge Cox model, prediction
  metrics.py            outcome status, IPCW, Turnbull estimator, AUC/IPA/net benefit, decision rule
  cv.py                 participant-level folds, nested lambda choice, repeated CV, paired bootstrap deltas
  analyses.py           declarative specs of the main, L1 and S1–S10 analyses
  risk.py               patient-level logic: applicable models, estimate with range, risk drivers
  manual_entry.py       hand-entered patients, validated against the pipeline's rules
  adni_import.py        ZIP import of ADNI exports
  store.py              SQLite patient store
  ui/                   one module per app page, plus shared helpers
tests/                  pytest suite
```

Development: `pip install -r requirements-dev.txt`, then `ruff check .` and `python -m pytest`.

## Intended use

For adults with confirmed MCI and no previous dementia diagnosis who are not receiving anti-amyloid antibodies.
Atrophy trends need at least two MRI scans on the same FreeSurfer version and field strength, 300 or more days apart.
The tool supports follow-up planning; it does not diagnose. It was developed and checked on the ADNI research cohort,
and its performance in other populations is not yet known.

## Data use

The ADNI data in `data/train` and the participant-level files in `runs/` are covered by the ADNI Data Use Agreement.
Check its terms before sharing this folder outside the approved project team.
