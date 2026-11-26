# MCI Dementia Risk

Clinical decision-support app that estimates the risk that a patient with mild cognitive impairment (MCI)
will receive a dementia diagnosis within 1, 2 or 3 years, and the research code that builds and checks the models
on ADNI data. The accompanying paper (Vietnamese, IEEE format) is in `docs/`.

## Run the app (Windows, Python 3.11)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Training data (80% of participants, `data/train`), fitted models and all results (`runs/`) are included, so the app
runs without retraining. All files are read and written as UTF-8.

## What the app offers

| Page | Purpose |
| --- | --- |
| Patient risk | Risk card, gauge and 3-year risk curve with a 95% range; how the patient compares with the study cohort; the factors driving the estimate; values and their dates; optional side-by-side comparison of all four models |
| Add patient | Sliders and number fields for one visit, plus tables for earlier cognitive tests and MRI scans; trends are computed automatically |
| Patient list | Search, star and delete patients; import an ADNI export (ZIP) at each participant's most recent MCI visit |
| Model performance | Accuracy, calibration, decision curve, model trade-off radar chart and the analysis of whether repeat MRI adds value |
| Study cohort | Diagnosis mix, outcomes, age and MMSE distributions, hippocampal volume by outcome, timing of diagnoses and data-quality checks |

Models, from simplest to most complete: **Clinical** (age, sex, education, MMSE, FAQ, ADAS-Cog13, delayed recall,
time since MCI diagnosis), **+ MRI** (hippocampal volume and entorhinal thickness), **+ cognitive trend** and
**+ atrophy trend** (yearly change from repeat MRI). The app uses the most complete model the patient's data allow.

## Reproduce the results

| Step | Command | Time (2 CPU cores) | Main outputs |
| --- | --- | --- | --- |
| 0 | `python tests/run_tests.py` | ~1 min | 11 regression tests |
| 1 | `python train.py` | ~2 min | `preprocessing.json`, `models.json` (paired set), `secondary_models.json` (available-data set), `train_inputs.csv`, `conflicts.csv`, `data_audit.json` |
| 2 | `python validation.py` | ~1–1.5 h | `runs/cv/*`: main cross-validation, leakage audit, sensitivity analyses, per-model results |
| 3 | `python uncertainty.py` | ~10 min | bootstrap draws for individual risk ranges |
| 4 | `python evaluate.py <test_folder>` | ~4 min | `test_*.csv`, `test_info.json` |
| 5 | `python report.py` | seconds | `summary.json`, `sensitivity_summary.csv` |

The analysis plan, thresholds and decision rule are fixed in `runs/protocol.json`.

## Intended use

For adults with confirmed MCI and no previous dementia diagnosis who are not receiving anti-amyloid antibodies.
Atrophy trends need at least two MRI scans on the same FreeSurfer version and field strength, 300 or more days apart.
The tool supports follow-up planning; it does not diagnose. It was developed and checked on the ADNI research cohort,
and its performance in other populations is not yet known.
