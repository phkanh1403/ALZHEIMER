# Changes in v8

Results change: the protocol amendment of 27/11/2026 (logged in `config/protocol.json`, made after the v7 results had
been seen) widens the lambda grid, so every model and analysis was rerun. Before any change, the v7 code was rerun in
the new environment and reproduced the v7 test results exactly (difference 0), so all changes in the numbers come from
the amendment.

## Penalty (lambda) selection

- v7 chose lambda = 10, the upper end of the grid [0.1, 1, 10], in 372 of 400 choices, so the optimum could lie
  outside the grid (on the side of *stronger* penalisation; the v7 report wrongly said the grid lacked small values).
  The grid is now a half-decade log grid 0.1–1000 (`config/protocol.json`). The chosen values are 30–300, inside the
  grid.
- The inner-loop score (`cv.interval_loglik`) now uses the same per-participant weights (1/n_i) as the fitted
  likelihood; before, participants with many visits dominated the choice of lambda.
- `cv.lambda_scores` / `best_lambda` / `choose_lambda` take the grid as an argument; `run_cv` records the share of
  choices at the edges of the grid (`lambda_at_grid_edge` in `*_info.json`) and warns when it exceeds 50%.

## Confusion matrix

- `metrics.confusion` (IPCW-weighted shares of TP/FP/FN/TN) and `metrics.confusion_rates` (sensitivity, specificity,
  PPV, NPV, F1, share flagged). `summarize` reports the cells at the two risk-band cut-offs (15% and 30%) and PR-AUC, so
  every metrics file (cross-validation and test) carries them. Descriptive secondary metrics; the decision rule is
  unchanged.
- The band cut-offs are defined once (`config.BAND_THRESHOLDS`) and shared by the app bands and the confusion matrix.
- New *Confusion matrix* tab on the Model performance page: model, time frame, cut-off and data source; matrix per 100
  patients with known status, the five rates, and a table of all models at both cut-offs.

## App text

- Every chart has a caption written from the numbers it shows.
- Patient risk: one overview paragraph for the selected patient (risk, 95% range, band, cohort position, main drivers,
  warning when the range crosses a band boundary, model used). The "Risk over the next 1–3 years" curve is removed.
- "Compare all four models": an explanation for the selected patient (step with the largest change, band agreement,
  overlap of the 95% ranges). Text comes from pure functions in `mci_risk/narrative.py`, with tests.

## Data protection and packaging

- `make_release.py` / `mci_risk/release.py` build the distributable ZIP without ADNI tables or participant-level files,
  and refuse to build if any remaining CSV header or JSON key holds a participant identifier.
- The cohort comparison on the Patient risk page uses an aggregated distribution (`runs/cohort_risk.json`: quantiles
  and a 2-point histogram per model and time frame) instead of participant rows, so the app works without ADNI data.
  The Study cohort page and the ADNI-participant picker appear only when local data exist.
- `.gitignore` excludes ADNI data and participant-level outputs.

## Report

- `build_report.py` + `reportgen/`: every result number (LaTeX macro `\R{key}`), table and figure of the report is
  generated from `runs/`; the report is compiled to PDF (XeLaTeX) and DOCX (pandoc, with section, figure, table,
  equation and reference numbers taken from the PDF build).
- `train.py` records the MRI QC audit (records without an overall QC rating) used in the report.

# Changes in v7 (code review and clean-up of v6)

All results are unchanged. The refactored code was run against the v6 code on the same data and gives identical
outputs: training (all 8 files in `runs/`), all 17 cross-validation analyses, the test evaluation, bootstrap draws and
the report. The only addition to `models.json` / `secondary_models.json` is a `converged` flag per model.

## Fixes in the app

- **Manual entry is validated** (`mci_risk/manual_entry.py`). v6 accepted, without warning, entries the models were
  never trained on: earlier test results dated after the visit (a future value then shaped the trend), an MRI dated
  after the visit (negative MRI age), an MRI years older than the visit (the pipeline only uses an MRI from the
  365 days before the visit), and earlier scans at another field strength. These are now rejected with an
  explanation, as are incomplete rows, an incomplete current MRI and out-of-range values. Earlier MRI scans now have
  a field-strength column.
- **Model performance page no longer crashes** when test or report results are missing; it says which script to run.
- **Patient store** opens a short connection per operation (safe across Streamlit sessions; v6 opened a new
  connection on every rerun and never closed it) and lives in `instance/` instead of the code folder.
- v6 copied the whole training cohort into the patient table on first start but never showed it; that copy is no
  longer made, and existing copies are removed once by a migration.

## Structure

- Library code moved into the `mci_risk` package; the scripts at the root only orchestrate. v6 had training and test
  evaluation importing from the `validation.py` script; shared CV code now lives in `mci_risk/cv.py`.
- One source of configuration: `mci_risk/config.py` reads time frames, follow-up window, folds, lambda grid,
  bootstrap size and decision rule from `config/protocol.json` (moved out of the results folder). v6 hard-coded
  5 folds and repeated time frames and thresholds in several files.
- `app.py` (658 lines) split into one module per page under `mci_risk/ui/`; patient-level logic (`risk.py`) is
  separate from Streamlit and tested. The app reuses the model's prediction function instead of its own copy.
- `registry.py` split by responsibility into `store.py`, `adni_import.py` and `manual_entry.py`.
- The 17 cross-validation analyses are a declarative table (`mci_risk/analyses.py`) used by both `validation.py`
  and `report.py`, replacing an if/elif chain and a second label list.
- `Sources` answers as-of-visit queries (`demographics_at`, `clinical_at`, `mri_at`, `bsi_at`); `raw_points`
  dropped from cyclomatic complexity 29 to 7, and no function is above 17 (v6: three above 20).

## Robustness and clean code

- Optimiser status is checked for both models: the interval model raises on a hard failure and warns
  (`ConvergenceWarning`) on an early line-search stop; the Cox model warns. Each model records `converged`.
- The pandas fragmentation warning is fixed at its source instead of being silenced globally at import.
- `metrics.status` rejects an unknown imputation mode instead of silently treating it as 'R'.
- Named constants replace magic numbers (IPCW floor, days per year/month, S9 centring, hash constant, file-size
  limit); unused imports and variables removed; the field-strength label is in English like the others.
- `ruff check .` passes with the rules in `pyproject.toml`.

## Tests

- `tests/run_tests.py` (11 checks in one script) replaced by a pytest suite of 58 tests: the original regression
  checks plus unit tests of the models (synthetic data, analytic gradient), Turnbull, net benefit, the decision rule,
  folds, manual-entry validation, the patient store and migration, ZIP import, configuration consistency, and every
  app page rendered with Streamlit's AppTest (including missing results).
