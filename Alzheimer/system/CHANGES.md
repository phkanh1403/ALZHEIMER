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
