"""Train the models on data/train.

Outputs (runs/):
  preprocessing.json      head-size/BLUP parameters from all training data (final models and test)
  models.json             PAIRED set: M0–M3 trained on the same visits with complete M3 inputs (added-value comparison)
  secondary_models.json   AVAILABLE-DATA set: each model trained on all visits with its own inputs (used for prediction)
  cohort_risk.json        aggregated predicted-risk distribution of the cohort (quantiles, histogram) per model
  coverage.csv, data_audit.json
Participant-level outputs (covered by the ADNI Data Use Agreement, never distributed; see make_release.py):
  train_landmarks.csv     labelled visits; train_inputs.csv: every MCI visit (including the latest, unlabelled)
  conflicts.csv           data-quality log
"""
import json

import numpy as np
import pandas as pd

from mci_risk.config import HORIZONS, MODELS, RUNS_DIR, TRAIN_DIR
from mci_risk.cv import FoldSpec, choose_lambda
from mci_risk.data import Sources, apply_params, eligible_points, labelled_raw, public, raw_points, read_table
from mci_risk.files import save_json
from mci_risk.metrics import categories, censoring_model
from mci_risk.model import fit_interval
from mci_risk.risk import risk_distribution

LAMBDA_SEED = 2026
S9_MRI_WINDOW = 180


def _complete(cols):
    return lambda frame: frame.dropna(subset=cols)


def fit_paired(src, raw, df, prep, all_rids):
    """M0–M3 on the same visits (complete M3 inputs), lambda chosen by nested CV."""
    shared = df.dropna(subset=MODELS['M3'])
    spec = FoldSpec(frozenset(), 'fold', prep, _complete(MODELS['M3']))
    lam = choose_lambda(src, raw.loc[shared.index], set(shared.RID), all_rids, MODELS, spec, seed=LAMBDA_SEED)
    models = {n: fit_interval(shared, c, lam=lam[n]) for n, c in MODELS.items()}
    models['null'] = fit_interval(shared, [], lam=1.0)
    models['censoring'] = censoring_model(shared)
    for n in MODELS:
        models[n].update(set='Paired comparison set', persons_train=int(shared.RID.nunique()))
    return models, shared, lam


def fit_available(src, raw, df, prep, all_rids, lam_paired):
    """Each model on every visit with its own inputs (used by the app); returns models and coverage rows."""
    models, coverage = {}, []
    for n, cols in MODELS.items():
        own = df.dropna(subset=cols)
        spec = FoldSpec(frozenset(), 'fold', prep, _complete(cols))
        lam = choose_lambda(src, raw.loc[own.index], set(own.RID), all_rids, {n: cols}, spec, seed=LAMBDA_SEED)[n]
        models[n] = fit_interval(own, cols, lam=lam)
        models[n].update(set='Available-data set', persons_train=int(own.RID.nunique()))
        coverage.append(dict(model=n, landmarks=len(own), persons=own.RID.nunique(),
                             persons_with_event_in_window=int(own.groupby('RID').event.max().sum()),
                             lam_secondary=lam, lam_paired=lam_paired[n]))
    complete_m0 = df.dropna(subset=MODELS['M0'])
    models['null'] = fit_interval(complete_m0, [], lam=1.0)
    models['censoring'] = censoring_model(complete_m0)
    return models, pd.DataFrame(coverage)


def cohort_risk(models, inputs):
    """Risk distribution at each participant's latest MCI visit, for every available-data model."""
    latest = inputs.sort_values('t0').drop_duplicates('RID', keep='last')
    return {n: risk_distribution(models[n], latest.dropna(subset=cols), HORIZONS) for n, cols in MODELS.items()}


def data_audit(src, all_rids, inputs, df, shared, lam):
    cats = {h: pd.Series(categories(shared.L.to_numpy(), shared.R.to_numpy(), h)).value_counts().to_dict()
            for h in HORIZONS}
    conflicts = pd.DataFrame(src.conflicts).groupby(['table', 'action']).size()
    return dict(
        train_rids=len(all_rids), eligible_points=len(inputs), labelled_landmarks=len(df),
        latest_points_without_followup=int((inputs.has_followup == 0).sum()),
        shared_landmarks=len(shared), shared_persons=int(shared.RID.nunique()),
        mri_measurement_error=src.meas_error,
        conflicts={f'{table} — {action}': int(v) for (table, action), v in conflicts.items()},
        demographic_conflict_rids=len(src.demo_conflict),
        status_categories_shared=cats,
        straddle_1095_kept=int(((shared.L < max(HORIZONS)) & (shared.R > max(HORIZONS))
                                & np.isfinite(shared.R)).sum()),
        mri_lag_days_quantiles=shared.mri_age_days.quantile([.1, .25, .5, .75, .9]).to_dict(),
        mri_same_day_share=float((shared.mri_age_days == 0).mean()),
        freesurfer_stream='Cross-Sectional FreeSurfer (7.x) per DATADIC: each image segmented independently; '
                          'no later images used',
        rundate_after_landmark=int((pd.to_datetime(shared.mri_run) > shared.t0).sum()),
        lambda_paired=lam,
        mri_qc=mri_qc_audit(),
    )


def mri_qc_audit():
    """How many FreeSurfer records carry an overall QC rating (records with an empty rating are kept)."""
    qc = read_table(TRAIN_DIR, 'UCSFFSX7').OVERALLQC
    return dict(records=len(qc), overall_qc_missing=int(qc.isna().sum()),
                overall_qc_counts={str(k): int(v) for k, v in qc.value_counts().items()})


def train():
    out = RUNS_DIR
    src = Sources(TRAIN_DIR)
    all_rids = set(src.vis.RID)
    prep = src.fit_params(all_rids)
    save_json(prep, out / 'preprocessing.json')
    pd.DataFrame(src.conflicts).to_csv(out / 'conflicts.csv', index=False)

    raw = labelled_raw(src)
    df = apply_params(raw, prep)
    public(df).to_csv(out / 'train_landmarks.csv', index=False)
    inputs = apply_params(raw_points(src, eligible_points(src.vis)), prep)
    public(inputs).to_csv(out / 'train_inputs.csv', index=False)

    paired, shared, lam = fit_paired(src, raw, df, prep, all_rids)
    save_json(paired, out / 'models.json')
    available, coverage = fit_available(src, raw, df, prep, all_rids, lam)
    save_json(available, out / 'secondary_models.json')
    save_json(cohort_risk(available, inputs), out / 'cohort_risk.json')

    # S9: coverage when the MRI must be at most 180 days old
    recent = apply_params(labelled_raw(src, mri_window=S9_MRI_WINDOW), prep).dropna(subset=MODELS['M3'])
    coverage.loc[len(coverage)] = dict(model=f'M3, MRI <= {S9_MRI_WINDOW} days (S9)', landmarks=len(recent),
                                       persons=recent.RID.nunique(),
                                       persons_with_event_in_window=int(recent.groupby('RID').event.max().sum()))
    coverage.to_csv(out / 'coverage.csv', index=False)

    audit = data_audit(src, all_rids, inputs, df, shared, lam)
    save_json(audit, out / 'data_audit.json')
    print(coverage.to_string(index=False))
    print(json.dumps(audit, ensure_ascii=False, indent=1, default=str))


if __name__ == '__main__':
    train()
