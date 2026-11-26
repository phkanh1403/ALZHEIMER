"""Score the 20% test set with models and preprocessing parameters frozen from training. Nothing is refitted.

Usage:  python evaluate.py <unzipped_test_folder>
Only aggregate results are saved to runs/ (no individual test records).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from mci_risk.config import (
    BOOTSTRAP,
    HORIZONS,
    MIN_SUBGROUP,
    MODELS,
    REQUIRED_TABLES,
    RUNS_DIR,
    SENSITIVITY_BOOTSTRAP,
    TRAIN_DIR,
)
from mci_risk.cv import PAIRS, paired_deltas, summary_table
from mci_risk.data import Sources, apply_params, has_table, labelled_raw, read_table
from mci_risk.files import load_json, save_json
from mci_risk.metrics import categories, status, turnbull_calibration
from mci_risk.model import predict

IMPUTE_SCOPES = ['imp_L', 'imp_mid', 'imp_R', 'imp_M3']


def roster(root):
    """Every participant ID in the core tables of a data folder."""
    ids = set()
    for table in REQUIRED_TABLES:
        if has_table(root, table):
            ids.update(read_table(root, table).RID.dropna().astype(int))
    return ids


def score(frame, models, replicates, rng, modes=('ref',)):
    """Predictions and outcome status, one random visit per participant in each replicate."""
    rows = []
    for r in range(replicates):
        ev = frame.sample(frac=1, random_state=int(rng.integers(1e9))).drop_duplicates('RID')
        for h in HORIZONS:
            for mode in modes:
                ref_model = models['M3'] if mode == 'M3' else models['M0']
                y, w, cat = status(ev, h, ref_model, np.random.default_rng(rng.integers(1e9)), models['censoring'],
                                   mode='ref' if mode in ('ref', 'M3') else mode)
                p_null = predict(models['null'], ev, h)
                for n in MODELS:
                    if n not in models:
                        continue
                    rows.append(pd.DataFrame(dict(repeat=r, scope='random' if mode == 'ref' else f'imp_{mode}',
                                                  RID=ev.RID.to_numpy(), h=h, model=n, y=y, w=w,
                                                  p=predict(models[n], ev, h), p_null=p_null, L=ev.L.to_numpy(),
                                                  R=ev.R.to_numpy(), cat=cat, mri_age=ev.mri_age_days.to_numpy())))
    return pd.concat(rows, ignore_index=True)


def _deltas(long, shared, models, rng):
    deltas = [paired_deltas(long, PAIRS)]
    for scope in IMPUTE_SCOPES:
        x = paired_deltas(long, [('M3', 'M2')], scope=scope, B=SENSITIVITY_BOOTSTRAP)
        x['comparison'] = x.comparison + f' [{scope}]'
        deltas.append(x)
    second = shared[shared.second_mci == 1]
    note = ''
    if second.RID.nunique() >= MIN_SUBGROUP:
        x = paired_deltas(score(second, models, 1, rng), [('M3', 'M2')], B=BOOTSTRAP)
        x['comparison'] = x.comparison + ' [second_mci]'
        deltas.append(x)
    else:
        note = (f'Second-MCI-visit subgroup not reported: only {second.RID.nunique()} participants '
                f'(< {MIN_SUBGROUP})')
    return pd.concat(deltas, ignore_index=True), note


def _calibration(long):
    rows = []
    first = long[(long.scope == 'random') & (long.repeat == 0)]
    for (h, n), g in first.groupby(['h', 'model']):
        c = turnbull_calibration(g.L.to_numpy(), g.R.to_numpy(), g.p.to_numpy(), h)
        for group, (pred, obs) in enumerate(zip(c['pred'], c['obs'], strict=True), start=1):
            rows.append(dict(h=h, model=n, group=group, predicted=pred, observed_turnbull=obs))
    return pd.DataFrame(rows)


def _coverage(df, secondary, replicates, seed):
    """Available-data models, each on every test visit with its own inputs."""
    cover = []
    for n, cols in MODELS.items():
        own = df.dropna(subset=cols)
        models = {**{k: secondary[k] for k in ['M0', 'null', 'censoring']}, n: secondary[n]}
        _, metrics = summary_table(score(own, models, replicates, np.random.default_rng(seed)))
        cover.append(metrics[metrics.model == n].assign(persons=own.RID.nunique(), set='Available-data set'))
    return pd.concat(cover, ignore_index=True)


def evaluate(testdir, replicates=20, seed=2026, save=True):
    testdir = Path(testdir)
    overlap = roster(TRAIN_DIR) & roster(testdir)
    if overlap:
        raise ValueError(f'{len(overlap)} participants appear in both training and test data; evaluation stopped')
    prep = load_json(RUNS_DIR / 'preprocessing.json')
    models = load_json(RUNS_DIR / 'models.json')
    secondary = load_json(RUNS_DIR / 'secondary_models.json')
    src = Sources(testdir)
    df = apply_params(labelled_raw(src), prep)
    shared = df.dropna(subset=MODELS['M3']).reset_index(drop=True)
    rng = np.random.default_rng(seed)
    long = score(shared, models, replicates, rng, modes=('ref', 'L', 'mid', 'R', 'M3'))
    _, metrics = summary_table(long)
    deltas, note = _deltas(long, shared, models, rng)
    calibration = _calibration(long)
    cover = _coverage(df, secondary, replicates, seed)

    one = shared.drop_duplicates('RID')
    info = dict(
        test_rids_with_diagnosis=int(src.vis.RID.nunique()), test_landmarks=len(df), test_persons=int(df.RID.nunique()),
        shared_landmarks=len(shared), shared_persons=int(shared.RID.nunique()),
        status_categories_one_per_person={h: pd.Series(categories(one.L.to_numpy(), one.R.to_numpy(), h))
                                          .value_counts().to_dict() for h in HORIZONS},
        conflicts=len(src.conflicts), mri_measurement_error=src.meas_error, note_second_mci=note,
        overlap_rids_with_train=0)
    if save:
        metrics.to_csv(RUNS_DIR / 'test_metrics.csv', index=False)
        deltas.to_csv(RUNS_DIR / 'test_deltas.csv', index=False)
        calibration.to_csv(RUNS_DIR / 'test_calibration.csv', index=False)
        cover.to_csv(RUNS_DIR / 'test_coverage.csv', index=False)
        save_json(info, RUNS_DIR / 'test_info.json')
    return metrics, deltas, info


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('test_directory', type=Path)
    args = parser.parse_args()
    metrics, deltas, info = evaluate(args.test_directory)
    print(json.dumps(info, ensure_ascii=False, indent=1, default=str))
    columns = ['h', 'model', 'n', 'case_certain', 'control_certain', 'ambiguous', 'censored_early', 'auc', 'ipa',
               'oe_turnbull', 'nb_core']
    print(metrics[metrics.scope == 'random'][columns].round(4).to_string(index=False))
    print(deltas.round(4).to_string(index=False))


if __name__ == '__main__':
    main()
