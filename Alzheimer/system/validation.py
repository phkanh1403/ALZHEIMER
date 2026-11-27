"""Repeated cross-validation on training data (5 folds x 20 repeats, split by participant). Never reads test data.

Usage:
    python validation.py                 # run every analysis (2 parallel processes)
    python validation.py --only main S1  # run selected analyses
    python validation.py --repeats 5     # quick trial run
"""
import argparse
import time
from multiprocessing import Pool

import pandas as pd

from mci_risk.analyses import ANALYSES
from mci_risk.config import BOOTSTRAP, CV_DIR, CV_REPEATS, SENSITIVITY_BOOTSTRAP, TRAIN_DIR
from mci_risk.cv import paired_deltas, run_cv, summary_table
from mci_risk.data import Sources, labelled_raw
from mci_risk.files import save_json

MAIN_EXTRA_SCOPES = ['second_mci', 'imp_L', 'imp_mid', 'imp_R', 'imp_M3']
MRI_AGE_SUBGROUPS = [('MRI <= 180 days', lambda age: age <= 180), ('MRI 181–365 days', lambda age: age > 180)]


def _main_outputs(long, deltas):
    """Extra outputs of the primary analysis: M3 − M2 by scope, S9a subgroups by MRI age, out-of-fold predictions."""
    extra = [deltas]
    for scope in MAIN_EXTRA_SCOPES:
        if (long.scope == scope).any():
            x = paired_deltas(long, [('M3', 'M2')], scope=scope, B=SENSITIVITY_BOOTSTRAP)
            x['comparison'] = x.comparison + f' [{scope}]'
            extra.append(x)
    subgroups = []
    for label, keep in MRI_AGE_SUBGROUPS:
        x = long[keep(long.mri_age) & (long.scope == 'random')]
        _, metrics = summary_table(x.assign(scope='random'), calib_scopes=())
        metrics['subgroup'] = label
        subgroups.append(metrics)
    pd.concat(subgroups).to_csv(CV_DIR / 'S9_mri_age_subgroups.csv', index=False)
    long[long.scope == 'random'].to_csv(CV_DIR / 'main_oof.csv.gz', index=False)
    return pd.concat(extra, ignore_index=True)


def analysis(name, repeats):
    """Run one analysis and write runs/cv/<name>_metrics.csv, _deltas.csv and _info.json."""
    started = time.time()
    spec = ANALYSES[name]
    is_main = name == 'main'
    src = Sources(TRAIN_DIR)
    raw = labelled_raw(src, **spec.raw_options)
    long, info = run_cv(src, raw, spec.models, repeats, name, params=spec.params,
                        transform=spec.transform(src) if spec.transform else None, extra=spec.extra,
                        second=is_main, impute_modes=('L', 'mid', 'R', 'M3') if is_main else ())
    CV_DIR.mkdir(parents=True, exist_ok=True)
    _, metrics = summary_table(long)
    metrics.to_csv(CV_DIR / f'{name}_metrics.csv', index=False)
    if spec.pairs:
        deltas = paired_deltas(long, spec.pairs, B=BOOTSTRAP if is_main else SENSITIVITY_BOOTSTRAP)
        if is_main:
            deltas = _main_outputs(long, deltas)
        deltas.to_csv(CV_DIR / f'{name}_deltas.csv', index=False)
    info['seconds'] = round(time.time() - started)
    save_json(info, CV_DIR / f'{name}_info.json')
    return name, info['seconds']


def _job(args):
    return analysis(*args)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--only', nargs='*', default=list(ANALYSES), choices=list(ANALYSES), metavar='NAME')
    parser.add_argument('--repeats', type=int, default=CV_REPEATS)
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    with Pool(args.workers) as pool:
        for name, seconds in pool.imap_unordered(_job, [(n, args.repeats) for n in args.only]):
            print(f'{name}: finished in {seconds} s', flush=True)


if __name__ == '__main__':
    main()
