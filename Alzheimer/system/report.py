"""Combine cross-validation, sensitivity and test results into runs/summary.json and runs/sensitivity_summary.csv.

Conclusions are reported SEPARATELY for cross-validation (80% training data) and the 20% test set, using the rule
locked in config/protocol.json.
"""
import json

import pandas as pd

from mci_risk.analyses import ANALYSES
from mci_risk.config import CV_DIR, DELTA_AUC, HORIZONS, NB_MARGIN, PRIMARY, PROTOCOL, RUNS_DIR
from mci_risk.cv import pair_label
from mci_risk.files import load_json, save_json

PRIMARY_PAIR = pair_label('M3', 'M2')
SCOPE_LABELS = {  # extra scopes of the main analysis, reported as S3 and S6
    'second_mci': 'S3 · second MCI visit only', 'imp_L': 'S6 · onset at last negative visit',
    'imp_mid': 'S6 · onset at interval midpoint', 'imp_R': 'S6 · onset at diagnosis visit',
    'imp_M3': 'S6 · onset drawn from M3',
}
_DELTA_COLUMNS = ['d_auc', 'auc_lo', 'auc_hi', 'd_nb_core', 'nb_lo', 'nb_hi']


def row(df, h, comparison):
    x = df[(df.h == h) & (df.comparison == comparison)]
    return x.iloc[0] if len(x) else None


def _tier(x):
    return dict(d_auc=x.d_auc, auc_ci=[x.auc_lo, x.auc_hi], d_nb=x.d_nb_core, nb_ci=[x.nb_lo, x.nb_hi],
                conclusion=x.conclusion)


def _delta_fields(x):
    return {**{c: x[c] for c in _DELTA_COLUMNS}, 'rule': x.conclusion}


def sensitivity_rows(main):
    rows = []
    for name, spec in ANALYSES.items():
        path = CV_DIR / f'{name}_deltas.csv'
        if spec.label is None or not path.exists():
            continue
        deltas = pd.read_csv(path)
        info = load_json(CV_DIR / f'{name}_info.json')
        for comparison in spec.reported:
            x = row(deltas, PRIMARY, comparison)
            if x is not None:
                rows.append(dict(analysis=spec.label, comparison=comparison, persons=info['persons'],
                                 landmarks=info['landmarks'], **_delta_fields(x)))
    for scope, label in SCOPE_LABELS.items():
        x = row(main, PRIMARY, f'{PRIMARY_PAIR} [{scope}]')
        if x is not None:
            rows.append(dict(analysis=label, comparison=PRIMARY_PAIR, **_delta_fields(x)))
    return pd.DataFrame(rows)


def build():
    out = dict(protocol=dict(delta_auc=DELTA_AUC, nb_margin=NB_MARGIN, primary_horizon=PRIMARY,
                             locked_at=PROTOCOL['locked_at'], amendments=PROTOCOL.get('amendments', [])))
    main = pd.read_csv(CV_DIR / 'main_deltas.csv')
    test = pd.read_csv(RUNS_DIR / 'test_deltas.csv')
    out['tiers'] = {str(h): dict(cv=_tier(row(main, h, PRIMARY_PAIR)), test=_tier(row(test, h, PRIMARY_PAIR)))
                    for h in HORIZONS}
    both_value = all(out['tiers'][str(PRIMARY)][k]['conclusion'] == 'Added value' for k in ('cv', 'test'))
    out['overall'] = ('Repeat MRI added prognostic value in both cross-validation and test data' if both_value else
                      'Not enough evidence that the atrophy rate from repeat MRI adds consistent prognostic value')
    out['test_note'] = ('The test set is an internal check on an ADNI split used in earlier analyses; '
                        'the main conclusion rests on cross-validation, and the test set checks consistency.')

    sens = sensitivity_rows(main)
    sens.to_csv(RUNS_DIR / 'sensitivity_summary.csv', index=False)
    l1, mn = row(pd.read_csv(CV_DIR / 'L1_deltas.csv'), PRIMARY, PRIMARY_PAIR), row(main, PRIMARY, PRIMARY_PAIR)
    out['leakage_audit'] = dict(d_auc_fold=mn.d_auc, d_auc_global=l1.d_auc, difference=l1.d_auc - mn.d_auc)
    save_json(out, RUNS_DIR / 'summary.json')
    print(json.dumps(out, ensure_ascii=False, indent=1, default=str))
    print(sens.round(4).to_string(index=False))


if __name__ == '__main__':
    build()
