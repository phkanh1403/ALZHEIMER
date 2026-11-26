"""Combine cross-validation, sensitivity and test results into runs/summary.json and runs/sensitivity_summary.csv.

Conclusions are reported SEPARATELY for cross-validation (80% training data) and the 20% test set, using the rule locked in protocol.json.
"""
import json
import pandas as pd
from pipeline import ROOT, PRIMARY, save_json
from metrics import PROTOCOL

R = ROOT / 'runs'
CV = R / 'cv'
LABELS = {
    'main': 'Primary analysis', 'L1': 'L1 · preprocessing fitted on all training data',
    'S1': 'S1 · standard Cox (visit date as event date)', 'S2a': 'S2a · two-scan atrophy rate', 'S2b': 'S2b · hippocampal BSI (ADNI-GO/2)',
    'S4': 'S4 · confirmed dementia only', 'S5': 'S5 · without FAQ', 'S7': 'S7 · excluding inconsistent demographics',
    'S8_1': 'S8 · MRI ≥ 1 day before visit', 'S8_7': 'S8 · MRI ≥ 7 days before visit', 'S8_30': 'S8 · MRI ≥ 30 days before visit',
    'S9': 'S9 · MRI age interaction', 'S10': 'S10 · same scanner model',
}


def row(df, h, comp):
    x = df[(df.h == h) & (df.comparison == comp)]
    return x.iloc[0] if len(x) else None


def build():
    out = dict(protocol=dict(delta_auc=PROTOCOL['delta_auc'], nb_margin=PROTOCOL['nb_margin'],
                             primary_horizon=PRIMARY, locked_at=PROTOCOL['locked_at'],
                             amendments=PROTOCOL.get('amendments', [])))
    main = pd.read_csv(CV / 'main_deltas.csv')
    test = pd.read_csv(R / 'test_deltas.csv')
    tiers = {}
    for h in PROTOCOL['horizons_days']:
        c, t = row(main, h, 'M3 − M2'), row(test, h, 'M3 − M2')
        tiers[str(h)] = dict(
            cv=dict(d_auc=c.d_auc, auc_ci=[c.auc_lo, c.auc_hi], d_nb=c.d_nb_core, nb_ci=[c.nb_lo, c.nb_hi], conclusion=c.conclusion),
            test=dict(d_auc=t.d_auc, auc_ci=[t.auc_lo, t.auc_hi], d_nb=t.d_nb_core, nb_ci=[t.nb_lo, t.nb_hi], conclusion=t.conclusion))
    out['tiers'] = tiers
    both_value = all(tiers[str(PRIMARY)][k]['conclusion'] == 'Added value' for k in ('cv', 'test'))
    out['overall'] = ('Repeat MRI added prognostic value in both cross-validation and test data' if both_value else
                      'Not enough evidence that the atrophy rate from repeat MRI adds consistent prognostic value')
    out['test_note'] = ('The test set is an internal check on an ADNI split used in earlier analyses; '
                        'the main conclusion rests on cross-validation, and the test set checks consistency.')

    rows = []
    for name, lab in LABELS.items():
        p = CV / f'{name}_deltas.csv'
        if not p.exists():
            continue
        d = pd.read_csv(p)
        info = json.loads((CV / f'{name}_info.json').read_text(encoding='utf-8'))
        comps = {'S1': ['M3_cox − M3', 'M3 − M2'], 'S2a': ['M3_twopoint − M2', 'M3 − M2'], 'S2b': ['M2_bsi − M2', 'M3 − M2'],
                 'S9': ['M1_age − M1']}.get(name, ['M3 − M2'])
        for comp in comps:
            x = row(d, PRIMARY, comp)
            if x is None:
                continue
            rows.append(dict(analysis=lab, comparison=comp, persons=info['persons'], landmarks=info['landmarks'],
                             d_auc=x.d_auc, auc_lo=x.auc_lo, auc_hi=x.auc_hi, d_nb_core=x.d_nb_core,
                             nb_lo=x.nb_lo, nb_hi=x.nb_hi, rule=x.conclusion))
    for sc in ['second_mci', 'imp_L', 'imp_mid', 'imp_R', 'imp_M3']:
        x = row(main, PRIMARY, f'M3 − M2 [{sc}]')
        if x is not None:
            rows.append(dict(analysis={'second_mci': 'S3 · second MCI visit only', 'imp_L': 'S6 · onset at last negative visit', 'imp_mid': 'S6 · onset at interval midpoint',
                                       'imp_R': 'S6 · onset at diagnosis visit', 'imp_M3': 'S6 · onset drawn from M3'}[sc],
                             comparison='M3 − M2', d_auc=x.d_auc, auc_lo=x.auc_lo, auc_hi=x.auc_hi, d_nb_core=x.d_nb_core,
                             nb_lo=x.nb_lo, nb_hi=x.nb_hi, rule=x.conclusion))
    sens = pd.DataFrame(rows)
    sens.to_csv(R / 'sensitivity_summary.csv', index=False)
    l1, mn = row(pd.read_csv(CV / 'L1_deltas.csv'), PRIMARY, 'M3 − M2'), row(main, PRIMARY, 'M3 − M2')
    out['leakage_audit'] = dict(d_auc_fold=mn.d_auc, d_auc_global=l1.d_auc, difference=l1.d_auc - mn.d_auc)
    save_json(out, R / 'summary.json')
    print(json.dumps(out, ensure_ascii=False, indent=1, default=str))
    print(sens.round(4).to_string(index=False))


if __name__ == '__main__':
    build()
