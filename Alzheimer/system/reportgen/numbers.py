"""Every result number quoted in the report text, keyed for the LaTeX macro \\R{key}."""
from mci_risk.config import BAND_THRESHOLDS, CV_FOLDS, CV_REPEATS, HORIZONS, LAMBDAS, MODEL_PAIRS, MODELS, PRIMARY
from mci_risk.cv import edge_share, pair_label

from .fmt import ci, conclusion, integer, num, pct
from .results import Results

PAIR_KEYS = {pair_label(a, b): f'{a}{b}' for a, b in MODEL_PAIRS}   # 'M3 − M2' -> 'M3M2'


def _delta_numbers(prefix, table, out):
    for comparison, key in PAIR_KEYS.items():
        for h in HORIZONS:
            d = Results.delta(table, comparison, h)
            k = f'{prefix}-{key}-{h}'
            out[f'{k}-dauc'] = num(d.d_auc)
            out[f'{k}-auc-ci'] = ci(d.auc_lo, d.auc_hi)
            out[f'{k}-dnb'] = num(d.d_nb_core)
            out[f'{k}-nb-ci'] = ci(d.nb_lo, d.nb_hi)
            out[f'{k}-rule'] = conclusion(d.conclusion)
            out[f'{k}-rule-lc'] = conclusion(d.conclusion).lower()


def _range(values, digits=3):
    return f'{num(min(values), digits)}–{num(max(values), digits)}'


def _lambda_numbers(r, out):
    info = r.info('main')
    counts = info['lambdas']
    total = sum(counts['M0'].values())
    out['lambda-grid'] = '; '.join(num(v, 1 if v < 1 else 0) for v in LAMBDAS)
    out['lambda-choices'] = integer(total * len(MODELS))
    chosen = {}
    for m in MODELS:
        for lam, c in counts[m].items():
            chosen[float(lam)] = chosen.get(float(lam), 0) + c
    top = sorted(chosen.items(), key=lambda kv: -kv[1])
    out['lambda-top'] = ', '.join(f'λ = {num(lam, 0)} ({integer(c)} lần)' for lam, c in top[:3])
    edge = edge_share(counts)
    out['lambda-edge-max'] = pct(max(edge.values()), 0)
    final = r.audit['lambda_paired']
    out['lambda-final'] = ', '.join(f'{m}: {num(final[m], 0)}' for m in MODELS)


def _cv_numbers(r, out):
    main = r.main
    for (h, m), row in main.iterrows():
        k = f'cv-{m}-{h}'
        out[f'{k}-auc'] = num(row.auc)
        out[f'{k}-oe'] = num(row.oe_turnbull, 2)
        out[f'{k}-slope'] = num(row.cal_slope_turnbull, 2)
        out[f'{k}-nb20'] = num(row.nb_20)
        out[f'{k}-ipa'] = num(row.ipa)
        out[f'{k}-brier'] = num(row.brier)
    out['cv-auc-range'] = _range(main.auc)
    out['cv-oe-range'] = _range(main.oe_turnbull, 2)
    out['cv-slope-range'] = _range(main.cal_slope_turnbull, 2)
    prim = main.loc[PRIMARY]
    out['cv-nb20-range'] = _range(prim.nb_20)
    out['cv-treatall20'] = num(prim.nb_treat_all_20.iloc[0])
    out['cv-observed-ipcw'] = num(prim.observed_ipcw.iloc[0])
    out['cv-observed-turnbull'] = num(prim.observed_turnbull.iloc[0])
    m3 = prim.loc['M3']
    for cat in ('case_certain', 'control_certain', 'ambiguous', 'censored_early'):
        out[f'cv-status-{cat.replace("_", "")}'] = num(m3[cat], 1)


def _test_numbers(r, out):
    test = r.test.loc[PRIMARY]
    out['test-auc-range'] = _range(test.auc)
    out['test-oe-range'] = _range(test.oe_turnbull, 2)
    out['test-slope-range'] = _range(test.cal_slope_turnbull, 1)
    info = r.json('test_info.json')
    out['test-persons'] = integer(info['shared_persons'])
    out['test-landmarks'] = integer(info['shared_landmarks'])
    out['test-labelled-persons'] = integer(info['test_persons'])
    out['test-labelled-landmarks'] = integer(info['test_landmarks'])
    out['test-per-group'] = integer(round(info['shared_persons'] / 10))


def _sensitivity_numbers(r, out):
    primary = pair_label('M3', 'M2')
    l1 = Results.delta(r.deltas('L1'), primary)
    out['L1-dauc-global'] = num(l1.d_auc, 4)
    out['L1-dauc-fold'] = num(Results.delta(r.main_deltas, primary).d_auc, 4)
    s1 = r.metrics('S1').loc[365]
    out['S1-oe-cox'] = num(s1.loc['M3_cox'].oe_turnbull, 2)
    out['S1-oe-interval'] = num(s1.loc['M3'].oe_turnbull, 2)
    out['S1-ipa-cox'] = num(s1.loc['M3_cox'].ipa)
    out['S1-ipa-interval'] = num(s1.loc['M3'].ipa)
    s2b = r.deltas('S2b')
    d = Results.delta(s2b, primary)
    out['S2b-persons'] = integer(r.info('S2b')['persons'])
    out['S2b-dauc'] = num(d.d_auc)
    out['S2b-auc-ci'] = ci(d.auc_lo, d.auc_hi)
    out['S2b-bsi-dauc'] = num(Results.delta(s2b, pair_label('M2_bsi', 'M2')).d_auc)
    for lag in (1, 7, 30):
        out[f'S8-{lag}-persons'] = integer(r.info(f'S8_{lag}')['persons'])
        out[f'S8-{lag}-dauc'] = num(Results.delta(r.deltas(f'S8_{lag}'), primary).d_auc)
    sub = r.csv('S9_mri_age_subgroups.csv', cv=True)
    sub = sub[(sub.h == PRIMARY) & (sub.model == 'M3')].set_index('subgroup')
    out['S9-auc-recent'] = num(sub.loc['MRI <= 180 days'].auc)
    out['S9-auc-older'] = num(sub.loc['MRI 181–365 days'].auc)
    zero = r.sensitivity
    zero = zero[zero.analysis != 'Primary analysis']
    out['sens-count'] = integer(len(zero))
    out['sens-include-zero'] = integer(((zero.auc_lo <= 0) & (zero.auc_hi >= 0)).sum())


def _confusion_numbers(r, out):
    for label, table in (('cv', r.main), ('test', r.test)):
        for t in BAND_THRESHOLDS:
            for m in MODELS:
                rates = r.rates(table, m, PRIMARY, t)
                for name, value in rates.items():
                    out[f'cm-{label}-{m}-{round(t * 100)}-{name}'] = num(value, 2)


def _data_numbers(r, out):
    events_persons, events_landmarks = r.shared_events()
    info = r.info('main')
    out['main-persons'] = integer(info['persons'])
    out['main-landmarks'] = integer(info['landmarks'])
    out['cv-per-group'] = integer(round(info['persons'] / 10))
    out['events-persons'] = integer(events_persons)
    out['events-landmarks'] = integer(events_landmarks)
    for m in MODELS:
        out[f'available-persons-{m}'] = integer(r.info(f'SEC_{m}')['persons'])
    qc = r.audit['mri_qc']
    out['qc-records'] = integer(qc['records'])
    out['qc-missing'] = integer(qc['overall_qc_missing'])
    out['qc-missing-pct'] = pct(qc['overall_qc_missing'] / qc['records'], 2)
    out['cv-design'] = f'{CV_FOLDS} × {CV_REPEATS}'


def build(r: Results):
    out = {}
    _lambda_numbers(r, out)
    _cv_numbers(r, out)
    _test_numbers(r, out)
    _delta_numbers('cv', r.main_deltas, out)
    _delta_numbers('test', r.test_deltas, out)
    _sensitivity_numbers(r, out)
    _confusion_numbers(r, out)
    _data_numbers(r, out)
    return {k: v.replace('%', '\\%') for k, v in out.items()}    # values are LaTeX text


def to_latex(numbers):
    """Macro definitions: \\R{key} expands to the formatted number."""
    lines = ['% generated by build_report.py from runs/ — do not edit',
             '\\newcommand{\\R}[1]{\\ifcsname R@#1\\endcsname\\csname R@#1\\endcsname'
             '\\else\\errmessage{No number for key #1}\\fi}']
    lines += [f'\\expandafter\\def\\csname R@{k}\\endcsname{{{v}}}' for k, v in sorted(numbers.items())]
    return '\n'.join(lines) + '\n'
