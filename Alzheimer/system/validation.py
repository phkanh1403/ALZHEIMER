"""Repeated cross-validation on training data (5 folds x 20 repeats, split by participant). Never reads test data.

Within-fold preprocessing: head-size adjustment and BLUP parameters are estimated on participants outside
the evaluated part, in both the outer loop (scoring) and the inner loop (choosing lambda).

Usage:
    python validation.py                 # run every analysis (2 parallel processes)
    python validation.py --only main S1  # run selected analyses
    python validation.py --repeats 5     # quick trial run
"""
import argparse
import json
import time
from multiprocessing import Pool
import numpy as np
import pandas as pd
from pipeline import (ROOT, MODELS, HORIZONS, Sources, apply_params, labelled_raw, save_json)
from model import fit_interval, predict, fit_cox, cox_predict, _exposure, linear_predictor
from metrics import (censoring_model, status, summarize, decide, turnbull_calibration, PROTOCOL)

OUT = ROOT / 'runs' / 'cv'
PAIRS = [('M1', 'M0'), ('M2', 'M1'), ('M3', 'M2')]
LAMBDAS = PROTOCOL['cv']['lambda_grid']


# --------------------------------------------------------------------------- utilities
def folds_by_rid(rids, k, seed):
    ids = np.array(sorted(set(rids)))
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    return {rid: i % k for i, rid in enumerate(ids)}


def interval_loglik(m, te):
    u = np.exp(linear_predictor(m, te))
    hz = np.exp(m['log_hazard'])
    A = _exposure(te.L.to_numpy()) @ hz * u
    ev = np.isfinite(te.R.to_numpy())
    B = _exposure(np.where(ev, te.R, te.L)) @ hz * u
    return float((-A + np.where(ev, np.log(-np.expm1(-np.maximum(B - A, 1e-300))), 0)).sum())


class Fold:
    """Features for a set of fitting participants, with preprocessing parameters from that set only."""

    def __init__(self, src, raw, fit_rids, params_mode, global_params, transform):
        self.params = global_params if params_mode == 'global' else src.fit_params(fit_rids)
        self.frame = transform(apply_params(raw, self.params))


def choose_lambda(src, raw, rids_train, all_rids, models, cfg, seed):
    """Nested 3-fold CV on training participants; preprocessing parameters re-estimated in the inner loop."""
    part = folds_by_rid(rids_train, PROTOCOL['cv']['inner_folds'], seed)
    score = {n: {l: 0.0 for l in LAMBDAS} for n in models}
    for f in range(PROTOCOL['cv']['inner_folds']):
        ev_ids = {r for r, k in part.items() if k == f}
        fold = Fold(src, raw, all_rids - ev_ids - cfg['outer_eval'], cfg['params'], cfg['global_params'], cfg['transform'])
        d = fold.frame[fold.frame.RID.isin(rids_train)]
        tr, te = d[~d.RID.isin(ev_ids)], d[d.RID.isin(ev_ids)]
        for n, cols in models.items():
            for l in LAMBDAS:
                score[n][l] += interval_loglik(fit_interval(tr, cols, lam=l), te)
    return {n: max(s, key=s.get) for n, s in score.items()}


def run_cv(src, raw, models, repeats, name, params='fold', transform=None, extra=None,
           impute_modes=('L', 'mid', 'R', 'M3'), second=True, seed=42):
    transform = transform or (lambda f: f.dropna(subset=sorted(set(sum(models.values(), [])))))
    global_params = src.fit_params(set(src.vis.RID))
    base = transform(apply_params(raw, global_params))            # defines the analysis set (does not depend on parameters)
    persons = sorted(base.RID.unique())
    raw = raw.loc[base.index]                                      # compute features only for analysis visits
    all_rids = set(src.vis.RID)
    rows, lambdas = [], []
    for r in range(repeats):
        part = folds_by_rid(persons, 5, seed + r)
        rng = np.random.default_rng(seed + 1000 + r)
        for f in range(5):
            ev_ids = {p for p, k in part.items() if k == f}
            tr_ids = set(persons) - ev_ids
            cfg = dict(outer_eval=ev_ids, params=params, global_params=global_params, transform=transform)
            lam = choose_lambda(src, raw, tr_ids, all_rids, models, cfg, seed=r * 10 + f)
            lambdas.append(dict(repeat=r, fold=f, **lam))
            fold = Fold(src, raw, all_rids - ev_ids, params, global_params, transform)
            d = fold.frame[fold.frame.RID.isin(persons)]
            tr, va = d[d.RID.isin(tr_ids)], d[d.RID.isin(ev_ids)]
            fitted = {n: fit_interval(tr, c, lam=lam[n]) for n, c in models.items()}
            if extra:
                fitted.update({n: fn(tr) for n, fn in extra.items()})
            ref = fitted['M0'] if 'M0' in fitted else fit_interval(tr, MODELS['M0'], lam=1.0)
            null = fit_interval(tr, [], lam=1.0)
            cm = censoring_model(tr)
            scopes = {'random': va.sample(frac=1, random_state=int(rng.integers(1e9))).drop_duplicates('RID')}
            if second and (va.second_mci == 1).any():
                scopes['second_mci'] = va[va.second_mci == 1]
            for scope, ev in scopes.items():
                for h in HORIZONS:
                    modes = [('ref', scope)] + ([(m, f'imp_{m}') for m in impute_modes] if scope == 'random' else [])
                    for mode, tag in modes:
                        refm = fitted.get('M3', ref) if mode == 'M3' else ref
                        y, w, cat = status(ev, h, refm, np.random.default_rng(rng.integers(1e9)), cm,
                                           mode='ref' if mode == 'M3' else mode)
                        pn = predict(null, ev, h)
                        for n, m in fitted.items():
                            p = cox_predict(m, ev, h) if m.get('kind') == 'cox' else predict(m, ev, h)
                            rows.append(pd.DataFrame(dict(repeat=r, fold=f, scope=tag, RID=ev.RID.to_numpy(), h=h, model=n,
                                                          y=y, w=w, p=p, p_null=pn, L=ev.L.to_numpy(), R=ev.R.to_numpy(),
                                                          cat=cat, mri_age=ev.mri_age_days.to_numpy())))
    long = pd.concat(rows, ignore_index=True)
    info = dict(name=name, persons=len(persons), landmarks=int(len(base)), events_window=int(base.drop_duplicates('RID').event.sum()),
                lambdas=pd.DataFrame(lambdas).drop(columns=['repeat', 'fold']).apply(lambda s: s.value_counts().to_dict()).to_dict())
    return long, info


def summary_table(long, calib_scopes=('random',)):
    from metrics import turnbull_risk
    out, cache = [], {}
    for (scope, rep, h, name), g in long.groupby(['scope', 'repeat', 'h', 'model']):
        calib = scope in calib_scopes
        if calib and (scope, rep, h) not in cache:
            cache[(scope, rep, h)] = turnbull_risk(g.L.to_numpy(), g.R.to_numpy(), h)
        s = summarize(g.y.to_numpy(), g.w.to_numpy(), g.p.to_numpy(), g.p_null.to_numpy(),
                      g.L.to_numpy(), g.R.to_numpy(), h, g['cat'].to_numpy(), calib=calib,
                      overall=cache.get((scope, rep, h)))
        out.append(dict(scope=scope, repeat=rep, h=h, model=name, **s))
    per = pd.DataFrame(out)
    num = per.select_dtypes('number').columns.drop(['repeat', 'h'])
    return per, per.groupby(['scope', 'h', 'model'])[list(num)].mean().reset_index()


def paired_deltas(long, pairs, scope='random', B=1000, seed=7):
    from metrics import summarize as S
    rng = np.random.default_rng(seed)
    d = long[long.scope == scope]
    rows = []
    for h in sorted(d.h.unique()):
        dh = d[d.h == h]
        reps = sorted(dh.repeat.unique())
        tabs = {r: dh[dh.repeat == r].pivot_table(index='RID', columns='model', values=['p', 'y', 'w', 'p_null'], aggfunc='first')
                for r in reps}
        for a, b in pairs:
            def delta(t, idx):
                y, w, pn = t['y'][a].to_numpy()[idx], t['w'][a].to_numpy()[idx], t['p_null'][a].to_numpy()[idx]
                sa = S(y, w, t['p'][a].to_numpy()[idx], pn, calib=False)
                sb = S(y, w, t['p'][b].to_numpy()[idx], pn, calib=False)
                return sa['auc'] - sb['auc'], sa['nb_core'] - sb['nb_core'], sa['ipa'] - sb['ipa']
            point = np.nanmean([delta(tabs[r], np.arange(len(tabs[r]))) for r in reps], axis=0)
            draws = np.array([delta(t, rng.integers(len(t), size=len(t)))
                              for t in (tabs[reps[rng.integers(len(reps))]] for _ in range(B))])
            lo, hi = np.nanquantile(draws, 0.025, axis=0), np.nanquantile(draws, 0.975, axis=0)
            rows.append(dict(h=h, comparison=f'{a} − {b}', d_auc=point[0], auc_lo=lo[0], auc_hi=hi[0],
                             d_nb_core=point[1], nb_lo=lo[1], nb_hi=hi[1], d_ipa=point[2], ipa_lo=lo[2], ipa_hi=hi[2]))
    out = pd.DataFrame(rows)
    out['conclusion'] = [decide(r.d_auc, r.auc_lo, r.auc_hi, r.d_nb_core, r.nb_lo, r.nb_hi) for r in out.itertuples()]
    return out


# --------------------------------------------------------------------------- analyses
def analysis(name, repeats):
    t0 = time.time()
    src = Sources(ROOT / 'data' / 'train')
    raw = labelled_raw(src)
    m23 = {k: MODELS[k] for k in ['M2', 'M3']}
    B = PROTOCOL['cv']['bootstrap'] if name == 'main' else 500
    pairs = [('M3', 'M2')]
    kw = {}
    if name == 'main':
        models, pairs = MODELS, PAIRS
    elif name == 'L1':
        models, kw = m23, dict(params='global')
    elif name == 'S1':
        def cox_m3(tr):
            ev = np.isfinite(tr.R.to_numpy())
            return fit_cox(tr, MODELS['M3'], np.where(ev, tr.R, tr.L), ev)
        models, kw, pairs = m23, dict(extra={'M3_cox': cox_m3}), [('M3', 'M2'), ('M3_cox', 'M3')]
    elif name == 'S2a':
        models = {**m23, 'M3_twopoint': MODELS['M2'] + ['hippo_rate2', 'entor_rate2']}
        pairs = [('M3', 'M2'), ('M3_twopoint', 'M2')]
    elif name == 'S2b':
        models = {**m23, 'M2_bsi': MODELS['M2'] + ['hbsi_rate']}
        pairs = [('M3', 'M2'), ('M2_bsi', 'M2')]
    elif name == 'S4':
        raw, models = labelled_raw(src, confirmed=True), m23
    elif name == 'S5':   # same participants as the main analysis, FAQ removed from predictors
        models = {k: [c for c in MODELS[k] if not c.startswith('faq')] for k in ['M2', 'M3']}
        kw = dict(transform=lambda f: f.dropna(subset=MODELS['M3']))
    elif name == 'S7':
        models = m23
        kw = dict(transform=lambda f: f[~f.RID.isin(src.demo_conflict)].dropna(subset=MODELS['M3']))
    elif name.startswith('S8_'):
        raw, models = labelled_raw(src, lag=int(name.split('_')[1])), m23
    elif name == 'S9':
        def tf(f):
            f = f.dropna(subset=MODELS['M3']).copy()
            f['hv_x_mriage'] = (f.hippo_adj - 6800) / 1000 * f.mri_age_days / 365.25
            f['mri_age_y'] = f.mri_age_days / 365.25
            return f
        models = {'M1': MODELS['M1'], 'M1_age': MODELS['M1'] + ['mri_age_y', 'hv_x_mriage']}
        kw, pairs = dict(transform=tf), [('M1_age', 'M1')]
    elif name == 'S10':
        raw, models = labelled_raw(src, series_key='version_model'), m23
    elif name.startswith('SEC_'):
        k = name.split('_')[1]
        models, pairs = {k: MODELS[k]}, []
    else:
        raise ValueError(name)
    long, info = run_cv(src, raw, models, repeats, name, second=(name == 'main'),
                        impute_modes=('L', 'mid', 'R', 'M3') if name == 'main' else (), **kw)
    OUT.mkdir(parents=True, exist_ok=True)
    per, agg = summary_table(long)
    agg.to_csv(OUT / f'{name}_metrics.csv', index=False)
    if pairs:
        dl = paired_deltas(long, pairs, B=B)
        if name == 'main':
            extra_rows = [dl]
            for sc in ['second_mci', 'imp_L', 'imp_mid', 'imp_R', 'imp_M3']:
                if (long.scope == sc).any():
                    x = paired_deltas(long, [('M3', 'M2')], scope=sc, B=500)
                    x['comparison'] = x.comparison + f' [{sc}]'
                    extra_rows.append(x)
            dl = pd.concat(extra_rows, ignore_index=True)
            # S9a: performance by MRI age on the main out-of-fold predictions
            sub = []
            for lab, mask in [('MRI <= 180 days', long.mri_age <= 180), ('MRI 181–365 days', long.mri_age > 180)]:
                x = long[mask & (long.scope == 'random')]
                _, a = summary_table(x.assign(scope='random'), calib_scopes=())
                a['subgroup'] = lab
                sub.append(a)
            pd.concat(sub).to_csv(OUT / 'S9_mri_age_subgroups.csv', index=False)
            long[long.scope == 'random'].to_csv(OUT / 'main_oof.csv.gz', index=False)
        dl.to_csv(OUT / f'{name}_deltas.csv', index=False)
    info['seconds'] = round(time.time() - t0)
    save_json(info, OUT / f'{name}_info.json')
    return name, info['seconds']


ALL = ['main', 'L1', 'S1', 'S2a', 'S2b', 'S4', 'S5', 'S7', 'S8_1', 'S8_7', 'S8_30', 'S9', 'S10',
       'SEC_M0', 'SEC_M1', 'SEC_M2', 'SEC_M3']


def _job(args):
    return analysis(*args)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--only', nargs='*', default=ALL)
    p.add_argument('--repeats', type=int, default=PROTOCOL['cv']['repeats'])
    p.add_argument('--workers', type=int, default=2)
    a = p.parse_args()
    with Pool(a.workers) as pool:
        for name, sec in pool.imap_unordered(_job, [(n, a.repeats) for n in a.only]):
            print(f'{name}: finished in {sec} s', flush=True)
