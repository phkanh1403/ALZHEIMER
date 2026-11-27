"""Cross-validation building blocks shared by training, validation and the test evaluation.

Folds are split by participant. Preprocessing parameters (head-size adjustment, BLUP) are re-estimated on the
participants outside the evaluated part, in both the outer loop (scoring) and the inner loop (choosing lambda).
"""
import warnings
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import CV_FOLDS, HORIZONS, INNER_FOLDS, LAMBDAS, MODEL_PAIRS, MODELS
from .data import apply_params
from .metrics import censoring_model, decide, status, summarize, turnbull_risk
from .model import exposure, fit_interval, linear_predictor, person_weights, predict

PAIRS = MODEL_PAIRS
_SEED_STEP = 1000     # offsets keep the fold, scoring and lambda random streams apart
LAMBDA_EDGE_WARNING = 0.5   # warn when more than this share of choices sit at an end of the lambda grid


def pair_label(a, b):
    return f'{a} − {b}'


def folds_by_rid(rids, k, seed):
    """Map each participant to one of k folds (shuffled, balanced)."""
    ids = np.array(sorted(set(rids)))
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    return {rid: i % k for i, rid in enumerate(ids)}


def interval_loglik(model, frame):
    """Unpenalised log-likelihood of held-out data (inner-loop score).

    Visits carry the same per-participant weights (1/n_i) as in fitting, so the score targets the population the
    model is fitted for (people, not visits).
    """
    u = np.exp(linear_predictor(model, frame))
    hazard = np.exp(model['log_hazard'])
    A = exposure(frame.L.to_numpy()) @ hazard * u
    event = np.isfinite(frame.R.to_numpy())
    B = exposure(np.where(event, frame.R, frame.L)) @ hazard * u
    contribution = -A + np.where(event, np.log(-np.expm1(-np.maximum(B - A, 1e-300))), 0)
    return float((person_weights(frame) * contribution).sum())


def dropna_transform(models):
    """Default analysis set: visits with every input any of the models needs."""
    needed = sorted({c for cols in models.values() for c in cols})
    return lambda frame: frame.dropna(subset=needed)


@dataclass(frozen=True)
class FoldSpec:
    """How features are built inside a fold.

    outer_eval     participants evaluated in the outer loop; never used for parameters in the inner loop
    params_mode    'fold' re-estimates preprocessing on the fitting participants; 'global' uses global_params (L1)
    """
    outer_eval: frozenset
    params_mode: str
    global_params: dict
    transform: Callable[[pd.DataFrame], pd.DataFrame]


def fold_frame(src, raw, fit_rids, spec):
    """Features for every visit, with preprocessing parameters estimated on fit_rids only."""
    params = spec.global_params if spec.params_mode == 'global' else src.fit_params(fit_rids)
    return spec.transform(apply_params(raw, params))


def lambda_scores(src, raw, rids_train, all_rids, models, spec, seed, grid=LAMBDAS):
    """Inner-CV score of every lambda: {model: {lambda: summed held-out log-likelihood}}.

    Preprocessing parameters are re-estimated in each inner fold without the inner and outer evaluation parts.
    """
    part = folds_by_rid(rids_train, INNER_FOLDS, seed)
    score = {n: dict.fromkeys(grid, 0.0) for n in models}
    for f in range(INNER_FOLDS):
        eval_ids = {r for r, k in part.items() if k == f}
        frame = fold_frame(src, raw, all_rids - eval_ids - spec.outer_eval, spec)
        d = frame[frame.RID.isin(rids_train)]
        train, test = d[~d.RID.isin(eval_ids)], d[d.RID.isin(eval_ids)]
        for n, cols in models.items():
            for lam in grid:
                score[n][lam] += interval_loglik(fit_interval(train, cols, lam=lam), test)
    return score


def best_lambda(scores):
    """Highest score; ties go to the value listed first in the grid."""
    return max(scores, key=scores.get)


def choose_lambda(src, raw, rids_train, all_rids, models, spec, seed, grid=LAMBDAS):
    """Nested CV on training participants: the lambda with the best inner-CV score for each model."""
    scores = lambda_scores(src, raw, rids_train, all_rids, models, spec, seed, grid)
    return {n: best_lambda(s) for n, s in scores.items()}


def edge_share(lambda_counts, grid=LAMBDAS):
    """Share of choices at the smallest or largest grid value, per model ({model: {lambda: count}})."""
    edges = {float(min(grid)), float(max(grid))}
    return {n: sum(c for lam, c in counts.items() if float(lam) in edges) / max(sum(counts.values()), 1)
            for n, counts in lambda_counts.items()}


def _prediction_rows(fitted, ev, h, null, y, w, cat, **tags):
    p_null = predict(null, ev, h)
    return [pd.DataFrame(dict(**tags, RID=ev.RID.to_numpy(), h=h, model=n, y=y, w=w, p=predict(m, ev, h),
                              p_null=p_null, L=ev.L.to_numpy(), R=ev.R.to_numpy(), cat=cat,
                              mri_age=ev.mri_age_days.to_numpy()))
            for n, m in fitted.items()]


def _score_fold(fitted, ref, null, cm, scopes, impute_modes, rng, repeat, fold):
    """Out-of-fold predictions with outcome status for every scope, time point and imputation mode."""
    rows = []
    for scope, ev in scopes.items():
        for h in HORIZONS:
            modes = [('ref', scope)] + ([(m, f'imp_{m}') for m in impute_modes] if scope == 'random' else [])
            for mode, tag in modes:
                ref_model = fitted.get('M3', ref) if mode == 'M3' else ref
                y, w, cat = status(ev, h, ref_model, np.random.default_rng(rng.integers(1e9)), cm,
                                   mode='ref' if mode == 'M3' else mode)
                rows += _prediction_rows(fitted, ev, h, null, y, w, cat, repeat=repeat, fold=fold, scope=tag)
    return rows


def run_cv(src, raw, models, repeats, name, params='fold', transform=None, extra=None,
           impute_modes=('L', 'mid', 'R', 'M3'), second=True, seed=42):
    """Repeated participant-level CV. Returns long-format out-of-fold predictions and run information.

    extra: additional models as {name: fit_function(train_frame)} (e.g. the Cox model of S1).
    second: also score the subgroup of second MCI visits (S3).
    """
    transform = transform or dropna_transform(models)
    global_params = src.fit_params(set(src.vis.RID))
    base = transform(apply_params(raw, global_params))   # defines the analysis set (does not depend on parameters)
    persons = sorted(base.RID.unique())
    raw = raw.loc[base.index]                              # compute features only for analysis visits
    all_rids = set(src.vis.RID)
    rows, lambdas = [], []
    for r in range(repeats):
        part = folds_by_rid(persons, CV_FOLDS, seed + r)
        rng = np.random.default_rng(seed + _SEED_STEP + r)
        for f in range(CV_FOLDS):
            eval_ids = {p for p, k in part.items() if k == f}
            train_ids = set(persons) - eval_ids
            spec = FoldSpec(frozenset(eval_ids), params, global_params, transform)
            lam = choose_lambda(src, raw, train_ids, all_rids, models, spec, seed=r * 10 + f)
            lambdas.append(dict(repeat=r, fold=f, **lam))
            d = fold_frame(src, raw, all_rids - eval_ids, spec)
            d = d[d.RID.isin(persons)]
            train, held_out = d[d.RID.isin(train_ids)], d[d.RID.isin(eval_ids)]
            fitted = {n: fit_interval(train, c, lam=lam[n]) for n, c in models.items()}
            fitted.update({n: fit(train) for n, fit in (extra or {}).items()})
            ref = fitted['M0'] if 'M0' in fitted else fit_interval(train, MODELS['M0'], lam=1.0)
            null = fit_interval(train, [], lam=1.0)
            cm = censoring_model(train)
            one_visit = held_out.sample(frac=1, random_state=int(rng.integers(1e9))).drop_duplicates('RID')
            scopes = {'random': one_visit}
            if second and (held_out.second_mci == 1).any():
                scopes['second_mci'] = held_out[held_out.second_mci == 1]
            rows += _score_fold(fitted, ref, null, cm, scopes, impute_modes, rng, r, f)
    long = pd.concat(rows, ignore_index=True)
    lambda_counts = (pd.DataFrame(lambdas).drop(columns=['repeat', 'fold'])
                     .apply(lambda s: s.value_counts().to_dict()).to_dict())
    info = dict(name=name, persons=len(persons), landmarks=len(base),
                events_window=int(base.drop_duplicates('RID').event.sum()), lambdas=lambda_counts,
                lambda_at_grid_edge=edge_share(lambda_counts))
    for n, share in info['lambda_at_grid_edge'].items():
        if share > LAMBDA_EDGE_WARNING:
            warnings.warn(f'{name}/{n}: lambda chosen at the edge of the grid in {share:.0%} of folds; '
                          'the grid may not contain the optimum', stacklevel=2)
    return long, info


def summary_table(long, calib_scopes=('random',)):
    """Metrics per scope, repeat, time point and model; and their mean over repeats."""
    out, overall = [], {}
    for (scope, rep, h, name), g in long.groupby(['scope', 'repeat', 'h', 'model']):
        calib = scope in calib_scopes
        if calib and (scope, rep, h) not in overall:
            overall[(scope, rep, h)] = turnbull_risk(g.L.to_numpy(), g.R.to_numpy(), h)
        s = summarize(g.y.to_numpy(), g.w.to_numpy(), g.p.to_numpy(), g.p_null.to_numpy(),
                      g.L.to_numpy(), g.R.to_numpy(), h, g['cat'].to_numpy(), calib=calib,
                      overall=overall.get((scope, rep, h)))
        out.append(dict(scope=scope, repeat=rep, h=h, model=name, **s))
    per = pd.DataFrame(out)
    num = per.select_dtypes('number').columns.drop(['repeat', 'h'])
    return per, per.groupby(['scope', 'h', 'model'])[list(num)].mean().reset_index()


def _delta(table, idx, a, b):
    """Differences (a - b) in AUC, core net benefit and IPA on rows idx of a per-participant table."""
    y, w, p_null = (table[k][a].to_numpy()[idx] for k in ('y', 'w', 'p_null'))
    sa = summarize(y, w, table['p'][a].to_numpy()[idx], p_null, calib=False)
    sb = summarize(y, w, table['p'][b].to_numpy()[idx], p_null, calib=False)
    return sa['auc'] - sb['auc'], sa['nb_core'] - sb['nb_core'], sa['ipa'] - sb['ipa']


def paired_deltas(long, pairs, scope='random', B=1000, seed=7):
    """Paired differences between models with participant bootstrap intervals and the protocol decision."""
    rng = np.random.default_rng(seed)
    d = long[long.scope == scope]
    rows = []
    for h in sorted(d.h.unique()):
        dh = d[d.h == h]
        reps = sorted(dh.repeat.unique())
        tables = {r: dh[dh.repeat == r].pivot_table(index='RID', columns='model', values=['p', 'y', 'w', 'p_null'],
                                                    aggfunc='first')
                  for r in reps}
        for a, b in pairs:
            point = np.nanmean([_delta(tables[r], np.arange(len(tables[r])), a, b) for r in reps], axis=0)
            draws = []
            for _ in range(B):
                t = tables[reps[rng.integers(len(reps))]]
                draws.append(_delta(t, rng.integers(len(t), size=len(t)), a, b))
            lo, hi = np.nanquantile(draws, 0.025, axis=0), np.nanquantile(draws, 0.975, axis=0)
            rows.append(dict(h=h, comparison=pair_label(a, b), d_auc=point[0], auc_lo=lo[0], auc_hi=hi[0],
                             d_nb_core=point[1], nb_lo=lo[1], nb_hi=hi[1], d_ipa=point[2], ipa_lo=lo[2],
                             ipa_hi=hi[2]))
    out = pd.DataFrame(rows)
    out['conclusion'] = [decide(r.d_auc, r.auc_lo, r.auc_hi, r.d_nb_core, r.nb_lo, r.nb_hi) for r in out.itertuples()]
    return out
