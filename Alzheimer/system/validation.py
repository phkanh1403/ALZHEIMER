"""Kiểm định chéo lặp (5 fold x R lần, chia theo RID) trên train; KHÔNG đọc dữ liệu test.

- Mỗi fold: học M0–M3, mô hình null và mô hình kiểm duyệt trên phần train của fold.
- Đánh giá: một mốc ngẫu nhiên cho mỗi người ở fold giữ lại; ca bất định được rút T từ M0.
- Chênh lệch M3 − M2 với khoảng tin cậy bootstrap theo cụm RID; áp quy tắc kết luận định trước.
- Phân tích độ nhạy S1–S5 với số lần lặp ít hơn.
"""
import argparse
import numpy as np
import pandas as pd
from pipeline import ROOT, MODELS, HORIZONS, PRIMARY, save_json
from model import fit_interval, predict, fit_cox, cox_predict
from train import inner_lambda
from metrics import censoring_model, status, summarize, decide, NB_CORE

PAIRS = [('M1', 'M0'), ('M2', 'M1'), ('M3', 'M2')]


def folds_by_rid(frame, k, seed):
    ids = np.array(sorted(frame.RID.unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    part = {rid: i % k for i, rid in enumerate(ids)}
    return frame.RID.map(part).to_numpy()


def pick_one(frame, rng):
    return frame.sample(frac=1, random_state=int(rng.integers(1e9))).drop_duplicates('RID')


def run_cv(frame, model_cols, repeats, seed=42, lam=None, extra=None, ref='M0'):
    """Trả về bảng dài các dự báo ngoài fold: repeat, RID, h, model, y, w, p, p_null, scope."""
    rows = []
    for r in range(repeats):
        k = folds_by_rid(frame, 5, seed + r)
        rng = np.random.default_rng(seed + 1000 + r)
        for f in range(5):
            tr, va = frame[k != f], frame[k == f]
            fitted = {}
            for name, cols in model_cols.items():
                l = lam if lam is not None else inner_lambda(tr, cols, seed=r * 10 + f)   # CV lồng
                fitted[name] = fit_interval(tr, cols, lam=l)
            null = fit_interval(tr, [], lam=1.0)
            cm = censoring_model(tr)
            if extra:
                fitted.update({n: fn(tr) for n, fn in extra.items()})
            ref_model = fitted[ref] if ref in fitted else fit_interval(tr, MODELS['M0'], lam=1.0)
            scopes = {'random': pick_one(va, rng)}
            second = va[va.second_mci == 1]
            if len(second):
                scopes['second_mci'] = second
            for scope, ev in scopes.items():
                for h in HORIZONS:
                    y, w = status(ev, h, ref_model, np.random.default_rng(rng.integers(1e9)), cm)
                    pn = predict(null, ev, h)
                    for name, m in fitted.items():
                        p = cox_predict(m, ev, h) if m.get('kind') == 'cox' else predict(m, ev, h)
                        rows.append(pd.DataFrame(dict(repeat=r, fold=f, scope=scope, RID=ev.RID.to_numpy(),
                                                      h=h, model=name, y=y, w=w, p=p, p_null=pn,
                                                      L=ev.L.to_numpy(), R=ev.R.to_numpy())))
    return pd.concat(rows, ignore_index=True)


def summary_table(long):
    out = []
    for (scope, rep, h, name), g in long.groupby(['scope', 'repeat', 'h', 'model']):
        s = summarize(g.y.to_numpy(), g.w.to_numpy(), g.p.to_numpy(), g.p_null.to_numpy(),
                      g.L.to_numpy(), g.R.to_numpy(), h)
        out.append(dict(scope=scope, repeat=rep, h=h, model=name, **s))
    per = pd.DataFrame(out)
    num = per.select_dtypes('number').columns.drop(['repeat', 'h'])
    agg = per.groupby(['scope', 'h', 'model'])[list(num)].mean().reset_index()
    return per, agg


def paired_deltas(long, pairs, scope='random', B=1000, seed=7):
    """Δ = trung bình qua các lần lặp; KTC 95% bằng bootstrap cụm RID (mỗi vòng rút một lần lặp)."""
    rng = np.random.default_rng(seed)
    d = long[long.scope == scope]
    rows = []
    for h in sorted(d.h.unique()):
        dh = d[d.h == h]
        reps = sorted(dh.repeat.unique())
        tables = {r: dh[dh.repeat == r].pivot_table(index='RID', columns='model', values=['p', 'y', 'w', 'p_null'],
                                                      aggfunc='first') for r in reps}
        for a, b in pairs:
            def delta(t, idx):
                y, w = t['y'][a].to_numpy()[idx], t['w'][a].to_numpy()[idx]
                pn = t['p_null'][a].to_numpy()[idx]
                sa = summarize(y, w, t['p'][a].to_numpy()[idx], pn)
                sb = summarize(y, w, t['p'][b].to_numpy()[idx], pn)
                return sa['auc'] - sb['auc'], sa['nb_core'] - sb['nb_core'], sa['ipa'] - sb['ipa']
            point = np.nanmean([delta(tables[r], np.arange(len(tables[r]))) for r in reps], axis=0)
            draws = []
            for _ in range(B):
                t = tables[reps[rng.integers(len(reps))]]
                draws.append(delta(t, rng.integers(len(t), size=len(t))))
            draws = np.array(draws)
            lo, hi = np.nanquantile(draws, 0.025, axis=0), np.nanquantile(draws, 0.975, axis=0)
            rows.append(dict(h=h, comparison=f'{a} − {b}', d_auc=point[0], auc_lo=lo[0], auc_hi=hi[0],
                             d_nb_core=point[1], nb_lo=lo[1], nb_hi=hi[1], d_ipa=point[2], ipa_lo=lo[2], ipa_hi=hi[2]))
    out = pd.DataFrame(rows)
    out['conclusion'] = [decide(r.d_auc, r.auc_lo, r.auc_hi, r.d_nb_core, r.nb_lo, r.nb_hi)
                         if r.comparison == 'M3 − M2' else '' for r in out.itertuples()]
    return out


def sensitivity(shared, confirmed, repeats):
    res = []
    base = {k: MODELS[k] for k in ['M2', 'M3']}

    def add(tag, desc, long, pairs):
        _, agg = summary_table(long)
        dl = paired_deltas(long, pairs, B=300)
        for r in dl.itertuples():
            res.append(dict(analysis=tag, description=desc, h=r.h, comparison=r.comparison,
                            d_auc=r.d_auc, auc_lo=r.auc_lo, auc_hi=r.auc_hi,
                            d_nb_core=r.d_nb_core, nb_lo=r.nb_lo, nb_hi=r.nb_hi, d_ipa=r.d_ipa))
        return agg

    # S1: Cox kiểm duyệt phải (như v1) so với mô hình khoảng, cùng biến M3
    def cox_m3(tr):
        ev = np.isfinite(tr.R.to_numpy())
        return fit_cox(tr, MODELS['M3'], np.where(ev, tr.R, tr.L), ev)
    long = run_cv(shared, {'M2': MODELS['M2'], 'M3': MODELS['M3']}, repeats, lam=1.0, extra={'M3_cox': cox_m3})
    agg = add('S1', 'Cox kiểm duyệt phải (v1) so với mô hình kiểm duyệt khoảng', long, [('M3_cox', 'M3')])
    s1_cal = agg[agg.model.isin(['M3', 'M3_cox']) & (agg.scope == 'random')][
        ['h', 'model', 'auc', 'ipa', 'oe_km', 'cal_slope_km', 'mean_pred']]
    # S2a: tốc độ hai điểm (v1) thay cho BLUP
    two = shared.dropna(subset=['hippo_rate2', 'entor_rate2'])
    cols2 = MODELS['M2'] + ['hippo_rate2', 'entor_rate2']
    long = run_cv(two, {'M2': MODELS['M2'], 'M3': MODELS['M3'], 'M3_twopoint': cols2}, repeats, lam=1.0)
    add('S2a', 'Tốc độ hai điểm (v1) so với slope BLUP', long, [('M3', 'M2'), ('M3_twopoint', 'M2')])
    # S2b: BSI hồi hải mã (ADNIGO/2)
    bsi = shared.dropna(subset=['hbsi_rate'])
    if bsi.RID.nunique() >= 100:
        long = run_cv(bsi, {'M2': MODELS['M2'], 'M3': MODELS['M3'], 'M2_bsi': MODELS['M2'] + ['hbsi_rate']},
                      repeats, lam=1.0)
        add('S2b', f'BSI hồi hải mã, {bsi.RID.nunique()} người ADNIGO/2', long, [('M3', 'M2'), ('M2_bsi', 'M2')])
    # S4: Dementia được xác nhận ở các lượt sau
    long = run_cv(confirmed.dropna(subset=MODELS['M3']), base, repeats, lam=1.0)
    add('S4', 'Dementia được xác nhận (không quay lại MCI/CN)', long, [('M3', 'M2')])
    # S5: bỏ FAQ khỏi mọi mô hình
    nofaq = {k: [c for c in MODELS[k] if not c.startswith('faq')] for k in ['M2', 'M3']}
    long = run_cv(shared, nofaq, repeats, lam=1.0)
    add('S5', 'Bỏ FAQ (thiên lệch hoà nhập)', long, [('M3', 'M2')])
    return pd.DataFrame(res), s1_cal


def validate(repeats=20, sens_repeats=5):
    out = ROOT / 'runs'
    df = pd.read_csv(out / 'train_landmarks.csv', parse_dates=['t0'])
    shared = df.dropna(subset=MODELS['M3']).reset_index(drop=True)
    long = run_cv(shared, MODELS, repeats)
    long.to_csv(out / 'validation_oof.csv.gz', index=False)
    per, agg = summary_table(long)
    agg.to_csv(out / 'validation.csv', index=False)
    deltas = paired_deltas(long, PAIRS)
    second = paired_deltas(long, [('M3', 'M2')], scope='second_mci', B=500)
    second['comparison'] = second.comparison + ' (mốc MCI lần 2, S3)'
    pd.concat([deltas, second]).to_csv(out / 'validation_deltas.csv', index=False)
    print(agg[(agg.scope == 'random')][['h', 'model', 'auc', 'ipa', 'oe_km', 'cal_slope_km', 'nb_core']].round(4).to_string(index=False))
    print(pd.concat([deltas, second]).round(4).to_string(index=False))
    conf = pd.read_csv(out / 'train_landmarks_confirmed.csv', parse_dates=['t0'])
    sens, s1_cal = sensitivity(shared, conf, sens_repeats)
    sens.to_csv(out / 'sensitivity.csv', index=False)
    s1_cal.to_csv(out / 'sensitivity_s1_calibration.csv', index=False)
    print(sens.round(4).to_string(index=False))
    print(s1_cal.round(4).to_string(index=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--repeats', type=int, default=20)
    p.add_argument('--sens-repeats', type=int, default=5)
    a = p.parse_args()
    validate(a.repeats, a.sens_repeats)
