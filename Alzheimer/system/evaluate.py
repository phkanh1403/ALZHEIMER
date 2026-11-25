"""Chấm tập test 20% bằng mô hình và tham số tiền xử lý đã khóa từ train. Không học lại trên test.

Cách dùng:  python evaluate.py <thư_mục_test_đã_giải_nén>
Chỉ lưu kết quả tổng hợp vào runs/ (không lưu dữ liệu cá nhân của test).
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from pipeline import ROOT, MODELS, HORIZONS, Sources, labelled_raw, apply_params, read, has_table, save_json
from model import predict
from metrics import status, turnbull_calibration, categories
from validation import summary_table, paired_deltas, PAIRS

TABLES = ['DXSUM', 'MMSE', 'FAQ', 'PTDEMOG', 'UCSFFSX7', 'ADAS', 'NEUROBAT']
MIN_SUBGROUP = 60


def roster(root):
    ids = set()
    for t in TABLES:
        if has_table(root, t):
            ids.update(read(root, t).RID.dropna().astype(int))
    return ids


def score(frame, models, replicates, rng, modes=('ref',)):
    rows = []
    for r in range(replicates):
        ev = frame.sample(frac=1, random_state=int(rng.integers(1e9))).drop_duplicates('RID')
        for h in HORIZONS:
            for mode in modes:
                refm = models['M3'] if mode == 'M3' else models['M0']
                y, w, cat = status(ev, h, refm, np.random.default_rng(rng.integers(1e9)), models['censoring'],
                                   mode='ref' if mode in ('ref', 'M3') else mode)
                pn = predict(models['null'], ev, h)
                for n in MODELS:
                    if n not in models:
                        continue
                    rows.append(pd.DataFrame(dict(repeat=r, scope='random' if mode == 'ref' else f'imp_{mode}',
                                                  RID=ev.RID.to_numpy(), h=h, model=n, y=y, w=w,
                                                  p=predict(models[n], ev, h), p_null=pn, L=ev.L.to_numpy(),
                                                  R=ev.R.to_numpy(), cat=cat, mri_age=ev.mri_age_days.to_numpy())))
    return pd.concat(rows, ignore_index=True)


def evaluate(testdir, replicates=20, seed=2026, save=True):
    testdir = Path(testdir)
    overlap = roster(ROOT / 'data' / 'train') & roster(testdir)
    if overlap:
        raise ValueError(f'{len(overlap)} RID trùng giữa train và test — dừng đánh giá')
    prep = json.loads((ROOT / 'runs' / 'preprocessing.json').read_text())
    models = json.loads((ROOT / 'runs' / 'models.json').read_text())
    secondary = json.loads((ROOT / 'runs' / 'secondary_models.json').read_text())
    src = Sources(testdir)
    df = apply_params(labelled_raw(src), prep)
    shared = df.dropna(subset=MODELS['M3']).reset_index(drop=True)
    rng = np.random.default_rng(seed)
    long = score(shared, models, replicates, rng, modes=('ref', 'L', 'mid', 'R', 'M3'))
    _, metrics = summary_table(long)
    deltas = [paired_deltas(long, PAIRS)]
    for sc in ['imp_L', 'imp_mid', 'imp_R', 'imp_M3']:
        x = paired_deltas(long, [('M3', 'M2')], scope=sc, B=500)
        x['comparison'] = x.comparison + f' [{sc}]'
        deltas.append(x)
    second = shared[shared.second_mci == 1]
    note = ''
    if second.RID.nunique() >= MIN_SUBGROUP:
        ls = score(second, models, 1, rng)
        x = paired_deltas(ls, [('M3', 'M2')], B=1000)
        x['comparison'] = x.comparison + ' [second_mci]'
        deltas.append(x)
    else:
        note = f'Không báo cáo phân nhóm mốc MCI lần 2 vì chỉ có {second.RID.nunique()} người (< {MIN_SUBGROUP})'
    deltas = pd.concat(deltas, ignore_index=True)

    cal = []
    ev0 = long[(long.scope == 'random') & (long.repeat == 0)]
    for (h, n), g in ev0.groupby(['h', 'model']):
        c = turnbull_calibration(g.L.to_numpy(), g.R.to_numpy(), g.p.to_numpy(), h)
        for q, (a, b) in enumerate(zip(c['pred'], c['obs'])):
            cal.append(dict(h=h, model=n, group=q + 1, predicted=a, observed_turnbull=b))

    cover = []
    for n, cols in MODELS.items():
        own = df.dropna(subset=cols)
        sec = dict(secondary)
        lo = score(own, {**{k: sec[k] for k in ['M0', 'null', 'censoring']}, n: sec[n]}, replicates, np.random.default_rng(seed))
        _, a = summary_table(lo)
        a = a[a.model == n].assign(persons=own.RID.nunique(), set='theo dữ liệu hiện có')
        cover.append(a)
    cover = pd.concat(cover, ignore_index=True)

    one = shared.drop_duplicates('RID')
    info = dict(
        test_rids_with_diagnosis=int(src.vis.RID.nunique()), test_landmarks=len(df), test_persons=int(df.RID.nunique()),
        shared_landmarks=len(shared), shared_persons=int(shared.RID.nunique()),
        status_categories_one_per_person={h: pd.Series(categories(one.L.to_numpy(), one.R.to_numpy(), h)).value_counts().to_dict()
                                          for h in HORIZONS},
        conflicts=len(src.conflicts), mri_measurement_error=src.meas_error, note_second_mci=note,
        overlap_rids_with_train=0)
    if save:
        out = ROOT / 'runs'
        metrics.to_csv(out / 'test_metrics.csv', index=False)
        deltas.to_csv(out / 'test_deltas.csv', index=False)
        pd.DataFrame(cal).to_csv(out / 'test_calibration.csv', index=False)
        cover.to_csv(out / 'test_coverage.csv', index=False)
        save_json(info, out / 'test_info.json')
    return metrics, deltas, info


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('test_directory', type=Path)
    a = p.parse_args()
    m, d, info = evaluate(a.test_directory)
    print(json.dumps(info, ensure_ascii=False, indent=1, default=str))
    print(m[m.scope == 'random'][['h', 'model', 'n', 'case_certain', 'control_certain', 'ambiguous', 'censored_early',
                                  'auc', 'ipa', 'oe_turnbull', 'nb_core']].round(4).to_string(index=False))
    print(d.round(4).to_string(index=False))
