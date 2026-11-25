"""Huấn luyện v2 trên data/train: tiền xử lý, bảng mốc, mô hình cuối M0–M3, mô hình kiểm duyệt.

Phân tích chính: cả bốn mô hình học trên CÙNG tập mốc đủ biến M3 (so sánh ghép cặp).
Mô hình phụ: mỗi mô hình học trên tập mốc đủ biến của riêng nó (độ bao phủ, dùng trong ứng dụng).
"""
import json
import numpy as np
import pandas as pd
from pipeline import ROOT, MODELS, build, fit_preprocessing, save_json
from model import fit_interval
from metrics import censoring_model

LAMBDA_GRID = [0.1, 1.0, 10.0]


def inner_lambda(frame, cols, seed=0, folds=3):
    """Chọn lambda theo log-likelihood kiểm duyệt khoảng ngoài fold (CV lồng, chia theo RID)."""
    from model import _exposure, linear_predictor
    ids = frame.RID.unique()
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    part = {rid: i % folds for i, rid in enumerate(ids)}
    k = frame.RID.map(part).to_numpy()
    score = {}
    for lam in LAMBDA_GRID:
        total = 0.0
        for f in range(folds):
            m = fit_interval(frame[k != f], cols, lam=lam)
            te = frame[k == f]
            u = np.exp(linear_predictor(m, te))
            hz = np.exp(m['log_hazard'])
            A = _exposure(te.L.to_numpy()) @ hz * u
            ev = np.isfinite(te.R.to_numpy())
            B = _exposure(np.where(ev, te.R, te.L)) @ hz * u
            total += float((-A + np.where(ev, np.log(-np.expm1(-np.maximum(B - A, 1e-300))), 0)).sum())
        score[lam] = total
    return max(score, key=score.get)


def analysis_set(df):
    return df.dropna(subset=MODELS['M3']).reset_index(drop=True)


def train():
    out = ROOT / 'runs'
    out.mkdir(exist_ok=True)
    data = ROOT / 'data' / 'train'
    prep = fit_preprocessing(data)
    save_json(prep, out / 'preprocessing.json')
    df, conflicts = build(data, prep)
    df.to_csv(out / 'train_landmarks.csv', index=False)
    conflicts.to_csv(out / 'conflicting_dates.csv', index=False)
    conf, _ = build(data, prep, confirmed=True)       # S4: Dementia được xác nhận
    conf.to_csv(out / 'train_landmarks_confirmed.csv', index=False)

    shared = analysis_set(df)
    coverage = []
    models, secondary = {}, {}
    for name, cols in MODELS.items():
        lam = inner_lambda(shared, cols)
        models[name] = fit_interval(shared, cols, lam=lam)
        own = df.dropna(subset=cols)
        secondary[name] = fit_interval(own, cols, lam=lam)
        coverage.append(dict(model=name, landmarks=len(own), persons=own.RID.nunique(),
                             events_in_window=int(own.event.sum()), lam=lam))
        print(name, 'lambda', lam, 'shared', len(shared), 'own', len(own))
    models['null'] = fit_interval(shared, [], lam=1.0)
    secondary['null'] = models['null']
    models['censoring'] = censoring_model(shared)
    save_json(models, out / 'models.json')
    save_json(secondary, out / 'secondary_models.json')
    cov = pd.DataFrame(coverage)
    cov.loc[len(cov)] = dict(model='Phân tích chính (đủ M3)', landmarks=len(shared),
                             persons=shared.RID.nunique(), events_in_window=int(shared.event.sum()), lam=np.nan)
    cov.to_csv(out / 'coverage.csv', index=False)
    print(cov.to_string(index=False))


if __name__ == '__main__':
    train()
