"""Bootstrap theo cụm người (200 lần) cho khoảng dự báo cá nhân — cho CẢ HAI bộ mô hình.

bootstrap_paired.json     : bộ ghép cặp (models.json), cùng tập mốc đủ M3
bootstrap_secondary.json  : bộ theo dữ liệu hiện có (secondary_models.json), tập đủ biến của từng mô hình
"""
import json
import numpy as np
import pandas as pd
from pipeline import ROOT, MODELS, save_json
from model import fit_interval


def draws_for(frame, cols, lam, B, rng):
    ids = frame.RID.unique()
    groups = dict(tuple(frame.groupby('RID')))
    out = []
    for _ in range(B):
        pick = rng.choice(ids, size=len(ids), replace=True)
        sample = pd.concat([groups[i].assign(RID=f'{i}_{k}') for k, i in enumerate(pick)], ignore_index=True)
        m = fit_interval(sample, cols, lam=lam)
        out.append({k: m[k] for k in ['columns', 'mean', 'scale', 'log_hazard', 'beta']})
    return out


def build(B=200, seed=42):
    rng = np.random.default_rng(seed)
    df = pd.read_csv(ROOT / 'runs' / 'train_landmarks.csv', parse_dates=['t0'])
    paired = json.loads((ROOT / 'runs' / 'models.json').read_text())
    secondary = json.loads((ROOT / 'runs' / 'secondary_models.json').read_text())
    shared = df.dropna(subset=MODELS['M3'])
    save_json({n: draws_for(shared, c, paired[n]['lam'], B, rng) for n, c in MODELS.items()},
              ROOT / 'runs' / 'bootstrap_paired.json')
    save_json({n: draws_for(df.dropna(subset=c), c, secondary[n]['lam'], B, rng) for n, c in MODELS.items()},
              ROOT / 'runs' / 'bootstrap_secondary.json')


if __name__ == '__main__':
    build()
