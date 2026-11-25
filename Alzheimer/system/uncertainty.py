"""Bootstrap theo cụm người (200 lần) để có khoảng biến thiên cho dự báo cá nhân trong ứng dụng."""
import json
import numpy as np
import pandas as pd
from pipeline import ROOT, MODELS, save_json
from model import fit_interval


def build(draws=200, seed=42):
    rng = np.random.default_rng(seed)
    df = pd.read_csv(ROOT / 'runs' / 'train_landmarks.csv', parse_dates=['t0'])
    models = json.loads((ROOT / 'runs' / 'secondary_models.json').read_text())
    out = {}
    for name, cols in MODELS.items():
        own = df.dropna(subset=cols)
        ids = own.RID.unique()
        groups = dict(tuple(own.groupby('RID')))
        fits = []
        for b in range(draws):
            pick = rng.choice(ids, size=len(ids), replace=True)
            sample = pd.concat([groups[i].assign(RID=f'{i}_{k}') for k, i in enumerate(pick)], ignore_index=True)
            m = fit_interval(sample, cols, lam=models[name]['lam'])
            fits.append({k: m[k] for k in ['columns', 'mean', 'scale', 'log_hazard', 'beta']})
        out[name] = fits
        print(name, 'xong')
    save_json(out, ROOT / 'runs' / 'bootstrap.json')


if __name__ == '__main__':
    build()
