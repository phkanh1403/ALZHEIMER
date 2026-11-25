"""Huấn luyện ver5 trên data/train.

Đầu ra (runs/):
  preprocessing.json      tham số ICV/BLUP ước lượng trên toàn train (dùng cho mô hình cuối và test)
  models.json             bộ GHÉP CẶP: M0–M3 cùng học trên tập mốc đủ biến M3 (so sánh giá trị gia tăng)
  secondary_models.json   bộ THEO DỮ LIỆU HIỆN CÓ: mỗi mô hình học trên tập đủ biến của riêng nó (dự báo)
  train_landmarks.csv     mốc có nhãn; train_inputs.csv: mọi lượt MCI (kể cả lượt mới nhất, chưa có nhãn)
  conflicts.csv, coverage.csv, data_audit.json
"""
import json
import numpy as np
import pandas as pd
from pipeline import (ROOT, MODELS, Sources, eligible_points, raw_points, apply_params, labelled_raw,
                      public, save_json, MIN_SPAN)
from model import fit_interval
from metrics import censoring_model, categories, PROTOCOL
from validation import choose_lambda


def train():
    out = ROOT / 'runs'
    src = Sources(ROOT / 'data' / 'train')
    all_rids = set(src.vis.RID)
    prep = src.fit_params(all_rids)
    save_json(prep, out / 'preprocessing.json')
    pd.DataFrame(src.conflicts).to_csv(out / 'conflicts.csv', index=False)

    raw = labelled_raw(src)
    df = apply_params(raw, prep)
    public(df).to_csv(out / 'train_landmarks.csv', index=False)
    inputs = apply_params(raw_points(src, eligible_points(src.vis)), prep)
    public(inputs).to_csv(out / 'train_inputs.csv', index=False)

    shared = df.dropna(subset=MODELS['M3'])
    cfg = dict(outer_eval=set(), params='fold', global_params=prep,
               transform=lambda f: f.dropna(subset=MODELS['M3']))
    lam = choose_lambda(src, raw.loc[shared.index], set(shared.RID), all_rids, MODELS, cfg, seed=2026)
    models = {n: fit_interval(shared, c, lam=lam[n]) for n, c in MODELS.items()}
    models['null'] = fit_interval(shared, [], lam=1.0)
    models['censoring'] = censoring_model(shared)
    for n in MODELS:
        models[n].update(set='ghép cặp', persons_train=int(shared.RID.nunique()))
    save_json(models, out / 'models.json')

    secondary, coverage = {}, []
    for n, cols in MODELS.items():
        own = df.dropna(subset=cols)
        cfg_own = dict(cfg, transform=lambda f, c=cols: f.dropna(subset=c))
        l = choose_lambda(src, raw.loc[own.index], set(own.RID), all_rids, {n: cols}, cfg_own, seed=2026)[n]
        secondary[n] = fit_interval(own, cols, lam=l)
        secondary[n].update(set='theo dữ liệu hiện có', persons_train=int(own.RID.nunique()))
        coverage.append(dict(model=n, landmarks=len(own), persons=own.RID.nunique(),
                             persons_with_event_in_window=int(own.groupby('RID').event.max().sum()), lam_secondary=l,
                             lam_paired=lam[n]))
    secondary['null'] = fit_interval(df.dropna(subset=MODELS['M0']), [], lam=1.0)
    secondary['censoring'] = censoring_model(df.dropna(subset=MODELS['M0']))
    save_json(secondary, out / 'secondary_models.json')

    # S9: bao phủ khi giới hạn tuổi MRI 180 ngày
    raw180 = apply_params(labelled_raw(src, mri_window=180), prep).dropna(subset=MODELS['M3'])
    cov = pd.DataFrame(coverage)
    cov.loc[len(cov)] = dict(model='M3, MRI <= 180 ngày (S9)', landmarks=len(raw180), persons=raw180.RID.nunique(),
                             persons_with_event_in_window=int(raw180.groupby('RID').event.max().sum()))
    cov.to_csv(out / 'coverage.csv', index=False)

    cats = {h: pd.Series(categories(shared.L.to_numpy(), shared.R.to_numpy(), h)).value_counts().to_dict()
            for h in PROTOCOL['horizons_days']}
    audit = dict(
        train_rids=len(all_rids), eligible_points=len(inputs), labelled_landmarks=len(df),
        latest_points_without_followup=int((inputs.has_followup == 0).sum()),
        shared_landmarks=len(shared), shared_persons=int(shared.RID.nunique()),
        mri_measurement_error=src.meas_error,
        conflicts={f'{k[0]} — {k[1]}': int(v) for k, v in pd.DataFrame(src.conflicts).groupby(['table', 'action']).size().items()},
        demographic_conflict_rids=len(src.demo_conflict),
        status_categories_shared=cats,
        straddle_1095_kept=int(((shared.L < 1095) & (shared.R > 1095) & np.isfinite(shared.R)).sum()),
        mri_lag_days_quantiles=shared.mri_age_days.quantile([.1, .25, .5, .75, .9]).to_dict(),
        mri_same_day_share=float((shared.mri_age_days == 0).mean()),
        freesurfer_stream='Cross-Sectional FreeSurfer (7.x) theo DATADIC — phân đoạn từng ảnh độc lập, không dùng ảnh sau mốc',
        rundate_after_landmark=int((pd.to_datetime(shared.mri_run) > shared.t0).sum()),
        lambda_paired=lam,
    )
    save_json(audit, out / 'data_audit.json')
    print(cov.to_string(index=False))
    print(json.dumps(audit, ensure_ascii=False, indent=1, default=str))


if __name__ == '__main__':
    train()
