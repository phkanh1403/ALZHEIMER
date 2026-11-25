"""Gộp kết quả CV, độ nhạy và test thành runs/summary.json và runs/sensitivity_summary.csv.

Kết luận được báo RIÊNG cho từng tầng (CV trên train 80%, test 20%) theo quy tắc đã khóa trong protocol.json.
"""
import json
import pandas as pd
from pipeline import ROOT, PRIMARY, save_json
from metrics import PROTOCOL

R = ROOT / 'runs'
CV = R / 'cv'
LABELS = {
    'main': 'Phân tích chính (tham số trong fold)', 'L1': 'L1 · tham số tiền xử lý trên toàn train (như ver4)',
    'S1': 'S1 · Cox kiểm duyệt phải', 'S2a': 'S2a · tốc độ hai điểm (v1)', 'S2b': 'S2b · BSI hồi hải mã (ADNIGO/2)',
    'S4': 'S4 · Dementia được xác nhận', 'S5': 'S5 · bỏ FAQ', 'S7': 'S7 · loại RID nhân khẩu không nhất quán',
    'S8_1': 'S8 · MRI có trước mốc ≥ 1 ngày', 'S8_7': 'S8 · MRI có trước mốc ≥ 7 ngày', 'S8_30': 'S8 · MRI có trước mốc ≥ 30 ngày',
    'S9': 'S9 · tương tác HV × tuổi MRI (M1_age − M1)', 'S10': 'S10 · chuỗi MRI cùng hãng/model máy',
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
    both_value = all(tiers[str(PRIMARY)][k]['conclusion'] == 'Có giá trị gia tăng' for k in ('cv', 'test'))
    out['overall'] = ('Có bằng chứng về giá trị gia tăng của tốc độ MRI ở cả CV và test' if both_value else
                      'Chưa đủ bằng chứng về giá trị gia tăng ổn định của tốc độ MRI')
    out['test_note'] = ('Test là kiểm chứng nội bộ trên phân hoạch ADNI đã dùng ở các phiên bản trước; '
                        'kết luận chính dựa trên CV, test dùng để kiểm tra tính nhất quán.')

    rows = []
    for name, lab in LABELS.items():
        p = CV / f'{name}_deltas.csv'
        if not p.exists():
            continue
        d = pd.read_csv(p)
        info = json.loads((CV / f'{name}_info.json').read_text())
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
            rows.append(dict(analysis={'second_mci': 'S3 · mốc MCI lần 2', 'imp_L': 'S6 · gán T = L', 'imp_mid': 'S6 · gán T = điểm giữa',
                                       'imp_R': 'S6 · gán T = R', 'imp_M3': 'S6 · rút T từ M3'}[sc],
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
