"""LaTeX bodies (tabular only) of the result tables; captions and placement live in the report source."""
import numpy as np

from mci_risk.config import BAND_THRESHOLDS, HORIZONS, MODEL_PAIRS, MODELS, PRIMARY
from mci_risk.cv import pair_label

from .fmt import ci, conclusion, integer, num
from .results import Results

HR_LABELS = {
    'tau': 'τ (năm từ MCI đầu)', 'tau2': 'τ²', 'age': 'Tuổi', 'sex': 'Giới (1 nam, 2 nữ)', 'education': 'Học vấn',
    'mmse': 'MMSE', 'faq': 'FAQ', 'adas13': 'ADAS-Cog13', 'ravlt_delay': 'RAVLT trì hoãn',
    'hippo_adj': 'HV$_{\\text{adj}}$', 'entor': 'Độ dày nội khứu', 'field3t': 'MRI 3T',
    'adas13_slope': 'Slope ADAS-Cog13', 'faq_slope': 'Slope FAQ', 'hippo_slope': 'Slope HV$_{\\text{adj}}$',
    'entor_slope': 'Slope nội khứu',
}


def _sd(value):
    """Three significant digits: 2,17 · 20,7 · 0,425 · 1.135."""
    return num(value, max(0, 2 - int(np.floor(np.log10(abs(value))))))


def hazard_ratios(r: Results):
    m0, m3 = r.paired_models['M0'], r.paired_models['M3']
    hr0 = dict(zip(m0['columns'], np.exp(m0['beta']), strict=True))
    rows = []
    for c, sd, b in zip(m3['columns'], m3['scale'], m3['beta'], strict=True):
        rows.append(f'{HR_LABELS[c]} & {_sd(sd)} & {num(hr0[c], 2) if c in hr0 else "—"} & {num(np.exp(b), 2)}\\\\')
    return ('\\begin{tabular}{@{}lrrr@{}}\n\\toprule\nĐặc trưng & ĐLC & HR (M0) & HR (M3)\\\\\n\\midrule\n'
            + '\n'.join(rows) + '\n\\bottomrule\n\\end{tabular}\n')


def cv_performance(r: Results):
    rows = []
    for h in HORIZONS:
        for i, m in enumerate(MODELS):
            x = r.main.loc[(h, m)]
            first = f'\\multirow{{4}}{{*}}{{{integer(h)}}}' if i == 0 else ''
            rows.append(f'{first} & {m} & {num(x.auc)} & {num(x.brier)} & {num(x.ipa)} & {num(x.oe_turnbull, 2)} & '
                        f'{num(x.cal_slope_turnbull, 2)} & {num(x.nb_core)}\\\\')
        if h != HORIZONS[-1]:
            rows.append('\\midrule')
    return ('\\begin{tabular}{@{}llrrrrrr@{}}\n\\toprule\n$h$ & Mô hình & AUC & Brier & IPA & O/E & Slope & '
            'NB$_{\\text{core}}$\\\\\n\\midrule\n' + '\n'.join(rows) + '\n\\bottomrule\n\\end{tabular}\n')


def paired_deltas(r: Results):
    rows = []
    for tier, table in (('CV', r.main_deltas), ('Test', r.test_deltas)):
        for a, b in MODEL_PAIRS:
            for h in HORIZONS:
                d = Results.delta(table, pair_label(a, b), h)
                rows.append(f'{tier} & {a} − {b} & {integer(h)} & {num(d.d_auc)} {ci(d.auc_lo, d.auc_hi)} & '
                            f'{num(d.d_nb_core)} {ci(d.nb_lo, d.nb_hi)} & {conclusion(d.conclusion)}\\\\')
        if tier == 'CV':
            rows.append('\\midrule')
    return ('\\begin{tabular}{@{}llllll@{}}\n\\toprule\nTầng & So sánh & $h$ & ΔAUC & ΔNB$_{\\text{core}}$ & '
            'Kết luận\\\\\n\\midrule\n' + '\n'.join(rows) + '\n\\bottomrule\n\\end{tabular}\n')


def test_and_available(r: Results):
    rows = []
    for m in MODELS:
        t = r.test.loc[(PRIMARY, m)]
        sec = r.secondary(m).loc[(PRIMARY, m)]
        ta = r.test_available.loc[(PRIMARY, m)]
        rows.append(f'{m} & {num(t.auc)} & {num(t.oe_turnbull, 2)} & {num(t.cal_slope_turnbull, 2)} & '
                    f'{integer(r.info(f"SEC_{m}")["persons"])} & {num(sec.auc)} & {integer(ta.persons)} & '
                    f'{num(ta.auc)}\\\\')
    return ('\\begin{tabular}{@{}lrrrrrrr@{}}\n\\toprule\n& \\multicolumn{3}{c}{Test, ghép cặp} & '
            '\\multicolumn{2}{c}{CV, hiện có} & \\multicolumn{2}{c}{Test, hiện có}\\\\\n'
            '\\cmidrule(lr){2-4}\\cmidrule(lr){5-6}\\cmidrule(l){7-8}\n'
            'Mô hình & AUC & O/E & Slope & Người & AUC & Người & AUC\\\\\n\\midrule\n'
            + '\n'.join(rows) + '\n\\bottomrule\n\\end{tabular}\n')


def confusion(r: Results):
    rows = []
    for t in BAND_THRESHOLDS:
        for i, m in enumerate(MODELS):
            x = r.rates(r.main, m, PRIMARY, t)
            first = f'\\multirow{{4}}{{*}}{{$\\geq {round(t * 100)}\\%$}}' if i == 0 else ''
            cells = ' & '.join(num(x[k], 2) for k in ('sensitivity', 'specificity', 'ppv', 'npv', 'f1'))
            rows.append(f'{first} & {m} & {cells} & {num(100 * x["flagged"], 0)}\\%\\\\')
        if t != BAND_THRESHOLDS[-1]:
            rows.append('\\midrule')
    return ('\\begin{tabular}{@{}llrrrrrr@{}}\n\\toprule\nNgưỡng & Mô hình & Sens & Spec & PPV & NPV & F1 & '
            'Gắn cờ\\\\\n\\midrule\n' + '\n'.join(rows) + '\n\\bottomrule\n\\end{tabular}\n')


TABLES = {'hazard_ratios': hazard_ratios, 'cv_performance': cv_performance, 'paired_deltas': paired_deltas,
          'test_and_available': test_and_available, 'confusion': confusion}
