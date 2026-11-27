"""Report figures, drawn from runs/ and saved as PDF (LaTeX) and PNG (DOCX)."""
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

from mci_risk.config import (
    BAND_THRESHOLDS,
    DAYS_PER_MONTH,
    DAYS_PER_YEAR,
    HORIZONS,
    MODELS,
    PRIMARY,
    THRESHOLDS,
    WINDOW,
)
from mci_risk.model import CUTS

from .fmt import num
from .results import Results

COLUMN = 3.45                      # inches, IEEE column width
BLUE, RED, PINK, LIGHT, SLATE, GREY = '#1f5fa8', '#c0392b', '#e8a0a0', '#cfd8e3', '#7a869a', '#9aa5b1'
MODEL_COLORS = dict(zip(MODELS, ['#9aa5b1', '#7fa7de', '#4a7fcf', BLUE], strict=True))
YEARS = {365: '1 năm', 730: '2 năm', 1095: '3 năm'}


def _style():
    plt.rcParams.update({
        'font.family': 'serif', 'font.serif': ['TeX Gyre Termes', 'DejaVu Serif'], 'font.size': 8,
        'axes.titlesize': 8, 'axes.labelsize': 8, 'legend.fontsize': 7, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
        'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': 0.6, 'lines.linewidth': 1.2,
        'axes.formatter.use_locale': False, 'pdf.fonttype': 42,
    })


def _comma(ax, axis='both', digits=None):
    """Vietnamese decimal commas on tick labels."""
    def fmt(v, _pos):
        d = digits if digits is not None else (0 if float(v).is_integer() else len(f'{v:g}'.split('.')[-1]))
        return num(v, d)
    if axis in ('x', 'both'):
        ax.xaxis.set_major_formatter(plt.FuncFormatter(fmt))
    if axis in ('y', 'both'):
        ax.yaxis.set_major_formatter(plt.FuncFormatter(fmt))


def interval_diagram():
    fig, axes = plt.subplots(2, 1, figsize=(COLUMN, 1.25))
    for ax, case in zip(axes, (True, False), strict=True):
        ax.set_xlim(0, 10)
        ax.set_ylim(-0.6, 1.2)
        ax.axis('off')
        ax.annotate('', xy=(10, 0), xytext=(0.3, 0), arrowprops=dict(arrowstyle='->', lw=0.8))
        ax.text(10, -0.05, '$t$', ha='left', va='center')
        ax.plot([0.6], [0], 'o', color='#1a2a6c', ms=4)
        ax.text(0.6, -0.45, 's (MCI)', ha='center', fontsize=7)
        if case:
            ax.plot([1.6, 2.4, 4.6], [0] * 3, 'o', color='#8ea2e8', ms=3.5)
            ax.text(2.4, -0.45, 'các lượt MCI', ha='center', fontsize=7)
            ax.plot([5.2], [0], 'o', color='#1a2a6c', ms=3.5)
            ax.plot([6.6], [0], 'o', color=RED, ms=4)
            ax.text(5.2, -0.45, '$L$', ha='center', fontsize=7)
            ax.text(6.6, -0.45, '$R$ (sa sút trí tuệ)', ha='left', fontsize=7)
            ax.plot([5.2, 5.2, 6.6, 6.6], [0.35, 0.55, 0.55, 0.35], color='k', lw=0.6)
            ax.text(5.9, 0.75, '$(L, R]$', ha='center', fontsize=7)
            ax.plot([5.9, 5.9], [-0.3, 0.3], ls=':', color='k', lw=0.6)
            ax.text(5.95, 0.1, '$h$', fontsize=6)
        else:
            ax.plot([5.2], [0], 'o', color='#1a2a6c', ms=3.5)
            ax.text(5.2, -0.45, 'khám cuối = $L$', ha='center', fontsize=7)
            ax.plot([5.2, 5.2, 9.6], [0.35, 0.55, 0.55], color='k', lw=0.6)
            ax.text(7.2, 0.75, '$(L, \\infty)$: kiểm duyệt phải', ha='center', fontsize=7)
    fig.subplots_adjust(hspace=0.2, left=0.02, right=0.97, top=0.98, bottom=0.02)
    return fig


def heaping(r: Results):
    lm = r.landmarks
    R = lm.R.astype(float)
    events = R[np.isfinite(R) & (R <= WINDOW)] / DAYS_PER_MONTH
    fig, ax = plt.subplots(figsize=(COLUMN, 1.9))
    ax.hist(events, bins=np.arange(0, 61, 1), color=BLUE, alpha=0.85)
    for m in range(6, 61, 6):
        ax.axvline(m, ls=':', lw=0.6, color='#555')
    ax.set_xlabel('Tháng từ mốc đến lần khám ghi nhận sa sút trí tuệ (R)')
    ax.set_ylabel('Số mốc')
    return fig


def reliability(r: Results):
    sigma = r.audit['mri_measurement_error']['sigma_robust']
    d22 = np.sqrt(r.json('preprocessing.json')['hippo_adj']['D'][1][1])
    lm = r.landmarks
    points = (lm.pair_days.median(), lm.dropna(subset=MODELS['M3']).pair_days.median())   # all / analysis set
    gap = np.linspace(90, 1100, 300)
    rho = lambda d: d22 ** 2 / (d22 ** 2 + 2 * sigma ** 2 * (DAYS_PER_YEAR / d) ** 2)   # noqa: E731
    fig, ax = plt.subplots(figsize=(COLUMN, 1.8))
    ax.plot(gap, rho(gap), color=BLUE)
    for d in points:
        ax.plot(d, rho(d), 'o', color=RED, ms=3.5)
        ax.annotate(f'{d:.0f} ngày: {num(rho(d), 2)}', (d, rho(d)), xytext=(6, -3), textcoords='offset points',
                    fontsize=7)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Khoảng cách giữa hai lần chụp Δd (ngày)')
    ax.set_ylabel('Độ tin cậy ρ(Δd)')
    _comma(ax, 'y', 1)
    return fig


def baseline_hazard(r: Results):
    m3 = r.paired_models['M3']
    rate = np.exp(m3['log_hazard'])[:-1] * DAYS_PER_YEAR
    edges = CUTS[:-1] / DAYS_PER_YEAR
    fig, ax = plt.subplots(figsize=(COLUMN, 1.6))
    ax.step(edges, np.r_[rate, rate[-1]], where='post', color=BLUE)
    ax.set_xlabel('Năm từ mốc')
    ax.set_ylabel('α(t) (/năm)')
    ax.set_ylim(0, rate.max() * 1.2)
    _comma(ax, 'y', 2)
    return fig


def status(r: Results):
    m3 = r.main.xs('M3', level='model')
    cats = [('case_certain', 'Ca chắc chắn ($R \\leq h$)', RED), ('ambiguous', 'Khoảng vắt qua $h$', PINK),
            ('control_certain', 'Chắc chắn chưa ($L \\geq h$)', LIGHT), ('censored_early', 'Ngừng theo dõi sớm', SLATE)]
    fig, ax = plt.subplots(figsize=(COLUMN, 1.7))
    left = np.zeros(len(HORIZONS))
    ys = np.arange(len(HORIZONS))[::-1]
    for key, label, color in cats:
        values = m3.loc[HORIZONS, key].to_numpy()
        ax.barh(ys, values, left=left, color=color, label=label, height=0.55)
        left += values
    ax.set_yticks(ys, [YEARS[h] for h in HORIZONS])
    ax.set_xlabel('Số người (trung bình mỗi lần lặp; một mốc ngẫu nhiên mỗi người)')
    ax.legend(ncol=2, loc='upper center', bbox_to_anchor=(0.5, -0.32), frameon=False)
    return fig


def architecture():
    fig, ax = plt.subplots(figsize=(COLUMN, 2.4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    boxes = [
        (0.2, 7.9, 4.5, 1.8, 'Giao diện\napp.py, mci_risk/ui/*'),
        (5.2, 7.9, 4.6, 1.8, 'Script điều phối\ntrain, validation, evaluate,\nuncertainty, report, make_release'),
        (0.2, 5.3, 4.5, 1.8, 'Nghiệp vụ bệnh nhân\nrisk, narrative, manual_entry,\nadni_import, store'),
        (0.2, 2.6, 9.6, 1.9,
         'Lõi thống kê\ndata (Mục III–IV) · model (Mục V) · metrics (Mục VI)\ncv, analyses (Mục VII)'),
        (0.2, 0.1, 9.6, 1.7, 'Cấu hình\nconfig.py ← config/protocol.json (giao thức khóa)'),
    ]
    for x, y, w, h, text in boxes:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.05,rounding_size=0.25', fc='#eef3fb',
                                    ec=BLUE, lw=0.7))
        ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=6.5, linespacing=1.3)
    arrows = [((2.45, 7.9), (2.45, 7.1)), ((2.45, 5.3), (2.45, 4.5)), ((7.5, 7.9), (7.5, 4.5)),
              ((5.0, 2.6), (5.0, 1.8))]
    for start, end in arrows:
        ax.annotate('', xy=end, xytext=start, arrowprops=dict(arrowstyle='->', lw=0.7, color='#333'))
    return fig


def decision_curve(r: Results):
    prim = r.main.loc[PRIMARY]
    percents = [100 * t for t in THRESHOLDS]
    observed = float(prim.observed_ipcw.iloc[0])
    fig, ax = plt.subplots(figsize=(COLUMN, 2.0))
    ax.axvspan(100 * BAND_THRESHOLDS[0], 100 * BAND_THRESHOLDS[1], color='#eef3fb', lw=0)
    for m in MODELS:
        ax.plot(percents, [prim.loc[m][f'nb_{round(t * 100)}'] for t in THRESHOLDS], marker='o', ms=2.5,
                color=MODEL_COLORS[m], label=m)
    ax.plot(percents, [observed - (1 - observed) * t / (1 - t) for t in THRESHOLDS], color='#d68910', ls='--',
            label='Theo dõi tất cả')
    ax.axhline(0, color='#333', lw=0.8, label='Không theo dõi ai')
    ax.set_xlabel('Ngưỡng nguy cơ $t$ (%)')
    ax.set_ylabel('Net benefit')
    ax.set_ylim(-0.02, None)
    ax.legend(ncol=3, loc='upper center', bbox_to_anchor=(0.5, -0.28), frameon=False)
    _comma(ax, 'y', 2)
    return fig


FOREST_LABELS = {
    'Primary analysis': 'Phân tích chính', 'L1': 'L1: tiền xử lý ước lượng toàn train',
    'S1': 'S1: Cox chuẩn so với M3 (M3cox − M3)', 'S2a': 'S2a: tốc độ hai điểm (thay BLUP)',
    'S2b': 'S2b: ADNI-GO/2, BSI thay BLUP', 'S4': 'S4: chỉ sa sút trí tuệ được xác nhận', 'S5': 'S5: bỏ FAQ',
    'S7': 'S7: loại nhân khẩu không nhất quán', 'S9': 'S9: tương tác HV × tuổi MRI (M1)',
    'S10': 'S10: cùng model máy chụp', 'S3': 'S3: chỉ lượt MCI thứ hai',
}
S6_LABELS = {'last negative': 'S6: khởi phát tại L', 'midpoint': 'S6: khởi phát tại (L + R)/2',
             'diagnosis visit': 'S6: khởi phát tại R', 'from M3': 'S6: rút khởi phát từ M3'}


def _forest_label(analysis, comparison):
    """Vietnamese label of one row of sensitivity_summary.csv."""
    code = analysis.split(' · ')[0]
    if code == 'S2b' and comparison == 'M3 − M2':
        return 'S2b: ADNI-GO/2, M3 − M2'
    if code == 'S8':
        return 'S8: MRI trước mốc ≥ ' + analysis.split('≥ ')[1].split(' ')[0] + ' ngày'
    if code == 'S6':
        return next(v for k, v in S6_LABELS.items() if k in analysis)
    return FOREST_LABELS.get(code, FOREST_LABELS.get(analysis, analysis))


def forest(r: Results):
    s = r.sensitivity.iloc[::-1].reset_index(drop=True)
    label = [_forest_label(a, c) for a, c in zip(s.analysis, s.comparison, strict=True)]
    fig, ax = plt.subplots(figsize=(7.1, 0.22 * len(s) + 0.6))
    ax.axvspan(-0.02, 0.02, color='#eef3fb', lw=0)
    ax.axvline(0, color='#555', lw=0.7)
    ax.errorbar(s.d_auc, range(len(s)), xerr=[s.d_auc - s.auc_lo, s.auc_hi - s.d_auc], fmt='o', ms=3, color=BLUE,
                elinewidth=0.9, capsize=1.5)
    ax.set_yticks(range(len(s)), label)
    ax.set_xlabel(f'ΔAUC tại {num(PRIMARY, 0)} ngày (KTC 95%); vùng tô: ±δ = ±0,02')
    _comma(ax, 'x', 2)
    return fig


def calibration(r: Results):
    cal = r.csv('test_calibration.csv')
    cal = cal[cal.h == PRIMARY]
    fig, ax = plt.subplots(figsize=(COLUMN * 0.8, COLUMN * 0.8))
    ax.plot([0, 1], [0, 1], ls='--', color='#777', lw=0.8, label='Lý tưởng')
    for m in ('M0', 'M3'):
        c = cal[cal.model == m].sort_values('predicted')
        ax.plot(c.predicted, c.observed_turnbull, marker='o', ms=3, color=MODEL_COLORS[m], label=m)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Nguy cơ dự báo')
    ax.set_ylabel('Tỷ lệ quan sát (Turnbull)')
    pct = plt.FuncFormatter(lambda v, _p: f'{v * 100:.0f}%')
    ax.xaxis.set_major_formatter(pct)
    ax.yaxis.set_major_formatter(pct)
    ax.legend(frameon=False, loc='upper left')
    return fig


FIGURES = {
    'fig1_interval': lambda r: interval_diagram(), 'fig2_heaping': heaping, 'fig3_reliability': reliability,
    'fig4_baseline': baseline_hazard, 'fig5_status': status, 'fig6_architecture': lambda r: architecture(),
    'fig7_decision': decision_curve, 'fig8_forest': forest, 'fig9_calibration': calibration,
}


def build(r: Results, out: Path):
    """Write every figure as <name>.pdf and <name>.png into out."""
    _style()
    out.mkdir(parents=True, exist_ok=True)
    for name, draw in FIGURES.items():
        fig = draw(r)
        for ext in ('pdf', 'png'):
            fig.savefig(out / f'{name}.{ext}', bbox_inches='tight', dpi=300 if ext == 'png' else None)
        plt.close(fig)
