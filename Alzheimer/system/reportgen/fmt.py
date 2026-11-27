"""Vietnamese number formatting for the report: decimal comma, dot as thousands separator, true minus sign."""
import math

MINUS = '−'
_SWAP = str.maketrans({',': '.', '.': ','})


def num(x, digits=3):
    """0.8714 -> '0,871'; 1176 -> '1.176' (digits=0); -0.006 -> '−0,006'; NaN -> '—'."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return '—'
    text = f'{abs(x):,.{digits}f}'.translate(_SWAP)
    return (MINUS if round(x, digits) < 0 else '') + text


def integer(x):
    return num(float(x), 0)


def ci(lo, hi, digits=3):
    """'(−0,006; 0,014)'."""
    return f'({num(lo, digits)}; {num(hi, digits)})'


def pct(x, digits=1):
    """0.8929 -> '89,3%'."""
    return f'{num(100 * x, digits)}%'


CONCLUSIONS = {'Added value': 'Có giá trị gia tăng', 'Equivalent': 'Tương đương', 'Inconclusive': 'Bất định'}


def conclusion(rule):
    return CONCLUSIONS[rule]
