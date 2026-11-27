"""Unit tests of the report generator: Vietnamese number format and the LaTeX rewriting for the Word version."""
import pytest

from reportgen.fmt import ci, num, pct

docx = pytest.importorskip('reportgen.docx', reason='python-docx not installed (dev dependency)')


@pytest.mark.parametrize(('value', 'digits', 'text'), [
    (0.8714, 3, '0,871'), (1176, 0, '1.176'), (1445236, 0, '1.445.236'), (-0.0062, 3, '−0,006'),
    (-0.0001, 3, '0,000'), (float('nan'), 3, '—'),
])
def test_num(value, digits, text):
    assert num(value, digits) == text


def test_ci_and_pct():
    assert ci(-0.006, 0.014) == '(−0,006; 0,014)'
    assert pct(8984 / 10062, 2) == '89,29%'


def test_replace_command_handles_nested_braces():
    assert docx.replace_command(r'a \R{x} b \R{y}', 'R', str.upper) == 'a X b Y'
    assert docx.replace_command(r'\caption{A {B} C}', 'caption', lambda c: f'[{c}]') == '[A {B} C]'


def test_number_sections_follows_ieee_style():
    text = r'\section{Intro}\subsection{One}\subsection{Two}\section{Next}\appendices\section{Code}'
    out = docx.number_sections(text)
    assert r'\section*{I. Intro}' in out and r'\subsection*{B. Two}' in out
    assert r'\section*{II. Next}' in out and r'\section*{Phụ lục A: Code}' in out


def test_simple_columns():
    assert docx._simple_columns(r'@{}lL{0.68\columnwidth}r@{}') == 'llr'


def test_equation_number_moves_to_the_end():
    out = docx.equations('\\begin{equation}\\label{eq:a}\nx=1\n\\end{equation}', {'eq:a': '7'})
    assert out == '\\begin{equation}\nx=1\\qquad(7)\n\\end{equation}'


def test_references_and_citations_resolved():
    labels, cites = {'fig:x': '3', 'eq:y': '12'}, {'k1': '1', 'k2': '5'}
    out = docx.resolve_refs(r'Hình~\ref{fig:x}, \eqref{eq:y} \cite{k1, k2}', labels, cites)
    assert out == 'Hình~3, (12) [1], [5]'
