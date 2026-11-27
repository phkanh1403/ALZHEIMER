"""Build the technical report from the results in runs/.

Writes docs/generated/ (number macros, table bodies, figures), then compiles docs/CoSoLyThuyet_v8.pdf with XeLaTeX
and converts it to docs/CoSoLyThuyet_v8.docx with pandoc. Every result number in the report comes from runs/.

Usage:
    python build_report.py            # numbers, tables, figures, PDF and DOCX
    python build_report.py --no-docx
"""
import argparse
import re
import shutil
import subprocess
import sys

from mci_risk.config import ROOT
from reportgen import docx, figures, numbers, tables
from reportgen.results import Results

DOCS = ROOT / 'docs'
SOURCE = DOCS / 'CoSoLyThuyet_v8.tex'
GENERATED = DOCS / 'generated'
BUILD = DOCS / 'build'


def count_tests():
    out = subprocess.run([sys.executable, '-m', 'pytest', '--collect-only', '-q'], cwd=ROOT, capture_output=True,
                         text=True, check=True).stdout
    return int(re.search(r'(\d+) tests? collected', out).group(1))


def generate(r):
    values = numbers.build(r)
    values['tests'] = str(count_tests())
    (GENERATED / 'tables').mkdir(parents=True, exist_ok=True)
    (GENERATED / 'numbers.tex').write_text(numbers.to_latex(values), encoding='utf-8')
    for name, make in tables.TABLES.items():
        (GENERATED / 'tables' / f'{name}.tex').write_text(make(r), encoding='utf-8')
    figures.build(r, GENERATED / 'figures')
    return values


def compile_pdf():
    BUILD.mkdir(exist_ok=True)
    for _ in range(2):    # second pass resolves references
        subprocess.run(['xelatex', '-interaction=nonstopmode', '-halt-on-error', f'-output-directory={BUILD}',
                        SOURCE.name], cwd=DOCS, check=True, capture_output=True)
    shutil.copy(BUILD / SOURCE.with_suffix('.pdf').name, DOCS)
    return DOCS / SOURCE.with_suffix('.pdf').name


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--no-docx', action='store_true')
    args = parser.parse_args()
    values = generate(Results())
    pdf = compile_pdf()
    print(f'PDF: {pdf}')
    if not args.no_docx:
        out = docx.build(SOURCE, values, BUILD / SOURCE.with_suffix('.aux').name, SOURCE.with_suffix('.docx'),
                         resource_dir=DOCS)
        print(f'DOCX: {out}')


if __name__ == '__main__':
    main()
