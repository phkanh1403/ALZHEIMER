"""Build the distributable ZIP: code, fitted models and aggregated results, without participant-level ADNI data.

Usage:
    python make_release.py                       # writes dist/MCI_Dementia_Risk.zip
    python make_release.py --out path/to/file.zip
"""
import argparse
from pathlib import Path

from mci_risk.config import ROOT
from mci_risk.release import RESTRICTED, build

NAME = 'MCI_Dementia_Risk'


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out', type=Path, default=ROOT / 'dist' / f'{NAME}.zip')
    args = parser.parse_args()
    files = build(ROOT, args.out, NAME)
    print(f'{len(files)} files written to {args.out}')
    print('Left out (participant-level, ADNI Data Use Agreement): ' + ', '.join(RESTRICTED))


if __name__ == '__main__':
    main()
