"""Read the result files in runs/ that the report draws on."""
from dataclasses import dataclass
from functools import cached_property

import pandas as pd

from mci_risk.config import BAND_THRESHOLDS, CV_DIR, MODELS, PRIMARY, RUNS_DIR
from mci_risk.files import load_json
from mci_risk.metrics import CELLS, cm_key, confusion_rates


@dataclass
class Results:
    runs: object = RUNS_DIR
    cv: object = CV_DIR

    # ------------------------------------------------------------------ files
    def csv(self, name, cv=False):
        return pd.read_csv((self.cv if cv else self.runs) / name)

    def json(self, name, cv=False):
        return load_json((self.cv if cv else self.runs) / name)

    @cached_property
    def main(self):
        m = self.csv('main_metrics.csv', cv=True)
        return m[m.scope == 'random'].set_index(['h', 'model'])

    @cached_property
    def main_deltas(self):
        return self.csv('main_deltas.csv', cv=True)

    @cached_property
    def test_deltas(self):
        return self.csv('test_deltas.csv')

    @cached_property
    def test(self):
        m = self.csv('test_metrics.csv')
        return m[m.scope == 'random'].set_index(['h', 'model'])

    @cached_property
    def test_available(self):
        m = self.csv('test_coverage.csv')
        return m[m.scope == 'random'].set_index(['h', 'model'])

    @cached_property
    def sensitivity(self):
        """Rows of the sensitivity forest; drops the M3 − M2 rows of S1 and S2a, which repeat the main result."""
        s = self.csv('sensitivity_summary.csv')
        repeat = s.analysis.str.match(r'^(S1|S2a) ') & (s.comparison == 'M3 − M2')
        return s[~repeat].reset_index(drop=True)

    @cached_property
    def paired_models(self):
        return self.json('models.json')

    @cached_property
    def audit(self):
        return self.json('data_audit.json')

    @cached_property
    def landmarks(self):
        return self.csv('train_landmarks.csv')

    def info(self, name):
        return self.json(f'{name}_info.json', cv=True)

    def metrics(self, name):
        m = self.csv(f'{name}_metrics.csv', cv=True)
        return m[m.scope == 'random'].set_index(['h', 'model'])

    def deltas(self, name):
        return self.csv(f'{name}_deltas.csv', cv=True)

    def secondary(self, model):
        return self.metrics(f'SEC_{model}')

    # ------------------------------------------------------------------ derived
    @staticmethod
    def delta(table, comparison, h=PRIMARY):
        row = table[(table.comparison == comparison) & (table.h == h)]
        if row.empty:
            raise KeyError(f'{comparison} at {h} not found')
        return row.iloc[0]

    def rates(self, table, model, h=PRIMARY, t=BAND_THRESHOLDS[0]):
        """Sensitivity, specificity, PPV, NPV, F1 from the mean confusion-matrix shares."""
        row = table.loc[(h, model)]
        return confusion_rates(**{c: float(row[cm_key(c, t)]) for c in CELLS})

    def shared_events(self):
        """(participants with an event in the window, landmarks with an event) in the paired set."""
        shared = self.landmarks.dropna(subset=MODELS['M3'])
        event = shared.event == 1
        return int(shared[event].RID.nunique()), int(event.sum())
