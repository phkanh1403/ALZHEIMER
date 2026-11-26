"""Consistency of the protocol, model definitions and analysis specs."""
from itertools import pairwise

import numpy as np

from mci_risk.analyses import ANALYSES
from mci_risk.config import HORIZONS, MODEL_PAIRS, MODELS, NB_CORE, PRIMARY, THRESHOLDS, WINDOW
from mci_risk.cv import pair_label
from mci_risk.model import CUTS


def test_protocol_values_fit_together():
    assert PRIMARY in HORIZONS and max(HORIZONS) <= WINDOW
    assert set(NB_CORE) <= set(THRESHOLDS)
    assert WINDOW in CUTS and all(h in CUTS for h in HORIZONS)
    assert np.isinf(CUTS[-1])


def test_models_are_nested():
    names = list(MODELS)
    for simpler, richer in pairwise(names):
        assert MODELS[richer][:len(MODELS[simpler])] == MODELS[simpler]
    assert all((a, b) in MODEL_PAIRS for a, b in zip(names[1:], names[:-1], strict=True))


def test_analysis_specs_are_consistent():
    labels = [a.label for a in ANALYSES.values() if a.label]
    assert len(labels) == len(set(labels))
    for name, spec in ANALYSES.items():
        available = set(spec.models) | set(spec.extra or {})
        assert all(a in available and b in available for a, b in spec.pairs), name
        assert set(spec.reported) <= {pair_label(a, b) for a, b in spec.pairs}, name
