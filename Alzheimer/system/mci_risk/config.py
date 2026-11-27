"""Paths, the locked analysis protocol and fixed constants.

Every other module takes its settings from here. Values that the protocol fixes (time frames, follow-up window,
cross-validation design, decision rule) are read from config/protocol.json, so code and protocol cannot drift apart.
"""
import json
import os
from pathlib import Path

# --------------------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / 'config'
DATA_DIR = ROOT / 'data'
TRAIN_DIR = DATA_DIR / 'train'
PARTITION_FILE = DATA_DIR / 'partition.csv'
RUNS_DIR = ROOT / 'runs'
CV_DIR = RUNS_DIR / 'cv'
DB_PATH = Path(os.environ.get('MCI_RISK_DB', ROOT / 'instance' / 'profiles.sqlite'))

# --------------------------------------------------------------------------- locked protocol
PROTOCOL = json.loads((CONFIG_DIR / 'protocol.json').read_text(encoding='utf-8'))

HORIZONS = list(PROTOCOL['horizons_days'])          # evaluation time frames (days after the visit)
PRIMARY = PROTOCOL['primary_horizon_days']
WINDOW = PROTOCOL['label_window_days']              # follow-up used for model fitting is censored here
CV_FOLDS = PROTOCOL['cv']['folds']
CV_REPEATS = PROTOCOL['cv']['repeats']
INNER_FOLDS = PROTOCOL['cv']['inner_folds']
LAMBDAS = PROTOCOL['cv']['lambda_grid']
BOOTSTRAP = PROTOCOL['cv']['bootstrap']
NB_CORE = PROTOCOL['nb_core_thresholds']
DELTA_AUC = PROTOCOL['delta_auc']
NB_MARGIN = PROTOCOL['nb_margin']
# risk bands of the app (and confusion-matrix thresholds): the lowest and highest core net-benefit thresholds
BAND_THRESHOLDS = (min(NB_CORE), max(NB_CORE))

# --------------------------------------------------------------------------- data rules (pipeline)
MRI_MAX_AGE = 365       # an MRI older than this (days) is not used at a visit
COG_MAX_AGE = 45        # a cognitive test older than this (days) is not used at a visit
MIN_SPAN = 300          # minimum time span (days) of a series used for a trend
TWO_POINT_MIN = 90      # minimum gap (days) between two scans for a two-point atrophy rate
LME_MIN_POINTS = 3      # per-participant series used to estimate mixed-model parameters ...
LME_MIN_SPAN = 365      # ... need this many points over at least this many days

# --------------------------------------------------------------------------- units
DAYS_PER_YEAR = 365.25
DAYS_PER_MONTH = 30.44

# --------------------------------------------------------------------------- evaluation
THRESHOLDS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]   # decision-curve thresholds
IPCW_MIN_SURVIVAL = 0.05     # censoring survival is floored here, capping IPCW weights at 20
CALIBRATION_GROUPS = 10
SENSITIVITY_BOOTSTRAP = 500  # bootstrap draws for sensitivity analyses (main analysis uses BOOTSTRAP)
MIN_SUBGROUP = 60            # smallest test subgroup (participants) that is reported

# --------------------------------------------------------------------------- ADNI tables
REQUIRED_TABLES = ['DXSUM', 'MMSE', 'FAQ', 'PTDEMOG', 'UCSFFSX7', 'ADAS', 'NEUROBAT']
OPTIONAL_TABLES = ['FOXLABBSI', 'All_Subjects_Structural_MRI_Images']

# --------------------------------------------------------------------------- models (nested: each adds to the previous)
_BLOCKS = {
    'M0': ['tau', 'tau2', 'age', 'sex', 'education', 'mmse', 'faq', 'adas13', 'ravlt_delay'],
    'M1': ['hippo_adj', 'entor', 'field3t'],
    'M2': ['adas13_slope', 'faq_slope'],
    'M3': ['hippo_slope', 'entor_slope'],
}
MODELS: dict[str, list[str]] = {}
for _name, _block in _BLOCKS.items():
    MODELS[_name] = (MODELS[list(MODELS)[-1]] if MODELS else []) + _block
del _name, _block

MODEL_PAIRS = [('M1', 'M0'), ('M2', 'M1'), ('M3', 'M2')]   # each richer model against the one below it

LABELS = {
    'tau': 'Years since first MCI diagnosis', 'tau2': 'Years since first MCI diagnosis (squared)',
    'age': 'Age (years)', 'sex': 'Sex', 'education': 'Education (years)',
    'mmse': 'MMSE score', 'faq': 'FAQ score', 'adas13': 'ADAS-Cog13 score',
    'ravlt_delay': 'RAVLT 30-min delayed recall (words)',
    'hippo_adj': 'Hippocampal volume, head-size adjusted (mm³)', 'entor': 'Entorhinal cortex thickness (mm)',
    'field3t': 'MRI field strength (1 = 3T, 0 = 1.5T)',
    'adas13_slope': 'ADAS-Cog13 change (points/year)', 'faq_slope': 'FAQ change (points/year)',
    'hippo_slope': 'Hippocampal volume change (mm³/year)', 'entor_slope': 'Entorhinal thickness change (mm/year)',
}
