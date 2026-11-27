"""Small file helpers shared by the scripts and the app."""
import json
from pathlib import Path


def save_json(obj, path):
    """Write JSON atomically (temporary file, then rename); NaN is rejected so files stay valid JSON."""
    tmp = Path(str(path) + '.tmp')
    tmp.write_text(json.dumps(obj, allow_nan=False, ensure_ascii=False, default=str), encoding='utf8')
    tmp.replace(path)


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))
