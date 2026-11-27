"""The release builder leaves out participant-level data and refuses to ship identifier columns."""
import json
import zipfile

import pytest

from mci_risk.release import RestrictedDataError, build, identifier_columns, release_files


@pytest.fixture
def project(tmp_path):
    root = tmp_path / 'project'
    files = {
        'app.py': 'print(1)\n',
        'data/partition.csv': 'RID,set\n1,train\n',
        'data/train/Clinical_Data/DXSUM.csv': 'RID,DIAGNOSIS\n1,2\n',
        'runs/models.json': json.dumps({'M0': {'beta': [0.1]}}),
        'runs/train_inputs.csv': 'RID,age\n1,70\n',
        'runs/cv/main_metrics.csv': 'h,model,auc\n730,M0,0.87\n',
        'runs/cv/main_oof.csv.gz': '',
        'mci_risk/__pycache__/x.pyc': '',
    }
    for rel, text in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text)
    return root


def test_release_leaves_out_participant_level_files(project):
    assert release_files(project) == ['app.py', 'data/partition.csv', 'runs/cv/main_metrics.csv',
                                      'runs/models.json']


def test_release_zip_has_prefix(project, tmp_path):
    files = build(project, tmp_path / 'out.zip', 'MCI')
    with zipfile.ZipFile(tmp_path / 'out.zip') as z:
        assert sorted(z.namelist()) == [f'MCI/{f}' for f in files]


def test_release_refuses_identifier_columns(project, tmp_path):
    (project / 'runs' / 'cv' / 'extra_predictions.csv').write_text('repeat,RID,p\n0,1,0.2\n')
    (project / 'runs' / 'leak.json').write_text(json.dumps({'rows': [{'PTID': '002_S_0001'}]}))
    with pytest.raises(RestrictedDataError, match=r'extra_predictions\.csv: RID.*leak\.json: PTID'):
        build(project, tmp_path / 'out.zip', 'MCI')
    assert not (tmp_path / 'out.zip').exists()


def test_identifier_columns_reads_headers_only(tmp_path):
    p = tmp_path / 'a.csv'
    p.write_text('h,model,auc\n730,RID,0.8\n')     # a value, not a column, named RID
    assert identifier_columns(p) == []
