import pytest
import pandas as pd
import yaml
import sys
sys.path.append('/home/dosky/Documents/churner_ml')
from churner.ml.utils import loader


@pytest.fixture
def datapath():
    with open('/home/dosky/Documents/churner_ml/config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    # Load data
    path = config['data']['datapath']
    return path


def test_loader(datapath):
    assert(isinstance(datapath, str))
    assert(isinstance(loader(datapath), pd.DataFrame))