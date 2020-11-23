import gzip
import os
from typing import Dict

import pandas
import pickle
from xgboost import Booster


def xgb_latest() -> (Booster, Dict[str, pandas.Categorical]):
    base_path = '/var/opt/pcsml/devel/training_data/dumps/debug004/2017-12-27T18-30-59'

    model = Booster()
    model.load_model(
        os.path.join(base_path, 'model_2017-12-27T18-30-59.xgb'))

    with gzip.open(os.path.join(base_path, 'model_2017-12-27T18-30-59_column_categories.pickle.gz'), 'rb') as f:
        column_categories = pickle.load(f)

    return model, column_categories


def _temp_zip_pickle():
    categories_pickle_path = '/var/opt/pcsml/devel/training_data/dumps/debug004/2017-12-27T18-30-59/model_2017-12-27T18-30-59_column_categories.pickle'
    with open(categories_pickle_path, 'rb') as f:
        categories = pickle.load(f)

    with gzip.open(f"{categories_pickle_path}.gz", 'wb') as gzf:
        pickle.dump(categories, gzf)


def _models_bin_file_path(file_name: str) -> str:
    return os.path.join(
        os.path.dirname(__file__),
        'models_bin',
        file_name
    )
