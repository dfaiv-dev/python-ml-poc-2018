####
# Loads data that is more or less ready for ML (read, not from SQL, etc)
####
import os
from typing import List

import pandas as pd
import pickle


data_dir_default = '/var/opt/pcsml/data'
exclude_columns = [
    'ProcessedLayerUid', 'YearId', 'Year', 'ImportBatchId', 'ProcessedLayerLastUpdated',
    'Area', 'Cost_Bu'
]


def group_cols():
    with open('data/group_cols.pickle', 'rb') as f:
        cols:List = pickle.load(f)
        cols = cols + ['MgmtZone']
        cols = [c for c in cols if c not in ['HarvDate#', 'PlntDate#']]

        return cols


def load_df_corn_pkl_smpl_25_20171018(include_all_columns: bool = False) -> pd.DataFrame:
    """
    Loads a gis pps corn 25% sample from a pickle.

    :return: DataFrame
    """
    file_name = 'df-corn-smpl_25-gis-pps-20171018.pkl'
    path = os.path.join(data_dir_default, file_name)
    df: pd.DataFrame = pd.read_pickle(path)

    if not include_all_columns:
        df.drop(exclude_columns, axis=1, inplace=True)

    return df
