####
# Loads data that is more or less ready for ML (read, not from SQL, etc)
####
import logging
import os
import pickle
from itertools import chain
from typing import List

import pandas as pd
import re

from modeling.preprocessing import encode_not_null, encode_date_to_biweek

log = logging.getLogger(__name__)

data_dir_default = '/opt/project/data/dumps'
exclude_columns = [
    'ProcessedLayerUid', 'YearId', 'Year', 'ImportBatchId', 'ProcessedLayerLastUpdated',
    'Area', 'Cost_Bu'
]


def group_cols():
    with open(os.path.join(os.path.dirname(__file__), 'group_cols.pickle'), 'rb') as f:
        cols: List = pickle.load(f)
        cols = cols + ['MgmtZone']
        cols = [c for c in cols if c not in ['HarvDate#', 'PlntDate#']]

        return cols


def load_df_corn_pkl_smpl_25_20171018(include_all_columns: bool = False) -> pd.DataFrame:
    """
    Loads a gis pps corn 25% sample from a pickle.

    :return: DataFrame
    """
    file_name = 'df-corn-smpl_25-gis-pps-20171018.pkl'
    return load_pickled(file_name, include_all_columns)


def load_pickled(file_name:str, include_all_columns: bool = False) -> pd.DataFrame:
    path = os.path.join(data_dir_default, file_name)
    df: pd.DataFrame = pd.read_pickle(path)

    if not include_all_columns:
        df.drop(exclude_columns, axis=1, inplace=True, errors='ignore')

    return df


def dump_sample(df: pd.DataFrame, file_name: str):
    df.to_pickle(os.path.join(data_dir_default, file_name))


def shape_gis_pps(df: pd.DataFrame,
                  includeSms=False) -> (pd.DataFrame, List[str]):
    # NOTE: these columns are from the old SQL tables, probably should update this somehow at some point?
    label_cols = group_cols()

    ###
    #  encode existence only columns (true, false)
    ###
    exists_cols = []
    exists_src_cols = [['Variety'], ['SampleDate']]
    if not includeSms:
        exists_src_cols.append(['SMS'])

    for src_cols in exists_src_cols:
        df, col = encode_not_null(df, src_cols)
        exists_cols.append(col)

    # add the new columns, remove the originals from group list
    label_cols = label_cols + exists_cols
    label_cols = [
        c for c in label_cols
        if c not in list(chain.from_iterable(exists_src_cols))]

    ###
    # encode date columns to year bi-weekly number (0 - ~26)
    ###
    date_src_cols = [c for c in df.columns if re.search(r'[aA]p(p)?[dD]ate', c)]
    date_cols = []
    for src_col in date_src_cols:
        df, col = encode_date_to_biweek(df, src_col)
        date_cols.append(col)

    # add the new columns, remove the originals from group list
    label_cols = label_cols + date_cols
    label_cols = [
        c for c in label_cols if c not in date_src_cols
    ]

    log.debug("Label Columns: %s", label_cols)
    return df, label_cols

