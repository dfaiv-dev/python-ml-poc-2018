####
# Loads data that is more or less ready for ML (read, not from SQL, etc)
####
import logging
import os
import pickle
import re
from typing import List

import numpy as np
import pandas as pd

from data.df_encoding_util import encode_not_null, encode_date_to_biweek

log = logging.getLogger(__name__)

data_dir_default = '/opt/project/data/dumps'
exclude_columns = [
    'YearID', 'Year', 'YearUID', 'FieldUID', 'UID',
    'Area', 'CreateDate', 'LastUpdated'
]
yield_dep_columns = [
    'PRem',
    'KRem',
    'SdUsage',
    'Ndex',
    'NUsage',
    'KRem2Yrs',
    'PRem2Yrs'
]
numeric_label_columns = [
    'Replant'
]


def group_cols():
    with open(os.path.join(os.path.dirname(__file__), 'resources/group_cols.pickle'), 'rb') as f:
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


def load_pickled(file_name: str, include_all_columns: bool = False) -> pd.DataFrame:
    path = os.path.join(data_dir_default, file_name)
    df: pd.DataFrame = pd.read_pickle(path)

    if not include_all_columns:
        df.drop(exclude_columns, axis=1, inplace=True, errors='ignore')

    return df


def dump_sample(df: pd.DataFrame, file_name: str):
    df.to_pickle(os.path.join(data_dir_default, file_name))


def gis_pps_encode_raw_csv(df: pd.DataFrame,
                           null_values=None,
                           includeSms=False) -> pd.DataFrame:
    # NOTE: these columns are from the old SQL tables, probably should update this somehow at some point?
    if null_values is None:
        null_values = ['', 'none', 'nan', 'null', None]

    ###
    #  encode existence only columns (true, false)
    ###
    exists_src_cols = [['Variety'], ['SampleDate']]
    if not includeSms:
        exists_src_cols.append(['SMS'])

    for src_cols in exists_src_cols:
        print(f"encode_not_null: {src_cols}")
        df = encode_not_null(df, src_cols)

    ###
    # encode date columns to year bi-weekly number (0 - ~26)
    ###
    date_src_cols = [c for c in df.columns if re.search(r'[aA]p(p)?[dD]ate', c)]
    for src_col in date_src_cols:
        print(f"encode_date_to_biweek: {src_col}")
        df = encode_date_to_biweek(df, src_col)

    ##
    # create category columns
    ##
    label_cols = df.select_dtypes(include=['bool', 'object']).columns
    label_cols = [c for c in label_cols if c not in exclude_columns]
    for label_col in label_cols + numeric_label_columns:
        print(f"encoding category column: {label_col}")

        df[label_col] = df[label_col].astype(str).str.lower()
        df[label_col] = df[label_col].replace(null_values, np.nan)
        df[label_col] = df[label_col].astype('category')

    ##
    # fill nan
    ##
    numeric_cols = df.select_dtypes(include=np.number).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)

    return df


def elb_year_ids() -> List[int]:
    df = pd.read_csv(
        os.path.join(
            os.path.dirname(__file__),
            'resources/elb_yearids.csv'))

    return df.iloc[:, 0].values
