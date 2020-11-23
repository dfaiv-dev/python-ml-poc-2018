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

from data import gis_pps
from data.gis_pps import encode_not_null, encode_date_to_biweek, TripColumnSet

log = logging.getLogger(__name__)

data_dir_default = '/opt/project/data/dumps'
exclude_columns = [
    'YearID', 'Year', 'YearUID', 'FieldUID', 'UID', 'FieldID'
    'Area', 'CreateDate', 'LastUpdated', 'HarvDate', 'GridRow',
    'GridColumn',
    'PlntDate'
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
existence_columns = [
    'Variety', 'SampleDate', 'SMS',
    'CheckTest',
    'Chmresis',
    'Disresis',
    'PostMethod',
    'PreAdd1', 'PreAdd6',
    'PriNAdd',
    'PstAdd1', 'PstAdd2', 'PstAdd30', 'PstAdd40',
    'Pstresis',
    'Sc1NAdd', 'Sc2NAdd', 'Sc3NAdd',
    'SeedTreat',
    'Spectrt',
    'StrType'
]
existence_column_sets = {
    'Micro': ['MicroApMthd', 'MicroName', 'MicroTime'],
    'Manure': ['ManAdd', 'ManureType'],
    'PriN': ['PriNTime', 'PriNType', 'PriNApMthd']
}
trip_count_columns = [
    TripColumnSet(
        'Foliar',
        product_columns=['Foliar1', 'Foliar2', 'Foliar3'],
        date_columns=['Foliar1Time', 'Foliar2Time', 'Foliar3Time']),
    TripColumnSet(
        'Fung',
        rate_columns=['Fung1Rate', 'Fung2Rate', 'Fung3Rate'],
        date_columns=['Fung1ApDate_biweek', 'Fung2ApDate_biweek', 'Fung3ApDate_biweek'],
        ap_method_columns=['Fung1ApMthd', 'Fung2ApMthd', 'Fung3ApMthd'],
        product_columns=['Fungicide1', 'Fungicide2', 'Fungicide3']),
    TripColumnSet(
        'Isc',
        rate_columns=['Isc1Rate', 'Isc2Rate', 'Isc3Rate'],
        date_columns=['Isc1ApDate_biweek', 'Isc2ApDate_biweek', 'Isc3ApDate_biweek'],
        product_columns=['Isc1Name', 'Isc2Name', 'Isc3Name']),
    TripColumnSet(
        'K',
        rate_columns=['KRate', 'K2Rate'],
        date_columns=['KTime', 'K2Time'],
        product_columns=['KType', 'K2Type'],
        ap_method_columns=['KApMthd', 'K2ApMthd']),
    TripColumnSet(
        'P',
        rate_columns=['PRate', 'P2Rate'],
        date_columns=['PTime', 'P2Time'],
        product_columns=['PType', 'P2Type'],
        ap_method_columns=['PApMthd', 'P2ApMthd']),
    TripColumnSet(
        'PN',
        rate_columns=['PN1Rate', 'PN2Rate']),
    TripColumnSet(
        'Pre',
        rate_columns=['Pre1Rate', 'Pre2Rate', 'Pre2Rate', 'Pre6Rate', 'Pre7Rate', 'Pre8Rate'],
        date_columns=['PreAppDate_biweek', 'Pre6AppDate_biweek'],
        product_columns=['Pre1Name', 'Pre2Name', 'Pre3Name', 'Pre6Name', 'Pre7Name', 'Pre8Name'],
        ap_method_columns=['PreMethod', 'Pre6Method']),
    TripColumnSet(
        'Pst',
        rate_columns=[
            'Pst1Rate', 'Pst2Rate', 'Pst30Rate', 'Pst31Rate', 'Pst32Rate', 'Pst33Rate', 'Pst3Rate',
            'Pst40Rate', 'Pst41Rate', 'Pst42Rate', 'Pst43Rate', 'Pst4Rate',
            'Pst5Rate', 'Pst6Rate', 'Pst7Rate', 'Pst8Rate'],
        date_columns=[
            'Pst1ApDate_biweek', 'Pst2ApDate_biweek', 'Pst30ApDate_biweek', 'Pst40ApDate_biweek'],
        product_columns=[
            'Pst1Name', 'Pst2Name', 'Pst30Name', 'Pst31Name', 'Pst32Name', 'Pst33Name', 'Pst3Name',
            'Pst40Name', 'Pst41Name', 'Pst42Name', 'Pst43Name', 'Pst4Name',
            'Pst5Name', 'Pst6Name', 'Pst7Name', 'Pst8Name'],
        ap_method_columns=['PstMethod2', 'PstMethod30', 'PstMethod40']),
    TripColumnSet(
        'ScN',
        rate_columns=['Sc1NRate', 'Sc2NRate', 'Sc3NRate'],
        date_columns=['Sc1NTime', 'Sc2NTime', 'Sc3NTime'],
        product_columns=['Sc1NType', 'Sc2NType', 'Sc3NType'],
        ap_method_columns=['Sc1NApMthd', 'Sc2NApMthd', 'Sc3NApMthd']),
    TripColumnSet(
        'Sulf',
        rate_columns=['SulfRate', 'Sulf2Rate'],
        product_columns=['SulfProd', 'Sulf2Prod'],
        ap_method_columns=['Sulf2ApMthd', 'SulfApMthd'],
        date_columns=['Sulf2Time', 'SulfTime']),
    TripColumnSet(
        'SulfN',
        rate_columns=['SulfNRate', 'SulfN2Rate'])
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
                           null_values=None) -> pd.DataFrame:
    # NOTE: these columns are from the old SQL tables, probably should update this somehow at some point?
    if null_values is None:
        null_values = ['', 'none', 'nan', 'null', None]

    ###
    #  encode existence only columns (true, false)
    ###
    for src_cols in existence_columns:
        print(f"encode_not_null: {src_cols}")
        df = encode_not_null(df, src_cols)
    for c in existence_column_sets.items():
        print(f"encoding existence for col set: {c}")
        df = encode_not_null(df, c)

    ###
    # encode trip columns
    ###
    for trip_col in trip_count_columns:
        print(f"encoding trip column: {trip_col.name}")
        trip_col.encode_df(df)

    ###
    # encode weather
    ###
    gis_pps.encode_plnt_wk_weather(df)
    
    ###
    # prev KRem, PRem (basically a proxy for prev yield)
    ###
    df['KRem_Prev'] = df['KRem2Yrs'] - df['KRem']
    df['PRem_Prev'] = df['PRem2Yrs'] - df['PRem']

    ###
    # encode date columns to year bi-weekly number (0 - ~26)
    ###
    date_src_cols = [c for c in df.columns if re.search(r'[aA]p(p)?[dD]ate', c)]
    for src_col in date_src_cols:
        print(f"encode_date_to_biweek: {src_col}")
        df = encode_date_to_biweek(df, src_col)

    ###
    # drop unused
    ###
    print(f"LOG: dropping exclude and yield dep columns: {exclude_columns + yield_dep_columns}")
    df.drop(exclude_columns + yield_dep_columns, axis=1, inplace=True, errors='ignore')

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
