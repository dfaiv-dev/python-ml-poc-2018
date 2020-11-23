import gzip
import pickle
import os
import sys
import math
import re
from datetime import datetime
from typing import List, Callable, Any, Optional, Union, Tuple, Dict

import dateutil.parser as dt_parser
import pandas
import numpy as np
import shutil

from h2o import H2OFrame
from scipy import sparse
import xgboost as xgb

from modeling import categorical_util
from util import mem_util


class TripColumnSet:
    def __init__(self,
                 name: str,
                 rate_columns: List[str] = None,
                 date_columns: List[str] = None,
                 product_columns: List[str] = None,
                 ap_method_columns: List[str] = None,
                 extra_columns: List[str] = None):
        self.name = name
        self.rate_columns = rate_columns if rate_columns is not None else []
        self.date_columns = date_columns if date_columns is not None else []
        self.ap_method_columns = ap_method_columns if ap_method_columns is not None else []
        self.product_columns = product_columns if product_columns is not None else []
        self.extra_columns = extra_columns if extra_columns is not None else []

    def all_columns(self) -> List[str]:
        return self.rate_columns + \
               self.date_columns + \
               self.ap_method_columns + \
               self.product_columns + \
               self.extra_columns

    def encode_df(self, data: pandas.DataFrame, remove_encoded=True):
        has_rate_columns_ = len(self.rate_columns) > 0
        if has_rate_columns_:
            tot_rate_col = self.name + "_TotalRate"
            data[tot_rate_col] = data.loc[:, self.rate_columns].sum(axis=1)

        # get count of existence
        exist_cols_ = self.product_columns

        if len(exist_cols_) == 0:
            print(f"LOG: will not create COUNT col for trip col: {self.name}.  No product columns found.")
        else:
            data[self.name + '_Count'] = (~data.loc[:, exist_cols_].isna()).sum(axis=1)

        if remove_encoded:
            print(f"LOG: dropping trip columns: {self.all_columns()}")
            data.drop(labels=self.all_columns(), axis=1, inplace=True, errors='ignore')

    def encode_h2o(self, data: H2OFrame) -> H2OFrame:
        has_rate_columns_ = len(self.rate_columns) > 0
        if has_rate_columns_:
            tot_rate_col = self.name + "_TotalRate"
            data[tot_rate_col] = data[:, self.rate_columns].sum(axis=1)

        # get count of existence
        exist_cols_ = self.product_columns

        if len(exist_cols_) == 0:
            print(f"LOG: will not create COUNT col for trip col: {self.name}.  No product columns found.")
        else:
            data[self.name + '_Count'] = (~data[:, exist_cols_].isna()).sum(axis=1)

        return data

    def __str__(self):
        return f"{self.name} -- rates: {self.rate_columns}, products: {self.product_columns}"

    def __repr__(self):
        return self.__str__()


class GisPpsDataSet:
    def __init__(self,
                 numeric: pandas.DataFrame,
                 dummies: pandas.SparseDataFrame,
                 dry_yield: pandas.Series,
                 year_ids: pandas.Series,
                 areas: pandas.Series,
                 column_categories: Dict[str, pandas.Categorical]):
        self.areas = areas
        self.column_categories = column_categories
        self.dry_yield = dry_yield
        self.year_ids = year_ids
        self.numeric = numeric
        self.dummies = dummies

    def pickle_all(self,
                   result_file_path_factory: Callable[[str, str], str],
                   compression='gzip'):
        def _get_result_file_path(file_suffix: str, sep='__') -> str:
            return result_file_path_factory(file_suffix, sep)

        def _save_pickle(pd_: Union[pandas.DataFrame, pandas.Series], suffix):
            path = _get_result_file_path(suffix)
            print(f"saving {suffix}: {path}")
            pd_.to_pickle(path, compression=compression)

        _save_pickle(self.areas, 'df_areas.pickle.gz')
        _save_pickle(self.dry_yield, 'df_dry_yield.pickle.gz')
        _save_pickle(self.year_ids, 'df_year_ids.pickle.gz')
        _save_pickle(self.numeric, 'df_numeric.pickle.gz')
        _save_pickle(self.dummies, 'df_dummies.pickle.gz')
        _save_pickle(
            pandas.Series(self.numeric.columns.append(self.dummies.columns)), 'columns.pickle.gz')

        column_cat_path = _get_result_file_path('column_categories.pickle.gz')
        print(f"saving.... {column_cat_path}")
        with gzip.open(column_cat_path, 'wb') as gz:
            pickle.dump(self.column_categories, gz)


class GisPpsLoader:
    def __init__(self,
                 base_name: str,
                 training_dir: str,
                 temp_dir: str):
        self.training_dir = training_dir
        self.temp_dir = temp_dir
        self.base_name = base_name

    def get_file_name(self, file_suffix, sep='__') -> str:
        return f"{self.base_name}{sep}{file_suffix}"

    def get_temp_file_path(self, file_suffix, sep='__') -> str:
        file_name = self.get_file_name(file_suffix, sep)
        return os.path.join(self.temp_dir, file_name)

    def ensure_copy_local(self, file_suffix, sep='__') -> str:
        temp_file_path = self.get_temp_file_path(file_suffix, sep)
        if os.path.isfile(temp_file_path):
            return temp_file_path

        # else, try and copy from training data dir
        src_path = os.path.join(self.training_dir, self.get_file_name(file_suffix, sep))
        print(f"copying data file to temp dir: {src_path}, {temp_file_path}")
        shutil.copy2(src_path, temp_file_path)

        return temp_file_path

    def year_ids(self) -> pandas.Series:
        return pandas.read_pickle(self.ensure_copy_local('df_year_ids.pickle.gz'))

    def feature_names(self) -> pandas.Series:
        return pandas.read_pickle(self.ensure_copy_local('columns.pickle.gz'))

    def dry_yield(self) -> pandas.Series:
        return pandas.read_pickle(self.ensure_copy_local('df_dry_yield.pickle.gz'))

    def numeric(self) -> pandas.DataFrame:
        return pandas.read_pickle(self.ensure_copy_local('df_numeric.pickle.gz'))

    def dummies(self) -> pandas.SparseDataFrame:
        return pandas.read_pickle(self.ensure_copy_local('df_dummies.pickle.gz'))

    def column_categories(self) -> Dict[str, pandas.Categorical]:
        path = self.ensure_copy_local('column_categories.pickle.gz')
        with gzip.open(path, 'rb') as gz:
            return pickle.load(gz)

    def areas(self) -> pandas.Series:
        return pandas.read_pickle(self.ensure_copy_local('areas.pickle.gz'))


meta_columns = ['YearID', 'FieldUID', 'Area']
exclude_columns = [
    'Year', 'YearUID', 'UID', 'FieldID',
    'CreateDate', 'LastUpdated', 'HarvDate', 'GridRow',
    'FieldID', 'CropYear', 'ShapeIndex',
    'GridRow', 'GridColumn', 'ProcessedLayerUID', 'ShapeX', 'ShapeY', 'GridColumn', 'PlntDate',
    'ProcessedLayerUid', 'ProcessedLayerLastUpdated', 'ImportBatchId'
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
        date_columns=[
            'Fung1ApDate_biweek', 'Fung1ApDate', 'Fung2ApDate_biweek', 'Fung2ApDate',
            'Fung3ApDate_biweek', 'Fung3ApDate'],
        ap_method_columns=['Fung1ApMthd', 'Fung2ApMthd', 'Fung3ApMthd'],
        product_columns=['Fungicide1', 'Fungicide2', 'Fungicide3']),
    TripColumnSet(
        'Isc',
        rate_columns=['Isc1Rate', 'Isc2Rate', 'Isc3Rate'],
        date_columns=[
            'Isc1ApDate_biweek', 'Isc2ApDate_biweek', 'Isc3ApDate_biweek',
            'Isc1ApDate', 'Isc2ApDate', 'Isc3ApDate'
        ],
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
        date_columns=[
            'PreAppDate_biweek', 'Pre6AppDate_biweek',
            'PreAppDate', 'Pre6AppDate'
        ],
        product_columns=['Pre1Name', 'Pre2Name', 'Pre3Name', 'Pre6Name', 'Pre7Name', 'Pre8Name'],
        ap_method_columns=['PreMethod', 'Pre6Method']),
    TripColumnSet(
        'Pst',
        rate_columns=[
            'Pst1Rate', 'Pst2Rate', 'Pst30Rate', 'Pst31Rate', 'Pst32Rate', 'Pst33Rate', 'Pst3Rate',
            'Pst40Rate', 'Pst41Rate', 'Pst42Rate', 'Pst43Rate', 'Pst4Rate',
            'Pst5Rate', 'Pst6Rate', 'Pst7Rate', 'Pst8Rate'],
        date_columns=[
            'Pst1ApDate_biweek', 'Pst2ApDate_biweek', 'Pst30ApDate_biweek', 'Pst40ApDate_biweek',
            'Pst1ApDate', 'Pst2ApDate', 'Pst30ApDate', 'Pst40ApDate'],
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


def encode_not_null(
        df: pandas.DataFrame,
        source_columns: Union[Tuple[str, List[str]], str],
        drop=True) -> pandas.DataFrame:
    """
    Encodes a column to a new category column as 0, 1: as either having a value or not having a value.
    Modifies the DF inplace.

    :param source_columns:
    :param drop:
    :param df:
    :return:
    """

    if isinstance(source_columns, str):
        src_name = source_columns
        src_cols = np.array([source_columns])
    else:
        src_name = source_columns[0]
        src_cols = np.array(source_columns[1])

    exists_mask = np.isin(src_cols, df.columns)
    if not exists_mask.any():
        print(f"LOG: no src cols exist: {src_cols}")
        return df

    src_cols = src_cols[exists_mask]
    cols: pandas.DataFrame = df.loc[:, src_cols]
    exists_column_name = src_name + "_exists"
    df[exists_column_name] = (~cols.isnull()).all(axis=1)
    if drop:
        df.drop(src_cols, axis=1, inplace=True)

    return df


def encode_date_to_biweek(X: pandas.DataFrame, src_col: str, drop=True) -> pandas.DataFrame:
    """
    Encodes a string column to bi-week number ranges category column.
    Modifies DF inplace.
    Any unparsable values will be left as is.

    :param X:
    :param src_col:
    :param drop:
    :return:
    """
    encoded_col_name = src_col + "_biweek"

    def _encode(val: pandas.Series):
        # https://stackoverflow.com/a/29882676/79113
        # use caching so we don't encode multiples
        encoded = {
            v: _encode_date(v, lambda d: math.ceil(d.timetuple().tm_yday / 14))
            for v in val.fillna('').unique()
        }

        return val.map(encoded)

    X[encoded_col_name] = _encode(X[src_col])
    if drop:
        X.drop([src_col], axis=1, inplace=True)

    return X


def encode_plnt_wk_weather(gis_pps_df: pandas.DataFrame, drop_wk_wx_cols=True):
    # encode plnt week from day
    gis_pps_df['PlntWk'] = (gis_pps_df['PlntDate#'] / 7).round()

    # rename weather columns for consistent casing
    wx_col_re = re.compile(r'^wk\d+(gdd|rain)$', re.IGNORECASE)
    gis_pps_df.rename(
        lambda _c: str(_c).lower() if wx_col_re.match(_c) is not None else _c,
        axis='columns',
        inplace=True)

    # get weather by week from plant date #
    # use gdd and rain specific dfs for masking what the source col is for each row
    rain_df = gis_pps_df[sorted([c for c in gis_pps_df.columns if re.match(r'wk\d+rain', c) is not None])]
    gdd_df = gis_pps_df[sorted([c for c in gis_pps_df.columns if re.match(r'wk\d+gdd', c) is not None])]
    plnt_wk = gis_pps_df['PlntWk']
    plnt_wk = plnt_wk[~plnt_wk.isna()].astype('int')
    rain_df = rain_df.loc[plnt_wk.index]
    gdd_df = gdd_df.loc[plnt_wk.index]

    wx_agg_range = range(0, 20)
    for wx_agg_wk in wx_agg_range:
        print(f"mapping wx for plntwk: {wx_agg_wk}")

        wk_offset = plnt_wk - 14 + wx_agg_wk
        wk_offset[wk_offset < 0] = 0
        wk_offset[wk_offset >= rain_df.shape[1]] = rain_df.shape[1] - 1
        mask = np.zeros(rain_df.shape).astype('bool')
        mask[(np.arange(len(wk_offset)), wk_offset)] = True

        wx_agg_wk_token = str(wx_agg_wk).rjust(2, '0')
        gis_pps_df[f"PlntWk{wx_agg_wk_token}Rain"] = rain_df.where(mask).sum(axis=1)
        gis_pps_df[f"PlntWk{wx_agg_wk_token}Gdd"] = gdd_df.where(mask).sum(axis=1)

        if wx_agg_wk == 0:
            continue

        # create aggregates
        agg_cols = [f"PlntWk{str(i).rjust(2, '0')}" for i in range(0, wx_agg_wk + 1)]

        agg_cols_rain = [f"{col}Rain" for col in agg_cols]
        gis_pps_df[f"PlntWk{wx_agg_wk_token}TotalRain"] = gis_pps_df[agg_cols_rain].sum(axis=1, min_count=1)

        agg_cols_gdd = [f"{col}Gdd" for col in agg_cols]
        gis_pps_df[f"PlntWk{wx_agg_wk_token}TotalGdd"] = gis_pps_df[agg_cols_gdd].sum(axis=1, min_count=1)

    if drop_wk_wx_cols:
        print(f"dropping wk rain cols: {rain_df.columns}")
        gis_pps_df.drop(rain_df.columns, axis=1, inplace=True)
        print(f"dropping wk gdd cols: {gdd_df.columns}")
        gis_pps_df.drop(gdd_df.columns, axis=1, inplace=True)


def transform_to_np_csr(
        data_numeric: pandas.DataFrame,
        data_dummies: pandas.SparseDataFrame) -> sparse.csr_matrix:
    data = data_numeric

    ###
    # np sparse matrices
    ###
    print("converting numeric training data to sparse df")
    data: pandas.SparseDataFrame = data.to_sparse(fill_value=0)
    mem_util.print_mem_usage()

    print("combining sparse numeric with sparse dummies (as data frame)")
    data: pandas.SparseDataFrame = data.join(data_dummies)
    mem_util.print_mem_usage()

    print("converting to float")
    data: pandas.SparseDataFrame = data.astype('float32')
    mem_util.print_mem_usage()

    print("converting to coo matrix")
    data: sparse.coo_matrix = data.to_coo()
    mem_util.print_mem_usage()

    print("converting to csr")
    data: sparse.csr_matrix = data.tocsr()
    mem_util.print_mem_usage()

    return data


def fillna_numeric_with_mean(data: pandas.DataFrame):
    cols = set(data.select_dtypes(include=np.number).columns.values)
    cols = cols - set(exclude_columns + yield_dep_columns)
    for c in cols:
        print(f"LOG: filling numeric col NaN: {c}")
        data[c].fillna(data[c].mean(), inplace=True)


def clean(data: pandas.DataFrame):
    # clean up some column names
    rename_cols = {
        'YearId': 'YearID'
    }
    for c in rename_cols:
        if c in data.columns:
            data[rename_cols[c]] = data[c]
            data.drop(c, axis=1, inplace=True)

    # drop $ columns
    cost_columns = [c for c in data.columns if '$' in c.lower()]
    cost_columns += [c for c in data.columns if 'cost' in c.lower()]
    print(f"found {len(cost_columns)} cost columns.  Dropping.  Head: {sorted(cost_columns)[:25]}...")
    drop_columns(data, cost_columns)

    # drop unused weather
    re_wx_col_num = re.compile('^WK(\d\d?)(Gdd|Rain)$', re.IGNORECASE)
    wx_columns = [c for c in data.columns if re_wx_col_num.search(c)]
    valid_wx_weeks = list(range(14, 40))
    invalid_wx_columns = [
        c for c in wx_columns
        if int(re_wx_col_num.match(c).group(1)) not in valid_wx_weeks
    ]
    print(f"dropping unused wx columns: {invalid_wx_columns[:20]}...")
    drop_columns(data, invalid_wx_columns)


def shape(data: pandas.DataFrame, encode_app_date_bi_weeks=False, drop_shaped_cols=False):

    if encode_app_date_bi_weeks:
        ###
        # encode date columns to year bi-weekly number (0 - ~26)
        ###
        date_src_cols = [c for c in data.columns if re.search(r'[aA]p(p)?[dD]ate', c)]
        for src_col in date_src_cols:
            print(f"encode_date_to_biweek: {src_col}")
            encode_date_to_biweek(data, src_col, drop=drop_shaped_cols)

    ###
    #  encode existence only columns (true, false)
    ###
    for src_cols in existence_columns:
        print(f"encode_not_null: {src_cols}")
        encode_not_null(data, src_cols, drop=drop_shaped_cols)
    for c in existence_column_sets.items():
        print(f"encoding existence for col set: {c}")
        encode_not_null(data, c, drop=drop_shaped_cols)

    ###
    # encode trip columns
    ###
    for trip_col in trip_count_columns:
        print(f"encoding trip column: {trip_col.name}")
        trip_col.encode_df(data, remove_encoded=drop_shaped_cols)

    ###
    # encode weather
    ###
    encode_plnt_wk_weather(data, drop_wk_wx_cols=drop_shaped_cols)

    ###
    # prev KRem, PRem (basically a proxy for prev yield)
    ###
    _add_krem_prev(data, drop_src_cols=drop_shaped_cols)
    _add_prev_prem(data, drop_src_cols=drop_shaped_cols)


def clean_shape_encode_sep(
        data: pandas.DataFrame,
        null_values: List[str] = None) -> GisPpsDataSet:
    if null_values is None:
        null_values = ['', 'none', 'nan', 'null', None]

    clean(data)

    shape(data, drop_shaped_cols=True)

    # cleanup numeric
    print("LOG> filling na with mean")
    fillna_numeric_with_mean(data)
    print("LOG> dropping numeric outliers")
    data = remove_outliers(data)

    print("extracting and saving meta columns")
    data_areas: pandas.Series = data.pop('Area')

    data_year_ids: pandas.Series = data.pop('YearID')

    data_yield: pandas.Series = data.pop('Dry_Yield')

    excl_cols_all = exclude_columns + yield_dep_columns
    print(f"dropping excluded columns: {excl_cols_all}")
    drop_columns(data, excl_cols_all)

    ##
    # encode category columns
    ##
    label_cols = data.select_dtypes(include=['bool', 'object']).columns
    label_cols = [c for c in label_cols if c not in exclude_columns]
    for label_col in label_cols + numeric_label_columns:
        print(f"encoding category column: {label_col}")

        data[label_col] = data[label_col].astype(str).str.lower()
        data[label_col] = data[label_col].replace(null_values, np.nan)
        data[label_col] = data[label_col].astype('category')

    ###
    # dummy encoding
    ###
    print(f"getting column category lookup")
    column_categories = categorical_util.get_categories_lookup(data)

    print("getting dummies, this can take a very long time...")
    df_cat = data.select_dtypes(include='category')
    data.drop(labels=df_cat.columns, axis=1, inplace=True)
    data_dummies = pandas.get_dummies(
        df_cat, prefix_sep=categorical_util.DUMMY_COL_SEP, sparse=True)
    del df_cat

    return GisPpsDataSet(data, data_dummies, data_yield, data_year_ids, data_areas, column_categories)


def remove_outliers(
        data: pandas.DataFrame,
        quantile_min=.002, quantile_max=.998,
        max_targ_pop=80_000, reindex=True) -> pandas.DataFrame:
    # remove outliers
    # explicitly remove targ pop > X
    cols = set(data.select_dtypes(include=np.number).columns.values)
    cols = sorted(list(cols - set(exclude_columns + yield_dep_columns)))

    rows = data.shape[0]
    data = data[data['TargPop'] <= max_targ_pop]
    quantile_min = data[cols].quantile(quantile_min)
    quantile_max = data[cols].quantile(quantile_max)

    for c in cols:
        min_val = quantile_min[c]
        max_val = quantile_max[c]

        print(f"LOG removing outliers: {c} >>> min: {min_val}, max: {max_val} ")

        if np.any(np.isnan([min_val, max_val])):
            print("LOG min/max is NaN, skipping")
            continue

        rows_prev_ = data.shape[0]
        data = data[(data[c] >= min_val) & (data[c] <= max_val)]
        print(f"dropped: {rows_prev_ - data.shape[0]}")

    dropped_count = rows - data.shape[0]
    print(f"dropped outlier rows: {dropped_count}, {(dropped_count/rows) * 100:.1f}%")

    if reindex:
        print(f"reindexing after dropped rows")
        data.reset_index(drop=True, inplace=True)

    return data


def _encode_date(val, date_enc: Callable[[datetime], Any]) -> str:
    date = _try_parse_date(val)
    if date is None:
        val = str(val)
        return "_" + val + "_" if len(val) > 0 else val

    return str(date_enc(date))


def _try_parse_date(s: str) -> Optional[datetime]:
    if not isinstance(s, str):
        return None
    if re.match(r'^\s*\d+\s*$', s):
        return None

    try:
        return dt_parser.parse(s)
    except (ValueError, TypeError):
        return None


def _add_prev_prem(data: pandas.DataFrame, drop_src_cols=True):
    prem_df = data[data['PRem2Yrs'] > data['PRem']][['PRem', 'PRem2Yrs']]
    data['PRem_Prev'] = prem_df['PRem2Yrs'] - prem_df['PRem']

    if drop_src_cols:
        data.drop(['PRem2Yrs', 'PRem'], axis=1, inplace=True)


def _add_krem_prev(data: pandas.DataFrame, drop_src_cols=True):
    krem_df = data[data['KRem2Yrs'] > data['KRem']][['KRem', 'KRem2Yrs']]
    data['KRem_Prev'] = krem_df['KRem2Yrs'] - krem_df['KRem']

    if drop_src_cols:
        data.drop(['KRem2Yrs', 'KRem'], axis=1, inplace=True)


def drop_columns(_df: pandas.DataFrame, columns: List[str]):
    _df.drop(labels=columns, axis=1, inplace=True, errors='ignore')
