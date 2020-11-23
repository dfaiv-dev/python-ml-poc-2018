###
# 2018-01-07
#
# Extract a number of rows from a text file.
#
# remote exec with:
# bash run_docker_image.sh -d ./ -s gis_pps_raw_csv_encoder__grp80158 -- \
#           python -u data/scripts/gis_pps_cells/gis_pps_cells_raw_csv_encoder.py --run-id gis_pps_raw_csv_encoder__grp80158
#
# NOTE: this has memory issues on a raw 25GB csv (~13m rows) on a 128GB machine.
#   look at the mem_constrained versions if you need that size
###

import argparse
import gzip
import os
import sys
from datetime import datetime
import re
from typing import List, Union

import gc
import numpy as np
import pandas
import pickle

import shutil
from pandas import DataFrame
from scipy import sparse
import xgboost as xgb

import data.pcsml_data_loader as dl
from data import gis_pps
from util import mem_util
from modeling import categorical_util, xgb_util

raw_csv_path = '/var/opt/pcsml/devel/training_data/dumps/racer_gis_pps_corn_2016__20180106.txt.gz'
output_dir = '/var/opt/pcsml/devel/results/_scratch'
temp_dir = '/var/opt/pcsml/devel/_temp'
run_id = 'dev'
sample_n: Union[int, None] = 400_000
csv_sep = '\t'
# mem_util.disable_print_mem()
mem_util.enable_print_mem()

if __name__ == '__main__' and not "pydevconsole" in sys.argv[0] and not '--debug' in sys.argv[1]:
    parser = argparse.ArgumentParser()
    # racer_gis_pps_corn_2016_smpl_2k__20180106.txt.gz
    # gis_pps_corn_2016__20180112.zip
    parser.add_argument('--run-id', '-r', type=str, default=f'{datetime.utcnow():%Y-%m-%dT%H-%M-%S}')
    opt = parser.parse_args()
    run_id = opt.run_id

    raw_csv_path = '/var/opt/pcsml/training-data/gis-pps/gis_pps_corn_grp_80158__20180116.zip'
    output_dir = '/var/opt/pcsml/remote-exec-out'
    temp_dir = '/opt/pcsml/py-scikit-spike/_tmp'
    mem_util.enable_print_mem()
    sample_n = None

result_dir = os.path.join(output_dir, run_id)
print(f"result_dir: {result_dir}")

os.makedirs(result_dir, exist_ok=True)

# ensure temp dir for storing local copy of data
shutil.rmtree(temp_dir, ignore_errors=True)
os.makedirs(temp_dir, exist_ok=True)

encoded_file_base_name = 'gis_pps_cells_corn_2016__grp80158'
if sample_n is not None and sample_n > 0:
    encoded_file_base_name += f"_sample_{sample_n/1000:.0f}k"

print(f"opts: ")
print({
    'raw_csv_path': raw_csv_path,
    'encoded_file_base_name': encoded_file_base_name,
    'output_dir': output_dir,
    'temp_dir': temp_dir,
    'result_dir': result_dir,
    'pwd': os.getcwd(),
    'sample_n': sample_n
})


def _get_encoded_file_name(file_suffix: str, sep='__') -> str:
    return f"{encoded_file_base_name}{sep}{file_suffix}"


def _get_result_file_path(file_suffix: str, sep='__') -> str:
    return os.path.join(
        result_dir,
        _get_encoded_file_name(file_suffix, sep))


def _get_temp_data_path(file_suffix: str, sep='__') -> str:
    return os.path.join(
        temp_dir,
        _get_encoded_file_name(file_suffix, sep)
    )


mem_util.print_mem_usage()

print(f"reading in training data, may take a while: {raw_csv_path}")
if sample_n is not None and sample_n > 0:
    data: DataFrame = pandas.read_csv(
        raw_csv_path,
        sep=csv_sep,
        nrows=sample_n)
else:
    print("READING ENTIRE FILE")
    data: DataFrame = pandas.read_csv(
        raw_csv_path,
        sep=csv_sep)

# make our indexes for all encoded files start at 0 so we can index into non-pandas versions
data.reset_index(drop=True, inplace=True)

dataset = gis_pps.clean_shape_encode_sep(data)
print("saving gis pps datasets")
dataset.pickle_all(_get_result_file_path)

dmatrix = xgb_util.dmatrix_from_gis_pps(dataset.numeric, dataset.dummies, dataset.dry_yield)
dmatrix_temp_path = _get_temp_data_path('.dmatrix', sep='')
print(f"saving xgb DMatrix: {dmatrix_temp_path}")
dmatrix.save_binary(dmatrix_temp_path, silent=False)

dmatrix_gz_path = _get_result_file_path('.dmatrix.gz', sep='')
print(f"compressing xgb matrix: {dmatrix_gz_path}")
with open(dmatrix_temp_path, 'rb') as f:
    with gzip.open(dmatrix_gz_path, 'wb') as gz:
        shutil.copyfileobj(f, gz)

print(f"deleting uncompressed dmatrix: {dmatrix_temp_path}")
os.remove(dmatrix_temp_path)

# print("Read raw csv, writing out column headers and stats")
# data.dtypes.to_csv(_get_result_file_path('raw_column_dtypes.csv'), header=True)
#
# print(f"raw csv df size: {sys.getsizeof(data) / 1024 / 1024:.0f} MB")
# print(f"year ids: {len(data['YearID'].unique())}")
#
# # drop $ columns
# cost_columns = [c for c in data.columns if '$' in c.lower()]
# cost_columns += [c for c in data.columns if 'cost' in c.lower()]
# print(f"found {len(cost_columns)} cost columns.  Dropping.  Head: {sorted(cost_columns)[:25]}...")
# _drop_columns(data, cost_columns)
#
# # drop unused weather
# re_wx_col_num = re.compile('^WK(\d\d?)(Gdd|Rain)$', re.IGNORECASE)
# wx_columns = [c for c in data.columns if re_wx_col_num.search(c)]
# valid_wx_weeks = list(range(14, 40))
# invalid_wx_columns = [
#     c for c in wx_columns
#     if int(re_wx_col_num.match(c).group(1)) not in valid_wx_weeks
# ]
# print(f"dropping unused wx columns: {invalid_wx_columns[:20]}...")
# _drop_columns(data, invalid_wx_columns)
#
# # drop known unused
# known_unused_cols = [
#     'FieldID',
#     'CropYear',
#     'ShapeIndex',
#     'GridRow',
#     'GridColumn',
#     'ProcessedLayerUID',
#     'ShapeX',
#     'ShapeY'
# ]
# print(f"dropping known unused columns")
# _drop_columns(data, known_unused_cols)
#
# # encode
# data = dl.gis_pps_encode_raw_csv(data)
#
# print("saving col csv")
# data.dtypes.to_csv(_get_result_file_path('unencoded_column_dtypes.csv'), header=True)
#
# # split categorical columns, since we always need to treat them differently
# print(f"getting column category lookup")
# column_categories = categorical_util.get_categories_lookup(data)
# column_cat_path = _get_result_file_path('column_categories.pickle.gz')
# print(f"saving.... {column_cat_path}")
# with gzip.open(column_cat_path, 'wb') as gz:
#     pickle.dump(column_categories, gz)
#
# df_cat = data.select_dtypes(include='category')
# data.drop(labels=df_cat.columns, axis=1, inplace=True)
# df_cat_path = _get_result_file_path('df_categorical.pickle.gz')
# print(f"saving category columns dataframe... ")
# df_cat.to_pickle(df_cat_path, compression='gzip')
#
# print("getting dummies, this can take a very long time...")
# data_dummies = pandas.get_dummies(
#     df_cat, prefix_sep='__DUMMY__', sparse=True, drop_first=True, dummy_na=True)
# df_dummies_path = _get_result_file_path('df_dummies.pickle.gz')
# print(f"saving dummies... {df_dummies_path}")
# data_dummies.to_pickle(df_dummies_path, compression='gzip')
# dummy_columns = data_dummies.columns
# del df_cat
#
# print("extracting and saving meta columns")
# data_areas: pandas.Series = data.pop('Area')
# data_areas.to_pickle(_get_result_file_path('df_areas.pickle.gz'), compression='gzip')
# data_year_ids: pandas.Series = data.pop('YearID')
# data_year_ids.to_pickle(_get_result_file_path('df_year_ids.pickle.gz'), compression='gzip')
# data_labels: pandas.Series = data.pop('Dry_Yield')
# data_labels.to_pickle(_get_result_file_path('df_dry_yield.pickle.gz'), compression='gzip')
#
# print("dropping and saving any excluded columns")
# data_excluded: pandas.DataFrame = \
#     data.loc[:, data.columns.intersection(dl.exclude_columns + dl.yield_dep_columns)]
# data_excluded.to_pickle(_get_result_file_path('df_excluded.pickle.gz'))
# data_excluded_columns = data_excluded.columns
# del data_excluded
#
# print("saving numeric columns df")
# _drop_columns(data, data_excluded_columns)
# data.to_pickle(_get_result_file_path('df_numeric.pickle.gz'), compression='gzip')
# print("saving numeric+cat column names")
#
# data_feature_names = data.columns.append(dummy_columns).values
# pandas.Series(data=data_feature_names).to_pickle(_get_result_file_path('columns.pickle.gz'), compression='gzip')
# pandas.Series(data=data_feature_names).to_csv(_get_result_file_path('columns.csv'), index=True)
#
# print("converting numeric training data to sparse df")
# data: pandas.SparseDataFrame = data.to_sparse(fill_value=0)
# mem_util.print_mem_usage()
#
# print("combining sparse numeric with sparse dummies (as data frame)")
# data: pandas.SparseDataFrame = data.join(data_dummies)
# mem_util.print_mem_usage()
#
# print("converting to float")
# data: pandas.SparseDataFrame = data.astype('float32')
# mem_util.print_mem_usage()
#
# print("converting to coo matrix")
# data: sparse.coo_matrix = data.to_coo()
# mem_util.print_mem_usage()
#
# coo_matrix_path = _get_temp_data_path("coo.npz")
# print(f"saving coo data, temp: {coo_matrix_path}")
# sparse.save_npz(coo_matrix_path, data, compressed=True)
#
# coo_matrix_gz_path = _get_result_file_path("coo.npz.gz")
# print(f"gzip and saving coo: {coo_matrix_gz_path}")
# with open(coo_matrix_path, 'rb') as f:
#     with gzip.open(coo_matrix_gz_path, 'wb') as gz:
#         shutil.copyfileobj(f, gz)
#
# os.remove(coo_matrix_path)
#
# print("converting to csr")
# data = data.tocsr()
# mem_util.print_mem_usage()
#
# csr_matrix_temp_path = _get_temp_data_path("csr.npz")
# print(f"saving uncompressed csr matrix, temp: {csr_matrix_temp_path}")
# sparse.save_npz(csr_matrix_temp_path, data, compressed=True)
#
# csr_matrix_gz_path = _get_result_file_path("csr.npz.gz")
# print(f"gzip csr to: {csr_matrix_gz_path}")
# with gzip.open(csr_matrix_gz_path, 'wb') as gz:
#     with open(csr_matrix_temp_path, 'rb') as f:
#         shutil.copyfileobj(f, gz)
#
# print(f"deleting csr temp: {csr_matrix_temp_path}")
# os.remove(csr_matrix_temp_path)
#
# print("converting to xgb DMatrix")
# data: xgb.DMatrix = xgb.DMatrix(data, data_labels, feature_names=data_feature_names)
# mem_util.print_mem_usage()
#
# dmatrix_temp_path = _get_temp_data_path('.dmatrix', sep='')
# print(f"saving xgb DMatrix: {dmatrix_temp_path}")
# data.save_binary(dmatrix_temp_path, silent=False)
#
# dmatrix_gz_path = _get_result_file_path('.dmatrix.gz', sep='')
# print(f"compressing xgb matrix: {dmatrix_gz_path}")
# with open(dmatrix_temp_path, 'rb') as f:
#     with gzip.open(dmatrix_gz_path, 'wb') as gz:
#         shutil.copyfileobj(f, gz)
#
# print(f"deleting uncompressed dmatrix: {dmatrix_temp_path}")
# os.remove(dmatrix_temp_path)
