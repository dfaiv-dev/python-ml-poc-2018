###
# 2018-01-16
#
# remote exec with:
# bash run_docker_image.sh -d ./ -s csv_gis_pps_aggr_loader -- \
#           python -u data/scripts/csv_gis_pps_aggr_loader.py --run-id csv_gis_pps_aggr_loader
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

raw_path = '/var/opt/pcsml/devel/training_data/dumps/gis-pps-aggr/gis-pps-agg-20171018-corn-smpl-25__df.pkl'
output_dir = '/var/opt/pcsml/devel/results/_scratch'
temp_dir = '/var/opt/pcsml/devel/_temp'
run_id = 'dev'
sample_n: Union[int, None] = 100_000

# mem_util.disable_print_mem()
mem_util.enable_print_mem()

if __name__ == '__main__' and not "pydevconsole" in sys.argv[0] and not '--debug' in sys.argv[1]:
    parser = argparse.ArgumentParser()
    # racer_gis_pps_corn_2016_smpl_2k__20180106.txt.gz
    # gis_pps_corn_2016__20180112.zip
    parser.add_argument('--run-id', '-r', type=str, default=f'{datetime.utcnow():%Y-%m-%dT%H-%M-%S}')
    opt = parser.parse_args()
    run_id = opt.run_id

    raw_path = '/var/opt/pcsml/training-data/gis-pps/gis_pps_corn_2016__20180112.zip'
    training_data_dir = '/var/opt/pcsml/training-data/gis-pps/gis_pps_corn_2016'
    training_data_name = 'gis_pps_corn_2016'
    output_dir = '/var/opt/pcsml/remote-exec-out'
    temp_dir = '/opt/pcsml/py-scikit-spike/_tmp'
    mem_util.enable_print_mem()
    sample_n = None

result_dir = os.path.join(output_dir, run_id)
print(f"result_dir: {result_dir}")

os.makedirs(result_dir, exist_ok=True)
os.makedirs(temp_dir, exist_ok=True)

encoded_file_base_name = 'gis_pps_aggr_corn_2016'
if sample_n is not None and sample_n > 0:
    encoded_file_base_name += f"_sample_{sample_n/1000:.0f}k"

print(f"opts: ")
print({
    'raw_csv_path': raw_path,
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

print(f"reading in raw data, may take a while: {raw_path}")
data: DataFrame = pandas.read_pickle(raw_path)
if sample_n is not None and sample_n > 0:
    print(f"sampling: {sample_n}")
    data = data.sample(n=sample_n)

# make our indexes for all encoded files start at 0 so we can index into non-pandas versions
data.reset_index(drop=True, inplace=True)

print("cleaning...")
gis_pps.clean(data)
print("shaping...")
gis_pps.shape(data, encode_app_date_bi_weeks=True, drop_shaped_cols=True)

print(f"dropping excluded columns: {gis_pps.exclude_columns}")
gis_pps.drop_columns(data, gis_pps.exclude_columns)
print(f"dropping yield dep cols: {gis_pps.yield_dep_columns}")
gis_pps.drop_columns(data, gis_pps.yield_dep_columns)

df_csv_path = _get_result_file_path('df.csv.gz')
print(f"writing data csv: {df_csv_path}")
data.to_csv(
    df_csv_path,
    header=True, index=False, compression='gzip')

df_col_types_path = _get_result_file_path('df_cols.pickle')
print(f"writing col types: {df_col_types_path}")
with open(df_col_types_path, 'wb')as f:
    pickle.dump(data.dtypes, f)
