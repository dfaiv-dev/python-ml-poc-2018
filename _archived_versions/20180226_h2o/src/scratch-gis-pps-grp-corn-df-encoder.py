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

# raw_csv_path = '/opt/project/training_data/dumps/racer_gis_pps_corn_2016__20180106.txt.gz'
raw_csv_path = '/opt/project/training_data/dumps/gis_pps_corn_grp_80158__20180116.zip'
output_dir = '/opt/project/results/_scratch'
temp_dir = '/opt/project/_temp'
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

gis_pps.clean(data)
gis_pps.shape(data, encode_app_date_bi_weeks=True, drop_shaped_cols=True)

excl_cols_all = gis_pps.exclude_columns + gis_pps.yield_dep_columns
gis_pps.drop_columns(data, excl_cols_all)

##
# encode category columns
##
label_cols = data.select_dtypes(include=['bool', 'object']).columns
label_cols = [c for c in label_cols if c not in (gis_pps.exclude_columns + gis_pps.meta_columns)]
null_values = ['', 'none', 'nan', 'null', None]
for label_col in label_cols + gis_pps.numeric_label_columns:
    print(f"encoding category column: {label_col}")

    data[label_col] = data[label_col].astype(str).str.lower()
    data[label_col] = data[label_col].replace(null_values, np.nan)
    data[label_col] = data[label_col].astype('category')

data.to_pickle(_get_result_file_path('df.pickle.gz'), compression='gzip')
