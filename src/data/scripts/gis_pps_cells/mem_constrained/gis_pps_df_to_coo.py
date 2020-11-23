###
# 2018-01-13
#
# Convert an already encoded numeric/dummies dataframe scipy coo matrix, save to disk.
#
# remote exec with:
# bash run_docker_image.sh -d ./ -s gis_pps_df_to_coo -- \
#           python -u data/scripts/gis_pps_df_to_coo.py --run-id gis_pps_df_to_coo
#
###

import argparse
import gzip
import os
import sys
from datetime import datetime
import re
from typing import List

import gc
import numpy as np
import pandas
import pickle

import shutil
from pandas import DataFrame
from scipy import sparse
import xgboost as xgb

import data.pcsml_data_loader as dl
from util import mem_util
from modeling import categorical_util

training_data_dir = '/var/opt/pcsml/devel/training_data/dumps/gis_pps_corn_2016_sample_50k'
training_data_name = 'gis_pps_corn_2016_sample_50k'
output_dir = '/var/opt/pcsml/devel/results/_scratch'
temp_dir = '/var/opt/pcsml/devel/_temp'
run_id = 'dev'
# mem_util.disable_print_mem()
mem_util.enable_print_mem()

if __name__ == '__main__' and not "pydevconsole" in sys.argv[0] and not '--debug' in sys.argv[1]:
    parser = argparse.ArgumentParser()
    # racer_gis_pps_corn_2016_smpl_2k__20180106.txt.gz
    # gis_pps_corn_2016__20180112.zip
    parser.add_argument('--run-id', '-r', type=str, default=f'{datetime.utcnow():%Y-%m-%dT%H-%M-%S}')
    opt = parser.parse_args()
    run_id = opt.run_id

    training_data_dir = '/var/opt/pcsml/training-data/gis-pps/gis_pps_corn_2016'
    training_data_name = 'gis_pps_corn_2016'
    output_dir = '/var/opt/pcsml/remote-exec-out'
    temp_dir = '/opt/pcsml/py-scikit-spike/_tmp'
    mem_util.enable_print_mem()

print(f"opts: ")
print({
    'training_data_dir': training_data_dir,
    'training_data_name': training_data_name,
    'output_dir': output_dir,
    'temp_dir': temp_dir,
    'pwd': os.getcwd()
})

result_dir = os.path.join(output_dir, run_id)
print(f"result_dir: {result_dir}")

os.makedirs(result_dir, exist_ok=True)
os.makedirs(temp_dir, exist_ok=True)


def _get_encoded_file_name(file_suffix: str, sep='__') -> str:
    return f"{training_data_name}{sep}{file_suffix}"


def _get_result_file_path(file_suffix: str, sep='__') -> str:
    return os.path.join(
        result_dir,
        _get_encoded_file_name(file_suffix, sep))


def _get_training_data_path(file_suffix: str, sep='__') -> str:
    return os.path.join(
        training_data_dir,
        _get_encoded_file_name(file_suffix, sep)
    )


def _get_temp_data_path(file_suffix: str, sep='__') -> str:
    return os.path.join(
        temp_dir,
        _get_encoded_file_name(file_suffix, sep)
    )


mem_util.print_mem_usage()

numeric_path = _get_training_data_path('df_numeric.pickle.gz')
print(f"loading numeric data frame: {numeric_path}")
data: pandas.DataFrame = pandas.read_pickle(numeric_path)
mem_util.print_mem_usage()

dummies_path = _get_training_data_path('df_dummies.pickle.gz')
print(f"loading dummies data frame: {dummies_path}")
data_dummies: pandas.SparseDataFrame = pandas.read_pickle(dummies_path)
mem_util.print_mem_usage()

labels_path = _get_training_data_path('df_dry_yield.pickle.gz')
print(f"loading labels: {labels_path}")
data_labels: pandas.Series = pandas.read_pickle(labels_path)
mem_util.print_mem_usage()

data_feature_names = data.columns.append(data_dummies.columns).values

###
# save DMatrix for xgboost
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
# data = data.to_coo().tocsr(copy=True)
coo: sparse.coo_matrix = data.to_coo()
mem_util.print_mem_usage()

coo_matrix_path = _get_temp_data_path("coo.npz")
print(f"saving coo data, temp: {coo_matrix_path}")
sparse.save_npz(coo_matrix_path, coo, compressed=True)

coo_matrix_gz_path = _get_result_file_path("coo.npz.gz")
print(f"gzip and saving coo: {coo_matrix_gz_path}")
with open(coo_matrix_path, 'rb') as f:
    with gzip.open(coo_matrix_gz_path, 'wb') as gz:
        shutil.copyfileobj(f, gz)
