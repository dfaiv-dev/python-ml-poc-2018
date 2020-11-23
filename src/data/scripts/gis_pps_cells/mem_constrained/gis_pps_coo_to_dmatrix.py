###
# 2018-01-13
#
# Convert an already encoded scipy coo matrix to a xgb dmatrix
#
# remote exec with:
# bash run_docker_image.sh -d ./ -s gis_pps_coo_to_dmatrix -- \
#           python -u data/scripts/gis_pps_coo_to_dmatrix.py --run-id gis_pps_coo_to_dmatrix
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

training_data_dir = '/var/opt/pcsml/devel/training_data/dumps/gis_pps_corn_2016_sample_200k'
training_data_name = 'gis_pps_corn_2016_sample_200k'
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

coo_matrix_path = _get_training_data_path("coo.npz.gz")
print(f"loading coo data: {coo_matrix_path}")
with gzip.open(coo_matrix_path, 'rb') as gz:
    data: sparse.coo_matrix = sparse.load_npz(gz)
mem_util.print_mem_usage()

print("converting to csr")
data = data.tocsr()
mem_util.print_mem_usage()

csr_matrix_temp_path = _get_temp_data_path("csr.npz")
print(f"saving uncompressed csr matrix, temp: {csr_matrix_temp_path}")
sparse.save_npz(csr_matrix_temp_path, data, compressed=True)

csr_matrix_gz_path = _get_result_file_path("csr.npz.gz")
print(f"gzip csr to: {csr_matrix_gz_path}")
with gzip.open(csr_matrix_gz_path, 'wb') as gz:
    with open(csr_matrix_temp_path, 'rb') as f:
        shutil.copyfileobj(f, gz)

print(f"deleting csr temp: {csr_matrix_temp_path}")
os.remove(csr_matrix_temp_path)

labels_path = _get_training_data_path('df_dry_yield.pickle.gz')
print(f"loading labels: {labels_path}")
data_labels: pandas.Series = pandas.read_pickle(labels_path)
mem_util.print_mem_usage()

feature_names_path = _get_training_data_path('columns.pickle.gz')
print(f"loading feature names: {feature_names_path}")
data_feature_names: pandas.Series = pandas.read_pickle(feature_names_path)
mem_util.print_mem_usage()

print("converting to xgb DMatrix")
data: xgb.DMatrix = xgb.DMatrix(data, data_labels, feature_names=data_feature_names)
mem_util.print_mem_usage()

dmatrix_temp_path = _get_temp_data_path('.dmatrix', sep='')
print(f"saving xgb DMatrix: {dmatrix_temp_path}")
data.save_binary(dmatrix_temp_path, silent=False)

dmatrix_gz_path = _get_result_file_path('.dmatrix.gz', sep='')
print(f"compressing xgb matrix: {dmatrix_gz_path}")
with open(dmatrix_temp_path, 'rb') as f:
    with gzip.open(dmatrix_gz_path, 'wb') as gz:
        shutil.copyfileobj(f, gz)

print(f"deleting uncompressed dmatrix: {dmatrix_temp_path}")
os.remove(dmatrix_temp_path)
