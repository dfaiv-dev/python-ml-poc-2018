###
# 2018-01-16
#
# remote exec with:
# bash run_docker_image.sh -d ./ -s h2o_stacked -- \
#           python -u modeling/scripts/h2o_stacked.py --run-id h2o_stacked
###

import argparse
import pickle
import os
import re
import sys
from datetime import datetime
from typing import Union

import h2o.automl
import numpy as np
import pandas
import h2o.estimators
from h2o.estimators import H2OGradientBoostingEstimator, H2OXGBoostEstimator
from h2o.grid import H2OGridSearch
from pandas import DataFrame

from data import gis_pps
from util import mem_util

try:
    cwd = os.path.dirname(os.path.realpath(__file__))
except NameError:
    cwd = os.getcwd()

training_data_dir = os.path.join(cwd, 'training_data/dumps/gis_pps_corn_grps')
training_data_name = 'gis_pps_cells_corn_2016__grp80158_sample_400k'
output_dir = os.path.join(cwd, 'results/_scratch')
temp_dir = os.path.join(cwd, '_temp')
run_id = 'dev'
is_debug = True

# mem_util.disable_print_mem()
mem_util.enable_print_mem()

if __name__ == '__main__' and not "pydevconsole" in sys.argv[0] and not '--debug' in sys.argv[1]:
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', '-r', type=str, default=f'{datetime.utcnow():%Y-%m-%dT%H-%M-%S}')
    opt = parser.parse_args()
    run_id = opt.run_id

    training_data_dir = '/var/opt/pcsml/training-data/h2o/gis_pps_aggr_corn__20171018'
    training_data_name = 'gis_pps_aggr_corn'
    output_dir = '/var/opt/pcsml/remote-exec-out'
    temp_dir = '/var/opt/pcsml/tmp'
    h2o_url = 'http://localhost:54321'
    h2o_max_mem = '63g'
    stop_tol = .1
    max_run_time_sec = 60 * 30  # 30 min per grid search
    max_models = 15
    max_trees = 500
    n_folds = 3
    stack_top_n_grid_results = 5
    is_debug = False

    mem_util.enable_print_mem()

result_dir = os.path.join(output_dir, run_id)
print(f"result_dir: {result_dir}")
os.makedirs(result_dir, exist_ok=True)

print(f"temp_dir: {temp_dir}")
os.makedirs(temp_dir, exist_ok=True)

print(f"opts: ")
print({
    'training_data_dir': training_data_dir,
    'training_data_name': training_data_name,
    'output_dir': output_dir,
    'temp_dir': temp_dir,
    'result_dir': result_dir,
    'pwd': os.getcwd()
})


def _get_encoded_file_name(file_suffix: str, sep='__') -> str:
    return f"{training_data_name}{sep}{file_suffix}"


def _get_result_file_path(file_suffix: str, sep='__') -> str:
    return os.path.join(
        result_dir,
        _get_encoded_file_name(file_suffix, sep))


def _get_training_data_path(file_suffix: str, sep='__') -> str:
    return os.path.join(
        training_data_dir,
        _get_encoded_file_name(file_suffix, sep))


df_path = _get_training_data_path('df.pickle.gz')
print(f"reading pandas df: {df_path}")

df: pandas.DataFrame = pandas.read_pickle(df_path)
print("Dropping FieldUID and Area")
df.drop(['FieldUID', 'Area'], axis=1, inplace=True, errors='ignore')

print("na perc calculation")
na_percents = df.isnull().sum() / df.shape[0]
for c in df.columns:
    if na_percents[c] > .9:
        print(f"dropping na column: {c}, {na_percents[c]}")
        df.drop(c, inplace=True, axis=1)
