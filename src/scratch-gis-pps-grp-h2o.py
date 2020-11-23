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
h2o_url = 'http://localhost:54321'
h2o_max_mem = '20g'
stop_tol = .1
max_run_time_sec = 30
max_models = 5
max_trees = 20
stack_top_n_grid_results = 3
n_folds = 3
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

# h2o.init(h2o_url)
h2o.init(url=h2o_url, max_mem_size=h2o_max_mem)
h2o.remove_all()

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
    'pwd': os.getcwd(),
    'n_folds': n_folds,
    'max_run_time_sec': max_run_time_sec,
    'stop_tol': stop_tol
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

col_types = {c: df[c].dtype for c in df.columns}
for ct in col_types:
    if col_types[ct] == np.number and ct not in gis_pps.numeric_label_columns:
        col_types[ct] = 'numeric'
    elif col_types[ct] == np.int and ct not in gis_pps.numeric_label_columns:
        col_types[ct] = 'numeric'
    elif ct not in gis_pps.exclude_columns:
        col_types[ct] = 'enum'
    else:
        col_types[ct] = 'string'
# manually set a couple columns
col_types_overrides = {
    'YearID': 'enum'
}
for cto in col_types_overrides:
    col_types[cto] = col_types_overrides[cto]

df_temp_parq_path = _get_result_file_path('df.parquet')
print(f"writing df to parquet for h2o import: {df_temp_parq_path}")
df.to_parquet(df_temp_parq_path)

df = h2o.import_file(df_temp_parq_path, col_types=col_types)

print(f"ensuring column types")
for c in col_types:
    c_type = col_types[c]
    prev_frame_id = df.frame_id
    if c_type == 'numeric' and not df[c].isnumeric()[0]:
        print(f"ensuring numeric: {c}")
        df[c] = df[c].asnumeric()
    elif c_type == 'enum' and not df[c].isfactor()[0]:
        print(f"ensuring enum: {c}, currently: {df.type(c)}")
        df[c] = df[c].ascharacter().asfactor()

    if prev_frame_id != df.frame_id:
        h2o.remove(prev_frame_id)

print("generating kfold col")
# df = h2o_frame_util.grouped_kfold(df, n_folds=n_folds, src_col_name='YearID')

print("exclude cols")
all_excl_cols = gis_pps.exclude_columns + gis_pps.yield_dep_columns
excl_cols = [c for c in df.col_names if c in all_excl_cols]
print(f"excl cols: {sorted(excl_cols)}")
train_cols = [c for c in df.col_names if c not in all_excl_cols]
print(f"train cols: {train_cols}")

# Search criteria
search_criteria = {
    'strategy': 'RandomDiscrete',
    'max_models': max_models,
    'max_runtime_secs': max_run_time_sec
}

# Grid search param options:
# http://docs.h2o.ai/h2o/latest-stable/h2o-docs/grid-search.html#xgboost-hyperparameters

gbm_params2 = {'learn_rate': [i * 0.01 for i in range(1, 14)],
               'max_depth': list(range(6, 11)),
               'sample_rate': [i * 0.1 for i in range(4, 11)],
               'col_sample_rate': [i * 0.1 for i in range(4, 11)]}

# Train and validate a random grid of GBMs
gbm_grid2 = H2OGridSearch(
    model=H2OGradientBoostingEstimator(
        ntrees=max_trees,
        stopping_rounds=3,
        stopping_tolerance=stop_tol,
        keep_cross_validation_predictions=True),
    hyper_params=gbm_params2,
    search_criteria=search_criteria)
print("grid searching gbm")
gbm_grid2.train(x=train_cols, y='Dry_Yield', training_frame=df, fold_column='YearID_KFold')
print(gbm_grid2)
gbm_best = gbm_grid2.get_grid(sort_by='mae').models[0]
print(gbm_best)
gbm_ids = [m.model_id for m in gbm_grid2.models[:stack_top_n_grid_results]]

# random forest
drf_params = {'max_depth': list(range(15, 30)),
              'sample_rate': [i * 0.1 for i in range(5, 11)],
              'col_sample_rate_per_tree': [i * 0.1 for i in range(5, 11)]}
drf_grid = H2OGridSearch(
    model=h2o.estimators.H2ORandomForestEstimator(
        ntrees=max_trees,
        stopping_rounds=3,
        stopping_tolerance=stop_tol,
        keep_cross_validation_predictions=True),
    hyper_params=drf_params,
    search_criteria=search_criteria)
print("grid search drf")
drf_grid.train(x=train_cols, y='Dry_Yield', training_frame=df, fold_column='YearID_KFold')
print(drf_grid)
drf_best = drf_grid.get_grid(sort_by='mae').models[0]
print(drf_best)
drf_ids = [m.model_id for m in drf_grid.models[:stack_top_n_grid_results]]

stack = h2o.estimators.H2OStackedEnsembleEstimator(
    metalearner_algorithm='gbm',
    metalearner_fold_column='YearID_KFold',
    base_models=gbm_ids + drf_ids
)
stack.train(training_frame=df, y='Dry_Yield')

save_path = h2o.save_model(stack, result_dir, force=True)
print(f"\n\n\n******\nSTACKED RESULTS\n******")
print(f"saved to: {save_path}")
print(stack)

print(f"\n*** META MODEL ***\n")
meta_model = h2o.get_model(stack.metalearner()['name'])
print(meta_model)

# xgb_params = {'learn_rate': [i * 0.01 for i in range(1, 30)],
#               'max_depth': list(range(5, 13)),
#               'sample_rate': [i * 0.1 for i in range(5, 11)],
#               'col_sample_rate': [i * 0.1 for i in range(4, 11)]}
#
# xgb_grid = H2OGridSearch(
#     model=H2OXGBoostEstimator(
#         ntrees=max_trees,
#         stopping_rounds=3,
#         stopping_tolerance=stop_tol,
#         categorical_encoding='enum',
#         keep_cross_validation_predictions=True),
#     hyper_params=xgb_params,
#     search_criteria=search_criteria)
#
# xgb_grid.train(x=train_cols, y='Dry_Yield', training_frame=df, fold_column='YearID_KFold')
# print(xgb_grid)
# xgb_best = xgb_grid.get_grid(sort_by='mae').models[0]
# print(xgb_best)
