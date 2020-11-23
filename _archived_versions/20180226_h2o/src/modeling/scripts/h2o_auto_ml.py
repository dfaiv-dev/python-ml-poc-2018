###
# 2018-01-16
#
# remote exec with:
# bash run_docker_image.sh -d ./ -s h2o_automl -- \
#           python -u modeling/scripts/h2o_auto_ml.py --run-id h2o_automl
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
from h2o.estimators import H2OGradientBoostingEstimator
from h2o.grid import H2OGridSearch
from pandas import DataFrame

from data import gis_pps, pcsml_data_loader as dl, h2o_frame_util
from util import mem_util

training_data_dir = '/var/opt/pcsml/devel/training_data/dumps/h2o/gis_pps_aggr_corn__201710'
training_data_name = 'gis_pps_aggr_corn_sample_10k'
output_dir = '/var/opt/pcsml/devel/results/_scratch'
temp_dir = '/var/opt/pcsml/devel/_temp'
run_id = 'dev'
h2o_url = 'http://localhost:54321'
h2o_max_mem = '7g'
stop_tol = .9
max_run_time_sec = 60
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
    h2o_max_mem = '64g'
    stop_tol = .9
    max_run_time_sec = 60 * 60 * 6
    n_folds = 3
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


def _get_temp_data_path(file_suffix: str, sep='__') -> str:
    return os.path.join(
        temp_dir,
        _get_encoded_file_name(file_suffix, sep)
    )


def _get_training_data_path(file_suffix: str, sep='__') -> str:
    return os.path.join(
        training_data_dir,
        _get_encoded_file_name(file_suffix, sep))


col_types_path = _get_training_data_path('h2o.col_types.pickle')
df_csv_path = _get_training_data_path('h2o.frame.csv.gz')
df = h2o_frame_util.load_gis_pps(df_csv_path, col_types_path)

print("generating elb holdout mask")
elb_year_ids = [int(i) for i in dl.elb_year_ids()]
year_ids = df['YearID']
mask = year_ids.isin(elb_year_ids)
prev_frame_id = df.frame_id

elb_frame = df[mask]
df = df[~mask]
print(f"elb frame: {elb_frame.frame_id}, df: {df.frame_id}")
h2o.remove(prev_frame_id)
h2o.remove(mask.frame_id)
h2o.remove(year_ids.frame_id)

print("generating kfold col")
df = h2o_frame_util.grouped_kfold(df, n_folds, 'YearID')

# we can't have train_cols and fold_columns
# see bug: https://0xdata.atlassian.net/browse/PUBDEV-5250?jql=issuetype%20%3D%20Bug%20AND%20status%20in%20(Backlog%2C%20%22In%20Progress%22%2C%20Open)%20AND%20text%20~%20%22automl%20fold_column%22
print("dropping exclude cols")
all_excl_cols = gis_pps.exclude_columns + gis_pps.yield_dep_columns
excl_cols = [c for c in df.col_names if c in all_excl_cols]
print(f"excl cols: {sorted(excl_cols)}")
df_clean = df.drop(excl_cols)

excl_cols = [c for c in elb_frame.col_names if c in all_excl_cols]
elb_frame_cleaned = elb_frame.drop(excl_cols)

# force eval...
print(f"training df id: {df_clean.frame_id}, elb val frame id: {elb_frame_cleaned.frame_id}")
h2o.remove(df)
h2o.remove(elb_frame)

print(h2o.ls())

auto_ml = h2o.automl.H2OAutoML(
    stopping_tolerance=stop_tol,
    max_runtime_secs=max_run_time_sec)
auto_ml.train(y='Dry_Yield', training_frame=df_clean, fold_column='_kfold')

top_model_ids = list(auto_ml.leaderboard.head(5).as_data_frame()['model_id'].values)
for i, m_id in enumerate(top_model_ids):
    print(f"\n\n********** MODEL STATS: (#{i}) {m_id} ************")
    model = h2o.get_model(m_id)
    print(model)

    print(f"saving model: {m_id}")
    h2o.save_model(model, path=os.path.join(result_dir, f"model_rank_{i}"), force=True)

print("\n\n***\nLeaderboard (top 20)\n***")
print(auto_ml.leaderboard.head(20))

top_model = h2o.get_model(top_model_ids[0])
print(f"\n\n\n******\nTOP MODEL\n******")
print(top_model)

# score elb
pred = top_model.predict(elb_frame_cleaned)
actual = elb_frame_cleaned['Dry_Yield']
diff = pred - actual
diff = diff.abs()
diff_per = diff / actual
q_95 = diff.quantile([.95])
print("\n\n**** ELB STATS ****")
print("**Abs**")
print(f"mean: {diff.mean()[0]:.1f}")
print(q_95)
print("**Percents**")
print(f"mean: {diff_per.mean()[0]*100:.1f}%")
print("q 95 %")
print(diff_per.quantile([.95]) * 100)

# split_frames = df.split_frame(ratios=[.7, .15])
# gbm_params2 = {'learn_rate': [i * 0.01 for i in range(1, 14)],
#                'max_depth': list(range(6, 11)),
#                'sample_rate': [i * 0.1 for i in range(4, 11)],
#                'col_sample_rate': [i * 0.1 for i in range(4, 11)]}
#
# # Search criteria
# search_criteria = {'strategy': 'RandomDiscrete', 'max_models': 15, 'max_runtime_secs': 360, 'seed': 1}
#
# # Train and validate a random grid of GBMs
# gbm_grid2 = H2OGridSearch(model=H2OGradientBoostingEstimator(ntrees=300, stopping_rounds=3),
#                           grid_id='gbm_grid2',
#                           hyper_params=gbm_params2,
#                           search_criteria=search_criteria)
# gbm_grid2.train(x=train_cols, y='Dry_Yield', training_frame=split_frames[0], validation_frame=split_frames[1])


# est = H2OGradientBoostingEstimator(fold_column='YearID', nfolds=3)
# est.train(x=train_cols, y='Dry_Yield', training_frame=data, max_runtime_secs=120)
#
#
#
# # GBM hyperparameters
# gbm_params1 = {'learn_rate': [0.01, 0.1],
#                'max_depth': [3, 5, 9],
#                'sample_rate': [0.8, 1.0],
#                'col_sample_rate': [0.2, 0.5, 1.0]}
#
# # Train and validate a cartesian grid of GBMs
# gbm_grid1 = H2OGridSearch(model=H2OGradientBoostingEstimator,
#                           grid_id='gbm_grid1',
#                           hyper_params=gbm_params1)
# gbm_grid1.train(x=train_cols, y='Dry_Yield',
#                 training_frame=data,
#                 ntrees=100,
#                 seed=1)
