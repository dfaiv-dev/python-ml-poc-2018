######
# sklearn random and extra trees and XGBoost CV search.
######
import argparse
import os
from typing import List

import numpy as np
import pandas
import psutil
import sys
import xgboost as xgb
from datetime import datetime
from pandas import DataFrame
from pympler.tracker import SummaryTracker
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.model_selection import GridSearchCV, GroupKFold, cross_validate, KFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.svm import SVR
from xgboost import XGBRegressor

from data import pcsml_data_loader as dl
from modeling import categorical_util, score_util

print_mem_enabled = False
tr = SummaryTracker()
mem_used = 0
prev_mem_used = 0

cv_results: List[GridSearchCV] = []

training_data = '/var/opt/pcsml/devel/training_data/dumps/transformed_sample_10000.pickle'
output_dir = '/var/opt/pcsml/devel/results/_scratch'
run_id = 'dev'
kf_folds = 3
kf_runs = 2
n_threads = 2
verbose_eval = True
debug_override = "--debug" in sys.argv[0]

if debug_override:
    print("LOG-DEBUG: running in debug mode, won't load cmd line params")

# remote exec with:
# bash run_docker_image.sh -d ./ -s skl_xgb_cv -- \
#           python -u modeling/scripts/skl_xgb_cv.py --run-id skl_xgb_cv

if not debug_override and __name__ == '__main__' and not "pydevconsole" in sys.argv[0]:
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', '-r', type=str, required=True)
    parser.add_argument('--n-threads', '-nt', type=int, required=True)
    opt = parser.parse_args()

    print("parsed opts:")
    print(opt)
    training_data = '/var/opt/pcsml/training-data/gis-pps/df-corn-gis-pps-20171018.transformed.pickle'
    output_dir = '/var/opt/pcsml/remote-exec-out'
    kf_folds = 9
    kf_runs = 2
    n_threads = opt.n_threads
    run_id = opt.run_id
    verbose_eval = False


def _print_mem_usage():
    global mem_used, prev_mem_used

    if not print_mem_enabled:
        return

    tr.print_diff()
    mem = psutil.virtual_memory()
    prev_mem_used = mem_used
    mem_used = (mem.total - mem.available) / 1024 / 1024
    print(f"virt_mem >> used: {mem_used:.0f}, prev: {prev_mem_used:.0f}, diff: {mem_used - prev_mem_used:.0f}")


print(f"""OPTIONS:
run_id: {run_id}
training_data: {training_data}
output_dir: {output_dir}
kf_folds, kf_runs: {kf_folds}, {kf_runs}
n_threads: {n_threads}
""")

result_dir = os.path.join(output_dir, run_id)
os.makedirs(result_dir, exist_ok=True)

print(f"reading in training data, may take a while: {training_data}")
data: DataFrame = pandas.read_pickle(training_data).sample(5000)

yld_mean = data['Dry_Yield'].mean()
yld_std = data['Dry_Yield'].std()
yld_rng_min = max(0, yld_mean - (yld_std * 4))
yld_rng_max = yld_mean + (yld_std * 4)
print(f"constraining yield range to: {yld_rng_min:.2f}, {yld_rng_max:.2f}")
data.loc[data['Dry_Yield'] >= yld_rng_max, 'Dry_Yield'] = yld_rng_max
data.loc[data['Dry_Yield'] <= yld_rng_min, 'Dry_Yield'] = yld_rng_min

year_ids: np.ndarray = data['YearId'].unique()
print(f"total year ids: {len(year_ids)}")
elb_year_ids = dl.elb_year_ids()
year_ids: np.ndarray = year_ids[~np.isin(year_ids, elb_year_ids)]
print(f"year ids without elbs: {len(year_ids)}")

elb_df: pandas.DataFrame = data.loc[data['YearId'].isin(elb_year_ids)]
elb_df = elb_df.drop(dl.exclude_columns + dl.yield_dep_columns, axis=1, errors='ignore')
categorical_util.encode_categories(elb_df)
elb_y = elb_df.pop('Dry_Yield')

data = data.loc[data['YearId'].isin(year_ids)]
df_year_id: np.ndarray = data.pop('YearId').values
data_label: np.ndarray = data.pop('Dry_Yield').values
data.drop(dl.exclude_columns + dl.yield_dep_columns, axis=1, inplace=True, errors='ignore')
column_categories = categorical_util.encode_categories(data)
dummy_enc = categorical_util.DummyEncoder(data.columns, column_categories)

print("Fitting dummy enc")

data: np.ndarray = dummy_enc.fit_transform(data.as_matrix())

kf_outer = GroupKFold()
split = next(kf_outer.split(data, groups=df_year_id))
train_idx, test_idx = split
train, train_y = data[train_idx], data_label[train_idx]
test, test_y = data[test_idx], data_label[test_idx]

model = XGBRegressor(max_depth=5, n_estimators=100, silent=False, n_jobs=2)
model.fit(train, train_y)
scr = score_util.ScoreReport(test_y, model.predict(test))
print(scr)

model = RandomForestRegressor(verbose=99, n_estimators=50, n_jobs=2)
model.fit(train, train_y)
scr = score_util.ScoreReport(test_y, model.predict(test))
print(scr)

model = ExtraTreesRegressor(verbose=99, n_estimators=50, n_jobs=2)
model.fit(train, train_y)
scr = score_util.ScoreReport(test_y, model.predict(test))
print(scr)

model = SVR(degree=5, verbose=99)
model.fit(train, train_y)
scr = score_util.ScoreReport(test_y, model.predict(test))
print(scr)


# for idx_outer, (train_outer_idx, test_outer_idx) in enumerate(kf_outer.split(data, groups=df_year_id)):
#     train_outer, test_outer = data[train_outer_idx], data[test_outer_idx]
#     train_year_ids = df_year_id[train_outer_idx]
#
#     kf_inner = GroupKFold()
#     for i, (train_idx, test_idx) in enumerate(kf_inner.split(train_outer, groups=train_year_ids)):
#         train = train_outer[train_idx]
#         train_y = data_label[train_idx]
#         test = train_outer[test_idx]
#         test_y = data_label[test_idx]
#
#         rf_model = RandomForestRegressor(verbose=99)
#         rf_model.fit(train, train_y)



