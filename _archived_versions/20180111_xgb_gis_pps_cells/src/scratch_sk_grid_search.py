import os
from typing import List

import numpy as np
import pandas
import psutil
import xgboost as xgb
from pandas import DataFrame
from pympler.tracker import SummaryTracker
from sklearn.model_selection import GridSearchCV, GroupKFold, cross_validate
from sklearn.pipeline import make_pipeline, Pipeline

from data import pcsml_data_loader as dl
from modeling import categorical_util, score_util

print_mem_enabled = False
tr = SummaryTracker()
mem_used = 0
prev_mem_used = 0

cv_results: List[GridSearchCV] = []


def _score_grid_search(est: Pipeline, X, y):
    predictions = est.predict(X)
    _print_mem_usage()
    return score_util.ScoreReport(y, predictions).abs_99


def _score_cv(est: GridSearchCV, X, y):
    if est.best_estimator_ not in cv_results:
        cv_results.append(est.best_estimator_)

    predictions = est.predict(X)
    return score_util.ScoreReport(y, predictions).abs_99


def _print_mem_usage():
    global mem_used, prev_mem_used

    if not print_mem_enabled:
        return

    tr.print_diff()
    mem = psutil.virtual_memory()
    prev_mem_used = mem_used
    mem_used = (mem.total - mem.available) / 1024 / 1024
    print(f"virt_mem >> used: {mem_used:.0f}, prev: {prev_mem_used:.0f}, diff: {mem_used - prev_mem_used:.0f}")


training_data = '/var/opt/pcsml/devel/training_data/dumps/transformed_sample_10000.pickle'
output_dir = '/var/opt/pcsml/devel/results/_scratch'
run_id = 'dev'

result_dir = os.path.join(output_dir, run_id)
os.makedirs(result_dir, exist_ok=True)

print(f"reading in training data, may take a while: {training_data}")
df: DataFrame = pandas.read_pickle(training_data).sample(5000)

year_ids: np.ndarray = df['YearId'].unique()
print(f"total year ids: {len(year_ids)}")
elb_year_ids = dl.elb_year_ids()
year_ids: np.ndarray = year_ids[~np.isin(year_ids, elb_year_ids)]
print(f"year ids without elbs: {len(year_ids)}")

elb_df: pandas.DataFrame = df.loc[df['YearId'].isin(elb_year_ids)]
elb_df = elb_df.drop(dl.exclude_columns, axis=1, errors='ignore')
categorical_util.encode_categories(elb_df)
elb_y = elb_df.pop('Dry_Yield')

df = df.loc[df['YearId'].isin(year_ids)]
df_year_id: pandas.Series = df.pop('YearId')
df.drop(dl.exclude_columns, axis=1, inplace=True, errors='ignore')
column_categories = categorical_util.encode_categories(df)
df_label = df.pop('Dry_Yield')

# DMF 2018-01-04
# there is no good way to get callback info from each trained pipeline
# also, the pipeline refits data for each loop?
xgb_gs = GridSearchCV(
    xgb.XGBRegressor(n_estimators=10),
    {
        'max_depth': [4, 5]
    },
    scoring=_score_grid_search,
    verbose=99,
    return_train_score=True,
    n_jobs=1)

cv_pipeline = Pipeline([
    ('dummy_encoder', categorical_util.DummyEncoder(df.columns, column_categories)),
    ('xgb_grid_search', xgb_gs)
])

_print_mem_usage()

cv = cross_validate(cv_pipeline, df.as_matrix(), df_label, groups=df_year_id,
                    fit_params={'xgb_grid_search__groups': df_year_id},
                    scoring=_score_cv,
                    return_train_score=True,
                    n_jobs=1,
                    verbose=True)

_print_mem_usage()

# print(cv)

# for r in sorted(gs.cv_results_.keys()):
#     print(f"{r}: {gs.cv_results_[r]}")
