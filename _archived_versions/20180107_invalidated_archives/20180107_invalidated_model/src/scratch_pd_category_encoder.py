import gc
import os
from importlib import reload

import pandas
import xgboost as xgb
from memory_profiler import profile
from pandas import DataFrame

import data.pcsml_data_loader as dl
from modeling import categorical_util, score_util

training_data = '/var/opt/pcsml/devel/training_data/dumps/transformed_sample_10000.pickle'
output_dir = '/var/opt/pcsml/devel/results/_scratch'
run_id = 'dev'

result_dir = os.path.join(output_dir, run_id)
os.makedirs(result_dir, exist_ok=True)

reload(categorical_util)

print(f"reading in training data, may take a while: {training_data}")
df: DataFrame = pandas.read_pickle(training_data).sample(200)
df.drop(dl.exclude_columns, axis=1, inplace=True, errors='ignore')

dtest: pandas.DataFrame = df.sample(frac=.25)
dtrain: pandas.DataFrame = df.loc[~df.index.isin(dtest.index)]

# @profile
# def _run_dummies():
column_categories = categorical_util.get_categories_lookup(dtrain)
categorical_util.set_categories(dtrain, column_categories)
categorical_util.set_categories(dtest, column_categories)

dtrain_y = dtrain.pop('Dry_Yield')
dtrain = categorical_util.encode_dummies(df)
dummy_cols = dtrain.columns
print(dummy_cols)
dtrain = dtrain.to_coo()

dtest_y = dtest.pop('Dry_Yield')
dtest = pandas.get_dummies(dtest, sparse=True, drop_first=True, dummy_na=False, prefix_sep='__DUMMY__').to_sparse().to_coo()

print(f"dmatrix sizes: {dtrain.shape[1]}, {dtest.shape[1]}")

dtrain: xgb.DMatrix = xgb.DMatrix(dtrain, dtrain_y)
dtest: xgb.DMatrix = xgb.DMatrix(dtest, label=dtest_y)

model: xgb.Booster = xgb.train({'max_depth': 2}, dtrain, num_boost_round=2)
pred = model.predict(dtest)

scr = score_util.ScoreReport(dtest_y, pred)
print(scr)


# _run_dummies()
