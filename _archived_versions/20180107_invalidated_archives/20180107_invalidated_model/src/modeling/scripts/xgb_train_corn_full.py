import argparse
import os
import pickle
import sys
from datetime import datetime

import pandas
import xgboost as xgb
from pandas import DataFrame

import data.pcsml_data_loader as dl
import modeling.categorical_util as categorical_util

# opts
# debug settings
# training_data = '/var/opt/pcsml/devel/training_data/dumps/df-corn-smpl_25-gis-pps-20171018.pkl'
training_data = '/var/opt/pcsml/devel/training_data/dumps/transformed_sample_10000.pickle'
output_dir = '/var/opt/pcsml/devel/results/_scratch'
run_id = 'dev'
sample_n = 5000
describe_data = False
cv_n_splits = 7
cv_n_runs = 2
xgb_n_threads = 2
xgb_max_depth = 4
xgb_n_rounds = 50
debug_override = "--debug" in sys.argv[0]

if debug_override:
    print("LOG-DEBUG: running in debug mode, won't load cmd line params")

if not debug_override and __name__ == '__main__' and not "pydevconsole" in sys.argv[0]:
    parser = argparse.ArgumentParser()
    parser.add_argument('--training-data', '-d', type=str,
                        default='/var/opt/pcsml/training-data/gis-pps/df-corn-gis-pps-20171018.transformed.pickle')
    parser.add_argument('--output-dir', '-o', type=str, default='/var/opt/pcsml/remote-exec-out')
    parser.add_argument('--cv-n-splits', '-cvs', type=int, default=10)
    parser.add_argument('--cv-n-runs', '-cvr', type=int, default=3)
    parser.add_argument('--xgb-n-threads', '-bnt', type=int, default=16)
    parser.add_argument('--xgb-max-depth', '-bmd', type=int, default=7)
    parser.add_argument('--xgb-n-rounds', '-br', type=int, default=2000)
    parser.add_argument('--run-id', '-r', type=str, default=f'{datetime.utcnow():%Y-%m-%dT%H-%M-%S}')
    parser.add_argument('--sample-n', '-s', type=int)

    opt = parser.parse_args()

    print("parsed opts:")
    print(opt)
    training_data = opt.training_data
    output_dir = opt.output_dir
    sample_n = opt.sample_n
    cv_n_splits = opt.cv_n_splits
    cv_n_runs = opt.cv_n_runs
    xgb_n_threads = opt.xgb_n_threads
    xgb_max_depth = opt.xgb_max_depth
    xgb_n_rounds = opt.xgb_n_rounds
    run_id = opt.run_id

###
# create run output dir
result_dir = os.path.join(output_dir, run_id)
os.makedirs(result_dir, exist_ok=True)

###
# - create data exploration outputs (graphs, etc)
# - run the xgboost model, saving model output
###
print(f"reading in training data, may take a while: {training_data}")
df: DataFrame = pandas.read_pickle(training_data)

if sample_n is not None:
    df = df.sample(sample_n)

print(f"training data original shape: {df.shape}")
# filter 2017
print(f"LOG: filtering data, dropping columns")
df = df[df.Year < 2017]
df.drop(dl.exclude_columns, axis=1, inplace=True, errors='ignore')
print(f"training data, filtered: {df.shape}")

###
# TRAIN MODEL
###

y = df.pop('Dry_Yield')
X = df

print(f"LOG: getting column categories")
column_categories = categorical_util.get_categories_lookup(X)
categorical_util.set_categories(X, column_categories)

print(f"LOG: encoding dummy columns")
dummy_enc = categorical_util.create_dummy_encoder(X, column_categories)
X = dummy_enc.fit_transform(categorical_util.encode_categories(X))

X = xgb.DMatrix(X, label=y, feature_names=dummy_enc.transformed_column_names)
params = {'max_depth': xgb_max_depth, 'subsample': 1, 'colsample_bytree': 1,
          'objective': 'reg:linear', 'eval_metric': 'mae',
          'nthread': xgb_n_threads,
          'silent': 1}

evallist = [(X, 'train')]
model: xgb.Booster = xgb.train(params, X, num_boost_round=xgb_n_rounds, evals=evallist)

# dump results
model.save_model(os.path.join(result_dir, f"model_{run_id}.xgb"))
with open(os.path.join(result_dir, f"model_{run_id}_column_categories.pickle"), 'wb') as f:
    pickle.dump(column_categories, f)
