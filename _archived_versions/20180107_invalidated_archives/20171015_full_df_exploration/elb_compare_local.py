import os
import pickle
from argparse import ArgumentParser
from datetime import datetime

import pandas
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler

from data_scripts import pcs_data_loader
from modeling import score_util

arg_parser_ = ArgumentParser(description="Train an ET model and test it against ELBs")
arg_parser_.add_argument('--use-full-dataframe', dest='use_full_dataframe', action='store_true')
arg_parser_.add_argument('--result-base-path', dest='result_base_path', type=str,
                         default='./results/20170918_et_elb_full_dataset')
arg_parser_.add_argument('--n-jobs', dest='n_jobs', default=3, type=int)
arg_parser_.add_argument('--n-estimators', dest='n_estimators', default=5, type=int)

arg_parser_.set_defaults(use_full_dataframe=False)

args_ = arg_parser_.parse_args()
use_full_df = args_.use_full_dataframe
result_base_path: str = args_.result_base_path
n_jobs: int = args_.n_jobs
n_estimators = args_.n_estimators

run_id = f'{datetime.now():%Y%m%d%H%m}'

print(f'running....: {run_id}')
print(args_)
print(f'use_full_dataframe: {use_full_df}')

train_df: pandas.DataFrame
if not use_full_df:
    train_df = pcs_data_loader.load_corn_rows_sample_shaped_pickle_gz()
else:
    train_df = pcs_data_loader.shape_pps_data(pcs_data_loader.load_corn_rows_pickle_gz())

# load training data and train et model
y = train_df['Dry_Yield']
X = train_df.drop(['Dry_Yield', 'Area'], axis=1)
scaler = StandardScaler()
scaler.fit(X)

print('fitting model')
model = ExtraTreesRegressor(n_jobs=n_jobs, n_estimators=n_estimators, verbose=99)
model.fit(scaler.transform(X), y)

model_path_ = f'{result_base_path}/et_model_{run_id}.pickle'
with open(model_path_, 'wb') as f:
    pickle.dump(model, f)
    print(f'model saved: {model_path_}')

scaler_path_ = f'{result_base_path}/et_scaler_{run_id}.pickle'
with open(scaler_path_, 'wb') as f:
    pickle.dump(scaler, f)
    print(f'model saved: {scaler_path_}')

results = []
for idx, elb_data in enumerate(pcs_data_loader.load_cached_elbs(df.columns)):
    year_id, elb_X, elb_y, extra_cols = elb_data
    print(f'comparing elb year id: {year_id}, index: {idx}')

    elb_score = score_util.score(model, scaler, elb_X, elb_y)
    print(elb_score)

    results.append((year_id, elb_score, extra_cols))

rdf: pandas.DataFrame = pandas.concat(
    [
        pandas.DataFrame([_id for (_id, _, _) in results], columns=['year_id']),
        score_util.create_data_frame([scr for (_, scr, _) in results]),
        pandas.DataFrame(pandas.Series([c for (_, _, c) in results], name='extra_cols'))
    ],
    axis=1
)

os.makedirs(result_base_path, exist_ok=True)

rdf.to_csv(f'{result_base_path}/elb_harvest_predictions_results_{run_id}.csv')
with open(f'{result_base_path}/elb_harvest_predictions_results_{run_id}.pickle', 'wb') as f:
    pickle.dump(results, f)
