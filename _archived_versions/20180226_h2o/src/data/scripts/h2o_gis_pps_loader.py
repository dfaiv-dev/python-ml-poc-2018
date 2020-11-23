###
# 2018-01-16
#
# remote exec with:
# bash run_docker_image.sh -d ./ -s gis_pps_aggr_h2o -- \
#           python -u data/scripts/h2o_gis_pps_loader.py --run-id gis_pps_aggr_h2o
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

from data import gis_pps
from util import mem_util

raw_path = '/var/opt/pcsml/devel/training_data/dumps/gis-pps-aggr/gis-pps-agg-20171018-corn-smpl-25__df.pkl'
output_dir = '/var/opt/pcsml/devel/results/_scratch'
temp_dir = '/var/opt/pcsml/devel/_temp'
run_id = 'dev'
sample_n: Union[int, None] = 10_000
h2o_url = 'http://localhost:54321'
h2o_max_mem = '10g'
is_debug = True

# mem_util.disable_print_mem()
mem_util.enable_print_mem()

if __name__ == '__main__' and not "pydevconsole" in sys.argv[0] and not '--debug' in sys.argv[1]:
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', '-r', type=str, default=f'{datetime.utcnow():%Y-%m-%dT%H-%M-%S}')
    opt = parser.parse_args()
    run_id = opt.run_id

    raw_path = '/var/opt/pcsml/training-data/gis-pps/df-corn-gis-pps-aggregates-20171018.pkl'
    output_dir = '/var/opt/pcsml/remote-exec-out'
    temp_dir = '/var/opt/pcsml/tmp'
    sample_n = None
    h2o_url = 'http://localhost:54321'
    h2o_max_mem = '125g'
    is_debug = False

    mem_util.enable_print_mem()

# h2o.init(h2o_url)
h2o.init(url=h2o_url, max_mem_size=h2o_max_mem)

result_dir = os.path.join(output_dir, run_id)
print(f"result_dir: {result_dir}")
os.makedirs(result_dir, exist_ok=True)

print(f"temp_dir: {temp_dir}")
os.makedirs(temp_dir, exist_ok=True)

encoded_file_base_name = 'gis_pps_aggr_corn'
if sample_n is not None and sample_n > 0:
    encoded_file_base_name += f"_sample_{sample_n/1000:.0f}k"

print(f"opts: ")
print({
    'raw_path': raw_path,
    'encoded_file_base_name': encoded_file_base_name,
    'output_dir': output_dir,
    'temp_dir': temp_dir,
    'result_dir': result_dir,
    'pwd': os.getcwd(),
    'sample_n': sample_n
})


def _get_encoded_file_name(file_suffix: str, sep='__') -> str:
    return f"{encoded_file_base_name}{sep}{file_suffix}"


def _get_result_file_path(file_suffix: str, sep='__') -> str:
    return os.path.join(
        result_dir,
        _get_encoded_file_name(file_suffix, sep))


def _get_temp_data_path(file_suffix: str, sep='__') -> str:
    return os.path.join(
        temp_dir,
        _get_encoded_file_name(file_suffix, sep)
    )


mem_util.print_mem_usage()

print(f"reading in data, may take a while: {raw_path}")
data_src: DataFrame = pandas.read_pickle(raw_path)
if sample_n is not None and sample_n > 0:
    print(f"sampling: {sample_n}")
    data_src = data_src.sample(n=sample_n)

print("cleaning...")
gis_pps.clean(data_src)
mem_util.print_mem_usage()

print("shaping...")
gis_pps.shape(data_src, encode_app_date_bi_weeks=True)
mem_util.print_mem_usage()

exclude_cols = [c for c in gis_pps.exclude_columns if c not in ['Area', 'YearID'] and c in data_src.columns]
print(f"dropping unused: {exclude_cols}")
data_src.drop(exclude_cols, inplace=True, axis=1)

# get h2o column types
col_names = list(data_src.columns)
col_types = {c: data_src[c].dtype for c in col_names}
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

df_csv_path = _get_result_file_path('h2o.frame.csv.gz')
print(f"saving df to csv: {df_csv_path}")
data_src.to_csv(df_csv_path, header=True, index=False, compression='gzip')

col_types_path = _get_result_file_path('h2o.col_types.pickle')
col_names_path = _get_result_file_path('h2o.col_names.pickle')
print(f"exporting col types pickle: {col_types_path}")
with open(col_types_path, 'wb') as f:
    pickle.dump(col_types, f)
print(f"exporting col names pickle: {col_names_path}")
with open(col_names_path, 'wb') as f:
    pickle.dump(col_names, f)



# spot check df is valid:
# df = h2o.import_file(df_csv_path, col_types=col_types)
# split_frames = df.split_frame(ratios=[.7, .15])
# gbm_params2 = {'learn_rate': [i * 0.01 for i in range(1, 14)],
#                'max_depth': list(range(6, 11)),
#                'sample_rate': [i * 0.1 for i in range(4, 11)],
#                'col_sample_rate': [i * 0.1 for i in range(4, 11)]}
#
# # Search criteria
# search_criteria = {'strategy': 'RandomDiscrete', 'max_models': 15, 'max_runtime_secs': 360, 'seed': 1}
#
# train_cols = [c for c in df.col_names if c not in (gis_pps.exclude_columns + gis_pps.yield_dep_columns)]
# # Train and validate a random grid of GBMs
# gbm_grid2 = H2OGridSearch(model=H2OGradientBoostingEstimator(ntrees=300, stopping_rounds=3),
#                           grid_id='gbm_grid2',
#                           hyper_params=gbm_params2,
#                           search_criteria=search_criteria)
# gbm_grid2.train(x=train_cols, y='Dry_Yield', training_frame=split_frames[0], validation_frame=split_frames[1])




# est = H2OGradientBoostingEstimator(fold_column='YearID', nfolds=3)
# est.train(x=train_cols, y='Dry_Yield', training_frame=data, max_runtime_secs=120)
#
# auto_ml = h2o.automl.H2OAutoML(nfolds=3, max_runtime_secs=100)
# auto_ml.train(x=train_cols, y='Dry_Yield', training_frame=data)
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
