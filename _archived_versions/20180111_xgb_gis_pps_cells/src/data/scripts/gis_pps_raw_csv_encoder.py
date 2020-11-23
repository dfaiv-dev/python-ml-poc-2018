###
# 2018-01-07
#
# Extract a number of rows from a text file.
#
# remote exec with:
# bash run_docker_image.sh -d ./ -s gis_pps_raw_csv_encoder -- \
#           python -u data/scripts/gis_pps_raw_csv_encoder.py --run-id gis_pps_raw_csv_encoder
###

import argparse
import gzip
import os
import sys
from datetime import datetime
import re
from typing import List

import numpy
import pandas
import pickle
from pandas import DataFrame
from scipy import sparse

import data.pcsml_data_loader as dl
from modeling import categorical_util

training_data = '/var/opt/pcsml/devel/training_data/dumps/racer_gis_pps_corn_2016__20180106.txt.gz'
output_dir = '/var/opt/pcsml/devel/results/_scratch'
run_id = 'dev'
sample_n = 50000

if __name__ == '__main__' and not "pydevconsole" in sys.argv[0]:
    parser = argparse.ArgumentParser()
    parser.add_argument('--training-data', type=str,
                        default='/var/opt/pcsml/training-data/gis-pps/racer_gis_pps_corn_2016__20180106.txt.gz')
    parser.add_argument('--output-dir', '-o', type=str, default='/var/opt/pcsml/remote-exec-out')
    parser.add_argument('--run-id', '-r', type=str, default=f'{datetime.utcnow():%Y-%m-%dT%H-%M-%S}')
    opt = parser.parse_args()

    print("parsed opts:")
    print(opt)
    training_data = opt.training_data
    output_dir = opt.output_dir
    run_id = opt.run_id
    sample_n = None


def _drop_columns(_df: pandas.DataFrame, columns: List[str]):
    _df.drop(labels=columns, axis=1, inplace=True, errors='ignore')


###
# create run output dir
result_dir = os.path.join(output_dir, run_id)
os.makedirs(result_dir, exist_ok=True)

col_csv_out_path = os.path.join(result_dir, 'cols.csv')
df_sample_out_path_base = os.path.join(
    result_dir,
    f"df_gis_pps_corn_2016")

print(f"reading in training data, may take a while: {training_data}")
if sample_n is not None and sample_n > 0:
    df_sample_out_path_base += f"_sample_{sample_n/1000:.0f}k"
    df: DataFrame = pandas.read_csv(
        training_data,
        sep='\t',
        nrows=sample_n)
else:
    df: DataFrame = pandas.read_csv(
        training_data,
        sep='\t')

# drop $ columns
cost_columns = [c for c in df.columns if '$' in c.lower()]
cost_columns += [c for c in df.columns if 'cost' in c.lower()]
print(f"found {len(cost_columns)} cost columns.  Dropping.  Head: {sorted(cost_columns)[:25]}...")
_drop_columns(df, cost_columns)

# drop unused weather
re_wx_col_num = re.compile('^WK(\d\d?)(Gdd|Rain)$', re.IGNORECASE)
wx_columns = [c for c in df.columns if re_wx_col_num.search(c)]
valid_wx_weeks = list(range(14, 40))
invalid_wx_columns = [
    c for c in wx_columns
    if int(re_wx_col_num.match(c).group(1)) not in valid_wx_weeks
]
print(f"dropping unused wx columns: {invalid_wx_columns[:20]}...")
_drop_columns(df, invalid_wx_columns)

# drop known unused
known_unused_cols = [
    'FieldID',
    'CropYear',
    'ShapeIndex',
    'GridRow',
    'GridColumn',
    'ProcessedLayerUID',
    'ShapeX',
    'ShapeY'
]
print(f"dropping known unused columns")
_drop_columns(df, known_unused_cols)

# encode
df = dl.gis_pps_encode_raw_csv(df)

print(f"df size: {sys.getsizeof(df) / 1024 / 1024:.0f} MB")
print(f"year ids: {len(df['YearID'].unique())}")

print("saving col csv")
df.dtypes.to_csv(col_csv_out_path, header=True)

# split categorical columns, since we always need to treat them differently
print(f"getting column category lookup")
column_categories = categorical_util.get_categories_lookup(df)
column_cat_path = f"{df_sample_out_path_base}_column_categories.pickle.gz"
print(f"saving.... {column_cat_path}")
with gzip.open(column_cat_path, 'wb') as gz:
    pickle.dump(column_categories, gz)

df_cat = df.select_dtypes(include='category')
df.drop(labels=df_cat.columns, axis=1, inplace=True)
df_cat_path = f"{df_sample_out_path_base}_categories.pickle.gz"
print(f"saving category columns dataframe... ")
df_cat.to_pickle(df_cat_path, compression='gzip')

print("getting dummies, this can take a very long time...")
df_dummies = pandas.get_dummies(
    df_cat, prefix_sep='__DUMMY__', sparse=True, drop_first=True, dummy_na=True)
df_dummies_path = f"{df_sample_out_path_base}_dummies.pickle.gz"
print(f"saving dummies... {df_dummies_path}")
df_dummies.to_pickle(df_dummies_path, compression='gzip')

df_pickle_path = df_sample_out_path_base + '.pickle.gz'
print("saving df pickle.gz")
df.to_pickle(df_pickle_path, compression='gzip')
