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

training_data = '/var/opt/pcsml/devel/training_data/dumps/transformed_sample_10000.pickle'
output_dir = '/var/opt/pcsml/devel/results/_scratch'
run_id = 'dev'

result_dir = os.path.join(output_dir, run_id)
os.makedirs(result_dir, exist_ok=True)

print(f"reading in training data, may take a while: {training_data}")
df: DataFrame = pandas.read_pickle(training_data)

df.drop(dl.exclude_columns, axis=1, inplace=True, errors='ignore')
label = df.pop('Dry_Yield')
df = pandas.get_dummies(df, dummy_na=True)

corrs = pandas.DataFrame(
    [(c, df[c].corr(label, method='spearman')) for c in df.columns],
    columns=['Feature', 'Corr'])
corrs['Corr'] = corrs['Corr'].abs()
