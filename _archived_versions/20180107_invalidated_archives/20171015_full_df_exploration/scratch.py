import logging
import os
import pickle
from datetime import datetime
from timeit import timeit

import numpy as np
import pandas
from sklearn.ensemble import AdaBoostRegressor, BaggingRegressor, ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import KFold, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler
from sklearn.svm import SVR

import modeling
from . import preprocessing
from data_scripts import pcs_data_loader
from modeling import score_util
from modeling.score_util import ScoreReport

logging.basicConfig(level=logging.DEBUG)

log = logging.getLogger(__name__)

_label_cols = pcs_data_loader.group_cols()
# remove plntdate#
_label_cols.remove('PlntDate#')
_label_cols.remove('HarvDate#')


def _shape_data(df: pandas.DataFrame, label_encoders: dict):
    df = df.drop(['Year', 'YearId', 'ProcessedLayerUID', 'Area'], axis=1, errors='ignore')

    _X = df.drop(['Dry_Yield'], axis=1)
    _y = df['Dry_Yield']
    for (_col, _enc) in label_encoders:
        _labels = _X[_col].str.lower()
        _labels = _labels.replace('none', None).replace('', None).astype(str)
        _encoded, _extra_cols = enc.transform(_labels)
        if len(_extra_cols) > 0:
            log.warning(f"Extra labels in column. ({_col}): {_extra_cols[:10]} ({len(_extra_cols)})")

        _X[_col] = _encoded

    return _X.fillna(value=0), _y


_df = pcs_data_loader.load_corn_rows_pickle_gz()
_label_encoders = {}
for c in _label_cols:
    col = np.array(_df[c].values)
    # col[pandas.isnull(col)] = "__none__"
    enc = preprocessing.FlexLabelEncoder()
    enc.fit(col)
    _label_encoders[c] = enc


X, y = _shape_data(pcs_data_loader.load_corn_rows_pickle_gz(), _label_encoders)
label_mask = np.isin(X.columns, _label_cols)

kf = KFold(random_state=871)
for idx, (train, test) in enumerate(kf.split(_df)):
    log.info("Running kfold: %s", idx)

    X_train, X_test, y_train, y_test = X.iloc[train], X.iloc[test], y.iloc[train], y.iloc[test]

    scaler = StandardScaler()
    ohe = OneHotEncoder(categorical_features=label_mask, handle_unknown='ignore')
    ohe.fit(X_train)
    scaler.fit(X_train)

    def _transform(_X):
        _X_scaled = scaler.transform(_X)
        # revert the labeled columns back to unscaled
        _X_scaled[:,label_mask] = _X.values[:,label_mask]
        return ohe.transform(_X_scaled)


    # model = ElasticNet()
    # model = RandomForestRegressor(verbose=99)

    # r2 .95, sq_mean 59, abs_stddev 7.5
    # model = MLPRegressor(verbose=99, alpha=1)

    # model = ExtraTreesRegressor(verbose=99, min_samples_leaf=3)
    model = BaggingRegressor(verbose=99)
    # model = SVR(verbose=99)
    model.fit(_transform(X_train), y_train)

    predictions = model.predict(_transform(X_test))
    score = ScoreReport(y_test, predictions)
    log.info(score)

# save the last model
with open(f'./results/20170918_et_elb_full_dataset/{datetime.now():%Y%m%d_%H%M}_{type(model).__name__}.pickle', 'wb') as f:
    pickle.dump(model, f);


def _load_cached_elbs(label_encoders, data_path='data/elbs'):
    for f in [f for f in os.listdir(data_path) if "_elb.pickle.gz" in f]:
        tokens = f.split('_')

        elb_df = pandas.read_pickle(f'{data_path}/{f}', compression='gzip')
        X, y = _shape_data(elb_df, label_encoders)

        year_id = int(tokens[0])
        yield year_id, X, y


elb_results = []
for idx, elb_data in enumerate(_load_cached_elbs(_label_encoders)):
    year_id, elb_X, elb_y = elb_data

    log.info(f'comparing elb year id: {year_id}, index: {idx}')

    predictions = model.predict(_transform(elb_X))
    elb_score = ScoreReport(elb_y.values, predictions)
    log.info(elb_score)

    elb_results.append((year_id, elb_score))


