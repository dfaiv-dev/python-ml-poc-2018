import logging

import itertools

import os
from pandas import DataFrame
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.externals import joblib
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

import env
import util.logging
from data_scripts import pcsml_data_loader as dl
from modeling import preprocessing, score_util
from modeling.preprocessing import make_one_hot_pipeline

util.logging.setup_default(env.result_path)

log = logging.getLogger(__name__)
log.info("Running...")
log.info("Env:\n%s", env.dump())

# load the data frame (a sample of the sample to make debugging faster...)
# df: DataFrame = dl.load_df_corn_pkl_smpl_25_20171018().sample(2000)

df: DataFrame = dl.load_df_corn_pkl_smpl_25_20171018()
logging.debug("data shape: %s", df.shape)

y = df.pop('Dry_Yield')
X = df

###
# transform
###

X, label_cols = preprocessing.shape_gis_pps(X)
transform_pipe = make_one_hot_pipeline(
    X, label_cols,
    [preprocessing.FillNaTransformer(), preprocessing.NumericTransformer(), StandardScaler()])

###
# Run Model
###
kcv = KFold(n_splits=10, shuffle=True, random_state=972)

kf_runs = itertools.islice(kcv.split(X), 3)

scores = []
for i, (train_split_idx, test_split_idx) in enumerate(kf_runs):
    log.info("Running kfold: %s", i)

    X_train_split, y_train_split = X.iloc[train_split_idx], y.iloc[train_split_idx]
    X_test_split, y_test_split = X.iloc[test_split_idx], y.iloc[test_split_idx]

    log.info("fitting transforms")
    transform_pipe.fit(X_train_split)
    log.info("transforming")
    X_train_transformed = transform_pipe.transform(X_train_split)

    logging.info("Running on input data shape: %s", X_train_transformed.shape)
    # model = ExtraTreesRegressor(verbose=99, n_jobs=4)
    # model = MLPRegressor(verbose=99, max_iter=150, tol=.01, learning_rate='constant', alpha=.1)
    model = GradientBoostingRegressor(verbose=99, n_estimators=100, max_depth=8)
    model.fit(X_train_transformed, y_train_split)
    logging.info("Scoring")
    scr = score_util.score(model, transform_pipe, X_test_split, y_test_split)

    joblib.dump(scr, os.path.join(env.result_path, f"score_{i}.pickle"))
    joblib.dump(scr, os.path.join(env.result_path, f"model_{i}.pickle"))

    scores.append(scr)

# for score in scores:
#     logging.info(score)

combined_score = score_util.combine(scores)
logging.info("kfold scores combined: %s", combined_score)
joblib.dump(combined_score, os.path.join(env.result_path, f"combined_score.pickle"))
