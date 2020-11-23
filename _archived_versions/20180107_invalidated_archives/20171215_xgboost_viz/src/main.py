import argparse
import itertools
import logging
import os

from pandas import DataFrame
from sklearn.externals import joblib
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from xgboost.sklearn import XGBRegressor

import env
import util.logging
from data_scripts import pcsml_data_loader as dl
from modeling import preprocessing, score_util
from modeling.preprocessing import make_one_hot_pipeline

parser = argparse.ArgumentParser()
parser.add_argument('--env', '-e', type=str)
parser.add_argument('--result-path', '-o', type=str)


util.logging.setup_default(env.result_path)

log = logging.getLogger(__name__)
log.info("Running...")
log.info("Env:\n%s", env.dump())

# load the data frame (a sample of the sample to make debugging faster...)
# df: DataFrame = dl.load_df_corn_pkl_smpl_25_20171018().sample(2000)
df: DataFrame = dl.load_df_corn_pkl_smpl_25_20171018()
log.debug("data shape: %s", df.shape)

y = df.pop('Dry_Yield')
X = df

###
# transform pipeline setup
###
X, label_cols = preprocessing.shape_gis_pps(X)
transform_pipe = make_one_hot_pipeline(
    X, label_cols,
    [preprocessing.FillNaTransformer(), preprocessing.NumericTransformer()])
# , StandardScaler()

###
# Run Model
###
est_pipeline = Pipeline([
    ('transform_pipeline', transform_pipe),
    ('model', XGBRegressor(n_jobs=4))
])

kcv = KFold(n_splits=7, shuffle=True, random_state=988)
kf_runs = itertools.islice(kcv.split(X), 2)

scores = []
clf = GridSearchCV(
    est_pipeline,
    {'model__max_depth': [6, 7],
     'model__n_estimators': [2000, 1600, 1200]},
    verbose=99,
    cv=kf_runs,
    scoring=make_scorer(score_util.abs_std_n_loss, greater_is_better=False, score_cache=scores))

clf.fit(X, y)
log.info("best params: %s", clf.best_params_)
log.info("best score: %d", clf.best_score_)

log.info("Grid Search Details:")
log.info(clf.cv_results_)

# get the best score objects
# for each grid search and train and test score are recorded, so multiply by two
# the test score is recorded first it seems
rng_start = clf.n_splits_ * 2 * clf.best_index_
rng_end = rng_start + clf.n_splits_ * 2
best_scores = [scores[i] for i in range(rng_start, rng_end, 2)]

best_score = score_util.combine(best_scores)
log.info(best_score)

joblib.dump(best_score, os.path.join(env.result_path, f"score_.pickle"))
joblib.dump(clf.best_estimator_, os.path.join(env.result_path, f"model_.pickle"))
