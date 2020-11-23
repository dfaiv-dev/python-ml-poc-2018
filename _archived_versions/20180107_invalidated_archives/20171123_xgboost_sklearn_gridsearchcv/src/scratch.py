import itertools
import logging

from pandas import DataFrame
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from xgboost.sklearn import XGBRegressor

from data_scripts import pcsml_data_loader as dl
from modeling import preprocessing, score_util
from modeling.preprocessing import make_one_hot_pipeline

logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)

result_path = 'results/20171103'

# load the data frame (a sample of the sample to make debugging faster...)
df: DataFrame = dl.load_df_corn_pkl_smpl_25_20171018().sample(1000)
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

kcv = KFold(n_splits=10, shuffle=True, random_state=972)
kf_runs = itertools.islice(kcv.split(X), 3)

scores = []
scorer = make_scorer(score_util.abs_std_n_loss, greater_is_better=False, score_cache=scores)

for i, (train_split_idx, test_split_idx) in enumerate(kf_runs):
    log.info("Running kfold: %s", i)

    X_train_split, y_train = X.iloc[train_split_idx], y.iloc[train_split_idx]
    X_test_split, y_test = X.iloc[test_split_idx], y.iloc[test_split_idx]

    log.info("fitting transforms")
    transform_pipe.fit(X_train_split)
    log.info("transforming")
    X_train = transform_pipe.transform(X_train_split)
    X_test = transform_pipe.transform(X_test_split)

    log.info("training model")
    model = XGBRegressor(n_jobs=4, max_depth=2, n_estimators=4)
    model.fit(X_train, y_train, eval_set=[X_test, y_test.values], eval_metric="mae", early_stopping_rounds=50, verbose=True)

    log.info("scoring")
    score = scorer(y_test, model.predict(X_test))
    log.info(score)
