import logging

from pandas import DataFrame
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from data_scripts import pcsml_data_loader as dl
from modeling import preprocessing, score_util
from modeling.preprocessing import make_one_hot_pipeline

logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)

result_path = 'results/20171103'

# load the data frame (a sample of the sample to make debugging faster...)
df: DataFrame = dl.load_df_corn_pkl_smpl_25_20171018().sample(100000)
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

scores = []
for train_split_idx, test_split_idx in kcv.split(X):
    X_train_split, y_train_split = X.iloc[train_split_idx], y.iloc[train_split_idx]
    X_test_split, y_test_split = X.iloc[test_split_idx], y.iloc[test_split_idx]

    transform_pipe.fit(X_train_split)
    X_train_transformed = transform_pipe.transform(X_train_split)
    logging.info("Running on input data: %s", X_train_transformed.shape)

    # model = MLPRegressor(verbose=99, max_iter=100000, tol=.01, learning_rate='constant', alpha=.01)
    model = GradientBoostingRegressor(verbose=99, n_estimators=500, min_samples_leaf=2, learning_rate=.2)
    model.fit(X_train_transformed, y_train_split)
    scr = score_util.score(model, transform_pipe, X_test_split, y_test_split)
    scores.append(scr)

# for score in scores:
#     logging.info(score)

combined_score = score_util.combine(scores)
logging.info("kfold scores combined: %s", combined_score)

