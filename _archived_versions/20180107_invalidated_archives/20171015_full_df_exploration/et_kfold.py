from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from data_scripts import pcs_data_loader as dl
from modeling import score_util

result_path = './results/20170828_et_optimizer'

df = dl.shape_pps_data(dl.load_corn_rows_mssql())

# ML
areas = df.pop('Area')
y = df['Dry_Yield'].values
X = df.drop(['Dry_Yield'], axis=1).values

kcv = KFold(n_splits=5, random_state=971)
pipeline = make_pipeline(StandardScaler(), ExtraTreesRegressor(n_jobs=2, verbose=99, n_estimators=10))
scores = []
for train_split_idx, test_split_idx in kcv.split(X):
    X_train_split, y_train_split = X[train_split_idx], y[train_split_idx]
    X_test_split, y_test_split = X[test_split_idx], y[test_split_idx]

    pipeline.fit(X_train_split, y_train_split)
    scr = score_util.score(pipeline, None, X_test_split, y_test_split)
    scores.append(scr)
    print(scr)