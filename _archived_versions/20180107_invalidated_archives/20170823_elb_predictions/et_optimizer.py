from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler

from data_scripts import pcs_data_loader as dl
from modeling import score_util

df = dl.load_corn_data_frame()

# ML
areas = df.pop('Area')
y = df['Dry_Yield']
X = df.drop(['Dry_Yield'], axis=1)
X_train, X_validation, y_train, y_validation = \
    train_test_split(X, y, test_size=.2, random_state=7)

scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)

extra_trees = ExtraTreesRegressor(n_jobs=-1, verbose=True)
extra_trees.fit(X_train_scaled, y_train)
scr = score_util.score(extra_trees, scaler, X_validation, y_validation)

grid_search_cv = GridSearchCV(
    estimator=extra_trees,
    param_grid={"n_estimators": [5,10,15,20,25,30,35,40]},
    error_score=0,
    n_jobs=2,
    verbose=99
)

grid_search_cv.fit(X_train_scaled, y_train)
print(grid_search_cv.best_params_)

scr = score_util.score(grid_search_cv.best_estimator_, scaler, X_validation, y_validation)
print(scr)
