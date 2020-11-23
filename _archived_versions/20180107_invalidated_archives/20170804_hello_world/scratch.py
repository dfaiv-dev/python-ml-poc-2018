import logging
import sys

import colorama
import coloredlogs
import numpy
import sklearn.ensemble as skl_ens
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error, median_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from data_scripts import pcs_data_loader as dl
from modeling import score_util

colorama.init()

root_log = logging.getLogger()
for h in root_log.handlers:
    root_log.removeHandler(h)
handler = logging.StreamHandler(stream=sys.stdout)

handler.setFormatter(coloredlogs.ColoredFormatter())
handler.addFilter(coloredlogs.HostNameFilter())
root_log.addHandler(handler)

logger = logging.getLogger(__name__)

logger.info("hello!")
logger.warning("warn!")

df = dl.load_corn_data_frame()

# ML
areas = df.pop('Area')
y = df['Dry_Yield']
X = df.drop(['Dry_Yield'], axis=1)
X_train, X_validation, y_train, y_validation = \
    train_test_split(X, y, test_size=.3, random_state=7)

scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)

logger.info("Fitting extra trees")
extra_trees = ExtraTreesRegressor(n_jobs=-1, verbose=True)
extra_trees.fit(X_train_scaled, y_train)
scr = score_util.score(extra_trees, scaler, X_validation, y_validation)

# explore
# seed = 11
# num_folds = 3
# model = Pipeline([('Scaler', StandardScaler()), ('ET', ExtraTreesRegressor())])
# kfold = KFold(n_splits=num_folds, random_state=seed)
#
# scoring = 'neg_mean_squared_error'
# cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring, n_jobs=-1)


def _score(m):
    scaler.fit(X_validation)
    predictions = m.predict(scaler.transform(X_validation))
    mean_sq = mean_squared_error(y_validation, predictions)
    r2 = r2_score(y_validation, predictions)
    median_abs = median_absolute_error(y_validation, predictions)
    err_abs = numpy.abs(y_validation - predictions)
    err_std = err_abs.std()
    err_max = err_abs.max()

    print("prediction > mean_sq: %f, r2: %f, median_abs: %f, err_std: %f, err_max: %f" % (
        mean_sq, r2, median_abs, err_std, err_max))


def _run_model(m):
    name, _m = m
    print('\n*** Running %s' % name)
    _m.fit(X_train_scaled, y_train)
    _score(_m)


ensembles = (
    ('AdaBoost', skl_ens.AdaBoostRegressor()),
    ('GradientBoost', skl_ens.GradientBoostingRegressor()),
    ('RandomForest', skl_ens.RandomForestRegressor()),
    ('ExtraTrees', skl_ens.ExtraTreesRegressor())
)

for m in ensembles:
    _run_model(m)


# scaler.fit(X_validation)
# predictions = extra_trees.predict(scaler.transform(X_validation))
# abs_errs = numpy.abs(y_validation - predictions)
# perc_errs = abs_errs / y_validation * 100
# val_results = X_validation.assign(err_abs=abs_errs, dry_yield=y_validation, pred=predictions, perc_errs=perc_errs)
# val_results.sort_values(by='perc_errs', ascending=False, inplace=True)
# val_results.head(50)
# worst_results = val_results.query('err_abs > 10')

# _run_model(extra_trees)

# Neural Networks
print("\n*** Fitting NN")

alphas = (.001, .01, .1, 1, 10, 100, 1000, 10000)
results = list()
for a in alphas:
    nn = MLPRegressor(alpha=a, max_iter=1000, verbose=True)
    _run_model(("NN (alpha=%f5)" % a, nn))
    results.append(nn)

for nn in results:
    _score(nn)
