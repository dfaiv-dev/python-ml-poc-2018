import logging
import sys

import colorama
import coloredlogs
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

df = dl.shape_pps_data(dl.load_corn_rows_mssql())

# ML
areas = df.pop('Area')
y = df['Dry_Yield']
X = df.drop(['Dry_Yield'], axis=1)
X_train, X_validation, y_train, y_validation = \
    train_test_split(X, y, test_size=.2, random_state=7)

scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)

nn = MLPRegressor(random_state=7, verbose=99, max_iter=5000)
nn.fit(X_train_scaled, y_train)

scr = score_util.score(nn, scaler, X_validation, y_validation)
print(scr)

# grid = GridSearchCV(
#     estimator=nn,
#     param_grid={
#         'activation': ['relu', 'tanh', 'logistic'],
#         'solver': ['adam'],
#         'alpha': [.0000001, .000001, .00001, .0001],
#         'learning_rate_init': [.0000001, .000001, .00001, .0001],
#         'hidden_layer_sizes': [
#             (2048, 1024, 512),
#             (512, 256, 128, 32, 16),
#             (256, 128, 32, 16, 8, 4),
#         ]},
#     verbose=100,
#     n_jobs=2,
#     cv=2
# )
#
# grid.fit(X_train, y_train)
#
# with open('./results/20170824_tf_dnn_reg/skmlp_grid_search.pickle', 'wb') as f:
#     pickle.dump(grid, f)
# with open('./results/20170824_tf_dnn_reg/skmlp_model.pickle', 'wb') as f:
#     pickle.dump(grid, f)


