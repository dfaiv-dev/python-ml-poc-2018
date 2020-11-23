import argparse
import itertools
import logging
import os

import seaborn as sns
import xgboost as xgb
from matplotlib import pyplot as plt
from pandas import DataFrame
from sklearn.model_selection import KFold

import env
import util.logging
from data_scripts import pcsml_data_loader as dl
from modeling import preprocessing, xgb_util
from modeling.preprocessing import make_one_hot_pipeline

parser = argparse.ArgumentParser()
parser.add_argument('--env', '-e', type=str)
parser.add_argument('--result-path', '-o', type=str)
opt, _ = parser.parse_known_args()

util.logging.setup_default(opt.result_path)

log = logging.getLogger(__name__)
log.info("Running...")
log.info("Env:\n%s", env.dump())

result_path = opt.result_path
if result_path is None:
    result_path = 'results/_scratch'

os.makedirs(result_path, exist_ok=True)


def create_result_file_path(filename: str):
    return os.path.join(result_path, filename)


# load the data frame (a sample of the sample to make debugging faster...)
# df: DataFrame = dl.load_df_corn_pkl_smpl_25_20171018().sample(2000)
# dl.dump_sample(df, 'df-corn-20171018-scratch-debug.pickle')
df: DataFrame = dl.load_pickled('df-corn-20171018-scratch-debug.pickle')

# df: DataFrame = dl.load_df_corn_pkl_smpl_25_20171018()
log.debug("data shape: %s", df.shape)

y = df.pop('Dry_Yield')
X = df

###
# transform pipeline setup
###
X, label_cols = dl.shape_gis_pps(X)
transform_pipe, one_hot_label_enc = make_one_hot_pipeline(
    X, label_cols,
    [preprocessing.FillNaTransformer(), preprocessing.NumericTransformer()])


# , StandardScaler()


###
# Run Model
###
def save_plot(name: str):
    plt.tight_layout(pad=1.25)
    plt.savefig(create_result_file_path(name))


kcv = KFold(n_splits=7, shuffle=True, random_state=988)
kf_runs = itertools.islice(kcv.split(X), 2)

scores = []
for (i, (idx_train, idx_test)) in enumerate(kf_runs):
    X_train, y_train = X.iloc[idx_train], y.iloc[idx_train]
    X_test, y_test = X.iloc[idx_test], y.iloc[idx_test]

    log.info("fitting transforms")
    transform_pipe.fit(X_train)
    log.info("transforming")
    X_train = transform_pipe.transform(X_train)
    X_test = transform_pipe.transform(X_test)

    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=one_hot_label_enc.column_names_)
    dtest = xgb.DMatrix(X_test, label=y_test, feature_names=one_hot_label_enc.column_names_)
    params = {'max_depth': 3, 'objective': 'reg:linear', 'eval_metric': 'mae', 'nthread': 3, 'silent': 1}

    evallist = [(dtest, 'test'), (dtrain, 'train')]
    eval_result = {}
    model: xgb.Booster = xgb.train(params, dtrain, num_boost_round=50, evals=evallist, evals_result=eval_result)
    feature_imp = xgb_util.merge_labeled_weight_importance(model, one_hot_label_enc)
    top_n = xgb_util.top_n_importance(feature_imp, 15)
    print("top N importance")
    print(top_n)

    plt.clf()
    sns.barplot(x='Importance', y='Feature', data=DataFrame(top_n, columns=['Feature', 'Importance']),
                palette='Blues_d')
    save_plot('feature_importance.png')

    eval_df = DataFrame({
        'train': eval_result['train']['mae'],
        'test': eval_result['test']['mae']
    }).melt()

    plt.clf()
    g = sns.FacetGrid(eval_df, hue='variable', size=5, aspect=1.5)
    g.map(plt.plot, 'value').add_legend()
    g.ax.set(xlabel='Iteration',
             ylabel='Mean Abs Err',
             title='')
    save_plot("eval.png")
