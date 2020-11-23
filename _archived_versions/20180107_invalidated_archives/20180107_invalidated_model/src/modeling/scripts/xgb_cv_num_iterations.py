import argparse
import itertools
import os
import pickle
import sys
from datetime import datetime

import numpy
import pandas
import seaborn as sns
import xgboost as xgb
from matplotlib import pyplot as plt
from pandas import DataFrame
from sklearn.model_selection import KFold, train_test_split

import data.pcsml_data_loader as dl
import modeling.categorical_util as dummy_util
from modeling import xgb_util, score_util
from util import plot_util

# opts
# debug settings
# training_data = '/var/opt/pcsml/devel/training_data/dumps/df-corn-smpl_25-gis-pps-20171018.pkl'
training_data = '/var/opt/pcsml/devel/training_data/dumps/transformed_sample_10000.pickle'
output_dir = '/var/opt/pcsml/devel/results/_scratch'
run_id = 'dev'
sample_n = 5000
describe_data = False
cv_n_splits = 7
cv_n_runs = 2
xgb_n_threads = 2
xgb_max_depth = 4
xgb_n_rounds = 50
debug_override = "--debug" in sys.argv[0]

if debug_override:
    print("LOG-DEBUG: running in debug mode, won't load cmd line params")

if not debug_override and __name__ == '__main__' and not "pydevconsole" in sys.argv[0]:
    parser = argparse.ArgumentParser()
    parser.add_argument('--training-data', '-d', type=str, default='/var/opt/pcsml/training-data/gis-pps/df-corn-gis-pps-20171018.transformed.pickle')
    parser.add_argument('--output-dir', '-o', type=str, default='/var/opt/pcsml/remote-exec-out')
    parser.add_argument('--cv-n-splits', '-cvs', type=int, default=10)
    parser.add_argument('--cv-n-runs', '-cvr', type=int, default=3)
    parser.add_argument('--xgb-n-threads', '-bnt', type=int, default=16)
    parser.add_argument('--xgb-max-depth', '-bmd', type=int, default=7)
    parser.add_argument('--xgb-n-rounds', '-br', type=int, default=2000)
    parser.add_argument('--run-id', '-r', type=str, default=f'{datetime.utcnow():%Y-%m-%dT%H-%M-%S}')
    parser.add_argument('--sample-n', '-s', type=int)

    opt = parser.parse_args()

    print("parsed opts:")
    print(opt)
    training_data = opt.training_data
    output_dir = opt.output_dir
    sample_n = opt.sample_n
    cv_n_splits = opt.cv_n_splits
    cv_n_runs = opt.cv_n_runs
    xgb_n_threads = opt.xgb_n_threads
    xgb_max_depth = opt.xgb_max_depth
    xgb_n_rounds = opt.xgb_n_rounds
    run_id = opt.run_id

xgb_early_stopping = max(10, int(round(xgb_n_rounds / 40)))

###
# create run output dir
result_dir = os.path.join(output_dir, run_id)
os.makedirs(result_dir, exist_ok=True)

###
# - create data exploration outputs (graphs, etc)
# - run the xgboost model, saving model output
###
print(f"reading in training data, may take a while: {training_data}")
df: DataFrame = pandas.read_pickle(training_data)

if sample_n is not None:
    df = df.sample(sample_n)

print(f"training data original shape: {df.shape}")
# filter 2017
print(f"LOG: filtering data, dropping columns")
df = df[df.Year < 2017]
df.drop(dl.exclude_columns, axis=1, inplace=True, errors='ignore')
print(f"training data, filtered: {df.shape}")

###
# TRAIN MODEL
###

# split outer test/train #
df_train, df_test = train_test_split(df, test_size=.15)

y = df_train.pop('Dry_Yield')
X = df_train
print(f"LOG: train set split: {df_train.shape}, {df_test.shape}")

kcv = KFold(n_splits=cv_n_splits, shuffle=True, random_state=9134)
kf_runs = itertools.islice(kcv.split(X), cv_n_runs)
scores = []
f_importances = []
models = []
for (i, (idx_train, idx_test)) in enumerate(kf_runs):
    print(f"LOG: running kfold: {i}")

    X_train, y_train = X.iloc[idx_train], y.iloc[idx_train]
    X_test, y_test = X.iloc[idx_test], y.iloc[idx_test]

    # clean up categoris
    print(f"LOG: encoding categories")
    column_categories = dummy_util.get_categories_lookup(X_train)
    dummy_util.set_categories(X_train, column_categories)
    dummy_util.set_categories(X_test, column_categories)

    dummy_enc = dummy_util.create_dummy_encoder(X_train, column_categories)
    print(f"LOG: dummy encoder -- fitting to training data")
    X_train = dummy_enc.fit_transform(dummy_util.encode_categories(X_train))
    print(f"LOG: dummy encoder -- tranforming test data")
    X_test = dummy_enc.transform(dummy_util.encode_categories(X_test))

    print(f"LOG: building train/test xgb DMatrix")
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=dummy_enc.transformed_column_names)
    dtest = xgb.DMatrix(X_test, label=y_test, feature_names=dummy_enc.transformed_column_names)
    params = {'max_depth': xgb_max_depth, 'subsample': 1, 'colsample_bytree': 1,
              'objective': 'reg:linear', 'eval_metric': 'mae',
              'nthread': xgb_n_threads,
              'silent': 1}

    #######
    # By putting dtest last in the eval list, it will be used for early stopping.
    # - this is probably not robust, since we're reporting test error on the early stopping of the
    #       test set.  Probably need a "validation" outer data split, and report error from that?
    #######
    evallist = [(dtrain, 'train'), (dtest, 'test')]
    eval_result = {}
    print(f"training xgb model")
    model: xgb.Booster = xgb.train(params, dtrain, num_boost_round=xgb_n_rounds, evals=evallist,
                                   evals_result=eval_result, early_stopping_rounds=xgb_early_stopping)
    models.append(model)

    ###
    # plot and output model stats
    ###
    feature_imp = xgb_util.merge_labeled_weight_importance(model,dummy_enc)
    f_importances.append(feature_imp)
    f_importance_top_n = xgb_util.top_n_importance(feature_imp, 20)
    print("top N importance")
    print(f_importance_top_n)

    plt.clf()
    sns.barplot(x='Importance', y='Feature', data=DataFrame(f_importance_top_n, columns=['Feature', 'Importance']),
                palette='Blues_d')
    plot_util.save_plot(os.path.join(result_dir, f"cv{i}_feature_importance.png"))

    eval_df = DataFrame({
        'train': eval_result['train']['mae'][10:],
        'test': eval_result['test']['mae'][10:]
    }).melt()

    plt.clf()
    g = sns.FacetGrid(eval_df, hue='variable', size=5, aspect=1.5)
    g.map(plt.plot, 'value').add_legend()
    g.ax.set(xlabel='Iteration',
             ylabel='Mean Abs Err',
             title='')
    plot_util.save_plot(os.path.join(result_dir, f"cv{i}_eval.png"))

    # eval perf
    predictions = model.predict(dtest, ntree_limit=model.best_ntree_limit)
    score = score_util.ScoreReport(y_test, predictions)
    scores.append(score)
    print(f"\n{score}")
    with open(os.path.join(result_dir, f"cv{i}_score.txt"), "w") as f_scr_txt:
        f_scr_txt.write(str(score))
    with open(os.path.join(result_dir, f"cv{i}_score.pickle"), "wb") as f_scr_pkl:
        pickle.dump(score, f_scr_pkl)
    # write model
    model.save_model(os.path.join(result_dir, f"cv{i}_model.bin"))

all_scores = score_util.combine(scores)

### run on test
# noinspection PyTypeChecker
best_iteration_mean = int(round(numpy.mean([m.best_iteration for m in models])))
# noinspection PyTypeChecker
best_ntree_limit_mean = int(round(numpy.mean([m.best_iteration for m in models])))

X_train = X
y_train = y
y_test = df_test.pop('Dry_Yield')
X_test = df_test

column_categories = dummy_util.get_categories_lookup(X_train)
dummy_util.set_categories(X_train, column_categories)

dummy_enc = dummy_util.create_dummy_encoder(X_train, column_categories)
X_train = dummy_enc.fit_transform(dummy_util.encode_categories(X_train))

dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=dummy_enc.transformed_column_names)
params = {'max_depth': xgb_max_depth, 'subsample': 1, 'colsample_bytree': 1,
          'objective': 'reg:linear', 'eval_metric': 'mae',
          'nthread': xgb_n_threads,
          'silent': 1}

evallist = [(dtrain, 'train')]
best_model: xgb.Booster = xgb.train(params, dtrain, num_boost_round=best_iteration_mean, evals=evallist)

dummy_util.set_categories(X_test, column_categories)
X_test = dummy_enc.transform(dummy_util.encode_categories(X_test))
dtest = xgb.DMatrix(X_test, label=y_test, feature_names=dummy_enc.transformed_column_names)
pred_test = best_model.predict(dtest)
test_score = score_util.ScoreReport(y_test, pred_test)


# dump results
best_model.save_model(os.path.join(result_dir, f"best_model.bin"))

print(f"\n\n**** RESULTS ****")
print(f"best_iteration_mean: {best_iteration_mean}")
print(f"\ncombined cv score:\n{all_scores}")
print(f"\ntest score:\n{test_score}")

# graph CV feature importance
total_feature_importance = {}
for imp in f_importances:
    for (name, val) in imp.items():
        total_feature_importance[name] = total_feature_importance.get(name, 0) + val

f_importance_top_n = xgb_util.top_n_importance(total_feature_importance, 20)
print("\nCV top N importance")
print(f_importance_top_n)

plt.clf()
sns.barplot(x='Importance', y='Feature', data=DataFrame(f_importance_top_n, columns=['Feature', 'Importance']),
            palette='Blues_d')
plot_util.save_plot(os.path.join(result_dir, f"cv_combined_f_importance.png"))
