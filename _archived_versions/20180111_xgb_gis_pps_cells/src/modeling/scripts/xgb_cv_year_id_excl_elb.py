import argparse
import gc
import gzip
import itertools
import os
import pickle
import sys
from datetime import datetime
from typing import List, Tuple, Union, Iterable

import numpy
import psutil
import numpy as np
import pandas
import xgboost as xgb
from pandas import DataFrame
from scipy import sparse
from sklearn.model_selection import KFold, ParameterGrid, GroupKFold
from pympler.tracker import SummaryTracker

import data.pcsml_data_loader as dl
from modeling import score_util, cv_util, xgb_util, categorical_util

# opts
# debug settings
# training_data = '/var/opt/pcsml/devel/training_data/dumps/df-corn-smpl_25-gis-pps-20171018.pkl'
training_data_dir = '/var/opt/pcsml/devel/training_data/dumps/'
training_data_name = 'df_gis_pps_corn_2016_sample_200k'
output_dir = '/var/opt/pcsml/devel/results/_scratch'
run_id = 'dev'
cv_n_outer_splits = 3
cv_n_inner_splits = 3
cv_n_outer_runs = 2
cv_n_inner_runs = 1
xgb_n_threads = 2
xgb_max_depth_min = 12
xgb_max_depth_max = 12
xgb_n_rounds = 500
verbose_eval = True
debug_override = "--debug" in sys.argv[0]


def _print_stats(y: Union[pandas.Series, np.ndarray, Iterable[float]]) -> str:
    return f"min: {np.min(y):.1f}, max: {np.max(y)}, avg: {np.average(y):.1f}, std: {np.std(y):.1f}"


if debug_override:
    print("LOG-DEBUG: running in debug mode, won't load cmd line params")

# remote exec with:
# bash run_docker_image.sh -d ./ -s nested_cv_excl_elb -- \
#           python -u modeling/scripts/xgb_cv_year_id_excl_elb.py --run-id nested_cv_excl_elb

if not debug_override and __name__ == '__main__' and not "pydevconsole" in sys.argv[0]:
    parser = argparse.ArgumentParser()
    parser.add_argument('--training-data-dir', '-d', type=str,
                        default='/var/opt/pcsml/training-data/gis-pps/')
    parser.add_argument('--training-data-name', '-n', type=str, default='df_gis_pps_corn_2016_smpl_2k')
    parser.add_argument('--output-dir', '-o', type=str, default='/var/opt/pcsml/remote-exec-out')
    parser.add_argument('--cv-n-outer-splits', '-cvos', type=int, default=3)
    parser.add_argument('--cv-n-outer-runs', '-cvor', type=int, default=2)
    parser.add_argument('--cv-n-inner-splits', '-cvis', type=int, default=3)
    parser.add_argument('--cv-n-inner-runs', '-cvir', type=int, default=1)
    parser.add_argument('--xgb-n-threads', '-bnt', type=int, default=16)
    parser.add_argument('--xgb-max-depth-min', '-bmdn', type=int, default=6)
    parser.add_argument('--xgb-max-depth-max', '-bmdx', type=int, default=6)
    parser.add_argument('--xgb-n-rounds', '-br', type=int, default=500)
    parser.add_argument('--run-id', '-r', type=str, default=f'{datetime.utcnow():%Y-%m-%dT%H-%M-%S}')
    parser.add_argument('--quiet-eval', '-qe', action='store_true', default=False)

    opt = parser.parse_args()

    print("parsed opts:")
    print(opt)
    training_data_dir = opt.training_data_dir
    training_data_name = opt.training_data_name
    output_dir = opt.output_dir
    cv_n_outer_splits = opt.cv_n_outer_splits
    cv_n_inner_splits = opt.cv_n_inner_splits
    cv_n_outer_runs = opt.cv_n_outer_runs
    cv_n_inner_runs = opt.cv_n_inner_runs
    xgb_n_threads = opt.xgb_n_threads
    xgb_max_depth_min = opt.xgb_max_depth_min
    xgb_max_depth_max = opt.xgb_max_depth_max
    xgb_n_rounds = opt.xgb_n_rounds
    run_id = opt.run_id
    verbose_eval = opt.quiet_eval is False

training_data_base_name = os.path.join(training_data_dir, training_data_name)
# xgb_early_stopping = max(10, int(round(xgb_n_rounds / 100)))
# xgb_early_stopping = min(50, xgb_early_stopping)
xgb_early_stopping = 10
print(f"early stopping rounds: {xgb_early_stopping}")

# xgb_max_depth_range = range(xgb_max_depth_max, xgb_max_depth_min - 1, -1)
xgb_max_depth_range = range(xgb_max_depth_min, xgb_max_depth_max + 1)

# 'eta': .015,
default_xgb_params = {'objective': 'reg:linear', 'eval_metric': 'mae',
                      'nthread': xgb_n_threads,
                      'silent': 1}

print("default params:")
print(default_xgb_params)

# 'subsample': [.05]
cv_params = ParameterGrid({
    'max_depth': xgb_max_depth_range,
    'subsample': [1]
})
print("cv_params")
print(list(cv_params))

###
# create run output dir
result_dir = os.path.join(output_dir, run_id, f"{datetime.utcnow():%Y%m%d_%H%M}")
os.makedirs(result_dir, exist_ok=True)

###
# - create data exploration outputs (graphs, etc)
# - run the xgboost model, saving model output
###
print(f"loading training data to find year ids: {training_data_base_name}.pickle.gz")
data: DataFrame = pandas.read_pickle(training_data_base_name + ".pickle.gz")
data_dummies: pandas.SparseDataFrame = pandas.read_pickle(training_data_base_name + "_dummies.pickle.gz")

# clean up yield vals
yld_mean = data['Dry_Yield'].mean()
yld_std = data['Dry_Yield'].std()
yld_rng_min = max(0, yld_mean - (yld_std * 3))
yld_rng_max = yld_mean + (yld_std * 3)
print(f"constraining yield range to: {yld_rng_min:.2f}, {yld_rng_max:.2f}")
data.loc[data['Dry_Yield'] >= yld_rng_max, 'Dry_Yield'] = yld_rng_max
data.loc[data['Dry_Yield'] <= yld_rng_min, 'Dry_Yield'] = yld_rng_min
print(f"training yield stats: {_print_stats(data.loc[:, 'Dry_Yield'])}")

year_ids: np.ndarray = data['YearID'].unique()
print(f"total year ids: {len(year_ids)}")
elb_year_ids = dl.elb_year_ids()
year_ids: np.ndarray = year_ids[~np.isin(year_ids, elb_year_ids)]
print(f"year ids without elbs: {len(year_ids)}")

elb_data: pandas.DataFrame = data.loc[data['YearID'].isin(elb_year_ids)]
elb_dummies: pandas.SparseDataFrame = data_dummies.loc[elb_data.index]

data = data.loc[data['YearID'].isin(year_ids)]
data_dummies: pandas.SparseDataFrame = data_dummies.loc[data.index]
data_areas: np.ndarray = data.pop('Area').values
data_year_ids: np.ndarray = data.pop('YearID').values
data_labels: np.ndarray = data.pop('Dry_Yield').values
data.drop(dl.exclude_columns + dl.yield_dep_columns, axis=1, inplace=True, errors='ignore')
data_feature_names = data.columns.append(data_dummies.columns).values

print("converting training data to sparse ndarray")
data: numpy.ndarray = data.as_matrix()
data = sparse.csr_matrix(data)
data_dummies = data_dummies.to_coo().tocsr()
data = sparse.hstack((data, data_dummies))
print("converting to sparse numpy (csr)")
data = data.tocsr()

print("Encoding elb dataset")
elb_data = elb_data.drop(dl.exclude_columns + dl.yield_dep_columns, axis=1, errors='ignore')
elb_y = elb_data.pop('Dry_Yield')
elb_data = sparse.hstack(
    (
        sparse.csr_matrix(elb_data.as_matrix()),
        elb_dummies.to_coo().tocsr()
    ))
elb_data = xgb.DMatrix(elb_data, label=elb_y, feature_names=data_feature_names)


def _test_elbs(model):
    if elb_data.num_row() == 0:
        print("no test elbs found, skipping scoring")
        return

    elb_pred = model.predict(elb_data)
    score = score_util.ScoreReport(elb_y, elb_pred)
    elb_cv_results.append(score.abs_99)
    print(score)
    print(f"elb yield stats: {_print_stats(elb_y)}")
    print(f"elb prediction stats: {_print_stats(elb_pred)}")


total_model_trainings = (cv_n_outer_runs * cv_n_inner_runs * len(cv_params)) + cv_n_outer_runs
curr_model_training = 0
outer_cv_results = []
inner_cv_results: List[List[cv_util.GridSearchCVResults]] = []
elb_cv_results = []
f_importance: List[DataFrame] = []
models: List[xgb.Booster] = []

kf_outer = GroupKFold(n_splits=cv_n_outer_splits)
kf_outer_runs = itertools.islice(kf_outer.split(data, groups=data_year_ids), cv_n_outer_runs)
for test_run_idx, (kf_outer_train_idx, kf_outer_test_idx) in enumerate(kf_outer_runs):
    train_outer, train_outer_y = data[kf_outer_train_idx], data_labels[kf_outer_train_idx]
    train_year_ids = data_year_ids[kf_outer_train_idx]
    test_outer, test_outer_y = data[kf_outer_test_idx], data_labels[kf_outer_test_idx]

    print(f"\nOUTER CV TRAINING {test_run_idx + 1} of {cv_n_outer_runs}\n")

    # INNER CV #
    kf_inner = GroupKFold(n_splits=cv_n_inner_splits)
    kf_inner_runs = itertools.islice(kf_inner.split(train_outer, groups=train_year_ids), cv_n_inner_runs)
    grid_cv_results = [cv_util.GridSearchCVResults(p) for p in cv_params]
    for idx, (train_idx, test_idx) in enumerate(kf_inner_runs):
        train, train_y = train_outer[train_idx], train_outer_y[train_idx]
        test, test_y = train_outer[test_idx], train_outer_y[test_idx]

        print(f"\n* INNER CV TRAINING *: ({test_run_idx + 1} of {cv_n_outer_runs}) {idx + 1} of {cv_n_inner_runs}")

        train = xgb.DMatrix(train, train_y, feature_names=data_feature_names)
        test = xgb.DMatrix(test, test_y, feature_names=data_feature_names)
        # run all params
        for i, p in enumerate(cv_params):
            default_xgb_params.update(p)

            curr_model_training += 1

            print(f"""
GRID SEARCH: ({curr_model_training} of total: {total_model_trainings}) 
{test_run_idx + 1}, {idx + 1}, {i + 1} of {len(cv_params)}, {p}""")

            #######
            # By putting test last in the eval list, it will be used for early stopping.
            # - OK, since we then CV in the outer loop too?
            #######
            eval_list = [(train, 'train'), (test, 'test')]
            eval_result = {}

            print(f"training yield stats: {_print_stats(train.get_label())}")


            def _eval(pred, d: xgb.DMatrix):
                _scr = score_util.ScoreReport(d.get_label(), pred)
                return '3std_100x', int(round(_scr.abs_99 * 100, 0))


            print(f"training xgb model")
            model: xgb.Booster = xgb.train(
                default_xgb_params, train, num_boost_round=xgb_n_rounds, early_stopping_rounds=xgb_early_stopping,
                evals=eval_list, evals_result=eval_result, feval=_eval, verbose_eval=verbose_eval)

            predictions = model.predict(test, ntree_limit=model.best_ntree_limit)
            score = score_util.ScoreReport(test.get_label(), predictions)
            _results = grid_cv_results[i]
            _results.add(
                score.abs_99,
                iteration=model.best_iteration,
                mean=score.abs_mean,
                std_dev=score.abs_std,
                y_min=np.min(train.get_label()),
                y_max=np.max(train.get_label()),
                pred_min=np.min(predictions),
                pred_max=np.max(predictions))

            print(f"""
GRID SEARCH RESULT: ({curr_model_training} of total: {total_model_trainings}) 
{test_run_idx + 1}, {idx + 1}, {i + 1} of {len(cv_params) - 1}, {p}
training yield stats: {_print_stats(train.get_label())}
pred yield stats: {_print_stats(predictions)}""")
            print(score)

        # clear memory
        del train
        del test

    inner_cv_results.append(grid_cv_results)

    # OUTER CV #
    # get best score from inner cv
    print("\nINNER CV RESULTS")
    print(sorted(grid_cv_results, key=lambda x: x.score_avg()))
    best_cv_result = min(grid_cv_results, key=lambda r: r.score_avg())
    best_params = best_cv_result.params
    print(
        f"best CV result: {best_params}\n"
        f"avg score: {best_cv_result.score_avg():.3F}\n"
        f"avg attrs:{best_cv_result.all_attr_avg()}")

    print(f"\n\n** TRAINING OUTER TEST **")
    curr_model_training += 1
    print(f"{curr_model_training} of total: {total_model_trainings}")

    train_outer = xgb.DMatrix(train_outer, label=train_outer_y, feature_names=data_feature_names)
    test_outer = xgb.DMatrix(test_outer, label=test_outer_y, feature_names=data_feature_names)

    default_xgb_params.update(best_params)
    best_iteration = int(round(best_cv_result.all_attr_avg()['iteration']))

    eval_list = [(train_outer, 'train')]
    model: xgb.Booster = xgb.train(default_xgb_params, train_outer, num_boost_round=best_iteration, evals=eval_list,
                                   verbose_eval=verbose_eval)
    models.append(model)
    model_path = f"{result_dir}/cv{test_run_idx}_model.xgb"
    print(f"saving model: {model_path}")
    model.save_model(model_path)

    f_imp_base_name = f"cv{test_run_idx}_f_imp"
    f_imp = xgb_util.feature_importance_dummy_merged_report(
        result_dir, f_imp_base_name, model)
    f_importance.append(f_imp)

    predictions = model.predict(test_outer)
    score = score_util.ScoreReport(test_outer_y, predictions)
    rdm_guesses = np.empty(len(test_outer_y), dtype=float)
    rdm_guesses.fill(yld_mean)
    random_guess_score = score_util.ScoreReport(test_outer_y, rdm_guesses)

    print(f"\nOUTER CV Score\nparams: {best_params}\niterations:{best_iteration}")
    print(score)
    print(f"random guess score: {random_guess_score}")
    print(f"training yield stats: {_print_stats(train_outer.get_label())}")
    print(f"prediction stats: {_print_stats(predictions)}")

    outer_cv_results.append(
        (best_params, best_iteration, score.abs_99, best_cv_result.score_avg(), random_guess_score.abs_99))

    _test_elbs(model)

    del train_outer
    del test_outer
    del train_outer_y
    del test_outer_y
    gc.collect()

# cv final result
print("\n*** FINAL CV RESULTS ***")

print("\n* INNER CV *")
for i in range(len(cv_params)):
    cv_results_ = [r[i] for r in inner_cv_results]
    for cv_r in cv_results_:
        print(f"{cv_r}, {cv_r.all_attr_avg()}")

print("\n* Feature Importance *")
fi_merged = f_importance[0]
for fi_df in f_importance[1:]:
    fi_merged = pandas.merge(fi_merged, fi_df, how='outer', on='Feature')

fi_total_df = pandas.DataFrame(fi_merged.loc[:, 'Feature'])
fi_total_df['Importance'] = fi_merged.fillna(0).sum(axis=1)
fi_total_df['Importance'] = fi_total_df['Importance'].astype(int)
fi_total_df.sort_values(by='Importance', ascending=False, inplace=True)
fi_total_df.to_csv(os.path.join(result_dir, 'cv_all_feature_importance.csv'))
xgb_util.plot_feature_importance(
    os.path.join(result_dir, 'cv_all_feature_importance.png'),
    {fi[0]: fi[1] for fi in fi_total_df.values})
# noinspection PyUnresolvedReferences
print([(fi[0], fi[1]) for fi in fi_total_df.head(20).values])

print("\n* ELB SCORING *")
print(elb_cv_results)

print("\n* OUTER CV TEST RESULTS *")
for r in outer_cv_results:
    print(f"params: {r[0]}, iterations: {r[1]}, "
          f"test abs 99: {r[2]:.2f}, "
          f"random guess abs 99: {r[4]:.2f}, "
          f"cv abs 99: {r[3]:.2f}")

print("\n** AVERAGES **")
cv_abs99_avg = np.average([r[2] for r in outer_cv_results])
guess_abs99_avg = np.average([r[4] for r in outer_cv_results])
print(f"iterations: {np.average([r[1] for r in outer_cv_results]):.2f}")
print(f"inner cv avg: {np.average([r[3] for r in outer_cv_results]):.2f}")
print(f"test abs_99: {cv_abs99_avg:.2f}")
print(f"rdm guess abs_99: {guess_abs99_avg:.2f}")
print(f"elb abs_99: {np.average(elb_cv_results):.2f}")
