###
# Run XGBoost directly loading DMatrix from disk
#  hopefully helps memory consumption?
#
# remote exec with:
# bash run_docker_image.sh -d ./ -s nested_cv_excl_elb -- \
#           python -u modeling/scripts/xgb_cv_dmatrix.py --run-id nested_cv_excl_elb
###


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
import shutil
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
# training_data/dumps/gis-pps-aggr/corn-2016-smpl-100k
# training_data_dir = '/var/opt/pcsml/devel/training_data/dumps/gis_pps_corn_2016_sample_200k'
# training_data_dir = '/var/opt/pcsml/devel/training_data/dumps/gis-pps-cells/corn_2016_sample_400k'
# training_data_name = 'gis_pps_cells_corn_2016_sample_400k'
# training_data_dir = '/var/opt/pcsml/devel/training_data/dumps/gis-pps-aggr/corn-2016-smpl-100k'
training_data_dir = '/var/opt/pcsml/devel/training_data/dumps/gis-pps-aggr/corn-2016-smpl-400k'
training_data_name = 'gis_pps_aggr_corn_2016_sample_400k'
output_dir = '/var/opt/pcsml/devel/results/_scratch'
temp_dir = '/var/opt/pcsml/devel/_temp'

run_id = 'dev'
cv_n_outer_splits = 10
cv_n_inner_splits = 10
cv_n_outer_runs = 2
cv_n_inner_runs = 1
xgb_n_threads = 2
xgb_max_depth_min = 30
xgb_max_depth_max = 30
xgb_n_rounds = 500
verbose_eval = True
xgb_early_stopping = 2
debug_override = "--debug" in sys.argv[0]


def _print_stats(y: Union[pandas.Series, np.ndarray, Iterable[float]]) -> str:
    return f"min: {np.min(y):.1f}, max: {np.max(y)}, avg: {np.average(y):.1f}, std: {np.std(y):.1f}"


if debug_override:
    print("LOG-DEBUG: running in debug mode, won't load cmd line params")

if not debug_override and __name__ == '__main__' and not "pydevconsole" in sys.argv[0]:
    parser = argparse.ArgumentParser()
    parser.add_argument('--training-data-dir', '-d', type=str,
                        default='/var/opt/pcsml/training-data/gis-pps/gis_pps_corn_2016_grp_80158')

    parser.add_argument('--training-data-name', '-n', type=str, default='gis_pps_cells_corn_2016__grp80158')
    parser.add_argument('--output-dir', '-o', type=str, default='/var/opt/pcsml/remote-exec-out')
    parser.add_argument('--cv-n-outer-splits', '-cvos', type=int, default=10)
    parser.add_argument('--cv-n-outer-runs', '-cvor', type=int, default=2)
    parser.add_argument('--cv-n-inner-splits', '-cvis', type=int, default=10)
    parser.add_argument('--cv-n-inner-runs', '-cvir', type=int, default=2)
    parser.add_argument('--xgb-n-threads', '-bnt', type=int, default=16)
    parser.add_argument('--xgb-max-depth-min', '-bmdn', type=int, default=7)
    parser.add_argument('--xgb-max-depth-max', '-bmdx', type=int, default=7)
    parser.add_argument('--xgb-n-rounds', '-br', type=int, default=2000)
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
    xgb_early_stopping = 20
    run_id = opt.run_id
    verbose_eval = opt.quiet_eval is False

# xgb_early_stopping = max(10, int(round(xgb_n_rounds / 100)))
# xgb_early_stopping = min(50, xgb_early_stopping)

print(f"early stopping rounds: {xgb_early_stopping}")

# xgb_max_depth_range = range(xgb_max_depth_max, xgb_max_depth_min - 1, -1)
xgb_max_depth_range = range(xgb_max_depth_min, xgb_max_depth_max + 1)

# 'eta': .015,
default_xgb_params = {'objective': 'reg:linear', 'eval_metric': 'mae',
                      'eta': .2,
                      'nthread': xgb_n_threads,
                      'colsample_bytree': 1,
                      # 'alpha': 10,
                      # 'lambda': 20,
                      'silent': 1}

print("default params:")
print(default_xgb_params)

# 'subsample': [.05]
cv_params = ParameterGrid({
    'max_depth': xgb_max_depth_range,
    'subsample': [.8]
})
print("cv_params")
print(list(cv_params))

# create run output dir
result_dir = os.path.join(output_dir, run_id, f"{datetime.utcnow():%Y%m%d_%H%M}")
os.makedirs(result_dir, exist_ok=True)

# ensure temp dir for storing local copy of data
shutil.rmtree(temp_dir, ignore_errors=True)
os.makedirs(temp_dir, exist_ok=True)


def _get_data_file(suffix: str, sep='__') -> str:
    _name = f"{training_data_name}{sep}{suffix}"
    _path = os.path.join(temp_dir, _name)
    if os.path.isfile(_path):
        return _path

    # else, try and copy from training data dir
    _src_path = os.path.join(training_data_dir, _name)
    print(f"copying data file to temp dir: {_src_path}, {_path}")
    shutil.copy2(_src_path, _path)

    return _path


###
# Train model
###
total_model_trainings = (cv_n_outer_runs * cv_n_inner_runs * len(cv_params)) + cv_n_outer_runs
curr_model_training = 0
outer_cv_results = []
inner_cv_results: List[List[cv_util.GridSearchCVResults]] = []
f_importance: List[DataFrame] = []
models: List[xgb.Booster] = []

data_year_ids: pandas.Series = pandas.read_pickle(_get_data_file('df_year_ids.pickle.gz'))
feature_names = pandas.read_pickle(_get_data_file('columns.pickle.gz'))
# feature_names = None


def _load_dmatrix() -> xgb.DMatrix:
    _name = f"{training_data_name}.dmatrix"
    _path = os.path.join(temp_dir, _name)
    print(f"Loading dmatrix: {_path}")

    if not os.path.isfile(_path):
        _gz_path = _get_data_file('.dmatrix.gz', sep='')
        print("extracting dmatrix gzip: " + _gz_path)
        with gzip.open(_gz_path) as gz:
            with open(_path, 'wb') as f:
                shutil.copyfileobj(gz, f)

    matrix = xgb.DMatrix(_path)
    return matrix


kf_outer = GroupKFold(n_splits=cv_n_outer_splits)
kf_outer_runs = itertools.islice(kf_outer.split(data_year_ids, groups=data_year_ids), cv_n_outer_runs)
for test_run_idx, (kf_outer_train_idx, kf_outer_test_idx) in enumerate(kf_outer_runs):
    outer_train_year_ids: pandas.Series = data_year_ids.iloc[kf_outer_train_idx]
    outer_test_year_ids: pandas.Series = data_year_ids.iloc[kf_outer_test_idx]

    print(f"\nOUTER CV TRAINING {test_run_idx + 1} of {cv_n_outer_runs}\n")

    # INNER CV #
    kf_inner = GroupKFold(n_splits=cv_n_inner_splits)
    kf_inner_runs = itertools.islice(kf_inner.split(outer_train_year_ids, groups=outer_train_year_ids), cv_n_inner_runs)
    grid_cv_results = [cv_util.GridSearchCVResults(p) for p in cv_params]
    for idx, (train_idx, test_idx) in enumerate(kf_inner_runs):
        train_year_ids: pandas.Series = outer_train_year_ids.iloc[train_idx]
        test_year_ids: pandas.Series = outer_train_year_ids.iloc[test_idx]

        data = _load_dmatrix()
        train: xgb.DMatrix = data.slice(train_year_ids.index)
        train_label: np.ndarray = train.get_label()
        test: xgb.DMatrix = data.slice(test_year_ids.index)

        del data

        print(f"\n* INNER CV TRAINING *: ({test_run_idx + 1} of {cv_n_outer_runs}) {idx + 1} of {cv_n_inner_runs}")

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

            print(f"training yield stats: {_print_stats(train_label)}")

            # def _eval(pred, d: xgb.DMatrix):
            #     _scr = score_util.ScoreReport(d.get_label(), pred)
            #     return '3std_100x', int(round(_scr.abs_std_3 * 100, 0))

            print(f"training xgb model")
            print(f"train size, test size = "
                  f"({train.num_row():,}, {train.num_col():,}); ({test.num_row():,}; {test.num_col():,})")
            # feval=_eval,
            model: xgb.Booster = xgb.train(
                default_xgb_params, train, num_boost_round=xgb_n_rounds, early_stopping_rounds=xgb_early_stopping,
                evals=eval_list, evals_result=eval_result, verbose_eval=verbose_eval)

            predictions = model.predict(test, ntree_limit=model.best_ntree_limit)
            score = score_util.ScoreReport(test.get_label(), predictions)
            _results = grid_cv_results[i]
            _results.add(
                score.abs_std_3,
                iteration=model.best_iteration,
                mean=score.abs_mean,
                std_dev=score.abs_std,
                y_min=np.min(train_label),
                y_max=np.max(train_label),
                pred_min=np.min(predictions),
                pred_max=np.max(predictions))

            print(f"""
GRID SEARCH RESULT: ({curr_model_training} of total: {total_model_trainings}) 
{test_run_idx + 1}, {idx + 1}, {i + 1} of {len(cv_params) - 1}, {p}
training yield stats: {_print_stats(train_label)}
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

    data = _load_dmatrix()
    train_outer = data.slice(outer_train_year_ids.index)
    test_outer = data.slice(outer_test_year_ids.index)
    del data

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
        result_dir, f_imp_base_name, model, feature_names)
    f_importance.append(f_imp)

    predictions = model.predict(test_outer)
    test_outer_label: np.ndarray = test_outer.get_label()

    score = score_util.ScoreReport(test_outer_label, predictions)
    rdm_guesses = np.empty(len(test_outer_label), dtype=float)
    rdm_guesses.fill(test_outer_label.mean())
    random_guess_score = score_util.ScoreReport(test_outer_label, rdm_guesses)

    print(f"\n**OUTER CV Score\nparams: {best_params}\niterations:{best_iteration}")
    print(score)
    print(f"\nrandom guess score: {random_guess_score}")
    print(f"\ntraining yield stats: {_print_stats(test_outer_label)}")
    print(f"prediction stats: {_print_stats(predictions)}")

    outer_cv_results.append(
        (best_params, best_iteration, score.abs_std_3, best_cv_result.score_avg(), random_guess_score.abs_std_3))

    del train_outer
    del test_outer
    del test_outer_label
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
