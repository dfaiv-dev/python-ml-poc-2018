###
# Run SKLearn Extra Trees directly loading np sparse csr matrix from disk
#  hopefully helps memory consumption?
#
# remote exec with:
# bash run_docker_image.sh -d ./ -s sklearn_extra_trees_cv -- \
#           python -u modeling/scripts/sklearn_extra_trees_cv.py --run-id sklearn_extra_trees_cv
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
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import KFold, ParameterGrid, GroupKFold
from pympler.tracker import SummaryTracker

import data.pcsml_data_loader as dl
from modeling import score_util, cv_util, xgb_util, categorical_util, sktrees_util
from util import mem_util

# opts
# debug settings
# training_data = '/var/opt/pcsml/devel/training_data/dumps/df-corn-smpl_25-gis-pps-20171018.pkl'
# training_data_dir = '/var/opt/pcsml/devel/training_data/dumps/gis_pps_corn_2016_sample_50k'
# training_data_name = 'gis_pps_corn_2016_sample_50k'
training_data_dir = '/var/opt/pcsml/devel/training_data/dumps/gis-pps-aggr/corn-2016-smpl-100k'
training_data_name = 'gis_pps_aggr_corn_2016_sample_100k'
# training_data_dir = '/var/opt/pcsml/devel/training_data/dumps/gis-pps-aggr/corn-2016-smpl-300k'
# training_data_name = 'gis_pps_aggr_corn_2016_sample_300k'
output_dir = '/var/opt/pcsml/devel/results/_scratch'
temp_dir = '/var/opt/pcsml/devel/_temp'

run_id = 'dev'
cv_n_outer_splits = 10
cv_n_inner_splits = 10
cv_n_outer_runs = 1
cv_n_inner_runs = 1
n_threads = 2
epochs_min = 2
epochs_max = 6
train_tol = 1
sample = 10000
debug_override = "--debug" in sys.argv[1]
# mem_util.disable_print_mem()
mem_util.enable_print_mem()

if debug_override:
    print("LOG-DEBUG: running in debug mode, won't load cmd line params")

if not debug_override and __name__ == '__main__' and not "pydevconsole" in sys.argv[0]:
    parser = argparse.ArgumentParser()
    parser.add_argument('--training-data-dir', '-d', type=str,
                        default='/var/opt/pcsml/training-data/gis-pps/gis_pps_corn_2016')

    parser.add_argument('--training-data-name', '-n', type=str, default='gis_pps_corn_2016')
    parser.add_argument('--output-dir', '-o', type=str, default='/var/opt/pcsml/remote-exec-out')
    parser.add_argument('--cv-n-outer-splits', '-cvos', type=int, default=10)
    parser.add_argument('--cv-n-outer-runs', '-cvor', type=int, default=1)
    parser.add_argument('--cv-n-inner-splits', '-cvis', type=int, default=10)
    parser.add_argument('--cv-n-inner-runs', '-cvir', type=int, default=1)
    parser.add_argument('--n-threads', '-bnt', type=int, default=16)
    parser.add_argument('--run-id', '-r', type=str, default=f'{datetime.utcnow():%Y-%m-%dT%H-%M-%S}')

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
    n_threads = opt.n_threads
    run_id = opt.run_id
    epochs_min = 1
    epochs_max = 6
    train_tol = .5
    sample = 500_000

    mem_util.enable_print_mem()


def _print_stats(y: Union[pandas.Series, np.ndarray, Iterable[float]]) -> str:
    return f"min: {np.min(y):.1f}, max: {np.max(y):.1f}, avg: {np.average(y):.1f}, std: {np.std(y):.1f}"


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
# remove outliers
###
data: pandas.DataFrame = pandas.read_pickle(_get_data_file('df_numeric.pickle.gz'))


###
# Train model
###
total_model_trainings = (cv_n_outer_runs * cv_n_inner_runs) + cv_n_outer_runs
curr_model_training = 0
outer_cv_results = []
inner_cv_scores: List[Tuple[int, score_util.ScoreReport]] = []
f_importance: List[DataFrame] = []
models: List[xgb.Booster] = []

data_year_ids: pandas.Series = pandas.read_pickle(_get_data_file('df_year_ids.pickle.gz'))
feature_names: pandas.Series = pandas.read_pickle(_get_data_file('columns.pickle.gz'))
data_labels: pandas.Series = pandas.read_pickle(_get_data_file('df_dry_yield.pickle.gz'))


def _load_data() -> sparse.csr_matrix:
    _name = f"{training_data_name}_csr.npz"
    _path = os.path.join(temp_dir, _name)

    if not os.path.isfile(_path):
        _gz_path = _get_data_file('csr.npz.gz')
        print("extracting dmatrix gzip: " + _gz_path)
        with gzip.open(_gz_path) as gz:
            with open(_path, 'wb') as f:
                shutil.copyfileobj(gz, f)

    print(f"Loading numpy csr: {_path}")
    return sparse.load_npz(_path)


# shuffle and sample year ids
# (shuffle, does the sample)
if sample <= 1:
    print(f"sampling year ids by frac: {sample}")
    year_ids = data_year_ids.sample(frac=sample)
else:
    print(f"sampling year ids by n: {sample}")
    year_ids = data_year_ids.sample(n=sample)

kf_outer = GroupKFold(n_splits=cv_n_outer_splits)
kf_outer_runs = itertools.islice(kf_outer.split(year_ids, groups=year_ids), cv_n_outer_runs)
for test_run_idx, (kf_outer_train_idx, kf_outer_test_idx) in enumerate(kf_outer_runs):
    outer_train_year_ids: pandas.Series = year_ids.iloc[kf_outer_train_idx]
    outer_test_year_ids: pandas.Series = year_ids.iloc[kf_outer_test_idx]

    print(f"\nOUTER CV TRAINING {test_run_idx + 1} of {cv_n_outer_runs}\n")

    # INNER CV #
    kf_inner = GroupKFold(n_splits=cv_n_inner_splits)
    kf_inner_runs = itertools.islice(kf_inner.split(outer_train_year_ids, groups=outer_train_year_ids), cv_n_inner_runs)
    for idx, (train_idx, test_idx) in enumerate(kf_inner_runs):
        train_year_ids: pandas.Series = outer_train_year_ids.iloc[train_idx]
        test_year_ids: pandas.Series = outer_train_year_ids.iloc[test_idx]

        data = _load_data()
        mem_util.print_mem_usage()

        print("creating train/test matrices")
        train: sparse.csr_matrix = data[train_year_ids.index]
        train_label: np.ndarray = data_labels.loc[train_year_ids.index].values
        test: sparse.csr_matrix = data[test_year_ids.index]
        test_label: np.ndarray = data_labels.loc[test_year_ids.index].values
        mem_util.print_mem_usage()

        print("deleting source matrix")
        del data
        mem_util.print_mem_usage()

        print(f"\n* INNER CV TRAINING *: ({test_run_idx + 1} of {cv_n_outer_runs}) {idx + 1} of {cv_n_inner_runs}")
        curr_model_training += 1

        print(f"training yield stats: {_print_stats(train_label)}")

        print(f"training extra trees model")
        print(f"train size, test size = "
              f"({train.shape[0]:,}; {train.shape[1]:,}), ({test.shape[0]:,}; {test.shape[1]:,})")

        model: ExtraTreesRegressor = ExtraTreesRegressor(n_estimators=0,
                                                         warm_start=True,
                                                         n_jobs=n_threads,
                                                         verbose=99)

        epoch = 0
        epoch_scores: List[score_util.ScoreReport] = []
        test_diff = float('inf')
        while epoch < epochs_max:
            print(f"model current estimators: {model.n_estimators}")
            n_estimators = model.n_estimators + n_threads
            print(f"setting estimators: {n_estimators}")
            model.n_estimators = n_estimators

            print("fitting model")
            model.fit(train, train_label)

            print("scoring")
            predictions = model.predict(test)
            score = score_util.ScoreReport(test_label, predictions, store_predictions=False)

            print(f"""
EPOCH RESULT: (training {curr_model_training} of total: {total_model_trainings})
epoch: {epoch}
n_estimators: {n_estimators}""")
            print(f"abs_99: {score.abs_99:.2f}")
            mem_util.print_mem_usage()

            if epoch > 0:
                test_diff = epoch_scores[epoch - 1].abs_std_3 - score.abs_std_3
                print(f"score diff: {test_diff:.2f}")

            epoch += 1
            if test_diff < train_tol and epoch >= epochs_min:
                print(f"ending early.  test diff: {test_diff:.2f}, epochs: {epoch}")
                break

            # else, we have a test diff that was more than train tol, add epoch score to results, and continue
            epoch_scores.append(score)

        # spot check that we didn't mess up the year id grouping.
        print(f"overlapping inner cv year ids (should be none): "
              f"{np.intersect1d(train_year_ids.unique(), test_year_ids.unique(), assume_unique=True)}")
        # clear memory
        del train
        del test

        inner_cv_scores.append((len(epoch_scores), epoch_scores[-1]))

    # OUTER CV #
    # get best score from inner cv
    print("\nINNER CV RESULTS")
    print(
        [(r[0], f"{r[1].abs_99:.2f}")
         for r in sorted(inner_cv_scores, key=lambda x: x[1].abs_99)])
    inner_cv_abs99_avg = np.average(
        [s[1].abs_std_3 for s in inner_cv_scores]
    )
    inner_cv_epoch_avg = int(round(np.average(
       [s[0] for s in inner_cv_scores]
    )))
    print(
        f"avg score: {inner_cv_abs99_avg:.2F}\n"
        f"avg epoch: {inner_cv_epoch_avg}")

    print(f"\n\n** TRAINING OUTER TEST **")
    curr_model_training += 1
    print(f"{curr_model_training} of total: {total_model_trainings}")

    data = _load_data()
    train_outer = data[outer_train_year_ids.index]
    train_outer_label = data_labels.loc[outer_train_year_ids.index].values
    test_outer = data[outer_test_year_ids.index]
    test_outer_label = data_labels.loc[outer_test_year_ids.index].values
    del data

    model: ExtraTreesRegressor = ExtraTreesRegressor(
        n_estimators=int(round(inner_cv_epoch_avg * n_threads)),
        n_jobs=n_threads,
        verbose=99)
    model.fit(train_outer, train_outer_label)

    model_path = f"{result_dir}/cv{test_run_idx}_skl_et.pickle.gz"
    print(f"saving model: {model_path}")
    with gzip.open(model_path, 'wb') as gz:
        pickle.dump(model, gz)

    f_imp_base_name = f"cv{test_run_idx}_f_imp"
    f_imp = sktrees_util.feature_importance_dummy_merged_report(
        result_dir, f_imp_base_name, model, feature_names)
    f_importance.append(f_imp)

    predictions = model.predict(test_outer)

    score = score_util.ScoreReport(test_outer_label, predictions, store_predictions=False)
    rdm_guesses = np.empty(len(test_outer_label), dtype=float)
    rdm_guesses.fill(test_outer_label.mean())
    random_guess_score = score_util.ScoreReport(test_outer_label, rdm_guesses, store_predictions=False)

    # spot check we didn't mess up the year id grouping
    print(f"overlapping outer cv year ids (should be none): "
          f"{np.intersect1d(outer_train_year_ids.unique(), outer_test_year_ids.unique(), assume_unique=True)}")

    print(f"\n**OUTER CV Score\nepochs: {inner_cv_epoch_avg}\nestimators:{model.n_estimators}")
    print(score)
    print(f"\nrandom guess score: {random_guess_score}")
    print(f"\ntraining yield stats: {_print_stats(test_outer_label)}")
    print(f"prediction stats: {_print_stats(predictions)}")
    mem_util.print_mem_usage()

    outer_cv_results.append(
        (model.n_estimators, score.abs_std_3, inner_cv_abs99_avg, random_guess_score.abs_std_3))

    print("deleting outer training data")
    del train_outer
    del test_outer
    del test_outer_label
    gc.collect()
    mem_util.print_mem_usage()

# cv final result
print("\n*** FINAL CV RESULTS ***")
mem_util.print_mem_usage()

print("\n* Feature Importance *")
fi_merged = f_importance[0]
for fi_df in f_importance[1:]:
    fi_merged = pandas.merge(fi_merged, fi_df, how='outer', on='Feature')

fi_total_df = pandas.DataFrame(fi_merged.loc[:, 'Feature'])
fi_total_df['Importance'] = fi_merged.fillna(0).sum(axis=1)
fi_total_df['Importance'] = fi_total_df['Importance']
fi_total_df.sort_values(by='Importance', ascending=False, inplace=True)
fi_total_df.to_csv(os.path.join(result_dir, 'cv_all_feature_importance.csv'))
xgb_util.plot_feature_importance(
    os.path.join(result_dir, 'cv_all_feature_importance.png'),
    {fi[0]: fi[1] for fi in fi_total_df.values})
# noinspection PyUnresolvedReferences
print([(fi[0], fi[1]) for fi in fi_total_df.head(20).values])

print("\n* OUTER CV TEST RESULTS *")
for r in outer_cv_results:
    print(f"n_estimators: {r[0]}, "
          f"cv abs 99: {r[2]:.2f}, "
          f"test abs 99: {r[1]:.2f}, "
          f"random guess abs 99: {r[3]:.2f}")

print("\n** AVERAGES **")
cv_n_estimators_avg = np.average([r[0] for r in outer_cv_results])
cv_abs99_avg = np.average([r[1] for r in outer_cv_results])
guess_abs99_avg = np.average([r[3] for r in outer_cv_results])
print(f"num est avg: {cv_n_estimators_avg:.2f}")
print(f"inner cv avg: {np.average([r[2] for r in outer_cv_results]):.2f}")
print(f"test abs_99: {cv_abs99_avg:.2f}")
print(f"rdm guess abs_99: {guess_abs99_avg:.2f}")
