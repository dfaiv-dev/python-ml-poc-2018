import operator
import os
from typing import Dict, Tuple, List, Union, Callable

import pandas
import seaborn as sns
import xgboost as xgb
from matplotlib import pyplot as plt
from memory_profiler import profile
from pandas import DataFrame
from xgboost import Booster, DMatrix

from modeling import categorical_util
from util import plot_util


def top_n_importance(importance: Dict, top_n: int) -> List[Tuple[str, Union[int, float]]]:
    sorted_importance = sorted(importance.items(), key=operator.itemgetter(1), reverse=True)
    score_cutoff = sorted_importance[:top_n][-1][1]
    return [(k, v) for k, v in sorted_importance if v >= score_cutoff]


def merge_labeled_weight_importance(model: Booster, dummy_col_sep=categorical_util.DUMMY_COL_SEP) -> Dict[str, int]:
    f_imp = model.get_score(importance_type='weight')

    merged: Dict[str, int] = {}
    for f in f_imp:
        src_feature = categorical_util.get_source_name_from_dummy(f, dummy_col_sep)
        merged[src_feature] = merged.get(src_feature, 0) + f_imp[f]

    return merged


def plot_feature_importance(file_path: str, f_imp: Dict[str, int], top_n: int = 20):
    top_n = top_n_importance(f_imp, top_n)
    plt.clf()
    sns.barplot(x='Importance', y='Feature', data=DataFrame(top_n, columns=['Feature', 'Importance']),
                palette='Blues_d')
    plot_util.save_plot(file_path)


def feature_importance_dummy_merged_report(
        output_dir: str, file_base_name: str, model: Booster,
        dummy_col_sep=categorical_util.DUMMY_COL_SEP) -> pandas.DataFrame:
    f_imp = merge_labeled_weight_importance(model, dummy_col_sep)
    df = pandas.DataFrame(
        [(i[0], i[1]) for i in f_imp.items()],
        columns=['Feature', 'Importance'])
    df = df.sort_values(by='Importance', ascending=False)

    csv_path = os.path.join(output_dir, f"{file_base_name}.csv")
    df.to_csv(csv_path)

    plot_feature_importance(os.path.join(output_dir, f"{file_base_name}.png"), f_imp)
    return df


def xgb_matrix_from_pcs_ml(
        df: DataFrame,
        label_col: str,
        column_categories: Dict[str, pandas.Categorical]) -> DMatrix:
    raise NotImplemented()


@profile
def train_test_from_df(
        df: pandas.DataFrame,
        label_col: str,
        test_split_func: Callable[[pandas.DataFrame], pandas.DataFrame],
        exclude_cols: List[str] = None) -> (
        DMatrix, DMatrix, Dict[str, pandas.Categorical]):
    dtrain, dtest, column_categories = \
        categorical_util.split_dummy_encode_df(df, test_split_func, exclude_cols)
    dtrain_y = dtrain.pop(label_col)
    dtest_y = dtest.pop(label_col)
    cols = dtrain.columns

    print(f"LOG: building train/test xgb DMatrix")
    dtrain_coo = dtrain.to_coo()
    dtest_coo = dtest.to_coo()
    dtrain_xgb = xgb.DMatrix(dtrain_coo, label=dtrain_y, feature_names=cols)
    dtest_xgb = xgb.DMatrix(dtest_coo, label=dtest_y, feature_names=cols)

    return dtrain_xgb, dtest_xgb, column_categories


def xgb_matrix_train_test_from_pcs_ml_disk(
        df_pickle_path: str,
        test_split_func: Callable[[pandas.DataFrame], pandas.DataFrame]) -> (
        DMatrix, DMatrix, Dict[str, pandas.Categorical]):
    print(f"LOG: reading df from disk: {df_pickle_path}")
    df = pandas.read_pickle(df_pickle_path)
    return train_test_from_df(df, test_split_func)
