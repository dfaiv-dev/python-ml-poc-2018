import operator
from typing import Dict, List, Tuple, Union

import os
import pandas
import seaborn as sns
from matplotlib import pyplot as plt
from memory_profiler import profile
from pandas import DataFrame
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble.forest import BaseForest

from modeling import categorical_util
from util import plot_util


def merge_labeled_weight_importance(
        model: BaseForest,
        feature_names: pandas.Series,
        dummy_col_sep=categorical_util.DUMMY_COL_SEP) -> Dict[str, float]:
    f_imp = {
        feature_names.iloc[i]: fi
        for (i, fi) in enumerate(model.feature_importances_)
    }

    merged: Dict[str, float] = {}
    for f in f_imp:
        src_feature = categorical_util.get_source_name_from_dummy(f, dummy_col_sep)
        merged[src_feature] = merged.get(src_feature, 0) + f_imp[f]

    return merged


def top_n_importance(importance: Dict, top_n: int) -> List[Tuple[str, Union[int, float]]]:
    sorted_importance = sorted(importance.items(), key=operator.itemgetter(1), reverse=True)
    score_cutoff = sorted_importance[:top_n][-1][1]
    return [(k, v) for k, v in sorted_importance if v >= score_cutoff]


def plot_feature_importance(file_path: str, f_imp: Dict[str, float], top_n: int = 20):
    top_n = top_n_importance(f_imp, top_n)
    plt.clf()
    sns.barplot(x='Importance', y='Feature', data=DataFrame(top_n, columns=['Feature', 'Importance']),
                palette='Blues_d')
    plot_util.save_plot(file_path)


def feature_importance_dummy_merged_report(
        output_dir: str, file_base_name: str,
        model: BaseForest,
        feature_names: pandas.Series,
        dummy_col_sep=categorical_util.DUMMY_COL_SEP) -> pandas.DataFrame:
    f_imp = merge_labeled_weight_importance(model, feature_names, dummy_col_sep)
    df = pandas.DataFrame(
        [(i[0], i[1]) for i in f_imp.items()],
        columns=['Feature', 'Importance'])
    df = df.sort_values(by='Importance', ascending=False)

    csv_path = os.path.join(output_dir, f"{file_base_name}.csv")
    df.to_csv(csv_path)

    plot_feature_importance(os.path.join(output_dir, f"{file_base_name}.png"), f_imp)
    return df
