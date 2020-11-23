import operator
from typing import Dict, Tuple, List, Union

from xgboost import Booster

from modeling.preprocessing import OneHotLabelEncoder


def top_n_importance(importance: dict, top_n: int) -> List[Tuple[str, Union[int, float]]]:
    sorted_importance = sorted(importance.items(), key=operator.itemgetter(1), reverse=True)
    score_cutoff = sorted_importance[top_n][1]
    return [(k, v) for k, v in sorted_importance if v >= score_cutoff]


def merge_labeled_weight_importance(model: Booster, label_encoder: OneHotLabelEncoder) -> Dict:
    f_imp = model.get_score(importance_type='weight')

    merged: Dict[str, int] = {}
    for f in f_imp:
        src_feature = label_encoder.source_column(f)
        merged[src_feature] = merged.get(src_feature, 0) + f_imp[f]

    return merged

