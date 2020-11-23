from typing import Callable, Dict, Any, Iterable, Tuple, List

import numpy
import pandas
from pandas import DataFrame, SparseDataFrame, Categorical

from modeling import categorical_util


class GridSearchCVResults:
    def __init__(self,
                 params: Dict):
        self.params = params
        self.attrs: List[Dict[str, Any]] = []
        self.scores = []

    def add(self, score, **attrs):
        self.scores.append(score)
        self.attrs.append(attrs)

    def score_avg(self) -> float:
        return numpy.average(self.scores) if len(self.scores) > 0 else numpy.nan

    def attr_avg(self, attr_name: str) -> float:
        if len(self.scores) is 0:
            return numpy.nan

        vals = [attrs[attr_name] for attrs in self.attrs]
        return numpy.average(vals)

    def all_attr_avg(self) -> Dict[str, float]:
        if len(self.scores) is 0:
            return numpy.nan

        # dict of all values for key in the first attribute
        # assume all attributes have the same key
        return {k: self.attr_avg(k) for k in self.attrs[0]}

    def avg(self) -> (float, Dict[str, float]):
        if len(self.scores) is 0:
            return numpy.nan, {}

        return self.score_avg(), self.all_attr_avg()

    def __str__(self):
        return f"{self.params}, score: {self.score_avg():.3f}"

    def __repr__(self):
        return f"({self.params}, {self.score_avg():.3f}"

