import typing
from typing import Callable, Dict, Any, Iterable, Tuple, List
import itertools

import numpy
import pandas
from pandas import DataFrame, SparseDataFrame, Categorical
from sklearn.model_selection import BaseCrossValidator

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


class CVProvider(BaseCrossValidator):
    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_runs if self.n_runs is not None  else self.cv.get_n_splits(X, y, groups)

    def _iter_test_indices(self, X=None, y=None, groups=None):
        return self.cv._iter_test_indices(X, y, groups)

    def __init__(self,
                 cv: BaseCrossValidator,
                 n_runs=None,
                 callbacks: List[Callable[[List[int], List[int]], typing.NoReturn]] = None):
        super().__init__()

        self.callbacks = callbacks
        self.cv = cv
        self.n_runs = n_runs
        self.train_idx_: List = None
        self.test_idx_: List = None

    def split(self, X, y=None, groups=None):
        splits = itertools.islice(self.cv.split(X, y, groups), self.get_n_splits(X, y, groups))
        for train_idx, test_idx in splits:
            self.train_idx_ = train_idx
            self.test_idx_ = test_idx

            if self.callbacks is not None:
                for c in self.callbacks:
                    c(train_idx, test_idx)

            yield train_idx, test_idx
