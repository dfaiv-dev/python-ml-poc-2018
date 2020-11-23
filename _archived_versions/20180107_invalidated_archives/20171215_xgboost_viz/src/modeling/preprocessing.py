import datetime
import logging
import math
import re
from itertools import chain
from typing import Any, Callable, Dict, Iterable, List, Union

import dateutil.parser as dt_parser
import numpy as np
import pandas
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.validation import check_is_fitted

from data_scripts import pcsml_data_loader as dl

_NONE_LABEL = '__NONE__'

log = logging.getLogger(__name__)


class OneHotLabelColumn:
    def __init__(self,
                 name: str,
                 source_name: str,
                 source_idx: int):
        super().__init__()

        self.source_name = source_name
        self.source_idx = source_idx
        self.name = name


class OneHotLabelEncoder(BaseEstimator, TransformerMixin):
    """
    Combines labelling and one hot encoding.  Returns sparse matrix, which means StandardScaler can't be used on the
    resulting transform.  See @MaskedTransformer for workaround.
    """

    def __init__(self,
                 column_mask,
                 source_column_names: List[str] = None,
                 ignore_case=True,
                 null_values=None):

        self._source_column_names = source_column_names
        self.column_mask = column_mask
        '''
        Transformed column_mask to actual indexes (where col_mask == 1)
        '''
        self.label_indexes_: List[int] = None

        self.null_values = null_values
        self.ignore_case = ignore_case

        '''
        {source col idx, [labels]} dict
        A list of known labels for each labeled column from the fit matrix
        '''
        self.col_labels_: Dict[int, List[str]] = None
        '''
        {source col idx, [unknown labels]} dict
        A list of unknown labels from the last transformed matrix
        '''
        self.col_unknown_labels_: Dict[int, List[str]] = None
        self.one_hot_enc_: OneHotEncoder = None
        self.columns_: List[OneHotLabelColumn] = None
        self.column_names_: List[str] = None

    def fit(self, X, y=None):
        self.fit_transform(X, y)
        return self

    def transform(self, X, y=None):
        log.debug("transforming with %s", type(self).__name__)
        check_is_fitted(self, ['one_hot_enc_', 'col_labels_'])

        labeled = self._transform_to_labeled(X)
        return self.one_hot_enc_.transform(labeled)

    def fit_transform(self, X, y=None, **fit_params):
        log.debug("fitting %s", type(self).__name__)
        self.col_labels_ = None
        self.col_unknown_labels_ = None

        self.null_values = self.null_values if self.null_values is not None else ['', 'none', 'nan', None, np.nan]

        self.label_indexes_ = np.ravel(np.where(self.column_mask == True))
        matrix = np.array(X)

        # for idx in self.label_indexes_:
        #     label_col = self._clean_label_col(matrix[:, idx])
        #     self.labels_[idx] = np.unique(label_col[~pandas.isnull(label_col)])

        m_labeled = self._transform_to_labeled(matrix)

        # setup the one hot encoder for the transform
        self.one_hot_enc_ = OneHotEncoder(
            categorical_features=self.column_mask,
            handle_unknown='ignore')

        m_transformed = self.one_hot_enc_.fit_transform(m_labeled)
        log.debug("fitted %s", type(self).__name__)

        ##
        # create new column names
        self.columns_: List[OneHotLabelColumn] = []
        for i, feature_idx in enumerate(self.one_hot_enc_.feature_indices_[1:]):
            src_col_idx = self.label_indexes_[i]
            src_col_name = self._source_column_names[src_col_idx]
            labels = self.col_labels_[src_col_idx]
            for label in labels:
                self.columns_.append(OneHotLabelColumn(
                    f"{src_col_name}__{label}",
                    src_col_name,
                    src_col_idx,
                ))

        non_label_col_idxs = np.ravel(np.where(self.column_mask == False))
        for idx in non_label_col_idxs:
            self.columns_.append(OneHotLabelColumn(
                self._source_column_names[idx],
                self._source_column_names[idx],
                idx
            ))

        self.column_names_ = [c.name for c in self.columns_]

        return m_transformed

    def source_columns(self, labeled_column_names: List[str]) -> List[str]:
        """
        Takes a list one hot labeled column names and returns the original source column names (unique)
        :param labeled_column_names:
        :return:
        """

        cols = [c for c in self.columns_ if c.name in labeled_column_names]
        names = []
        for c in cols:
            if c.source_name not in names:
                names.append(c)

        return names

    def source_column(self, labeled_column_name: str) -> str:
        return next(
            (c.source_name for c in self.columns_ if c.name == labeled_column_name),
            None)

    def _clean_label_col(self, col):
        labels = list(col)
        if self.ignore_case:
            labels = [str.lower(str(l)) if l is not None else None for l in list(col)]

        for null_val in self.null_values:
            labels = [None if label == null_val else label for label in labels]

        return np.array(labels, copy=False)

    def _transform_to_labeled(self, matrix) -> np.ndarray:
        fit_col_labels = self.col_labels_ is None
        if fit_col_labels:
            self.col_labels_ = {}
        else:
            self.col_unknown_labels_ = {}

        for idx in self.label_indexes_:
            # labels = self.labels_[idx]
            col = self._clean_label_col(matrix[:, idx])
            col[pandas.isnull(col)] = _NONE_LABEL
            labels = np.unique(col)
            matrix[:, idx] = np.searchsorted(labels, col)

            if fit_col_labels:
                self.col_labels_[idx] = labels
                continue

            # get the classes for the column
            known_labels = self.col_labels_[idx]
            if len(np.intersect1d(labels, known_labels)) < len(labels):
                diff = np.setdiff1d(labels, known_labels)
                # don't include none label as part of diff reporting
                diff = diff[~np.isin(diff, [_NONE_LABEL])]
                if len(diff) > 0:
                    # log.debug("col contains new labels: %s (%d), %s", self.column_names[idx], idx, str(diff))
                    self.col_unknown_labels_[idx] = diff

                labeled_idxs = np.in1d(col, known_labels)
                # set unknown classes to the next label ID
                matrix[:, idx][labeled_idxs == False] = len(known_labels)

        return np.nan_to_num(np.array(matrix, dtype=float, copy=False), copy=False)


class MaskedTransformer(BaseEstimator, TransformerMixin):
    """
    :param mask
    array of bools with True for columns that should be transformed.

    :param dtype
    this is work around for the fact that we label and one hot columns in a single
    transform.  If we labeled them (1, 2, 3, ...) first, then we could safely call
    it first and have a fully numeric matrix from there on out, and do the one hot encoding
    last.

    We can't just call the dummy encoder last because it produces a sparse matrix, which breaks
    things down the line.
    """

    def __init__(self,
                 transformer: Union[BaseEstimator, TransformerMixin],
                 mask,
                 dtype=float):
        self.transformer = transformer
        self.mask = mask
        self.mask_indexes_ = np.ravel(np.where(self.mask == True))
        self.mask_dtype = dtype

    def fit(self, X, y=None):
        X_masked = np.array(X)[:, self.mask_indexes_]
        self.transformer.fit(X_masked.astype(self.mask_dtype))

        return self

    def transform(self, X):
        check_is_fitted(self, 'mask_indexes_')

        X_transformed = np.array(X)
        X_masked = X_transformed[:, self.mask_indexes_]
        X_masked = self.transformer.transform(X_masked.astype(self.mask_dtype or float))
        X_transformed[:, self.mask_indexes_] = X_masked

        return X_transformed


class FillNaTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return np.nan_to_num(X.astype(float))


class NumericTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return np.array(X).astype(float)


class ExistsEncoderConfig:
    def __init__(self,
                 columns: Union[str, List[str]],
                 none_values: List = None,
                 exists_column_name: str = None,
                 true_val='true',
                 false_val='false',
                 drop_source=True):
        self.columns: List[str] = columns if isinstance(columns, list) else [columns]
        self.none_values = none_values if none_values is not None else [None, np.nan, ""]
        self.exists_column_name = exists_column_name if exists_column_name is not None else "_".join(
            self.columns) + "_exists"
        self.true_val = true_val
        self.false_val = false_val
        self.drop_source = drop_source


def encode_not_null(X: pandas.DataFrame, source_columns: List[str], drop=True) -> (pandas.DataFrame, str):
    """

    :param source_columns:
    :param drop:
    :param X:
    :return:
    """

    cols: pandas.DataFrame = X.loc[:, source_columns]
    exists_column_name = "_".join(source_columns) + "_exists"
    X[exists_column_name] = (~cols.isnull()).all(axis=1)
    if drop:
        X.drop(source_columns, axis=1, inplace=True)

    return X, exists_column_name


def encode_date_to_biweek(X: pandas.DataFrame, src_col: str, drop=True) -> (pandas.DataFrame, str):
    encoded_col_name = src_col + "_biweek"

    def _encode(val):
        return _encode_date(val, lambda d: math.ceil(d.timetuple().tm_yday / 14))

    X[encoded_col_name] = X.loc[:, src_col].fillna('').apply(_encode).astype(dtype=str)
    X.drop([src_col], axis=1, inplace=True)
    return X, encoded_col_name


def _encode_date(val, date_enc: Callable[[datetime.datetime], Any]) -> str:
    date = _try_parse_date(val)
    if date is None:
        val = str(val)
        return "_" + val + "_" if len(val) > 0 else val

    return str(date_enc(date))


def _try_parse_date(s: str) -> datetime:
    if not isinstance(s, str):
        return None
    if re.match(r'^\s*\d+\s*$', s):
        return None

    try:
        return dt_parser.parse(s)
    except (ValueError, TypeError):
        return None


def create_label_mask(data_cols: Union[np.ndarray, Iterable], label_cols):
    """
    Creates a mask (array of 0 and 1s) that mark which columns are label columns.

    :param data_cols:
    :param label_cols:
    :return:
    """
    mask = np.isin(data_cols, label_cols)
    return mask


def _fill_na(X: np.ndarray):
    return np.nan_to_num(X)


def make_one_hot_pipeline(
        X: pandas.DataFrame,
        label_cols: List[str],
        non_label_transformers: List[Union[BaseEstimator, TransformerMixin]],
        terminate=True) -> (Pipeline, OneHotLabelEncoder):
    """
    Handles masking transformers to only op on numeric (non-label) columns

    :param terminate: if True, then adds a None final step for the model estimator.  use to create a transformer only pipeline.
    :param X:
    :param label_cols:
    :param non_label_transformers:
    :return:
    """

    label_mask = create_label_mask(X.columns, label_cols)
    steps: List[(str, TransformerMixin)] = [
        (str(type(t)), MaskedTransformer(t, ~label_mask))
        for t in non_label_transformers]

    one_hot_label_encoder = OneHotLabelEncoder(label_mask, X.columns)
    steps.append(('one_hot_labeled', one_hot_label_encoder))

    if terminate:
        steps.append(('none_estimator', None))

    return Pipeline(steps), one_hot_label_encoder
