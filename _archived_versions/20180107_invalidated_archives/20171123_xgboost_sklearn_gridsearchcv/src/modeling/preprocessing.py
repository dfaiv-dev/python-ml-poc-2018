import datetime
import logging
import math
import re
from itertools import chain
from typing import Any, Callable, Iterable, List, Union

import dateutil.parser as dt_parser
import numpy as np
import pandas
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.validation import check_is_fitted

from data_scripts import pcsml_data_loader as dl

log = logging.getLogger(__name__)


class DummyEncoder(BaseEstimator, TransformerMixin):
    def __init__(self,
                 column_mask,
                 column_names: List[str] = None,
                 ignore_case=True,
                 null_values=None):

        self.column_names = column_names
        self.column_mask = column_mask

        self.null_values = null_values
        self.ignore_case = ignore_case

    def fit(self, X, y=None):
        log.debug("fitting %s", type(self).__name__)

        self.labels_ = {}
        self.label_indexes_ = np.ravel(np.where(self.column_mask == True))
        self.one_hot_enc_ = None
        self.null_values = self.null_values if self.null_values is not None else ['', 'none', None, np.nan]

        matrix = np.array(X, copy=True)

        for idx in self.label_indexes_:
            labels = self._format_label_col(matrix[:, idx])
            self.labels_[idx] = np.unique(labels[~pandas.isnull(labels)])

        labeled, _ = self._transform_labels(matrix)
        # setup the one hot encoder for the transform
        one_hot_enc = OneHotEncoder(
            categorical_features=self.column_mask,
            handle_unknown='ignore')

        one_hot_enc.fit(labeled)
        self.one_hot_enc_ = one_hot_enc
        log.debug("fitted %s", type(self).__name__)

        return self

    def _format_label_col(self, col):
        labels = list(col)
        if self.ignore_case:
            labels = [str.lower(str(l)) if l is not None else None for l in list(col)]

        for v in self.null_values:
            labels = [None if l == v else l for l in labels]

        return np.array(labels, copy=False)

    def _transform_labels(self, X):

        matrix = np.array(X, copy=True)
        # label each column index we have a labels for
        unknown_labels = {}
        for idx in self.label_indexes_:
            labels = self.labels_[idx]
            y = self._format_label_col(matrix[:, idx])
            # create a none label that is always last in search
            none_label = labels[-1] + '_ZZ' if len(labels) > 0 else '_ZZ'
            y[pandas.isnull(y)] = none_label

            labeled = np.searchsorted(labels, y)

            # get the classes for the column
            y_labels = np.unique(y)
            diff = []
            if len(np.intersect1d(y_labels, labels)) < len(y_labels):
                diff = np.setdiff1d(y_labels, labels)
                # don't include none label as part of diff reporting
                diff = diff[~np.isin(diff, [none_label])]
                if len(diff) > 0:
                    # log.debug("col contains new labels: %s (%d), %s", self.column_names[idx], idx, str(diff))
                    unknown_labels[idx] = diff

                _idx = np.in1d(y, labels)
                # set unknown classes to the next label ID
                labeled[_idx == False] = len(labels)

            matrix[:, idx] = labeled

            # raise ValueError("y contains new labels: %s" % str(diff))

        return np.nan_to_num(np.array(matrix, dtype=float, copy=False), copy=False), unknown_labels

    def transform(self, X):
        check_is_fitted(self, ['one_hot_enc_', 'labels_'])

        log.debug("transforming with %s", type(self).__name__)
        labeled, _ = self._transform_labels(X)
        return self.one_hot_enc_.transform(labeled)


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


def shape_gis_pps(df: pandas.DataFrame,
                  includeSms=False) -> (pandas.DataFrame, List[str]):
    # NOTE: these columns are from the old SQL tables, probably should update this somehow at some point?
    label_cols = dl.group_cols()

    ###
    #  encode existence only columns (true, false)
    ###
    exists_cols = []
    exists_src_cols = [['Variety'], ['SampleDate']]
    if not includeSms:
        exists_src_cols.append(['SMS'])

    for src_cols in exists_src_cols:
        df, col = encode_not_null(df, src_cols)
        exists_cols.append(col)

    # add the new columns, remove the originals from group list
    label_cols = label_cols + exists_cols
    label_cols = [
        c for c in label_cols
        if c not in list(chain.from_iterable(exists_src_cols))]

    ###
    # encode date columns to year bi-weekly number (0 - ~26)
    ###
    date_src_cols = [c for c in df.columns if re.search(r'[aA]p(p)?[dD]ate', c)]
    date_cols = []
    for src_col in date_src_cols:
        df, col = encode_date_to_biweek(df, src_col)
        date_cols.append(col)

    # add the new columns, remove the originals from group list
    label_cols = label_cols + date_cols
    label_cols = [
        c for c in label_cols if c not in date_src_cols
    ]

    log.debug("Label Columns: %s", label_cols)
    return df, label_cols


def make_one_hot_pipeline(
        X: pandas.DataFrame,
        label_cols: List[str],
        transformers: List[Union[BaseEstimator, TransformerMixin]]):
    """
    Handles masking transformers to only op on numeric (non-label) columns

    :param X:
    :param label_cols:
    :param transformers:
    :return:
    """

    label_mask = create_label_mask(X.columns, label_cols)
    masked_steps: List[(str, TransformerMixin)] = [
        (str(type(t)), MaskedTransformer(t, ~label_mask))
        for t in transformers]

    return Pipeline(masked_steps + [(str(type(DummyEncoder)), DummyEncoder(label_mask, X.columns))])
