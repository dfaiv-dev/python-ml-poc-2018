import logging
from typing import Iterable, List, Union

import numpy as np
import pandas
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import column_or_1d
from sklearn.utils.validation import check_is_fitted


class DummyEncoder(BaseEstimator, TransformerMixin):
    def __init__(self,
                 column_mask,
                 ignore_case=True,
                 null_values=None):

        self.column_mask_ = column_mask
        self.labels_ = {}
        self.label_indexes_ = np.ravel(np.where(self.column_mask_ == True))
        self.one_hot_enc_ = None
        self.null_values_ = null_values if null_values is not None else ['', 'none', None, np.nan]
        self.ignore_case_ = ignore_case

    def fit(self, X, y=None):
        matrix = np.array(X, copy=True)
        self.labels_ = {}

        for idx in self.label_indexes_:
            labels = self._format_label_col(matrix[:, idx])
            self.labels_[idx] = np.unique(labels[~pandas.isnull(labels)])

        labeled, _ = self._transform_labels(matrix)
        # setup the one hot encoder for the transform
        one_hot_enc = OneHotEncoder(
            categorical_features=self.column_mask_,
            handle_unknown='ignore')

        one_hot_enc.fit(labeled)
        self.one_hot_enc_ = one_hot_enc

    def _format_label_col(self, col):
        labels = list(col)
        if self.ignore_case_:
            labels = [str.lower(str(l)) if l is not None else None for l in list(col)]

        for v in self.null_values_:
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
                    logging.debug("col contains new labels: %s (%d), %s", self.labels_[idx], idx, str(diff))
                    unknown_labels[idx] = diff

                _idx = np.in1d(y, labels)
                # set unknown classes to the next label ID
                labeled[_idx == False] = len(labels)

            matrix[:, idx] = labeled

            # raise ValueError("y contains new labels: %s" % str(diff))

        return np.nan_to_num(np.array(matrix, dtype=float, copy=False), copy=False), unknown_labels

    def transform(self, X):
        check_is_fitted(self, ['one_hot_enc_', 'labels_'])

        labeled, _ = self._transform_labels(X)
        return self.one_hot_enc_.transform(labeled)


class MaskedTransformer(BaseEstimator, TransformerMixin):
    def __init__(self,
                 transformer: Union[BaseEstimator, TransformerMixin],
                 mask):
        self.transformer = transformer
        self.mask = mask
        self.mask_indexes_ = np.ravel(np.where(self.mask == True))

    def fit(self, X):
        logging.debug("Fit masked transform: %s", type(self.transformer))

        X_masked = np.array(X)[:, self.mask_indexes_]
        self.transformer.fit(X_masked)

        return self

    def transform(self, X):
        check_is_fitted(self, 'mask_indexes_')

        X_transformed = np.array(X)
        X_masked = X_transformed[:, self.mask_indexes_]
        X_masked = self.transformer.transform(X_masked)
        X_transformed[:, self.mask_indexes_] = X_masked

        return X_transformed


class FillNaTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return np.nan_to_num(X.astype(float))


class FlexLabelEncoder(BaseEstimator, TransformerMixin):
    """Encode labels with value between 0 and n_classes-1.  Add support for unknown labels

    See also
    --------
    sklearn.preprocessing.OneHotEncoder : encode categorical integer features
        using a one-hot aka one-of-K scheme.
    """

    def fit(self, y):
        """Fit label encoder

        Parameters
        ----------
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : returns an instance of self.
        """
        y = column_or_1d(y, warn=True)
        self.classes_ = np.unique(y[~pandas.isnull(y)])
        return self

    def transform(self, y):
        """Transform labels to normalized encoding.

        Parameters
        ----------
        y : array-like of shape [n_samples]
            Target values.

        Returns
        -------
        y : array-like of shape [n_samples]
        """
        check_is_fitted(self, 'classes_')
        y = column_or_1d(y, warn=True)
        none_label = self.classes_[-1] + '_ZZ' if self.classes_.size > 0 else '_ZZ'
        y[pandas.isnull(y)] = none_label

        labeled = np.searchsorted(self.classes_, y)

        classes = np.unique(y)
        diff = []
        if len(np.intersect1d(classes, self.classes_)) < len(classes):
            diff = np.setdiff1d(classes, self.classes_)
            # don't include none label as part of diff reporting
            diff = diff[~np.isin(diff, [none_label])]
            if len(diff) > 0:
                logging.warning("y contains new labels: %s" % str(diff))

            _idx = np.in1d(y, self.classes_)
            # set unknown classes to the next label ID
            labeled[_idx == False] = len(self.classes_)

            # raise ValueError("y contains new labels: %s" % str(diff))

        return labeled, diff

    def inverse_transform(self, y):
        """Transform labels back to original encoding.

        Parameters
        ----------
        y : numpy array of shape [n_samples]
            Target values.

        Returns
        -------
        y : numpy array of shape [n_samples]
        """
        check_is_fitted(self, 'classes_')

        diff = np.setdiff1d(y, np.arange(len(self.classes_)))
        if diff:
            raise ValueError("y contains new labels: %s" % str(diff))
        y = np.asarray(y)
        return self.classes_[y]


def create_dummy_encoder(X: pandas.DataFrame, label_cols: List[str]) -> DummyEncoder:
    """
    Creates a fitted DummyEncoder
    :param X:
    :param label_cols:
    :return:
    """

    mask = create_label_mask(X.columns, label_cols)
    enc = DummyEncoder(mask)
    enc.fit(X)


def create_label_mask(data_cols: Union[np.ndarray, Iterable], label_cols):
    """
    Creates a mask (array of 0 and 1s) that mark which columns are label columns.

    :param data_cols:
    :param label_cols:
    :return:
    """
    mask = np.isin(data_cols, label_cols)
    return mask


def transform_with_dummies(
        X: pandas.DataFrame,
        label_cols: List[str],
        transformers: List[Union[BaseEstimator, TransformerMixin]]):
    """

    :param X:
    :param label_cols:
    :param transformers:
    :return:
    """

    dummy_enc = create_dummy_encoder(X, label_cols)
    label_mask = dummy_enc.column_mask_
    X_transformed = X.fillna(0)
    for transformer in transformers:
        continuous_columns = X.loc[:, ~label_mask]
        continuous_columns.fillna(0)


def _fill_na(X: np.ndarray):
    return np.nan_to_num(X)
