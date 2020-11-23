import logging
from typing import Dict, List, Callable

import numpy as np
import pandas
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

_NONE_LABEL = '__NONE__'

log = logging.getLogger(__name__)

DUMMY_COL_SEP = '__DUMMY__'
NA_CATEGORY_NAME = '__na__'


class CategoricalColumn:
    def __init__(self,
                 category: pandas.Categorical,
                 name: str,
                 index: int):
        self.cat = category
        self.categories = list(category.categories)
        self.name = name
        self.index = index

        self.encoder: CategoryEncoder = None

    def codes(self):
        return self.cat.codes

    def add_na_category(self):
        if len(self.categories) is 0 or self.categories[0] is not NA_CATEGORY_NAME:
            self.categories.insert(0, NA_CATEGORY_NAME)


class CategoricalDummyEncoder(BaseEstimator, TransformerMixin):
    def __init__(self,
                 column_names: List[str],
                 category_columns: List[CategoricalColumn]):
        self._category_columns = category_columns
        self._column_names = column_names

        ###
        # indexes of category columns
        ###
        self._category_columns_lookup: Dict[int, CategoricalColumn] = None
        self._category_column_mask: List[bool] = None
        self._category_column_indexes: List[int] = None
        self._one_hot_enc: OneHotEncoder = None
        self._dummy_columns_map: Dict[str, CategoricalColumn] = None
        self.transformed_column_names: List[str] = None

    def column_categories(self) -> Dict[str, pandas.Categorical]:
        return {c.name: c.cat for c in self._category_columns}

    def transform(self, X: np.ndarray, y=None, **fit_params):
        if self._one_hot_enc is None:
            raise AssertionError("not fit")

        self._encode_categories(X)
        return self._one_hot_enc.transform(X)

    def fit_transform(self, X: np.ndarray, y=None, **fit_params):
        self._init_category_columns()
        self._encode_categories(X)

        self._one_hot_enc = OneHotEncoder(
            categorical_features=self._category_column_mask,
            handle_unknown='ignore',
            sparse=True,
            n_values=[len(c.categories) for c in self._category_columns])

        X_transformed = self._one_hot_enc.fit_transform(X)
        self._dummy_columns_map = {}
        self.transformed_column_names = []
        for i, feature_idx in enumerate(self._one_hot_enc.feature_indices_[1:]):
            category_col = self._category_columns[i]
            for code in category_col.encoder.classes_:
                label = category_col.categories[code]
                dummy_col_name = f"{category_col.name}__{label}"
                self._dummy_columns_map[dummy_col_name] = category_col
                self.transformed_column_names.append(dummy_col_name)

        # add non-category column names so we can do a full lookup
        non_label_col_idxs = np.ravel(np.where(self._category_column_mask == False))
        for idx in non_label_col_idxs:
            self.transformed_column_names.append(self._column_names[idx])

        return X_transformed

    def get_source_column_name(self, dummy_col_name) -> str:

        if dummy_col_name in self._dummy_columns_map:
            return self._dummy_columns_map[dummy_col_name].name
        else:
            return dummy_col_name

    def _init_category_columns(self):
        for c in self._category_columns:
            c.categories.append('__na__')

        self._category_columns_lookup = {
            c.index: c
            for c in self._category_columns
        }
        self._category_column_indexes = list(self._category_columns_lookup.keys())
        mask = np.zeros(len(self._column_names), dtype=bool)
        mask[self._category_column_indexes] = 1
        self._category_column_mask = mask

    def _encode_categories(self, X: np.ndarray):
        if X.shape[1] != len(self._column_names):
            raise ValueError(
                f"input matrix does not have same number of columns.  Expected: {len(self._column_names)}, "
                f"got: {X.shape[1]}")

        for idx in self._category_column_indexes:
            category_col = self._category_columns_lookup[idx]
            cat_len = len(category_col.categories)
            col = X[:, idx]
            col[col == -1] = cat_len - 1
            X[:, idx] = col


class DummyEncoder(BaseEstimator, TransformerMixin):
    def __init__(self,
                 feature_names: List[str],
                 feature_categories: Dict[str, pandas.Categorical]):

        self.feature_categories: Dict[str, pandas.Categorical] = feature_categories
        self.feature_names: List[str] = feature_names

        self.category_columns_: List[CategoricalColumn] = None
        self._category_columns_lookup: Dict[int, CategoricalColumn] = None
        self._category_column_mask: List[bool] = None
        self._one_hot_enc: OneHotEncoder = None

        # maps a dummy column back to its source CategoricalColumn
        self._dummy_columns_map: Dict[str, CategoricalColumn] = None
        # column names of the transformed matrix
        self.transformed_column_names_: List[str] = None

    def transform(self, X: np.ndarray, y=None, **fit_params):
        if self._one_hot_enc is None:
            raise AssertionError("not fit")

        self._encode_categories(X)
        return self._one_hot_enc.transform(X).tocsr()

    def fit_transform(self, X: np.ndarray, y=None, **fit_params):
        self.feature_names = list(self.feature_names)

        self._init_category_columns()
        self._encode_categories(X)

        self._one_hot_enc = OneHotEncoder(
            categorical_features=self._category_column_mask,
            handle_unknown='ignore',
            sparse=True,
            n_values=[len(c.encoder.classes_) for c in self.category_columns_])

        # noinspection PyPep8Naming
        X = self._one_hot_enc.fit_transform(X)

        # make a map of dummy columns to feature names
        self._dummy_columns_map = {}
        self.transformed_column_names_ = []
        for i, feature_idx in enumerate(self._one_hot_enc.feature_indices_[1:]):
            category_col = self.category_columns_[i]
            for label in category_col.categories:
                dummy_col_name = f"{category_col.name}__{label}"
                self._dummy_columns_map[dummy_col_name] = category_col
                self.transformed_column_names_.append(dummy_col_name)

        # add non-category column names so we can do a full lookup
        non_label_col_idxs = np.ravel(np.where(self._category_column_mask == False))
        for idx in non_label_col_idxs:
            self.transformed_column_names_.append(self.feature_names[idx])

        return X.tocsr()

    def get_source_feature_name(self, dummy_col_name) -> str:

        if dummy_col_name in self._dummy_columns_map:
            return self._dummy_columns_map[dummy_col_name].name
        else:
            return dummy_col_name

    def _init_category_columns(self):
        self.category_columns_ = [
            CategoricalColumn(cat, name, self.feature_names.index(name))
            for (name, cat) in self.feature_categories.items()
        ]

        self._category_columns_lookup = {
            c.index: c
            for c in self.category_columns_
        }
        idxs = [c.index for c in self.category_columns_]

        # create the mask of category columns, used by the OneHotEncoder
        mask = np.zeros(len(self.feature_names), dtype=bool)
        mask[idxs] = 1
        self._category_column_mask = mask

    def _encode_categories(self, X: np.ndarray):
        if X.shape[1] != len(self.feature_names):
            raise ValueError(
                f"input matrix does not have same number of columns.  Expected: {len(self.feature_names)}, "
                f"got: {X.shape[1]}")

        for col in self.category_columns_:
            if col.encoder is not None:
                X[:, col.index] = col.encoder.transform(X[:, col.index])
            else:
                # have not initialized, fit and transform
                enc = CategoryEncoder()
                X[:, col.index] = enc.fit_transform(X[:, col.index])
                col.encoder = enc

                # if we have a na class, add a category name for it
                # insert at beginning
                col.add_na_category()


class CategoryEncoder:
    def __init__(self,
                 # pandas sets nan categories to code -1
                 na_class=-1,
                 handle_unknown='ignore'):
        self.na_class = na_class
        self.handle_unknown = handle_unknown

        self.classes_: np.ndarray = None

    @property
    def has_na_class(self):
        return np.isin(self.classes_, self.na_class).any()

    def fit(self, col: np.array):
        self.classes_ = np.unique(col)

    def fit_transform(self, col: np.array):
        self.classes_, encoded = np.unique(col, return_inverse=True)
        return encoded

    def transform(self, col: np.array):
        classes = np.unique(col)

        # invert=True so that True values mark unknown classes
        unknown_classes_mask = np.isin(classes, self.classes_, assume_unique=True, invert=True)
        unknown_classes = classes[unknown_classes_mask]
        if len(unknown_classes) > 0:
            if self.handle_unknown is not 'ignore':
                raise ValueError(f"col has unknown category ids: {unknown_classes}")

            # set unknown classes to the next category class value
            # this will let the generated label be known_labels + 1
            classes[unknown_classes_mask] = self.classes_[-1] + 1

        return np.searchsorted(self.classes_, col)


def create_dummy_encoder(df: pandas.DataFrame,
                         column_categories: Dict[str, pandas.Categorical]) -> CategoricalDummyEncoder:
    categorical_columns = [
        CategoricalColumn(
            column_categories[c],
            c,
            df.columns.get_loc(c))
        for c in column_categories
    ]
    return CategoricalDummyEncoder(df.columns, categorical_columns)


def set_categories(df: pandas.DataFrame, column_categories: Dict[str, pandas.Categorical]):
    for c in df.select_dtypes(include='category').columns:
        df[c].cat.set_categories(column_categories[c].categories, inplace=True)


def get_categories_lookup(df: pandas.DataFrame, remove_unused=True) -> Dict[str, pandas.Categorical]:
    """
    Builds a dictionary of col_name->Categorical for all category columns in DF.
    :return:
    """

    categories = {}
    for c in df.select_dtypes(include='category').columns:
        category: pandas.Categorical = df[c].cat
        if remove_unused:
            category = category.remove_unused_categories().cat

        categories[c] = category

    return categories


def encode_categories(df: pandas.DataFrame) -> Dict[str, pandas.Categorical]:
    """
    Encodes, in place, all categorical columns to codes.
    :param df:

    :return: a column_name to categorical lookup
    """

    categories_lookup = {}

    for c in df.select_dtypes(include='category').columns:
        cat: pandas.Categorical = df[c].cat

        categories_lookup[c] = cat
        df.iloc[:, df.columns.get_loc(c)] = cat.codes

    return categories_lookup


def get_source_name_from_dummy(dummy: str, sep=DUMMY_COL_SEP) -> str:
    tokens = dummy.split(sep)
    return tokens[0]


def encode_dummies(
        df: pandas.DataFrame,
        prefix_sep=DUMMY_COL_SEP, sparse=True) -> pandas.SparseDataFrame:
    return pandas.get_dummies(df, prefix_sep=prefix_sep, sparse=sparse, drop_first=True, dummy_na=False).to_sparse()


def split_dummy_encode_df(
        df: pandas.DataFrame,
        test_split_func: Callable[[pandas.DataFrame], pandas.DataFrame],
        exclude_cols: List[str] = None) -> (
        pandas.SparseDataFrame, pandas.SparseDataFrame, Dict[str, pandas.Categorical]):
    dtest = pandas.DataFrame(test_split_func(df))
    dtrain = pandas.DataFrame(df.loc[~df.index.isin(dtest.index)])

    if exclude_cols is not None:
        dtrain.drop(exclude_cols, axis=1, inplace=True, errors='ignore')
        dtest.drop(exclude_cols, axis=1, inplace=True, errors='ignore')

    print(f"LOG: setting categories")
    column_categories = get_categories_lookup(dtrain)
    set_categories(dtrain, column_categories)
    set_categories(dtest, column_categories)

    print(f"LOG: encoding dummies (train)")
    dtrain_enc = encode_dummies(dtrain)
    print(f"LOG: encoding dummies (test)")
    dtest_enc = encode_dummies(dtest)

    return dtrain_enc, dtest_enc, column_categories
