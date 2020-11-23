import math
import re
from datetime import datetime
from typing import List, Callable, Any, Optional

import dateutil.parser as dt_parser
import pandas


def encode_not_null(X: pandas.DataFrame, source_columns: List[str], drop=True) -> pandas.DataFrame:
    """
    Encodes a column to a new category column as 0, 1: as either having a value or not having a value.
    Modifies the DF inplace.

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

    X[exists_column_name].astype('category')
    return X


def encode_date_to_biweek(X: pandas.DataFrame, src_col: str, drop=True) -> pandas.DataFrame:
    """
    Encodes a string column to bi-week number ranges category column.
    Modifies DF inplace.
    Any unparsable values will be left as is.

    :param X:
    :param src_col:
    :param drop:
    :return:
    """
    encoded_col_name = src_col + "_biweek"

    def _encode(val: pandas.Series):
        # https://stackoverflow.com/a/29882676/79113
        # use caching so we don't encode multiples
        encoded = {
            v: _encode_date(v, lambda d: math.ceil(d.timetuple().tm_yday / 14))
            for v in val.fillna('').unique()
        }

        return val.map(encoded).astype('category')

    X[encoded_col_name] = _encode(X[src_col])
    if drop:
        X.drop([src_col], axis=1, inplace=True)

    return X


def _encode_date(val, date_enc: Callable[[datetime], Any]) -> str:
    date = _try_parse_date(val)
    if date is None:
        val = str(val)
        return "_" + val + "_" if len(val) > 0 else val

    return str(date_enc(date))


def _try_parse_date(s: str) -> Optional[datetime]:
    if not isinstance(s, str):
        return None
    if re.match(r'^\s*\d+\s*$', s):
        return None

    try:
        return dt_parser.parse(s)
    except (ValueError, TypeError):
        return None
