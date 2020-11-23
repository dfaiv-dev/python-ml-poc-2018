"""
Visualizations for pcsml data

https://seaborn.pydata.org/tutorial/distributions.html#visualizing-pairwise-relationships-in-a-dataset
"""
import logging
import os

import pandas as pd
import pandas.api.types as pd_types
import seaborn as sns
from matplotlib import pyplot as plt

import data_scripts.pcsml_data_loader as dl

log = logging.getLogger(__name__)


def plot_all_pcsml(df: pd.DataFrame, out_dir: str, rug=False, max_categories=15):
    os.makedirs(out_dir, exist_ok=True)
    # plot one off, non histogram columns
    plot_lat_lon(df, out_dir)

    exclude_cols = dl.exclude_columns
    exclude_cols = [c for c in exclude_cols if c is not 'Year']
    exclude_cols += ['Lat', 'Lon']

    # histograms
    for col in [c for c in df.columns if c not in exclude_cols]:
        log.info("plotting dists: %s" % col)

        if df[col].dropna().count() == 0:
            log.warning("column is all NaN, nothing to plot")
            continue

        plot_distribution(df, col, out_dir, rug, max_categories)


def plot_distribution(df: pd.DataFrame, col: str, out_dir: str, rug=False, max_categories=15):
    kde = True
    if len(df[col].dropna().unique()) == 1:
        log.info("%s has only one unique value, kde=False", col)
        kde = False

    if pd_types.is_numeric_dtype(df[col]):
        sns.distplot(df[col].fillna(0), kde=kde, rug=rug)
        plt.title(col + ' Dist (NaN = 0)')
        _save_plot(out_dir, f'{col}_dist_nan0.png')

        sns.distplot(df[col].dropna(), kde=kde, rug=rug)
        plt.title(col + ' Dist (NaN Excluded)')
        _save_plot(out_dir, f'{col}_dist.png')
    else:
        counts: pd.Series = df[col].value_counts()
        counts_top: pd.Series = counts[:max_categories]
        others = counts[max_categories:]
        if len(others) > 0:
            counts_top = counts_top.append(pd.Series({'others': others.sum()}))

        plt.title(col + " - Counts")
        sns.barplot(x=counts_top.values, y=counts_top.index, palette='Blues_d')
        _save_plot(out_dir, f'{col}_dist.png')


def plot_lat_lon(df: pd.DataFrame, out_dir: str, lat_col="Lat", lon_col="Lon"):
    sns.jointplot(x=lon_col, y=lat_col, data=df, kind='kde')
    _save_plot(out_dir, 'lat_lon.png')


def _save_plot(out_dir: str, name: str, close=True):
    plt.tight_layout(pad=1.25)
    plt.savefig(os.path.join(out_dir, name))
    plt.close()
