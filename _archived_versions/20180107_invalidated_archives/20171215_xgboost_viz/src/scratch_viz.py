import argparse
import logging
import os

from pandas import DataFrame

import env
import util.logging
from data_scripts import pcsml_data_loader as dl, visualizer as vz

parser = argparse.ArgumentParser()
parser.add_argument('--env', '-e', type=str)
parser.add_argument('--result-path', '-o', type=str)
opt, _ = parser.parse_known_args()

util.logging.setup_default(opt.result_path)

log = logging.getLogger(__name__)
log.info("Running...")
log.info("Env:\n%s", env.dump())

result_path = opt.result_path
if result_path is None:
    result_path = 'results/_scratch'

os.makedirs(result_path, exist_ok=True)


def create_result_file_path(filename: str):
    return os.path.join(result_path, filename)


# load the data frame (a sample of the sample to make debugging faster...)
# df: DataFrame = dl.load_df_corn_pkl_smpl_25_20171018().sample(2000)
# dl.dump_sample(df, 'df-corn-20171018-scratch-debug.pickle')
# df: DataFrame = dl.load_pickled('df-corn-20171018-scratch-debug.pickle')

df: DataFrame = dl.load_df_corn_pkl_smpl_25_20171018().sample(100000)
log.info("data shape: %s", df.shape)

# y = df.pop('Dry_Yield')
# X = df

###
# transform pipeline setup
###
X, label_cols = dl.shape_gis_pps(df)
vz.plot_all_pcsml(X, out_dir='results/_scratch/distributions')
