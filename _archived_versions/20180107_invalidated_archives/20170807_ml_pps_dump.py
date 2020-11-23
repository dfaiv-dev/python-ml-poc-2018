import datetime as dt
import pickle
import zipfile
import pandas as pd

from data_scripts import pcs_data_loader as dl

df_raw = dl.load_corn_rows_mssql()
dump_base_path_ = f'./_temp/out/raw_{dt.datetime.now():%Y%m%d%_H%m}'

df_raw.to_csv((dump_base_path_ + f'.csv'), compression='gzip')
with zipfile.ZipFile(f'{dump_base_path_}.csv.zip', mode="w", compression=zipfile.ZIP_DEFLATED) as z:
    z.write(dump_base_path_ + ".csv", arcname=f'pcs_ml_raw_dump_{dt.datetime.now():%Y-%m-%d}.csv')

df_raw.to_pickle(f'{dump_base_path_}.pickle')
df_raw.to_hdf(f'{dump_base_path_}.hdf5', key='_')

df = dl.shape_pps_data(df_raw)
with zipfile.ZipFile(dump_base_path_ + "_pickle.zip", mode="w", compression=zipfile.ZIP_DEFLATED) as z:
    z.write(dump_base_path_ + ".pickle")

df.to_pickle(dump_base_path_ + ".pickle", compression='gzip')

##
# Reading from gzip pickle ~11 sec, dumping ~1 min, filesize ~50MB
# Reading from pickle ~3 sec, dumping ~14 sec, filesize ~2.5GB
#
# shaping df_raw ~1 min
##
