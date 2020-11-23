###
# 2017-12-22
#
# Extract 2017 observations into its own DF
###

import argparse
import os
import sys
from datetime import datetime

import pandas
from pandas import DataFrame

training_data = '/var/opt/pcsml/devel/training_data/dumps/transformed_sample_10000.pickle'
output_dir = '/var/opt/pcsml/devel/results/_scratch'
run_id = 'dev'

if __name__ == '__main__' and not "pydevconsole" in sys.argv[0]:
    parser = argparse.ArgumentParser()
    parser.add_argument('--training-data', type=str,
                        default='/var/opt/pcsml/training-data/gis-pps/df-corn-gis-pps-20171018.transformed.pickle')
    parser.add_argument('--output-dir', '-o', type=str, default='/var/opt/pcsml/remote-exec-out')
    parser.add_argument('--run-id', '-r', type=str, default=f'{datetime.utcnow():%Y-%m-%dT%H-%M-%S}')
    opt = parser.parse_args()

    print("parsed opts:")
    print(opt)
    training_data = opt.training_data
    output_dir = opt.output_dir
    run_id = opt.run_id

###
# create run output dir
result_dir = os.path.join(output_dir, run_id)
os.makedirs(result_dir, exist_ok=True)

print(f"reading in training data, may take a while: {training_data}")
df: DataFrame = pandas.read_pickle(training_data)

df2017 = df[df['Year'] == 2017]
result_path = os.path.join(result_dir, "df_gis_pps_extract_2017.pickle")
print(f"writing extracted 2017 data frame: {result_path}")
df2017.to_pickle(result_path)
