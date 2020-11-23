###
# 2017-12-22
#
# Script to run once to get pps_gis aggregate dumps into a ML expected form
###

import argparse
import os
import sys
from datetime import datetime

import pandas
from pandas import DataFrame

import data.pcsml_data_loader as data_loader
import data.visualizer as vz

training_data = '/var/opt/pcsml/devel/training_data/dumps/df-corn-smpl_25-gis-pps-20171018.pkl'
output_dir = '/var/opt/pcsml/devel/results/_scratch'
run_id = 'dev'
visualize_data = False
describe_data = True
sample_n = 1000

if __name__ == '__main__' and not "pydevconsole" in sys.argv[0]:
    parser = argparse.ArgumentParser()
    parser.add_argument('--training-data-dir', type=str, default='/var/opt/pcsml/training-data/gis-pps')
    parser.add_argument('--input-df', '-d', type=str, default='df-corn-gis-pps-20171018.pkl')
    parser.add_argument('--output-dir', '-o', type=str, default='/var/opt/pcsml/remote-exec-out')
    parser.add_argument('--describe-data', '-dd', action='store_true', default=False)
    parser.add_argument('--visualize-data', '-vd', action='store_true', default=False)
    parser.add_argument('--run-id', '-r', type=str, default=f'{datetime.utcnow():%Y-%m-%dT%H-%M-%S}')
    parser.add_argument('--sample-n', '-s', type=int)
    opt = parser.parse_args()

    print("parsed opts:")
    print(opt)
    training_data = os.path.join(opt.training_data_dir, opt.input_df)
    output_dir = opt.output_dir
    describe_data = opt.describe_data
    visualize_data = opt.visualize_data
    run_id = opt.run_id
    sample_n = opt.sample_n

###
# create run output dir
result_dir = os.path.join(output_dir, run_id)
os.makedirs(result_dir, exist_ok=True)

print(f"reading in training data, may take a while: {training_data}")
df: DataFrame = pandas.read_pickle(training_data)

if sample_n is not None:
    print(f"LOG: sampling data to: {sample_n}")
    df = df.sample(sample_n)

if visualize_data:
    print(f"LOG: visualizing data")
    vz.plot_all_pcsml(df, os.path.join(result_dir, 'distributions'))

print(f"LOG: transforming data to ML format")
df = data_loader.gis_pps_transform_numeric(df)
df.to_pickle(os.path.join(result_dir, 'transformed.pickle'))
