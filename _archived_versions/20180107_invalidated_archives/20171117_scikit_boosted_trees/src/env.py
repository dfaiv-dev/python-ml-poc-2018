import os
import shutil
from datetime import datetime

from util import config

cfg = config.load_settings()

timestamp = f'{datetime.utcnow():%Y-%m-%dT%H-%M}'

results_base_path = os.path.join(cfg['results_dir'], 'results')

result_path = f"results/_current/{timestamp}"
if os.path.exists(cfg['results_dir']):
    result_path = os.path.join(cfg['results_dir'], f'results/{datetime.utcnow():%Y-%m-%dT%H-%M}')
    os.makedirs(result_path, exist_ok=True)

data_dir = cfg['data_dir']


def dump():
    return f'result_path: {result_path}\ndata_dir: {data_dir}\nCFG:\n{cfg}'
