import csv
from _archived_versions._20170908_krmenec_reports import et_describer
import numpy
import os
import pandas
import pandas as pd
import pickle
from datetime import datetime
from sklearn import tree
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler

from data_scripts import pcs_data_loader as dl, elb_repo

out_base_path_ = 'results/20170908_et_stats_krmenec/'
graphviz_exe_path = f'c:\\Program Files (x86)\\Graphviz2.38\\bin\\dot.exe'


def load_training_data() -> (pandas.DataFrame, pandas.DataFrame):
    df_sample = dl.load_corn_rows_sample_shaped_pickle_gz()

    y = df_sample['Dry_Yield']
    X = df_sample.drop(['Area', 'Dry_Yield'], axis=1)
    return X, y


def train_model(X, y):
    scaler = StandardScaler()
    et = ExtraTreesRegressor(verbose=99, n_jobs=3)

    scaler.fit(X)
    et.fit(scaler.transform(X), y)
    return (et, scaler)


def dump_model(et, scaler, columns, base_path):
    with open(f'{base_path}/et_model.pickle', 'wb') as f:
        pickle.dump(et, f)

    with open(f'{base_path}/et_scaler.pickle', 'wb') as f:
        pickle.dump(scaler, f)

    ranks_ = pd.DataFrame({'feature': columns, 'importance': et.feature_importances_})
    ranks_.to_excel(f'{base_path}/feature_importance.xlsx', index=False)

    dump_graphviz(et, columns, out_dir=base_path)


def dump_graphviz(et, columns, out_dir=out_base_path_, max_depth=8):
    graphviz_out_dir = f'{out_dir}/graphviz'
    for idx, estimator in enumerate(et.estimators_):
        graphviz_out_base = f'{graphviz_out_dir}/et_{idx}_graph'
        graphviz_out = f'{graphviz_out_base}.dot'
        tree.export_graphviz(estimator, feature_names=columns, out_file=graphviz_out)

        graphviz_out_max_depth = f'{graphviz_out_base}_depth-{max_depth}.dot'
        tree.export_graphviz(estimator, feature_names=columns, out_file=graphviz_out_max_depth, max_depth=max_depth)
        os.system(
            f'"{graphviz_exe_path}" -Tpng {graphviz_out_max_depth} -o {graphviz_out_max_depth}.dot.png')


def load_model(dir=out_base_path_) -> (ExtraTreesRegressor, StandardScaler):
    with open(f'{dir}et_model.pickle', 'rb') as f_model:
        model = pickle.load(f_model)

    with open(f'{dir}et_scaler.pickle', 'rb') as f_scaler:
        scaler = pickle.load(f_scaler)

    return (model, scaler)


def train_and_dump():
    X, y = load_training_data()
    et, scaler = train_model(X, y)
    dump_model(et, scaler, X.columns)


def _main():
    result_path = f'{out_base_path_}{datetime.now():%Y%m%d_%H%M}'
    predictions_path = f'{result_path}/predictions'
    decision_trees_path = f'{result_path}/decision_trees'

    os.makedirs(result_path, exist_ok=True)
    os.makedirs(predictions_path, exist_ok=True)
    os.makedirs(decision_trees_path, exist_ok=True)
    os.makedirs(f'{result_path}/graphviz', exist_ok=True)

    et, scaler = load_model()
    X, y = load_training_data()
    dump_model(et, scaler, X.columns, result_path)

    X.to_csv(f'{result_path}/X_train.csv')
    y.to_csv(f'{result_path}/y_train.csv')

    for year_id, elb_X, elb_y in elb_repo.load_cached_elbs():
        results = []
        for idx, estimator in enumerate(et.estimators_):
            path = et_describer.tree_decision_path(estimator, elb_X, scaler)

            results = results + \
                      [[year_id, idx] + p for p in path]

            # with open(f'{result_path}/{year_id}_tree{idx:03d}.csv', 'w', newline='') as f:
            #     writer = csv.writer(f)
            #     writer.writerows(path)
        print(f'ran elb decision paths: {year_id}')

        with open(f'{decision_trees_path}/{year_id}_decision_trees.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'year_id', 'estimator_id', 'elb_row_id', 'tree_level',
                'node_id', 'feature_col_id', 'feature_col', 'feature_val_trans', 'threshold sign', 'threshold',
                'feature_val'])
            writer.writerows(results)

        predictions = et.predict(scaler.transform(elb_X))
        df = pandas.DataFrame.from_items([
            ('row_id', range(0, len(elb_y))),
            ('actual', elb_y),
            ('prediction', predictions),
            ('diff', numpy.abs(predictions - elb_y))
        ])
        df.to_csv(f'{predictions_path}/{year_id}_predictions.csv', header=True, index=True)
