import pandas
import seaborn as sns
from util import plot_util
import scipy.optimize
from pandas import DataFrame
import os

import data.model_loader as model_loader
from modeling import xgb_util

training_data = '/var/opt/pcsml/devel/training_data/dumps/transformed_sample_10000.pickle'
output_dir = '/var/opt/pcsml/devel/results/_scratch'

# model, cat_lookup = model_loader.xgb_dev_latest()
model, cat_lookup = model_loader.xgb_latest()

# df: DataFrame = pandas.read_pickle(training_data)
df: DataFrame = pandas.read_pickle(
    '/var/opt/pcsml/devel/training_data/dumps/df-corn-20171018-2017-transformed.pickle')

# targ pop range based on std-dev
targ_pop: pandas.Series = df.query('TargPop >= 20000 & TargPop <= 60000')['TargPop']
targ_pop_mean = int(round(targ_pop.mean()))
targ_pop_min = int(round(targ_pop.mean() - (targ_pop.std() * 2.5)))
targ_pop_max = int(round(targ_pop.mean() + (targ_pop.std() * 2.5)))


# predict and score all 2017 with model
# y = df.pop('Dry_Yield')
# X = df
# matrix = xgb_util.xgb_matrix_from_pcs_ml(X, y, cat_lookup)
# pred = model.predict(matrix)
# score = score_util.ScoreReport(y, pred)
# print(score)


def predictor(inputs, record_X, record_y):
    try:
        record_X['TargPop'] = inputs[0]
    except IndexError:
        record_X['TargPop'] = inputs

    matrix = xgb_util.xgb_matrix_from_pcs_ml(record_X, record_y, cat_lookup)

    pred = model.predict(matrix)
    return pred[0] * -1


# run target pop optimization for each year id (will have multiple records)
year_ids = df['YearId'].unique()
for year_id in year_ids:
    year_id_agg = df[df['YearId'] == year_id]
    print(year_id_agg[['YearId', 'TargPop', 'SOILNAME', 'SdUsage', 'Dry_Yield']])

    results = {
        'agr_grp_id': [],
        'target_pop': [],
        'yield_pred': []
    }
    # for each "agronomic group" of the year id (DF index)
    # run the optimized targ pop
    indexes = year_id_agg.index
    for i_label in indexes:
        record = pandas.DataFrame(year_id_agg.loc[[i_label], :])
        record_y = record.pop('Dry_Yield')
        ranges = [slice(targ_pop_min, targ_pop_max, 500)]
        print(f"agronomic group: {i_label} optimizing target pop")
        res: scipy.optimize.OptimizeResult = scipy.optimize.brute(
            predictor,
            args=[record, record_y],
            # x0=numpy.array([X['TargPop'].values[0]]),
            ranges=ranges,
            full_output=True,
            finish=None,
            disp=True)

        # add the agronomic group id (index label) for each estimate
        results['agr_grp_id'].extend([i_label] * len(res[2]))

        results['target_pop'].extend(res[2])
        results['yield_pred'].extend(res[3] * -1)
        print(res)

    result_df = pandas.DataFrame(results)
    sns.lmplot(x='target_pop', y='yield_pred', data=result_df, col='agr_grp_id', col_wrap=4, order=2)
    plot_util.save_plot(os.path.join(output_dir, f"{year_id}_targ_pop_response_curves.png"))

# opt_res: scipy.optimize.OptimizeResult = scipy.optimize.minimize(
#     predictor,
#     method='L-BFGS-B',
#     x0=numpy.array([targ_pop_mean]),
#     bounds=[(targ_pop_min, targ_pop_max)],
#     options={
#         'disp': True,
#         'eps': 500,
#         'gtol': 1e-08
#     })

# opt_res: scipy.optimize.OptimizeResult = scipy.optimize.basinhopping(
#     predictor,
#     # x0=numpy.array([X['TargPop'].values[0]]),
#     x0=numpy.array([targ_pop_min]),
#     stepsize=500,
#     T=1,
#     disp=True
# )
