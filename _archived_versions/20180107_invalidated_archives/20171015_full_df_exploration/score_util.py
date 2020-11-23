import logging
from collections import namedtuple

import numpy
from pandas import DataFrame
from sklearn.metrics import mean_squared_error, r2_score, median_absolute_error, mean_absolute_error

logger = logging.getLogger(__name__)


class ScoreReport:
    def __init__(self, y, predictions):
        err_sq_mean = mean_squared_error(y, predictions)
        r2 = r2_score(y, predictions)
        err_abs_mean = mean_absolute_error(y, predictions)
        err_abs_median = median_absolute_error(y, predictions)
        err_abs = numpy.abs(y - predictions)
        err_abs_std = err_abs.std()
        err_abs_max = err_abs.max()

        err_percents = numpy.array((err_abs / y) * 100)
        err_percent_std = err_percents.std()
        err_percent_max = err_percents.max()
        err_percent_mean = err_percents.mean()

        self.y = y
        self.predictions = predictions
        self.abs_errs = err_abs
        self.percent_errs = err_percents

        self.sq_mean = err_sq_mean
        self.r2 = r2
        self.abs_median = err_abs_median
        self.abs_mean = err_abs_mean
        self.abs_std = err_abs_std
        self.abs_max = err_abs_max
        self.percent_mean = err_percent_mean
        self.percent_std = err_percent_std
        self.percent_max = err_percent_max

    def __str__(self):
        return """score >>
r2: {}, sq_mean: {}, abs_median: {}
abs_mean: {}, abs_std: {}, abs_99: {}, abs_max: {}
per_mean: {}, per_std: {}, per_99: {}, per_max: {}""".format(
            self.r2, self.sq_mean, self.abs_median,
            self.abs_mean, self.abs_std, self.abs_mean + (self.abs_std * 3), self.abs_max,
            self.percent_mean, self.percent_std, self.percent_mean + (self.percent_std * 3), self.percent_max
        )


# noinspection PyPep8Naming
def score(model, scaler, X_validation, y_validation) -> ScoreReport:
    X_val = scaler.transform(X_validation) if scaler else X_validation
    predictions = model.predict(X_val)

    score_report = ScoreReport(y_validation, predictions)

    logger.info(score_report)
    return score_report


def create_data_frame(scores) -> DataFrame:
    return DataFrame(
        [(s.r2, s.sq_mean,
          s.abs_mean, s.abs_std, s.abs_max,
          s.abs_mean + s.abs_std * 2, s.abs_mean + s.abs_std * 3,
          s.percent_mean, s.percent_std, s.percent_max,
          s.percent_mean + s.percent_std * 2, s.percent_mean + s.percent_std * 3) for s in scores],
        columns=['r2', 'sq_mean',
                 'abs_mean', 'abs_std', 'abs_max', 'abs_2std_95', 'abs_3std_99',
                 'percent_mean', 'percent_std', 'percent_max', 'percent_2std_95', 'percent_3std_99']
    )
