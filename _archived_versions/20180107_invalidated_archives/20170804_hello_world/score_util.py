import logging
from collections import namedtuple

import numpy
from sklearn.metrics import mean_squared_error, r2_score, median_absolute_error, mean_absolute_error

logger = logging.getLogger(__name__)


class ScoreReport:
    def __init__(self, err_sq_mean,
                 r2, err_abs_median, err_abs_mean, err_abs_std, err_abs_max,
                 err_percent_mean, err_percent_std, err_percent_max):
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
abs_mean: {}, abs_std: {}, abs_max: {}
per_mean: {}, per_std: {}, per_max: {}""".format(
            self.r2, self.sq_mean, self.abs_median,
            self.abs_mean, self.abs_std, self.abs_max,
            self.percent_mean, self.percent_std, self.percent_max
        )


# noinspection PyPep8Naming
def score(model, scaler, X_validation, y_validation) -> ScoreReport:
    scaler.fit(X_validation)
    predictions = model.predict(scaler.transform(X_validation))
    mean_sq = mean_squared_error(y_validation, predictions)
    r2 = r2_score(y_validation, predictions)
    err_abs_mean = mean_absolute_error(y_validation, predictions)
    error_abs_median = median_absolute_error(y_validation, predictions)
    err_abs = numpy.abs(y_validation - predictions)
    err_abs_std = err_abs.std()
    err_abs_max = err_abs.max()

    err_percents = numpy.array((err_abs / y_validation) * 100)
    err_percent_std = err_percents.std()
    err_percent_max = err_percents.max()
    err_percent_mean = err_percents.mean()

    score_report = ScoreReport(
        mean_sq, r2, error_abs_median,
        err_abs_mean, err_abs_std, err_abs_max,
        err_percent_mean, err_percent_std, err_percent_max
    )

    logger.info(score_report)
    return score_report
