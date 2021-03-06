import logging
from typing import List

import numpy
import numpy as np
from pandas import DataFrame
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score

log = logging.getLogger(__name__)


class ScoreReport:
    def __init__(self, y, predictions, store_predictions=False):
        y = np.array(y)
        predictions = np.array(predictions)

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

        if store_predictions:
            self.y = y
            self.predictions = predictions
        else:
            self.y = self.predictions = None

        self.n_samples = len(y)

        self.sq_mean = err_sq_mean
        self.r2 = r2
        self.abs_median = err_abs_median
        self.abs_mean = err_abs_mean
        self.abs_std = err_abs_std
        self.abs_max = err_abs_max
        self.abs_std_3 = self.abs_mean + (self.abs_std * 3)
        self.percent_mean = err_percent_mean
        self.percent_std = err_percent_std
        self.percent_max = err_percent_max

        # percentiles
        self.abs_p90 = np.percentile(err_abs, 90)
        self.abs_p95 = np.percentile(err_abs, 95)
        self.abs_p99 = np.percentile(err_abs, 99)

    def __str__(self):
        return """SCORE REPORT >> abs_std3: {:.2f} (p95: {:.2f}) (r2: {:.6f}) <<
abs_p90: {:.1f}, abs_p95: {:.1f}, abs_p99: {:.1f}
abs_mean: {:.2f}, abs_std: {:.2f}, abs_std3: {:.2f}, abs_max: {:.2f}
r2: {:.6f}, sq_mean: {:.2f}, abs_median: {:.2f}""".format(
            self.abs_std_3, self.abs_p95, self.r2,
            self.abs_p90, self.abs_p95, self.abs_p99,
            self.abs_mean, self.abs_std, self.abs_std_3, self.abs_max,
            self.r2, self.sq_mean, self.abs_median,
        )


# noinspection PyPep8Naming
def score(model, scaler, X_validation, y_validation) -> ScoreReport:
    X_val = scaler.transform(X_validation) if scaler else X_validation
    predictions = model.predict(X_val)

    score_report = ScoreReport(y_validation, predictions)

    log.info(score_report)
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


def combine(scores: List[ScoreReport]) -> ScoreReport:
    predictions = [pred for s in scores for pred in s.predictions]
    actual = [act for s in scores for act in s.y]

    return ScoreReport(actual, predictions)


def abs_std_n_loss(y, predictions, stddevs=3, score_cache: List[ScoreReport] = None) -> float:
    scr_report = ScoreReport(y, predictions)
    if score_cache is not None:
        score_cache.append(scr_report)

    return scr_report.abs_mean + (scr_report.abs_std * stddevs)
