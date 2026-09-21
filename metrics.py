
import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)


def mae(y_true, y_pred):
    return mean_absolute_error(
        y_true,
        y_pred
    )


def rmse(y_true, y_pred):
    return np.sqrt(
        mean_squared_error(
            y_true,
            y_pred
        )
    )


def mase(
    y_true,
    y_pred,
    y_train,
    seasonal_period=7
):
    y_train = np.asarray(y_train)

    naive_errors = np.abs(
        y_train[seasonal_period:]
        -
        y_train[:-seasonal_period]
    )

    scale = np.mean(
        naive_errors
    )

    return (
        np.mean(
            np.abs(
                np.asarray(y_true)
                -
                np.asarray(y_pred)
            )
        )
        / scale
    )
