
import pandas as pd


def run_backtest(
    series,
    forecast_function,
    window_type="expanding",
    n_folds=6,
    horizon=28,
    rolling_window=730
):

    results = []

    total_test_size = (
        n_folds * horizon
    )

    first_test_start = (
        len(series)
        - total_test_size
    )

    for fold in range(n_folds):

        test_start = (
            first_test_start
            + fold * horizon
        )

        test_end = (
            test_start
            + horizon
        )

        if window_type == "expanding":
            train_start = 0

        elif window_type == "rolling":
            train_start = max(
                0,
                test_start - rolling_window
            )

        else:
            raise ValueError(
                "window_type must be "
                "'expanding' or 'rolling'"
            )

        train = series.iloc[
            train_start:test_start
        ]

        test = series.iloc[
            test_start:test_end
        ]

        predictions = forecast_function(
            train,
            test.index
        )

        fold_result = pd.DataFrame({
            "date": test.index,
            "actual": test.values,
            "predicted": predictions,
            "fold": fold + 1,
            "window_type": window_type,
            "train_start": train.index[0],
            "train_end": train.index[-1]
        })

        results.append(
            fold_result
        )

    return pd.concat(
        results,
        ignore_index=True
    )
