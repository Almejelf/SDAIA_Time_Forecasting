# Retail Demand Forecasting Capstone

This project was completed as part of the **SDAIA Academy – Time Series Forecasting for AI Systems** training programme.

**Author:** Faisal Almejel  
**Cohort:** September 2026  
**SDAIA Academy GitHub:** https://github.com/SDAIAAcademy

## Project Objective

This project develops and evaluates forecasting models for daily retail
demand using historical sales data.

The analysis focuses on **Store 1, Item 1** and demonstrates:

- time-series diagnostics and decomposition;
- classical forecasting;
- machine-learning forecasting;
- walk-forward validation;
- leakage prevention;
- probabilistic forecasting and prediction intervals.

## Data

The dataset comes from Kaggle's **Store Item Demand Forecasting Challenge**.

**Dataset:** Store Item Demand Forecasting Challenge  
**Source:** https://www.kaggle.com/competitions/demand-forecasting-kernels-only/data  
**Platform:** Kaggle

The full dataset contains daily sales observations across multiple stores
and items (N= 913000). This project focuses on the daily sales series for
**Store 1, Item 1**.

### Data Fields

| Field | Description |
|---|---|
| `date` | Date of the sales observation |
| `store` | Store identifier |
| `item` | Item identifier |
| `sales` | Number of items sold for that store-item combination on that date |

## Models

The project evaluates several forecasting approaches:

- Seasonal Naive
- SARIMAX
- Holt-Winters Exponential Smoothing
- LightGBM
- Prophet
- sktime Theta forecasting

SARIMAX and exponential smoothing provide classical statistical
approaches, while LightGBM provides a tree-based machine-learning
forecasting approach.

Prophet and sktime are additionally used to evaluate probabilistic
forecasts and prediction intervals.

## Feature Engineering

For LightGBM, temporal information is represented explicitly using:

- lagged sales features;
- rolling means and rolling standard deviations;
- day-of-week information;
- month;
- day of year;
- weekend indicators.

Rolling statistics are calculated only from historical observations to
prevent the current target value from leaking into its own predictors.

## Validation

Forecasting performance is evaluated using **walk-forward backtesting**
rather than a random train/test split.

The evaluation uses:

- **6 historical folds**
- **28-day forecasting horizon per fold**
- **expanding training windows**
- **rolling 730-day training windows**

Each model is fitted again within every fold.

No future observations are allowed to cross into a fold's training
period.

For LightGBM, multi-step forecasts are generated recursively so that
actual future target values are not used to construct later forecast
features.

Across four point-forecasting approaches, two window strategies, six
folds, and 28 forecast days, the evaluation produced:

**1,344 out-of-sample point forecasts.**

## Evaluation Metrics

Point forecasts are evaluated using:

- **MAE** — Mean Absolute Error
- **RMSE** — Root Mean Squared Error
- **MASE** — Mean Absolute Scaled Error

MASE is included because it provides a scale-safe comparison against a
seasonal-naive historical error benchmark.

Probabilistic forecasts are evaluated using:

- empirical interval coverage;
- average interval width;
- quantile forecasts;
- pinball loss.

## Main Point-Forecast Results

The best point-forecast results were obtained by the
**expanding-window LightGBM model**.

| Metric | LightGBM |
|---|---:|
| MAE | 4.110 |
| RMSE | 5.040 |
| MASE | 0.841 |

The expanding seasonal-naive benchmark produced:

| Metric | Seasonal Naive |
|---|---:|
| MAE | 5.875 |
| RMSE | 7.082 |
| MASE | 1.203 |

Compared with the seasonal-naive benchmark, expanding-window LightGBM
reduced:

- MAE by approximately **30.0%**
- RMSE by approximately **28.8%**

The expanding LightGBM model also achieved an MAE standard deviation of
approximately **0.350** across the six backtesting folds.

## Probabilistic Forecasting Results

Prophet and sktime Theta were evaluated using nominal **80% prediction
intervals** across the same six 28-day historical forecasting periods.

| Method | Nominal Coverage | Empirical Coverage | Average Width |
|---|---:|---:|---:|
| Prophet | 80% | 75% | 11.328 |
| sktime Theta | 80% | 75% | 12.757 |

Both approaches achieved empirical coverage of 75%.

Prophet produced the narrower average interval while maintaining the same
observed empirical coverage.

sktime was also used to generate 0.10, 0.50, and 0.90 quantile forecasts,
which were evaluated using pinball loss.

## Final Model Selection

For point forecasting, the project selects
**expanding-window LightGBM** based on the walk-forward results.

It achieved the lowest average MAE and RMSE among the evaluated
point-forecast models while also showing relatively low variability
across folds.

SARIMAX and exponential smoothing remained competitive classical
alternatives, while Prophet and sktime demonstrated how uncertainty can
be represented when a single point forecast is insufficient.

The selected model should not be interpreted as universally superior.
The conclusion applies to this dataset, store-item series, forecasting
horizon, feature set, and validation design.

## Limitations

The analysis focuses on a single store-item combination and therefore
does not establish that the same model will perform best across all
stores and items.

The dataset also does not include several potential demand drivers such
as price, promotions, inventory availability, weather, competitor
activity, or wider economic conditions.

Historical backtesting estimates performance under previously observed
conditions and does not guarantee future forecasting performance.

