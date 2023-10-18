"""Master File With List of Experiments to Run
"""
from src.util.ml_experiment import MLForecastingExperiment
from sklearn.linear_model import Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor

# models to run for each experiment
model_list = [
        Ridge(alpha = 1.0), 
        Lasso(alpha = 0.1),
        KNeighborsRegressor(n_neighbors = 5)
    ]

electricity_standard = MLForecastingExperiment(
    exp_name = 'electricity_initial',
    data_file = 'electricity_weekly.csv',
    models = model_list,
    lags = [1, 2, 3, 4, 5, 6, 7],
    calibration_windows = [7, 14, 28, 60],
    date_parts_to_encode = ['month'],
    target_transform = 'log_diff',
    encode_entity = True,
    train_size = 125,
)

covid_standard = MLForecastingExperiment(
    exp_name = 'covid_deaths_initial',
    data_file = 'covid_deaths_daily.csv',
    models = model_list,
    lags = [1, 2, 3, 4, 5, 6, 7],
    calibration_windows = [7, 30, 60, 120],
    date_parts_to_encode = ['dayofweek', 'quarter'],
    target_transform = 'log_diff',
    encode_entity = True,
    train_size = 160
)

fred_standard = MLForecastingExperiment(
    exp_name = 'fred_initial',
    data_file = 'fred_monthly.csv',
    models = model_list,
    lags = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    calibration_windows = [12, 36, 72, 144],
    date_parts_to_encode = ['month', 'quarter'],
    target_transform = 'diff',
    encode_entity = True,
    train_size = 500,
)

traffic_standard = MLForecastingExperiment(
    exp_name = 'traffic_initial',
    data_file = 'traffic_weekly.csv',
    models = model_list,
    lags = [1, 2, 3, 4],
    calibration_windows = [3, 12, 36],
    date_parts_to_encode = ['dayofweek', 'quarter', 'month'],
    target_transform = 'log_diff',
    encode_entity = True,
    train_size = 80,
)

hospital_standard = MLForecastingExperiment(
    exp_name = 'hospital_initial',
    data_file = 'hospital_monthly.csv',
    models = model_list,
    lags = [1, 2, 3, 4],
    calibration_windows = [3, 12, 24],
    date_parts_to_encode = ['month', 'quarter'],
    target_transform = 'log_diff',
    encode_entity = True,
    train_size = 65,
)

tourism_standard = MLForecastingExperiment(
    exp_name = 'tourism_initial',
    data_file = 'tourism_monthly.csv',
    models = model_list,
    lags = [1, 2, 3, 4],
    calibration_windows = [12, 24, 48, 72],
    date_parts_to_encode = ['month', 'quarter'],
    target_transform = 'log_diff_12',
    encode_entity = True,
    train_size = 240,
)

m1_standard = MLForecastingExperiment(
    exp_name = 'm1_initial',
    data_file = 'm1_monthly.csv',
    models = model_list,
    lags = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    calibration_windows = [3, 24, 30],
    date_parts_to_encode = ['month', 'quarter'],
    target_transform = 'log_diff',
    encode_entity = True,
    train_size = 74,
)

m3_standard = MLForecastingExperiment(
    exp_name = 'm3_initial',
    data_file = 'm3_monthly.csv',
    models = model_list,
    lags = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    calibration_windows = [3, 24, 30],
    date_parts_to_encode = ['month', 'quarter'],
    target_transform = 'log_diff',
    encode_entity = True,
    train_size = 74
)

m4_standard = MLForecastingExperiment(
    exp_name = 'm4_initial',
    data_file = 'm4_hourly.csv',
    models = model_list,
    lags = list(range(1, 25)),
    calibration_windows = [3, 24, 30],
    date_parts_to_encode = ['hour'],
    target_transform = 'log_diff',
    encode_entity = True,
    train_size = 700
)

