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

electricity_standard.run()