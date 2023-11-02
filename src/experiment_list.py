"""Master File With List of Experiments to Run
"""
from src.util.ml_experiment import MLForecastingExperiment
from sklearn.linear_model import Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

# models to run for each experiment
model_list = [
    ("ridge", Ridge(alpha=1.0)),
    ("lasso", Lasso(alpha=0.1)),
    ("neighbors_5", KNeighborsRegressor(n_neighbors=5)),
    ("neighbors_10", KNeighborsRegressor(n_neighbors=10)),
    (
        "random_forest_100",
        RandomForestRegressor(n_estimators=100, max_features=0.7, min_samples_leaf=10),
    ),
    ("decision_tree_5", DecisionTreeRegressor(min_samples_leaf=5)),
    ("decision_tree_25", DecisionTreeRegressor(min_samples_leaf=25)),
    ("xgb_100", XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1)),
    ("xgb_250", XGBRegressor(n_estimators=250, max_depth=3, learning_rate=0.1)),
]

standard_exps = [
    MLForecastingExperiment(
        exp_name="initial",
        data_file="electricity_weekly.csv",
        models=model_list,
        lags=[1, 2, 3, 4, 5, 6, 7],
        calibration_windows=[6, 12, 48, 60],
        date_parts_to_encode=["month"],
        target_transform="log_diff",
        encode_entity=True,
        train_size=125,
        training_step_size=3,
    ),
    MLForecastingExperiment(
        exp_name="initial",
        data_file="covid_deaths_daily.csv",
        models=model_list,
        lags=[1, 2, 3, 4, 5, 6, 7],
        calibration_windows=[3, 30, 60, 120],
        date_parts_to_encode=["dayofweek", "quarter"],
        target_transform="log_diff_1_1",
        encode_entity=True,
        train_size=160,
        training_step_size=3,
    ),
    MLForecastingExperiment(
        exp_name="initial",
        data_file="fred_monthly.csv",
        models=model_list,
        lags=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        calibration_windows=[6, 12, 36, 72, 144],
        date_parts_to_encode=["month", "quarter"],
        target_transform="diff_1_1",
        encode_entity=True,
        train_size=500,
        training_step_size=3,
    ),
    MLForecastingExperiment(
        exp_name="initial",
        data_file="traffic_weekly.csv",
        models=model_list,
        lags=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        calibration_windows=[6, 12, 24, 36],
        date_parts_to_encode=["dayofweek", "quarter", "month"],
        target_transform="log_diff",
        encode_entity=True,
        train_size=80,
        training_step_size=3,
    ),
    MLForecastingExperiment(
        exp_name="initial",
        data_file="hospital_monthly.csv",
        models=model_list,
        lags=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        calibration_windows=[3, 12, 24, 30],
        date_parts_to_encode=["month", "quarter"],
        target_transform="log_diff",
        encode_entity=True,
        train_size=65,
        training_step_size=3,
    ),
    MLForecastingExperiment(
        exp_name="initial",
        data_file="m1_monthly.csv",
        models=model_list,
        lags=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        calibration_windows=[3, 12, 24, 30],
        date_parts_to_encode=["month", "quarter"],
        target_transform="log_diff_1_1",
        encode_entity=True,
        train_size=74,
        training_step_size=3,
    ),
    MLForecastingExperiment(
        exp_name="initial",
        data_file="m3_monthly.csv",
        models=model_list,
        lags=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        calibration_windows=[3, 12, 24, 30],
        date_parts_to_encode=["month", "quarter"],
        target_transform="log_diff_1_1",
        encode_entity=True,
        train_size=74,
        training_step_size=3,
    ),
]

# will add this later
"""
    MLForecastingExperiment(
        exp_name="initial",
        data_file="m4_hourly.csv",
        models=model_list,
        lags=list(range(1, 25)),
        calibration_windows=[3, 24, 72, 144],
        date_parts_to_encode=["hour"],
        target_transform="log_diff_1_24",
        encode_entity=True,
        train_size=600,
        training_step_size=3,
    ),

    MLForecastingExperiment(
        exp_name="initial",
        data_file="tourism_monthly.csv",
        models=model_list,
        lags=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        calibration_windows=[6, 24, 48, 60],
        date_parts_to_encode=["month", "quarter"],
        target_transform="log_diff_1_12",
        encode_entity=True,
        train_size=240,
        training_step_size=3,
    ),
"""
