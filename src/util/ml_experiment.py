"""
    ExperimentClass for ML Models to run on datasets
"""
import os

from datetime import datetime

import pandas as pd
import numpy as np

# sktime imports
from sktime.forecasting.model_selection import SlidingWindowSplitter

# sklearn imports
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error

# category encoders
from category_encoders import OneHotEncoder, TargetEncoder

class MLForecastingExperiment():

    def __init__(self, 
                 data_file: str, 
                 exp_name: str,
                 models: list, 
                 target_transform: str, 
                 lags: list = [1, 2, 3, 7],
                 calibration_windows: list = [7, 14],
                 encode_entity: bool = True,
                 train_size: int = 120,

                 # list of date parts to encode for date features
                 date_parts_to_encode: list = ['month'],
                 window_transforms: dict = None):

        self.data_file = data_file
        self.exp_name = exp_name
        self.models    = models
        self.target_transform = target_transform
        self.lags = lags
        self.calibration_windows = calibration_windows
        self.window_transforms = window_transforms
        self.encode_entity = encode_entity
        self.train_size = train_size
        self.date_parts_to_encode = date_parts_to_encode

    def _load_data(self):
        """Load in dataset"""
        self.data = pd.read_csv(f'data/{self.data_file}', parse_dates = ['date'])

        # validate data
        self._validate_data()

        # set index
        self.data.set_index(['series', 'date'], inplace = True)

    def _validate_data(self):
        """Basic error checking for data inputs

        Assumes input data has the following columns:
            series: str, name of the entity being forecasted
            date:   str, date of the observation
            value:  float or int, value of the observation
        """

        # check that data has the right columns
        assert 'series' in self.data.columns, 'Data must have a series column'
        assert 'date' in self.data.columns, 'Data must have a date column'
        assert 'value' in self.data.columns, 'Data must have a value column'

        # check that data has no missing values
        assert self.data.isna().sum().sum() == 0, 'Data must have no missing values'

        # check that data has no duplicate rows
        assert self.data.duplicated().sum() == 0, 'Data must have no duplicate rows'

        # check that data has no duplicate (series, date) pairs
        assert self.data.groupby(['series', 'date']).size().max() == 1, 'Data must have no duplicate (series, date) pairs'

        # check that data has has the correct types
        assert self.data['series'].dtype == 'object', 'Series column must be of type object'
        assert self.data['date'].dtype == 'datetime64[ns]', 'Date column must be of type datetime64[ns]'
        assert self.data['value'].dtype in ['float64', 'int64'], 'Value column must be of type float64 or int64'

    def _transform_target(self):
        """Apply target transformation to data"""

        # difference target
        if self.target_transform == 'diff':
            self.data['value'] = self.data.groupby(level = 0)['value'].diff()

        # log transform target
        elif self.target_transform == 'log':
            self.data['value'] = self.data.groupby(level = 0)['value'].apply(np.log1p)

        # log difference target
        elif self.target_transform == 'log_diff':
            self.data['value'] = self.data.groupby(level = 0)['value'].apply(np.log1p).diff().values

        # no transformation
        elif self.target_transform == None:
            pass

    def _create_window_transforms(self):
        """Create window transforms based off of time based features in target"""

        for transform in self.window_transforms.keys():
            if hasattr(self.data['value'].rolling(0), transform):
                for window in self.window_transforms[transform]:
                    self.data[f'{transform}_{window}'] = self.data.groupby(level = 0)['value'].rolling(window).apply(getattr(self.data['value'].rolling(window), transform)).shift().values

    def _create_X(self):
        """Create X based off of time based features in target"""

        # create X based off of lags
        for lag in self.lags:
            self.data[f'lag_{lag}'] = self.data.groupby(level = 0)['value'].shift(lag).values

        # create X based off of window transforms
        if self.window_transforms is not None:
            self._create_window_transforms()

        if self.encode_entity:
            self.data['entity'] = self.data.index.get_level_values(0)

        if self.date_parts_to_encode is not None:
            self._add_date_features()

        # drop rows with missing values
        self.data.dropna(inplace = True)

    def _calc_naive_forecast(self):
        """Creates a Naive Baseline for the Dataset"""
        self.data['naive_forecast'] = self.data.groupby(level = 0)['value'].shift(1).values

    def _calculate_error_metrics(self, preds_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate error metrics for each model"""

        # drop rows with missing values in the predictions
        preds_df = preds_df.dropna()

        pred_cols = [col for col in preds_df if 'pred' in col]

        # store the results
        results = []
        for col in pred_cols:
            results.append({
                'model': col,
                'mae': mean_absolute_error(preds_df['y_true'], preds_df[col]),
                'rmse': mean_squared_error(preds_df['y_true'], preds_df[col], squared = False),
                'mape': mean_absolute_percentage_error(preds_df['y_true'], preds_df[col]),
                'r2': r2_score(preds_df['y_true'], preds_df[col])
            })

        # calculate the naive forecast error metrics
        naive_dates = self.data.index.get_level_values(1)
        mask = naive_dates.isin(preds_df['date'])
        naive_preds = self.data.loc[mask, 'naive_forecast'].values
        naive_y_true = self.data.loc[mask, 'value'].values

        results.append({
            'model': 'naive_forecast',
            'mae': mean_absolute_error(naive_y_true, naive_preds),
            'rmse': mean_squared_error(naive_y_true, naive_preds, squared = False),
            'mape': mean_absolute_percentage_error(naive_y_true, naive_preds),
            'r2': r2_score(naive_y_true, naive_preds)
        })

        # return results as a dataframe
        return pd.DataFrame(results)

    def _add_date_features(self):
        """Create date features based off of date_parts_to_encode"""

        # create date features, sugglested list of date parts to encode: 
        # ['year', 'month', 'day', 'dayofweek', 'dayofyear', 'weekofyear']
        # only use ones that are appropriate for your data
        # for more information on other date parts see: https://pandas.pydata.org/docs/user_guide/timeseries.html#time-date-components
        for date_part in self.date_parts_to_encode:
            if hasattr(self.data.index.get_level_values(1), date_part):
                self.data[date_part] = getattr(self.data.index.get_level_values(1), date_part)

    def _build_data_encoder(self):
        """Create data transformer based off of presence of different columns
        
        Standard data encoding scheme is as follows:
            date parts are one-hot encoded
            entity is target encoded
        """
        # create data encoder
        self.data_encoder = make_pipeline(
            OneHotEncoder(cols = self.date_parts_to_encode, use_cat_names = True))
        
        if self.encode_entity:
            self.data_encoder.steps.append(('target_encoder', TargetEncoder(cols = ['entity'])))

    def _create_X_y(self):
        """Create X and y for whole dataset"""

        # create X
        self._create_X()

        # create y
        self.y = self.data['value']

        # drop target from X
        self.X = self.data.drop(columns = ['value'])

        # create train and test
        self._create_train_and_test()

    def _create_train_and_test(self):
        """Create training and test sets for X & y"""

        # create training and test sets
        grouping = self.data.groupby(level = 0)
        train, test = grouping.apply(lambda x: x.iloc[:self.train_size]), grouping.apply(lambda x: x.iloc[self.train_size:])

        self.X_train, self.X_test = train.drop(columns = ['value']), test.drop(columns = ['value']) 
        self.y_train, self.y_test = train['value'], test['value']

    def _fit_data(self, encoder, model):
        """Fit data using sliding window one step ahead forecasting"""
        # store the results for each training window
        final_window_results = []

        # loop through each calibration window
        for window in self.calibration_windows:
            window_results = []
            print(f"Fitting model w/ a window size of: {window}")
            splitter = SlidingWindowSplitter(fh = [1], 
                                window_length = window, # this is the experimental variable we want to manipulate
                                step_length = 1)
            
            # use sliding window one step ahead forecasting for each
            # series simultaneously
            for train_idx, val_idx in splitter.split(self.X_train.index):

                # transform the training data
                X_train_temp = encoder.fit_transform(self.X_train.iloc[train_idx], self.y_train.iloc[train_idx])

                # fit the model
                model.fit(X_train_temp, self.y_train.iloc[train_idx])

                # transform the test data
                X_val_temp = encoder.transform(self.X_train.iloc[val_idx])

                # predict on the test data
                y_pred = model.predict(X_val_temp)

                # store the results
                window_results.append(pd.DataFrame({
                                                    'date': X_val_temp.index.get_level_values(2),
                                                    'series': X_val_temp.index.get_level_values(1),
                                                    'y_pred': y_pred, 
                                                    'y_true': self.y_train.iloc[val_idx].values
                                                    }))
                
            # concatenate the results
            window_results_df = pd.concat(window_results)
            window_results_df['window'] = window
            final_window_results.append(window_results_df)
                
        # format the results from each window
        final_exp_results = pd.concat(final_window_results, axis = 0)

        # format the results so they are easier to read
        pivoted_results = final_exp_results.pivot_table(
            index = ['series', 'date'], 
            columns = 'window')
        
        # flatten the column names
        pivoted_results.columns = [f'{col[0]}_{col[1]}' 
                                   for col in pivoted_results.columns]
        true_cols       = [col for col in pivoted_results if 'true' in col]
        pivoted_results = pivoted_results.drop(true_cols[1:], axis = 1)
        pivoted_results.rename({
            true_cols[0]: 'y_true'
            }, axis = 1, inplace = True)
        
        # reset the index
        pivoted_results.reset_index(inplace = True)

        return pivoted_results
    
    def _create_results_directory(self):
        """Create directory to save results for this dataset
        """
        dataset_name = self.data_file.split('.')[0]
        self.path_to_create = os.path.abspath(f'results/{dataset_name}_{self.exp_name}')
        os.makedirs(self.path_to_create, exist_ok = True)


    def run(self):
            """Function to run experiment with all available models"""

            # load data
            self._load_data()

            # create directory to save the results
            self._create_results_directory()

            # create naive forecast
            self._calc_naive_forecast()

            # transform target
            self._transform_target()

            # create X and y
            self._create_X_y()

            # create data encoder
            self._build_data_encoder()

            experiment_results = []

            # fit data
            for model in self.models:
                print(f"Fitting model for {model}")
                results = self._fit_data(self.data_encoder, model)
                metrics = self._calculate_error_metrics(results)

                # save results
                results.to_csv(f'{self.path_to_create}/{model}_preds.csv', index = False)
                metrics.to_csv(f'{self.path_to_create}/{model}_metrics.csv', 
                               index = False)