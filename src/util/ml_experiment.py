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

# category encoders
from category_encoders import OneHotEncoder, TargetEncoder

class MLForecastingExperiment():

    def __init__(self, 
                 data_file: str, 
                 models: list, 
                 target_transform: str, 
                 lags: list = [1, 2, 3, 7],
                 calibration_windows: list = [7, 14, 28],
                 encode_entity: bool = True,
                 train_size: int = 120,
                 # list of date parts to encode for date features
                 date_parts_to_encode: list = ['month'],
                 window_transforms: dict = None):

        self.data_file = data_file
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
            self.data[f'lag_{lag}'] = self.data.groupby(level = 0)['value'].shift(lag)

        # create X based off of window transforms
        if self.window_transforms is not None:
            self._create_window_transforms()

        if self.encode_entity:
            self.data['entity'] = self.data.index.get_level_values(0)

        if self.date_parts_to_encode is not None:
            self._add_date_features()

        # drop rows with missing values
        self.data.dropna(inplace = True)

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
                                                    'date': self.X_train.iloc[val_idx].index.get_level_values(1),
                                                    'series': self.X_train.iloc[val_idx].index.get_level_values(0),
                                                    f'y_pred_{window}': y_pred, 
                                                    'y_true': self.y_train.iloc[val_idx]
                                                    }))
                
            # concatenate the results
            window_results = pd.concat(window_results, axis = 0)
            final_window_results.append(window_results)
                
        # format the results from each window
        final_window_results = pd.concat(final_window_results, axis = 0)
        return final_window_results

    def run(self):
            """Function to run experiment with all available models"""

            # load data
            self._load_data()

            # transform target
            self._transform_target()

            # create X and y
            self._create_X_y()

            # create data encoder
            self._build_data_encoder()

            # fit data
            for model in self.models:
                print(f"Fitting model for {model}")
                results = self._fit_data(self.data_encoder, model)

                results['model'] = model

                # create directory to save the results
                current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
                dataset_name = self.data_file.split('.')[0]
                path_to_create = os.path.abspath(f'results/{dataset_name}')
                print(path_to_create)
                os.makedirs(path_to_create, exist_ok = True)
                results.to_csv(f'{path_to_create}/{model}_{current_time}.csv')

if __name__ == '__main__':

    # run experiment
    experiment = MLForecastingExperiment(data_file = 'weekly_electricity_demand_final.csv',
                                         target_transform = 'log_diff',
                                         models = [Ridge(alpha = 1.0), 
                                                   Lasso(alpha = 0.1), 
                                                   KNeighborsRegressor(n_neighbors = 5),
                                                   RandomForestRegressor(n_estimators = 100, 
                                                                         min_samples_leaf = 5,
                                                                         max_features = 0.8)])  
    experiment.run()                                      






    

    
