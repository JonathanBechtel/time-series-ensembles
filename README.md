# Time Series Ensembling ⏲️
This repository is meant to host the code and analysis for the project specified in this repository: https://github.com/DSML-Research-Group/public-projects/issues/2

The "big idea" behind the project is that combining different window sizes is a novel way of improving time series results, and allows models to improve both their bias and variance, since different window lengths contain both fundamentally different types of information about a time series, as well as serve the purpose of cancelling each other out.  

# Background Information 📖
Luckily for us, this problem is fairly novel, and represents a new way of looking at the domain of time series modeling.  However, the once precedent for using this technique and getting results with it was established in the N-Beats paper:  https://arxiv.org/abs/1905.10437

Ironically, this technique was NOT the main idea behind the paper, but they tried it and it worked.

# Helpful Links 🔗
 - N-Beats paper: https://arxiv.org/abs/1905.10437

# Roadmap 🗺️
The current roadmap is this:
 - [x] Collect initial datasets to use for the project
 - [ ] Write preprocessing steps that can do the necessary data transformations to get each dataset ready for modeling
 - [ ] Create some sort of scheme for the different datasets to note how to handle them for modeling
 - [x] Create starter notebooks with sample code for other people to visually explore the results and be able to grok the problem
 - [x] Write files that automate the modeling process and export results to a particular folder
 - [ ] Look at the results!  This is the most important part.  Look at the evals and parse what the results tell us.

# Current Questions ❔
Current issues that are being discussed are:

 - What rules should there be for pre-processing the data?
 - How much of the project can be done locally, vs. doing over the cloud?

# Installation 🖥️
To install the project on your computer locally, please follow these steps:

1).  Fork the repo to your own account

2).  Copy the URL of the git file (this is not the public url of the repo)

3).  From your command line or in Github Desktop run the command `git clone git_repo_location`

To install the dependencies, you can then run the commands:

### Regular Python 🐍
`cd time-series-ensembles`

`pip install -r requirements.txt`

### With Anaconda 🐍
If you'd like to create a development environment with anaconda, you can install everything you need with the `environment.yaml` file by following these steps:

First navigate into the root directory of the repo with this line:

`cd time-series-ensembles`

`conda env create -f environment.yaml`

If you then want to activate that environment you can do so with the command:

`conda activate time_series_ensembles`

### Creating Results

To recreate results for a particular experiment with a particular dataset, navigate to the root folder of this repo and run the following command:  `python -m src.experiment "experiment name" "dataset name"`

An example of this command would be: `python -m src.experiment "linear regression detrended demeaned" "car_parts"`

The current list of datasets and experiments that have been conducted are listed below:

| Dataset   | Experiment Name                      |
| --------- | -------------------------------------|
| car_parts | linear regression detrended demeaned |

Other arguments can be found in the `experiment.py` file to specify experiment parameters, if you wish.

# Project Outline

We'll cover the basic setup of this project's workflow with the following steps:

 - Data
 - Choice of models
 - Choice of tools
 - Modeling Approach
 - Pre-Processing
 - Feature Engineering
 - Fitting Routine
 - Evaluation

## Data

The datasets we'll be using for this project are listed in the following table.  Each one is listed with some of its basic attributes.  Each of these is a panel dataset, with multiple observations for each entity inside the dataset

| Dataset      | # Entities  | # Rows | Description                               | **Frequency** |
| ------------ | ----------- | ------ | ----------------------------------------- | ------------- |
| covid deaths | 266         | 56392  | daily covid deaths for different entities | Daily         |
| fred         | 107         | 77896  | economic time series collected from FRED  | Monthly       |
| traffic      | 862         | 89648  | traffic counts across cities              | Weekly        |
| tourism      | 366         | 109280 | tourist visits across cities              | Monthly       |
| m3           | 1428        | 167562 | monthly sales for m3 competition          | Monthly       |
| m1           | 617         | 55998  | monthly sales for m1 competition          | Monthly       |
| hospital     | 767         | 64428  | visits to different hospitals             | Monthly       |
| m4           | 414         | 373372 | hourly sales for m4 competition           | Hourly        |
| electricity  | 321         | 47508  | electricity usage for 321 households      | Weekly        |

## Choice of Models

Broadly speaking, there are three categories of models you can choose from when doing forecasting.  They are the following:

| Model Class        | Examples                           | # Strengths                                       | Weaknesses                                |
| -----------------  | ---------------------------------- | ------------------------------------------------- | ----------------------------------------- |
| Machine Learning   | KNN, LASSO, GBM                    | Global datasets, complicated, non-linear patterns | Not good for small series, univariate     |
| Statistical models | ARIMA, Exponential Smoothing, VAR  | autocorrelation, small time series                | Can't be applied globally                 |
| Deep Learning      | LSTM, TFT, N-BEATS                 | Can handle complex interdependencies better       | Too much for many problems                |

Each class of model may require different fitting routines, and might possibly display results that are different from the other.  Therefore, it's recommended that we only work on one class of models at a time, and proceed from there is we please.

## Choice of Tools
There are several options you have for doing forecasting with Python, such as the following:

 - SKTime
 - Nixtla
 - TSFresh
 - Statsmodels

These usually provide a combination of specific models, as well as utility functions like cross validation and feature extraction that are specific to time series.  This is in addition to libraries that provide the standard ML models we'll use such as GBM's and neural networks.  

The current scripts run experiments using `SKTime`, since it's pretty comprehensive, but this is open to change as time goes on.

## Modeling Approach
Every dataset in this project has multiple instances, and can therefore be modeled either as separate time series, or as a global dataset.  This project intends to take the global approach, for two reasons:
  - It's often most suitable for complicated datasets with many entities that have different characteristics
  - It's more suitable for ML based approaches, since they require more data for pattern extraction

This can easily be managed inside SKTime, as long as you setup the modeling pipeline in the correct way.  

The below code snippet is a brief example:  

`from sktime.transformations.series.difference import Differencer`
`from sktime.transformations.series.boxcox import LogTransformer`
`from sklearn.ensemble import RandomForestRegressor`

`kwargs = {`
    `'lag_feature': {`
        `"lag" : [1, 2],`
        `'mean': [[1, 2], [1, 3]],`
        `'std': [[1, 2], [1, 3]],`
        `'skew': [[1, 5]]`
    `}`
`}`

`summarizer =  WindowSummarizer(**kwargs)`

`forecaster = make_reduction(RandomForestRegressor(max_depth = 6),`
                            `window_length = None,` 
                            `transformers = [WindowSummarizer(**kwargs)],`
                            `pooling = "global")`

`pipeline = ForecastingPipeline(`
    `steps=[`
        `("log", LogTransformer(offset = 1)`,
        `("diff", Differencer(lags = 1)),`
        `("forecaster", forecaster),`
    `]`
`)`

In the above code, we're using the `kwargs` dictionary to specify different lag variables (these are the experimental variables in the dataset), and specifying the `global` option in the `make_reduction` function means the model will be fit on the entire dataset, as opposed to fitting one separately for each time series.

## Pre-Processing

To ensure consistent results, we want to make sure each dataset is transformed to make it as stationary as possible.  To do this, we'll do a combination of log transformations and differencing, for both previous values and seasonal terms if they exist.  

To check for stationarity afterwards, we'll apply the ADF test to each panel in the time series.  We'd like to shoot for at least a 90 - 95% portion of stationarity for each dataset.

If we're using linear models or deep learning, we'll want to standardize the data after this transformation to make coefficients consistent, and to avoid gradient overflows.

All of these pre-processing steps can be specified in the pipeline, but you should do a little bit of EDA beforehand in a notebook to demonstrate what pre-processing technique is most appropriate for the dataset.  See the notebook `BasicDataExploration -- Car Parts.ipynb` for an example of how you can do this.  

Once this is done, the pre-processing steps should be added to the pipelines that are used for that dataset.

## Feature Engineering

The standard feature to create for this experiment is a lag variable of the previous value over different horizons, but it'll probably be prudent to create additional window variables for ML models, since they typically need a lot of features to help them discern time based data.  These can be calculated inside the `WindowSummarizer` varialbe mentioned in the above section.  

We'll begin by calculating moving averages, moving standard deviations, and moving measures of skew to detect pattern drift in the data.  

## Fitting Routine

The `evaluate` function in sktime provides a nice utility for evaluating model results.  

The approach we'll be using to model data is one-step ahead cross validation, where we continually expand the training set by one sample for each entity and predict the next observation to calculate model metrics.  

A quick example can be seen below:

from sktime.forecasting.model_selection import temporal_train_test_split, ExpandingWindowSplitter
from sktime.forecasting.model_evaluation import evaluate

`cv      = ExpandingWindowSplitter(step_length = 1, fh=[1], initial_window = 100)`
`results = evaluate(forecaster = pipeline, y = data, cv = cv, strategy="refit", return_data=True)`

In the above example, `cv` is a data splitter that will incrementally walk through the time series with a step size of 1 and make a prediction one time step into the future.
`evaluate` uses the pipeline created earlier to fit the model on the data returned by `cv` and returns the validation results for each out of sample prediction.  

## Evaluation

At the end of each experiment, we'll ensemble the predictions from each of the lookbacks, and **only include results that are better than the naive baseline.**
                            
