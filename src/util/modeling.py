# to suppress warnings
import os
import warnings
import re
import pandas as pd
import numpy as np
import time
import plotly.express as px

from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, r2_score, mean_squared_error
from tqdm import tqdm
from sktime.forecasting.model_evaluation import evaluate

warnings.filterwarnings('ignore')

def build_lookback_predictions(forecaster, 
                               data, 
                               splitter,  
                               max_window_length: int = 10,
                               hierarchical: bool = True
                              ) -> pd.DataFrame:
    """Function to Build Prediction Ensembles From Different Window Lengths

    Args:
        forecaster (SKTime pipeline): An SKTime pipeline or forecasting object
        data (pd.DataFrame): Dataframe to that contains the time series data
        splitter (SKTime splitter): Window splitter used to create new training / test segments
        max_window_length (int, optional): biggest window length to use. Defaults to 10.
        hierarchical (bool, optional): if you have a hierarchical time series. Defaults to True.

    Returns:
        pd.DataFrame: contains true labels, and predicted values for each lookback window
    """
    
    # to store preds from all window lengths
    final_results = pd.DataFrame()
    
    # for each window length
    for idx, i in tqdm(enumerate(range(1, max_window_length))):
        print(f"Building predictions with a lookback length of {i}")
        
        # change the lookback window to the appropriate number
        # IMPORTANT -- Assumes you have a TransformedTargetForecaster via sktime for this step
        # if you don't, will need to adjust to something else
        forecaster.set_params(TransformedTargetForecaster__RecursiveTabularRegressionForecaster__window_length = i)
        
        # get out-of-sample results for each window length
        tmp_results = evaluate(forecaster, y = data, cv = splitter, strategy = 'refit', return_data = True)

        # extract the predictions -- they are not cleanly formatted in tmp_results
        tmp_preds = extract_train_and_test_preds(tmp_results, hierarchical = hierarchical)
        
        if idx == 0:
            final_results['y_true']  = tmp_preds['y_true']
            
        final_results[f'y_pred_{i}'] = tmp_preds['y_pred']
        
    return final_results

def extract_train_and_test_preds(results_df: pd.DataFrame, hierarchical: bool = False) -> pd.DataFrame:
    """Helper function to extract test predictions & true values to use for further calculations

    Args:
        results_df (pd.DataFrame): dataframe returned by evaluate() function in sktime
        hierarchical (bool, optional): if time series data is hierarchical or not. Defaults to False.

    Returns:
        pd.DataFrame: dataframe with labels and predictions for each day / entity in the time series data
    """
    
    # if dataset has a panel orientation with a pd.MultiIndex
    if hierarchical:
        return extract_hierarchical_train_test_preds(results_df)
    else:
        y_true = np.array([row['y_test'].iloc[0] for idx, row in results_df.iterrows()])
        y_pred = np.array([row['y_pred'].iloc[0] for idx, row in results_df.iterrows()])
    
    return pd.DataFrame({'y_true': y_true, 'y_pred': y_pred, 'Date': results_df['cutoff'].values})

def extract_hierarchical_train_test_preds(results_df: pd.DataFrame) -> pd.DataFrame:
    """_summary_Takes the output of the evaluate() function in sktime, and extracts individual 
    predictions for a hierarchical dataset

    Args:
        results_df (pd.DataFrame): output of evaluate() function in sktime, when data is hierarchical

    Raises:
        Exception: Not an equal number of test values and test predictions

    Returns:
        pd.DataFrame: dataframe with labels and predictions for each day / entity in the time series data
    """
    
    if results_df['y_test'].shape[0] != results_df['y_pred'].shape[0]:
        raise Exception("Not an equal number of test values and test predictions")
        
    # how many unique predicted dates are there?    
    n_preds = results_df['y_test'].shape[0]
    
    # list of dfs to concat at the end
    results = []
    
    # get the train & test predictions for each cutoff point
    for i in range(n_preds):
        test_vals = results_df['y_test'].iloc[i]
        test_preds = results_df['y_pred'].iloc[i]
        
        # rename columns for naming conventions -- for later functions to use
        test_vals.rename({'Values': 'y_true'}, axis = 1, inplace = True)
        test_preds.rename({'Values': 'y_pred'}, axis = 1, inplace = True)
        
        tmp_totals = test_vals.merge(test_preds, left_index = True, right_index = True)
        
        results.append(tmp_totals)
        
    # convert into a final dataframe
    return pd.concat(results).reset_index()

def get_valid_filename(file_name: str) -> str:
    """Transforms a string into a valid filename"""
    s = str(file_name).strip().replace(" ", "_")
    s = re.sub(r"(?u)[^-\w.]", "", s)
    
    return s

def calculate_metrics(y_true, y_preds, round_vals = False) -> dict:
    """Calculate performance metrics for a set of predictions / labels

    Args:
        y_true (np.array): ground truth labels
        y_preds (np.array): model predictions
        round_vals (bool, optional): if you want to round your predictions to integers. Defaults to False.

    Returns:
        dict: key / value for each metric / score used
    """

    if round_vals:
        y_preds = y_preds.round()
           
    results =  {
        'mape': mean_absolute_percentage_error(y_true, y_preds),
        'rmse': np.sqrt(mean_squared_error(y_true, y_preds)),
        'mae': mean_absolute_error(y_true, y_preds),
        'r2': r2_score(y_true, y_preds)
    }

    return results

def calc_metrics_for_preds_df(preds_df: pd.DataFrame, round_vals = False) -> pd.DataFrame:
    """Calculate performance metrics for each lookback period and ensemble

    Args:
        preds_df (pd.DataFrame): dataframe returned by calc_ensemble_predictions
        round_vals (bool, optional): if you want to round your predictions to integers. Defaults to False.

    Returns:
        pd.DataFrame: dataframe of metrics for each lookback period and ensemble
    """
    
    final_results = []
    
    pred_cols = [col for col in preds_df.columns if 'pred' in col]
    
    for col in pred_cols:
        pred_metrics = calculate_metrics(preds_df['y_true'], preds_df[col], round_vals = round_vals)
        pred_metrics['model'] = col
        final_results.append(pred_metrics)
        
    return pd.DataFrame(final_results)

def calc_ensembled_predictions(lookback_preds_df: pd.DataFrame) -> pd.DataFrame:
    """Builds ensembles of predictions from different lookback windows

    Args:
        lookback_preds_df (pd.DataFrame): dataframe returned by build_lookback_predictions

    Returns:
        pd.DataFrame: dataframe with predicted values for each ensemble, appened to lookback_preds_df
    """
    
    pred_cols = sorted([col for col in lookback_preds_df.columns 
                 if 'pred' in col])
    
    
    ensemble_preds_df = lookback_preds_df.copy()
    
    # ensemble all of the prediction columns starting with 2, then 3, etc....
    for i in range(2, len(pred_cols) + 1):
        ensemble_preds_df[f'ensemble_preds_{i}'] = lookback_preds_df.loc[:, pred_cols[:i]].mean(axis = 1)
        
    return ensemble_preds_df

def run_experiment(name: str, 
                   model, 
                   splitter, 
                   hierarchical: bool = True,
                   round_vals: bool = False,
                   max_window_length: int = 10,
                   data_file: str = 'car_parts_final.csv', 
                   output_dir: str = 'results') -> None:

    """meta function to load in a dataset and produce results.  Outputs all results & artifacts to results folder.

    HAS ONLY BEEN TESTED TO WORK ON THE CAR PARTS DATASET
    
    Args:
        name (str): name of the experiment to run
        model (SKTime pipeline): model to use in experiment.  Likely an sktime / sklearn pipeline or estimator
        splitter (SKTime splitter): window splitter to use for creating train / test sets
        hierarchical (bool): whether data is hierarchical or not.  Defaults to True
        round_vals (bool): whether to round predictions to integers.  Defaults to False.
        max_window_length (int):  maximum lookback window to use.  Function will create lookback windows from 1 to
        this value
        data_file (str): name of the data file to read in.  Assumes it's in the data folder
        output_dir (str): name of the folder to output the results to.

    Returns:
        None
    """

    # first, load in the dataset -- using os module to avoid relative filepaths
    # assumes you are running this from the root directory
    current_dir = os.getcwd()
    file_path   = os.path.join(f'{current_dir}/data', data_file)
    
    # check to make sure the specified file exists
    if not os.path.isfile(file_path):
        raise Exception(f"""Provided filepath does not exist on your computer: {file_path}.  
        It's recommended to run this function from the root directory of this repo""")
        
    # now load in the file
    df = pd.read_csv(file_path, parse_dates = ['Date'])
    
    # set the index to make it sktime compatible
    df.set_index(['Series', 'Date'], inplace = True)
    
    df.index = df.index.set_levels(df.index.levels[1].to_period('M'), level='Date')
    
    print(f"""Loaded in the file from path: {file_path}, 
    dataset has {df.shape[0]} rows and {df.shape[1]} columns""")
    
    print("Fitting model on different lookback windows")
    
    # this will take awhile to run -- maybe 1 - 3 hrs
    # this df will contain a column of predictions for each lookback window size
    lookback_preds = build_lookback_predictions(model, 
                                                df[:204], 
                                                splitter = splitter,
                                                max_window_length = max_window_length,
                                                hierarchical = hierarchical)
    
    # build ensembled predictions from the different lookback windows
    # ie, if total lookbacks is 9, will ensemble predictions from 2, 3, 4
    # 5, etc lookbacks
    ensemble_preds = calc_ensembled_predictions(lookback_preds)
    
    # calculate the metrics for each of the ensemble values
    metric_results = calc_metrics_for_preds_df(ensemble_preds, round_vals = round_vals)

    # graphs to demonstrate the results
    fig1 = px.line(x = range(2, max_window_length), y = metric_results.loc[max_window_length-1:, 'mae'], title = 'MAE vs # of Window Ensembles')
    fig1.update_layout(xaxis_title = '# of Window Ensembles', yaxis_title = 'MAE')
    
    fig2 = px.line(x = range(2, max_window_length), y = metric_results.loc[max_window_length-1:, 'rmse'], title = 'RMSE vs # of Window Ensembles')
    fig2.update_layout(xaxis_title = '# of Window Ensembles', yaxis_title = 'RMSE')
    
    print("Getting ready to export results to folder........")
    
    # transform experiment name into a valid directory name
    exp_name = get_valid_filename(name)
    
    # create the directory
    timestr  = time.strftime("%Y%m%d-%H%M%S")
    dir_name = os.path.join(f'{current_dir}/results', f'{exp_name}_{timestr}')
    
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)
        
    # export the files
    img1_path = os.path.join(dir_name, 'MAE_vs_Ensembles.png')
    fig1.write_image(img1_path)
    
    img2_path = os.path.join(dir_name, 'RMSE_vs_Ensembles.png')
    fig2.write_image(img2_path)
    
    lookback_preds_path = os.path.join(dir_name, 'lookback_preds.csv')
    
    lookback_preds.to_csv(lookback_preds_path, index = False)
    
    ensemble_preds_path = os.path.join(dir_name, 'ensemble_preds.csv')
    ensemble_preds.to_csv(ensemble_preds_path, index = False)
    
    metric_results_path = os.path.join(dir_name, 'metric_results.csv')
    metric_results.to_csv(metric_results_path, index = False)
    
    print("Finished.")
