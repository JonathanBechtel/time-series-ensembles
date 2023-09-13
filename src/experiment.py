# Loads Arguments Into Argparse, Runs Experiments From Command Line
import argparse
import warnings
from src.util.modeling import run_experiment

from sklearn.linear_model import LinearRegression
from sktime.forecasting.compose import make_reduction, TransformedTargetForecaster
from sktime.transformations.series.detrend import Detrender, Deseasonalizer
from sktime.transformations.series.difference import Differencer
from sktime.forecasting.model_selection import ExpandingWindowSplitter
from sktime.forecasting.naive import NaiveForecaster

from statsmodels.stats.diagnostic import acorr_ljungbox

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(
                    prog='Time Series Ensembles',
                    description='Executable To Run Experiments',
                    epilog='Text at the bottom of help')

# the name of the experiment & dataset name to run
parser.add_argument('experiment_name', 
                    type=str, 
                    help='The name of the experiment to run')

parser.add_argument('dataset_name',
                    type=str,
                    help='The name of the dataset to run')

# following are arguments to pass into run_car_parts_experiment function
parser.add_argument('-hi', '--hierarchical',  
                    action='store_false', default=True)
parser.add_argument('-r', '--round_vals',  action='store_false', default=False)
parser.add_argument('-m', '--max_window_length', 
                    type=int, default=10)
parser.add_argument('-d', '--data_path',  
                    type=str, default='car_parts_final.csv')
parser.add_argument('-o', '--output_dir',  type=str, default='results')

if __name__ == '__main__':
    args = parser.parse_args()

    if args.dataset_name == 'car_parts':
        if args.experiment_name == 'linear regression detrended demeaned':
            model = TransformedTargetForecaster(
                    [ Detrender() *
                      Deseasonalizer(sp = 12) *
                      Deseasonalizer(sp = 3) *
                      make_reduction(LinearRegression(), 
                                     window_length = 1, 
                                     strategy = 'recursive')])

            splitter = ExpandingWindowSplitter(step_length=1, fh=[1], initial_window=25)

            run_experiment(model = model,
                        splitter = splitter,
                        name = args.experiment_name, 
                        hierarchical = args.hierarchical,
                        round_vals = args.round_vals,
                        max_window_length = args.max_window_length,
                        data_file = 'car_parts_final.csv',
                        output_dir = args.output_dir)
            
        elif args.experiment_name == 'linear regression differenced':

            model = TransformedTargetForecaster(
                    [ Differencer(lags = 1) *
                     make_reduction(LinearRegression(), 
                                     window_length = 1, 
                                     strategy = 'recursive')])
            
            splitter = ExpandingWindowSplitter(step_length=1, fh=[1], initial_window=25)
            run_experiment(model = model,
                           splitter = splitter,
                           name = args.experiment_name, 
                           hierarchical = args.hierarchical,
                           round_vals = args.round_vals,
                           max_window_length = args.max_window_length,
                           data_file = 'car_parts_final.csv',
                           output_dir = args.output_dir)
            
        elif args.experiment_name == 'naive':
            model = TransformedTargetForecaster(
                [Differencer(lags = 1) * NaiveForecaster(strategy = 'last')]
                )
            
            splitter = ExpandingWindowSplitter(step_length=1, fh=[1], initial_window=25)
            run_experiment(model = model,
                            splitter = splitter,
                            name = args.experiment_name, 
                            hierarchical = args.hierarchical,
                            round_vals = args.round_vals,

                            # for naive this should just be set to 1
                            max_window_length = args.max_window_length,
                            data_file = 'car_parts_final.csv',
                            output_dir = args.output_dir)
