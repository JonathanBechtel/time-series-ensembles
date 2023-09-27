# Loads Arguments Into Argparse, Runs Experiments From Command Line
import argparse
import warnings
from src.experiment_list import *

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(
                    prog='Time Series Ensembles',
                    description='Executable To Run Experiments',
                    epilog='Text at the bottom of help')

# the name of the experiment & dataset name to run
parser.add_argument('-e', 
                    '--experiment',
                    type=str, 
                    help='The name of the experiment to run')

if __name__ == '__main__':
    args = parser.parse_args()

    if args.experiment == 'electricity_standard':
        electricity_standard.run()
    
