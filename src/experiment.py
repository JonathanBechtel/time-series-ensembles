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

    elif args.experiment == 'covid_standard':
        covid_standard.run()

    elif args.experiment == 'fred_standard':
        fred_standard.run()

    elif args.experiment == 'traffic_standard':
        traffic_standard.run()

    elif args.experiment == 'hospital_standard':
        hospital_standard.run()
    
    elif args.experiment == 'tourism_standard':
        tourism_standard.run()

    elif args.experiment == 'm1_standard':
        m1_standard.run()

    elif args.experiment == 'm3_standard':
        m3_standard.run()

    elif args.experiment == 'm4_standard':
        m4_standard.run()

    elif args.experiment == 'all':
        electricity_standard.run()
        covid_standard.run()
        fred_standard.run()
        traffic_standard.run()
        hospital_standard.run()
        tourism_standard.run()
        m1_standard.run()
        m3_standard.run()
        m4_standard.run()

    
