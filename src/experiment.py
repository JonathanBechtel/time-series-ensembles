# Loads Arguments Into Argparse, Runs Experiments From Command Line
import argparse
import warnings

from datetime import datetime
from src.experiment_list import *
from src.util.utilities import run_experiment

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(
    prog="Time Series Ensembles",
    description="Executable To Run Experiments",
    epilog="Text at the bottom of help",
)

# the name of the experiment & dataset name to run
parser.add_argument(
    "-e", "--experiment", type=str, help="The name of the experiment to run"
)

if __name__ == "__main__":
    args = parser.parse_args()

    if args.experiment == "all_standard":
        str_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_experiment(f"standard_exps_{str_date}", standard_exps)
