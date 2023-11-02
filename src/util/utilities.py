from statsmodels.tsa.stattools import adfuller


def check_stationarity(series):
    """Run ADF Test on a Time series and check for stationarity"""

    return adfuller(series, regression="c")[1]


def run_experiment(exp_name: str, exp_list: list) -> None:
    """Run all of the experiments grouped together under a common umbrella

    Args:
        exp_name (str): name of the master experiment
        exp_list (list): list of experiments to run
    """
    for exp in exp_list:
        exp.exp_name = exp_name
        exp.run()
