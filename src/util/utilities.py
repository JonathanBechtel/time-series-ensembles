def check_stationarity(series):
    """Run ADF Test on a Time series and check for stationarity"""
    
    return adfuller(series, regression = 'c')[1]