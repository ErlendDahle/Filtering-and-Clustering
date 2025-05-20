import numpy as np
import xarray as xr 
from sklearn.metrics import mean_squared_error


def moving_average(data, window_size):
    """
    Smooths the data using a moving average filter.
    """
    nbr_of_lines= data.shape[0]
    filtered_data=np.zeros((nbr_of_lines,data.shape[1]-window_size+1))
    for i in range(nbr_of_lines):  
        filtered_data[i]=np.convolve(data[i], np.ones(window_size)/window_size, mode='valid')

    return filtered_data

def interpolate_nan(signal):
    nans, x = np.isnan(signal), lambda z: z.nonzero()[0]
    signal[nans] = np.interp(x(nans), x(~nans), signal[~nans])
    return signal


def compare_signals(signal1, signal2):
    min_length = min(len(signal1), len(signal2))
    # Ensure both signals are of the same length
    signal1 = signal1[:min_length]
    signal2 = signal2[:min_length]
    mse = mean_squared_error(signal1, signal2)
    return mse