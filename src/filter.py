from scipy import signal
import numpy as np

from configurations import ModelingParameters as JP

# Implementation of first order low pass filter
def first_order_lpf(time_series_list,cutoff_hz=JP.lpf_cutoff_hz):
    
    """
    First order low pass filter implementation

    Args:
        time_series_list (list of numpy arrays): List of time series to be filtered
        cutoff_hz (float, optional): Cutoff frequency in Hz for the lpf . Defaults to JP.pos_lpf_cutoff_hz.

    Returns:
        list of numpy array: First order low pass filtered time series
    """
    
    filtered_time_series_list = time_series_list
    
    w0 = 2*np.pi*cutoff_hz; # pole frequency (rad/s)
    num = w0        # transfer function numerator coefficients
    den = [1,w0]    # transfer function denominator coefficients
    lowPass = signal.TransferFunction(num,den) # Transfer function
    
    discreteLowPass = lowPass.to_discrete(JP.resampled_dt_s,method='gbt',alpha=0.5) #generalized bilinear transform
    
    b = discreteLowPass.num
    a = -discreteLowPass.den
    
    for ts_idx in range(len(time_series_list)):
        for idx in range(3,len(time_series_list[ts_idx])):
            filtered_time_series_list[ts_idx][idx] = \
                a[1]*filtered_time_series_list[ts_idx][idx-1] + b[0]*time_series_list[ts_idx][idx] + b[1]*time_series_list[ts_idx][idx-1]
    
    return filtered_time_series_list
