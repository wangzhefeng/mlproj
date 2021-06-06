# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd 


def series_to_supervised(data, n_lag = 1, n_fut = 1, selLag = None, selFut = None, dropnan = True):
    """
    Converts a time series to a supervised learning data set by adding 
    time-shifted prior and future period data as input or output 
    (i.e., target result) columns for each period.
    
    :param data:  
        a series of periodic attributes as a list or NumPy array.
    :param n_lag: 
        number of PRIOR periods to lag as input (X); 
        generates: Xa(t-1), Xa(t-2); min= 0 --> nothing lagged.
    :param n_fut: 
        number of FUTURE periods to add as target output (y); 
        generates Yout(t+1); min= 0 --> no future periods.
    :param selLag:  
        only copy these specific PRIOR period attributes; 
        default= None; EX: ['Xa', 'Xb' ].
    :param selFut:  
        only copy these specific FUTURE period attributes; 
        default= None; EX: ['rslt', 'xx'].
    :param dropnan: 
        True= drop rows with NaN values; 
        default= True.
    :return: 
        a Pandas DataFrame of time series data organized for supervised learning.

    NOTES:
        (1) The current period's data is always included in the output.
        (2) A suffix is added to the original column names to indicate a relative time reference: 
            e.g., (t) is the currentperiod; 
                  (t-2) is from two periods in the past; 
                  (t+1) is from the next period.
        (3) This is an extension of Jason Brownlee's series_to_supervised() function, customized for MFI use
    """
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    origNames = df.columns
    cols, names = list(), list()
    # include all current period attributes
    cols.append(df.shift(0))
    names += [("%s" % origNames[j]) for j in range(n_vars)]
    # lag any past period attributes (t-n_lag, ..., t-1)
    n_lag = max(0, n_lag) # force valid number of lag periods
    # input sequence (t-n, ..., t-1)
    for i in range(n_lag, 0, -1):
        suffix = "(t-%d)" % i
        if (None == selLag):
            cols.append(df.shift(i))
            names += [("%s%s" % (origNames[j], suffix)) for j in range(n_vars)]
        else:
            for var in (selLag):
                cols.append(df[var].shift(i))
                names += [("%s%s" % (var, suffix))]
    # include future period attributes (t+1, ..., t+n_fut)
    n_fut = max(n_fut, 0)
    # forecast sequence (t, t+1, ..., t+n)
    for i in range(0, n_fut + 1):
        suffix = "(t+%d)" % i
        if (None == selFut):
            cols.append(df.shift(-i))
            names += [("%s%s" % (origNames[j], suffix)) for j in range(n_vars)]
        else:
            for var in (selFut):
                cols.append(df[var].shift(-i))
                names += [("%s%s" % (var, suffix))]
    # put it all together
    agg = pd.concat(cols, axis = 1)
    agg.columns = names

    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace = True)

    return agg



if __name__ == "__main__":
    pass