#!/usr/bin/env python3

# This file is part of krakenex.
# Licensed under the Simplified BSD license. See `examples/LICENSE.txt`.

# Pretty-print a pair's order book depth.


import krakenex
import pandas as pd
import kraken_tools as kt
import numpy as np
import datetime, time, os
from requests.exceptions import HTTPError

import utils

double_logger_monitor_pairs=utils.create_double_logger(logger_name='monitor_pairs.py',log_file='monitor_pairs.py.log')

since=1499000000

def push_longdate(d,unit='M',to='floor'):

    from pandas.tseries.offsets import MonthEnd, YearEnd

    z=pd.to_datetime(d).to_period(unit).to_timestamp()

    if unit=='M':
        end =  MonthEnd(1)
    if unit=='Y':
        end = YearEnd(1)

    if (to == 'ceil') or (to == 'ceiling'):
        return z + end
    if to == 'floor':
        return z

def filename(_pair,_interval,_from,_to):
    return "pairs/"+_pair+"/"+str(_interval)+"/"+_pair+"@"+str(_from)+'@'+str(_to)+".csv"


def date_interval_ceil2floor_bins(_min,_max,file_resolution='1h'):

    if file_resolution in ['1d','1h']:
        _from=_min.ceil(file_resolution)
        _to=_max.floor(file_resolution)

        dates_=pd.date_range(start=_from, end=_to, freq=file_resolution)#str( chunk )+"min" ) # a file for each

    else:
        if file_resolution=='1M':
            unit='M'
        if file_resolution=='1Y':
            unit='Y'
        # bins will be 1 month
        _from=push_longdate(_min,unit=unit,to='ceil')#ceil_longdate(_min)
        _to =push_longdate(_max,unit=unit,to='floor') #floor_longdate(_max)
        dates_=pd.date_range(start=_from, end=_to, freq=unit ) # one file for each

    double_logger_monitor_pairs.debug( dates_.shape )

    bins = np.array([dates_[:-1],dates_[1:]]).T
    return bins

def get_bars(pair=None,since=None,interval=None):
    #pd_bars={}
    #get_bars
    pd_bars=kt.get_history(pair=pair,since=since,interval=interval)

    # interpret result
    for _field in ['open', 'high', 'low', 'close', 'vwap', 'volume', 'count']:
        pd_bars[_field]=pd.to_numeric( pd_bars[_field])

    return pd_bars

def window_dataframe(df,_from,_to):
    '''
    df_to_save=pd_bars[_interval][(pd_bars[_interval]['time'] >= _from) & (pd_bars[_interval]['time'] < _to)]
    '''
    try:
        df_to_save=df[(df['time'] >= _from) & (df['time'] < _to)]
        return df_to_save
    except KeyError:
        return None

def pairs2csv(pairs=None,intervals=[1,15,60,1440],chunksize={1:60,15:24*60,1440:24*60,21600:24*60},api_length=720,file_resolution={1:"1h",15:"1d",1440:"1M",21600:"1Y"}):
    if pairs is not None:
        for _interval in intervals:
            #chunk=chunksize[_interval] #minutes #int(_interval*chunksize)
            for _pair in pairs:

                pd_bars=pd.DataFrame({})
                _min=pd.to_datetime( datetime.datetime.utcnow() - datetime.timedelta(days=0,hours=0,minutes=api_length*_interval) )
                _max=pd.to_datetime( datetime.datetime.utcnow() )

                bins=date_interval_ceil2floor_bins(_min,_max,file_resolution=file_resolution[_interval])

                for _from, _to in bins:

                    _filename_=filename(_pair,_interval,_from,_to) #"pairs/"+_pair+"/"+str(_interval)+"/"+_pair+"@"+str(_from)+'@'+str(_to)+".csv"
                    # check if file exists
                    if os.path.isfile(_filename_):
                        double_logger_monitor_pairs.debug(_filename_ + " file exist")
                    else:
                        double_logger_monitor_pairs.info(_filename_ + " file does not exist")
                        #check if need to ask for bars or we got this already

                        # try to make the time slice that was missing
                        df_to_save=window_dataframe(pd_bars,_from,_to)

                        if df_to_save is None:
                            double_logger_monitor_pairs.info("slice missing, fetching bars to fill "+str(_from)+"-->"+str(_to))

                            # get_bars
                            pd_bars=get_bars(pair=_pair,since=since,interval=_interval)
                        # making the time slice that was missing
                        df_to_save=window_dataframe(pd_bars,_from,_to)
                        # save result
                        df_to_save.to_csv(_filename_)
                time.sleep(0.1)
