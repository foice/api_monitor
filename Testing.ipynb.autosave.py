#!/usr/bin/env python
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc" style="margin-top: 1em;"><ul class="toc-item"></ul></div>

# In[23]:


import pandas as pd
import datetime
import numpy as np

def date_interval_ceil2floor_bins(_min,_max,chunk):
    _from=_min.ceil('1h')
    _to=_max.floor('1h')

    dates_=pd.date_range(start=_from, end=_to, freq=str( chunk )+"min" )

    bins = np.array([dates_[:-1],dates_[1:]]).T
    return bins

def window_dataframe(df,_from,_to):
    '''
    df_to_save=pd_bars[_interval][(pd_bars[_interval]['time'] >= _from) & (pd_bars[_interval]['time'] < _to)]
    '''
    df_to_save=df[(df['time'] >= _from) & (df['time'] < _to)]
    return df_to_save


# In[32]:


import requests

resp = requests.get('https://api.kraken.com/0/public/AssetPairs')
txt = resp.json(  )
available_pairs=pd.DataFrame(txt['result']).T


# In[36]:


all_pairs=sorted(list(available_pairs['altname'].values))


# In[35]:


sorted(all_pairs)


# In[31]:


intervals=[1,15,60,1440]
pairs=['XTZEUR']

for _interval in intervals:
    chunk=int(_interval*60)
    for _pair in pairs:

        pd_bars=pd.DataFrame({})
        _min=pd.to_datetime( datetime.datetime.now() - datetime.timedelta(days=0,hours=0,minutes=720*_interval) )
        _max=pd.to_datetime( datetime.datetime.now() )

        bins=date_interval_ceil2floor_bins(_min,_max,chunk)
        
        for _from, _to in bins:

            try:
                # making the time slice that was missing
                df_to_save=window_dataframe(pd_bars,_from,_to)        
            except KeyError:
                print( "get_bars")


# In[ ]:




