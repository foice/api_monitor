#!/usr/bin/env python
# coding: utf-8

# In[1]:


from IPython.core.display import display, HTML
display(HTML("<style>.container { width:99% !important; }</style>"))


# In[2]:


import krakenex
import kraken_tools as kt


# In[3]:


import pandas as pd
import numpy as np
from pandas.tseries.offsets import MonthEnd, YearEnd
import requests
import os, glob, os.path

import datetime, time
import utils, importlib


# In[4]:


from bokeh.io import show, output_file, save, export_png, export_svgs
from bokeh.layouts import column
from bokeh.models import ColumnDataSource, RangeTool
from bokeh.plotting import figure
from bokeh.plotting import output_notebook, figure, show
#
from bokeh.core.properties import value
from bokeh.palettes import Spectral5,Spectral4

output_notebook()


# In[5]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[6]:


plt.rcParams['figure.figsize'] = [8.0, 6.0]
plt.rcParams['figure.dpi'] = 80
plt.rcParams['savefig.dpi'] = 150

plt.rcParams['font.size'] = 18
plt.rcParams['legend.fontsize'] = 'large'
plt.rcParams['figure.titlesize'] = 'medium'


# In[11]:


importlib.reload(utils)


# In[122]:


importlib.reload(kt)


# In[9]:


ETHmonitor_double_logger=utils.create_double_logger(logger_name='ETHmonitor',log_file='ETHmonitor.log')


# In[10]:


ETHmonitor_double_logger.info('INFO to console and disk?')
ETHmonitor_double_logger.debug('INFO only to disk?')


# # Logging

# Levels are used for identifying the severity of an event. There are six logging levels:
# 
# - CRITICAL
# - ERROR
# - WARNING
# - INFO
# - DEBUG
# - NOTSET

# # Globals 

# In[7]:


since=1499000000
interval=1 #  (1, 5, 15, 30, 60, 240, 1440, 10080, 21600)


# # Pairs

# In[43]:


def get_pairs_info():
    res={}

    resp = requests.get('https://api.kraken.com/0/public/AssetPairs')
    txt = resp.json(  )
    available_pairs=pd.DataFrame(txt['result']).T
    all_intervals=[1, 5, 15, 30, 60, 240, 1440, 10080, 21600]
    all_pairs=sorted(list(available_pairs.index))

    res["all_pairs_list"]=all_pairs
    res["all_intervals_list"]=all_intervals
    res["available_pairs_df"]=available_pairs

    return res


# In[44]:


def initialize_directory(all_pairs,all_intervals):
    import os
    for pair in all_pairs:
        print(pair)
        os.system('mkdir pairs/'+pair)
        for interval_ in all_intervals:
            os.system('mkdir pairs/'+pair+"/"+str(interval_))


# In[81]:


pairs_info=pairs_info.get_pairs_info()
all_pairs_list=[p for p in pairs_info["all_pairs_list"] if ".d" not in p]
all_intervals_list=pairs_info["all_intervals_list"]
available_pairs_df = pairs_info["available_pairs_df"]


# In[50]:


initialize_directory(all_pairs_list,all_intervals_list)


# In[77]:


available_pairs_df.T['ANTUSD']


# In[225]:


available_pairs_df.T['XXBTZEUR']


# # Bars 

# In[137]:


push_longdate(datetime.datetime.now(),unit='Y',to='floor'), push_longdate(datetime.datetime.now(),unit='Y',to='ceiling')


# In[139]:


push_longdate(datetime.datetime.now(),unit='M',to='floor'), push_longdate(datetime.datetime.now(),unit='M',to='ceiling')


# In[211]:


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
        
    logger.info( dates_.shape )

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
        


# In[136]:


intervals=[1,15,1440,21600]
pairs=['XTZEUR']
pairs=all_pairs


# In[210]:


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
                        logger.info(_filename_ + " file exist")
                    else:
                        logger.info(_filename_ + " file does not exist")
                        #check if need to ask for bars or we got this already

                        # try to make the time slice that was missing
                        # logger.info(pd_bars)

                        df_to_save=window_dataframe(pd_bars,_from,_to) 

                        if df_to_save is None:
                            logger.info("slice missing, fetching bars to fill "+str(_from)+"-->"+str(_to))
                            # get_bars   
                            pd_bars=get_bars(pair=_pair,since=since,interval=_interval)
                        # making the time slice that was missing
                        df_to_save=window_dataframe(pd_bars,_from,_to) 
                        # save result
                        df_to_save.to_csv(_filename_)
                time.sleep(0.1)    


# In[74]:


_dt=pd.to_datetime( datetime.datetime.now() - datetime.timedelta(days=0,hours=3) )
print(_dt,"--->",_dt.floor('7d'))


# # Schedule 

# In[91]:


import pairs_info
import monitor_pairs


# In[12]:


importlib.reload(monitor_pairs)


# In[94]:


_pairs_info=pairs_info.get_pairs_info()
all_pairs_list=_pairs_info["all_pairs_list"]
all_pairs_list =[p for p  in  all_pairs_list if ".d" not in p ]
all_intervals_list=_pairs_info["all_intervals_list"]
available_pairs_df = _pairs_info["available_pairs_df"]


# In[13]:


all_intervals_list


# In[16]:


monitor_pairs.pairs2csv(pairs=all_pairs_list,intervals=[1])


# In[17]:


scheduler.print_jobs()


# In[66]:


#from apscheduler.scheduler import Scheduler
from apscheduler.schedulers.background import BackgroundScheduler
import logging
logging.basicConfig()
logging.getLogger('apscheduler').setLevel(logging.DEBUG)
import requests

# Start the scheduler
#sched = Scheduler()
#sched.start()

def hourly_job():
        import monitor_pairs
        monitor_pairs.pairs2csv(pairs=all_pairs_list,intervals=[1]) #2m when no file exists, 30 sec if all files exists
def daily_job():
        import monitor_pairs
        monitor_pairs.pairs2csv(pairs=all_pairs_list,intervals=[15])
def weekly_job():
        import monitor_pairs
        monitor_pairs.pairs2csv(pairs=all_pairs_list,intervals=[1440]) #30 sec if all files exists
        monitor_pairs.pairs2csv(pairs=all_pairs_list,intervals=[21600])

        
        
# Schedules job_function to be run on the third Friday
# of June, July, August, November and December at 00:00, 01:00, 02:00 and 03:00
#sched.add_cron_job(job_function, month='6-8,11-12', day='3rd fri', hour='0-3')
#sched.add_cron_job(job_function, month='*', day='*', hour='21',minute='14')


scheduler = BackgroundScheduler()
scheduler.start()
scheduler.add_job(hourly_job, trigger='cron', minute=10,id='hourly_job')
scheduler.add_job(daily_job, trigger='cron', hour=4,id='daily_job')
scheduler.add_job(weekly_job, trigger='cron', day_of_week='2',id='weekly_job')
print('here is the schedule')
scheduler.print_jobs()

#input("Press CTLR+C to exit \n\n")


# In[16]:


scheduler.remove_job('hourly_job')
scheduler.remove_job('daily_job')
scheduler.remove_job('weekly_job')


# In[9]:


scheduler.shutdown()


# In[8]:


scheduler.remove_all_jobs()


# # Fetch Local Data 

# In[18]:


all_pairs_list


# In[20]:


def read_pair_history(pair,interval):
    import glob
    file_re = monitor_pairs.filename(pair,interval,"*","*") 
    paths = glob.glob(file_re)
    df=pd.concat([ pd.read_csv(path) for path in paths ])
    df['time']=pd.to_datetime(df['time'])
    df['pair']=pair
    return df


# In[166]:


def columns_with_nans(df):
    rows_with_nan = []
    for index, row in df.T.iterrows():
        is_nan_series = row.isnull()
        if is_nan_series.any():
            rows_with_nan.append(index)

    #print(rows_with_nan)
    return rows_with_nan


# ## Simple Read

# In[22]:


all_pairs_df=pd.concat([read_pair_history(_pair,'15')  for _pair in all_pairs_list ])


# In[23]:


all_pairs_df.info()


# In[45]:


all_pairs_df.loc[0,'pair']


# ## Organized Read

# In[367]:


print([ _pair for _pair in all_pairs_list if "EUR" in  _pair ])


# In[180]:



pairs_analysis_df={}


# In[364]:


pairs_analysis={}
quantity='open'
criterion=lambda _string_ :  (("EUR" in _string_) and any([ a in _string_ for a in                                                            #["ETH","XTZ","EOS","OXT","BTX","ADA","WAVES","QTUM"]\
                                                           ['ADAEUR', 'ALGOEUR', 'ANTEUR', 'ATOMEUR', 'BALEUR', 'BATEUR', 'BCHEUR', 'COMPEUR', 'CRVEUR', 'DAIEUR', 'DASHEUR', 'DOTEUR', 'EOSEUR', 'FILEUR', 'GNOEUR', 'ICXEUR', 'KAVAEUR', 'KEEPEUR', 'KNCEUR', 'KSMEUR', 'LINKEUR', 'LSKEUR', 'NANOEUR', 'OMGEUR', 'OXTEUR', 'PAXGEUR', 'QTUMEUR', 'REPV2EUR', 'SCEUR', 'SNXEUR', 'STORJEUR', 'TBTCEUR', 'TRXEUR', 'UNIEUR', 'WAVESEUR', 'XDGEUR', 'XETCZEUR', 'XETHZEUR', 'XLTCZEUR', 'XMLNZEUR', 'XREPZEUR', 'XTZEUR', 'XXBTZEUR', 'XXLMZEUR', 'XXMRZEUR', 'XXRPZEUR', 'XZECZEUR', 'YFIEUR']
                                                          ]))
    

pairs_dfs=[read_pair_history(_pair,'15') for _pair in all_pairs_list if criterion(_pair)]#_pair=='ANTEUR' ] #5 secs for 4 days of 15-min data 

pairs_dfs_timeindex = [df.set_index(['time']) for df in pairs_dfs ]

pairs_analysis[quantity]=[ pair_df[[quantity]].rename(columns={quantity: list(pair_df['pair'].unique())[0] }) for pair_df in pairs_dfs_timeindex  if pair_df.shape[0]>0 ]

pairs_analysis_df[quantity]=pd.concat( pairs_analysis[quantity] , axis=1).reset_index()


# In[365]:


pairs_analysis_df['open']


# In[366]:


data_  =  pairs_analysis_df['open'].drop(columns=columns_with_nans(pairs_analysis_df['open']))

corr = data_.corr()
corr.style.background_gradient(cmap='coolwarm',vmin=-1,vmax=1).set_precision(2)
# 'RdBu_r' & 'BrBG' are other good diverging colormaps


# In[368]:


available_pairs_df.T['CRVEUR']


# In[369]:


_u, _s, _vh = np.linalg.svd(corr)


# In[370]:


fig, ax = plt.subplots(nrows=1, ncols=3,figsize=(16, 3)) #plt.subplots(figsize=(13, 3), ncols=3)
im=ax[0].imshow(_u,cmap='coolwarm')
fig.colorbar(im, ax=ax[0])
im=ax[1].imshow(_vh,cmap='coolwarm')
fig.colorbar(im, ax=ax[1])
im=ax[2].imshow(_u-_vh.T,cmap='coolwarm')
fig.colorbar(im, ax=ax[2])
plt.show()


# In[371]:


np.argsort(_s)


# In[372]:


print(_s)


# In[373]:


def color_dic(col,color_dic={}):
    try:
        res=color_dic[col]
    except  KeyError:
        res='k'
    return res
        


# In[377]:


u_,s_,vh_=np.linalg.svd(corr)
for k_,p_ in enumerate(range(0+1*len(corr.columns))):
    plt.figure(figsize=(50,10))
    ###
    plt.plot( np.abs(u_[:,p_]) )
    ###
    plt.ylabel(str(k_)+":"+str(s_[k_]))
    names_=[ l+":"+str(i) for i,l in enumerate(corr.columns)]
    plt.xticks(range(len(corr.columns)),names_,rotation='vertical');
    my_colors = [ color_dic(col,{"XETHZEUR":'r',"XXBTZEUR":'g'}) for col in corr.columns  ]

    for ticklabel, tickcolor in zip(plt.gca().get_xticklabels(), my_colors):
        ticklabel.set_color(tickcolor)
    plt.ylim([-1,1])
    plt.axhline(0, linestyle='-', color='k') # horizontal lines
    for hl in [-0.2,0.2]:
        plt.axhline(hl, linestyle='--', color='k') # horizontal lines
    for hl in [-0.4,0.4]:
        plt.axhline(hl, linestyle='-.', color='k') # horizontal lines    
    #ax.axvline(x, linestyle='--', color='k') # vertical lines
    plt.show()


# In[296]:


plt.hist(np.linalg.svd(corr)[1],bins=50 )


# In[198]:


np.linalg.svd(corr)[2].shape


# ## Tests

# In[117]:


get_ipython().run_line_magic('timeit', "[ [all_pairs_list[i], pair_df.iloc[0]['pair'] ]for i,pair_df in enumerate(pairs_dfs) if pair_df.shape[0]>0 ]")


# In[123]:


get_ipython().run_line_magic('timeit', "[ [all_pairs_list[i],list(pair_df['pair'].unique())[0] ]for i,pair_df in enumerate(pairs_dfs_timeindex) if pair_df.shape[0]>0 ]")


# In[156]:


pair1='WAVESEUR'
pair2='OXTEUR'
pair3='KAVAEUR'


# In[157]:


a_df=double_df.query('pair==\''+pair1+'\'')[['time','close']]
b_df=double_df.query('pair==\''+pair2+'\'')[['time','close']]
c_df=double_df.query('pair==\''+pair3+'\'')[['time','close']]


# In[158]:


merged=a_df.merge(c_df,on='time',suffixes=[pair1,pair3])


# In[159]:


merged


# In[160]:


merged.corr()


# # Time Series Plot

# In[102]:


from pandas.plotting import lag_plot
from pandas.plotting import autocorrelation_plot


# - first diff autocrrelation
# - second diff autocorrelation

# In[112]:


data=df['close']
plt.figure()
lag_plot(data)
plt.show()
plt.figure()
autocorrelation_plot(data)


# # Plots 

# In[11]:


np.arange(1,1.2,0.01)


# In[12]:


central = np.mean( pd_bars[15]['high'] + pd_bars[15]['low'])/2
delta = np.std( pd_bars[15]['high'] + pd_bars[15]['low'])/2/central
span=np.arange(1-delta,1+delta,0.006)
_bins=central*span


# In[13]:


fig, ax = plt.subplots()
plt.hist(pd_bars[15]['high'],bins=_bins );
plt.xticks(rotation=-90)
plt.xticks(_bins, map(lambda x: '{:.3f}'.format(x) ,span) )
secax = ax.secondary_xaxis('top')
secax.set_xlabel('EUR\n')
plt.show()


# In[23]:


SpectralN=Spectral4

#_data = _data[_data['time']>  pd.to_datetime( datetime.datetime.now() - datetime.timedelta(days=4) )  ]

xdata='time'
ylabel='EUR'
base_interval=1

# Making the main plot
p = figure(plot_height=600, plot_width=800, tools="xpan,reset,save,hover,box_zoom", toolbar_location='left',
           x_axis_type="datetime", x_axis_location="above",
           background_fill_color="#efefef", x_range=(pd_bars[base_interval][xdata].min(), pd_bars[base_interval][xdata].max()), y_range=(pd_bars[base_interval]['low'].min(), pd_bars[base_interval]['high'].max()))
p.yaxis.axis_label = ylabel



for i,_interval in enumerate(intervals):
    print(_interval)
    p.line(x=xdata, y='high', source=pd_bars[_interval], line_color=SpectralN[i],legend=str(_interval)) #legend_label=_source)
    p.line(x=xdata, y='low', source=pd_bars[_interval], line_color=SpectralN[i],legend=str(_interval)) #legend_label=_source)


# Making the selection tool
select = figure(title="Drag the middle and edges of the selection box to change the range above",
                plot_height=200, plot_width=800, y_range=p.y_range,
                x_axis_type="datetime", y_axis_type=None,
                tools="", toolbar_location=None, background_fill_color="#efefef")    


range_tool = RangeTool(x_range=p.x_range)
range_tool.overlay.fill_color = "navy"
range_tool.overlay.fill_alpha = 0.2

select.line(xdata, 'high', source=pd_bars[base_interval],color=SpectralN[i])#,legend_label=_source) #what to plot in the selection tool    

    
####
select.ygrid.grid_line_color = None
select.add_tools(range_tool)
select.toolbar.active_multi = range_tool
#select.legend.location = "top_left"
#select.legend.click_policy="hide"

p.legend.location = "top_left"
p.legend.click_policy="hide"



show(column(p, select))

del xdata,ylabel


# In[97]:


286.18*(1+2*0.0015)


# In[96]:


0.09/(286.18*1.01*0.2)


# # Do the plots

# In[8]:


fmts=['--','-','-.']
ymin=10
ymax=35
plt.figure(3)
plt.figure(figsize=(30, 30))
#plt.subplots(figsize=(9, 6))
for i,h in enumerate(timeSeries['sensor'].unique()):
    plt.subplot(3,3,1+i)
    data=timeSeries[timeSeries['sensor']==h].sort_values('time')
    data = data[data['time']>  pd.to_datetime( datetime.datetime.now() - datetime.timedelta(days=2) )  ]
    plt.plot_date(x='time',y='temperature',data=data,fmt=fmts[i%len(fmts)],label=h)
    plt.ylim([ymin,ymax])
    plt.title=h
    plt.xticks(rotation=-80)
    #print(data)
    #plt.yscale('log')
    plt.legend()#bbox_to_anchor=[1,1])
    _data=timeSeries[['time','temperature','sensor']]
    _data.columns=['Date','Temperature','sensor']


# In[ ]:





# In[ ]:





# In[ ]:




