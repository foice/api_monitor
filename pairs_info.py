#!/usr/bin/env python3

# This file is part of krakenex.
# Licensed under the Simplified BSD license. See `examples/LICENSE.txt`.

# Pretty-print a pair's order book depth.


import requests
import pandas as pd
import  os
import datetime
from requests.exceptions import HTTPError

import utils

double_logger_pairs_info=utils.create_double_logger(logger_name='pairs_info.py',log_file='pairs_info.py.log')



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

def initialize_directory(all_pairs,all_intervals):
    import os
    for pair in all_pairs:
        print(pair)
        os.system('mkdir pairs/'+pair)
        for interval_ in all_intervals:
            os.system('mkdir pairs/'+pair+"/"+str(interval_))
