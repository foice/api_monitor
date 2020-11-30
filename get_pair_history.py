#!/usr/bin/env python3

# This file is part of krakenex.
# Licensed under the Simplified BSD license. See `examples/LICENSE.txt`.

# Pretty-print a pair's order book depth.

from requests.exceptions import HTTPError
import decimal
import time
import pprint
import argparse

import krakenex
import pandas as pd
import kraken_tools as kt

parser = argparse.ArgumentParser(description='Reads the API of Kraken.com', usage='')


parser.add_argument('--pair',
                    default='',
                    help='pair to lookup. Names from https://api.kraken.com/0/public/AssetPairs (default: \'\'')

parser.add_argument('--since',
                    default='1499000000',
                    help='Unix time or YY-MM-DD-HH-MM (default: 1499000000, that is UTC 2017-07-02 12:53:20')

parser.add_argument('--interval',
                    default='60',
                    help='granularity of the data in minutes (1, 5, 15, 30, 60, 240, 1440, 10080, 21600)')

parser.add_argument('--csv',
                    default=False,
                    help='File name where to save CSV data in the name format %par_UTCstartdate-enddate_%granularity.csv')



args = parser.parse_args()
pair=args.pair
since = args.since # str(1499000000) # UTC 2017-07-02 12:53:20
interval = args.interval # str(1499000000) # UTC 2017-07-02 12:53:20
csv = args.csv

kraken = krakenex.API()


kt.get_book(pair=pair,counts=10)

pd_bars=kt.get_history(pair=pair,since=since,interval=interval)

print('***************OHLC data***************')
print(pd_bars)
date=str(kt.now())
csv_file=pair+"_"+date+"_"+str(interval)+".csv" 
pd_bars.to_csv(csv_file)

while False:
    lineprint(now())

    before = now()
    ret = kraken.query_public('OHLC' ,data = {'pair': pair, 'since': since,'interval': interval} )
    after = now()

    # comment out to track the same "since"
    #since = ret['result']['last']

    # TODO: don't repeat-print if list too short
    bars = ret['result'][pair]
    pd_bars=pd.DataFrame(bars,columns=['time', 'open', 'high', 'low', 'close', 'vwap', 'volume', 'count'])
    pd_bars['time']=pd.to_datetime(pd_bars['time'],unit='s')
    for b in bars[:5]:
        print(b)
    print('...')
    for b in bars[-5:]:
        print(b)
    print(pd_bars)
    lineprint(after - before)

    time.sleep(20)
