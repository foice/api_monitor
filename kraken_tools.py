#!/usr/bin/env python3

# This file is part of krakenex.
# Licensed under the Simplified BSD license. See `examples/LICENSE.txt`.

# Pretty-print a pair's order book depth.

from requests.exceptions import HTTPError
import decimal
import time
import pprint

import krakenex
import pandas as pd



kraken = krakenex.API()


def get_book(pair='',counts=10):

    try:
        response = kraken.query_public('Depth', {'pair': pair, 'count': counts})
        pprint.pprint(response)
    except HTTPError as e:
        print(str(e))


def now():
    return decimal.Decimal(time.time())

def lineprint(msg, targetlen = 72):
    line = '-'*5 + ' '
    line += str(msg)

    l = len(line)
    if l < targetlen:
        trail = ' ' + '-'*(targetlen-l-1)
        line += trail

    print(line)
    return


def get_history(pair=None, since=None, interval=None):

    try:
        ret = kraken.query_public('OHLC' ,data = {'pair': pair, 'since': since,'interval': interval} )
        try:
            bars = ret['result'][pair]
            pd_bars=pd.DataFrame(bars,columns=['time', 'open', 'high', 'low', 'close', 'vwap', 'volume', 'count'])
            pd_bars['time']=pd.to_datetime(pd_bars['time'],unit='s')
            return pd_bars
        except KeyError as k:
            print(k, "is missing")
            print(ret)
            print(k, "is missing")

    except HTTPError as e:
        print(str(e))
