# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 09:34:58 2019

@author: magut
"""

import datetime as dt
import time
import os
import numpy as np
import pandas as pd
import MetaTrader5 as mt5
from pytz import timezone
import matplotlib.pyplot as plt
from kaissandra.config import Config as C
from kaissandra.local_config import local_vars as LC

 # connect to MetaTrader 5
mt5.MT5Initialize()
# wait till MetaTrader 5 establishes connection to the trade server and synchronizes the environment
mt5.MT5WaitForTerminal()
utc_tz = timezone('Etc/GMT-3')#

first_day = '2013.01.01'
last_day = '2019.12.31'

#delta_dates = end_day-init_day
#dateTestDt = [init_day + dt.timedelta(i) for i in range(delta_dates.days + 1)]
#dateTest = [dt.date.strftime(d,'%Y.%m.%d') for d in dateTestDt]
samps = 110000
assets = [1,2,3,4,7,8,10,11,12,13,14,15,16,17,19,27,28,29,30,31,32]
# request ticks from AUDUSD within 2019.04.01 13:00 - 2019.04.02 13:00 
for ass in assets:
    init_day = dt.datetime.strptime(first_day,'%Y.%m.%d').astimezone(utc_tz)#.date()
    end_day = dt.datetime.strptime(last_day,'%Y.%m.%d').astimezone(utc_tz)#.date()
    day_index = 0
    # init current time
    counter_days = 0
    ticks = mt5.MT5CopyTicksFrom(C.AllAssets[str(ass)], init_day, 1, mt5.MT5_COPY_TICKS_ALL)
    # gauge real init day time
    init_day = init_day.replace(day=ticks[0].time.astimezone(utc_tz).day)
    current_time = init_day
    # create dir
    ass_dir = LC.data_dir_py+C.AllAssets[str(ass)]+'/'
    if not os.path.exists(ass_dir):
        os.makedirs(ass_dir)
    print(C.AllAssets[str(ass)])
    # loop over days
    while current_time<=end_day:
        # get first tick
        ticks = mt5.MT5CopyTicksFrom(C.AllAssets[str(ass)], current_time, 1, mt5.MT5_COPY_TICKS_ALL)
        # gauge current time
        new_time = ticks[0].time.astimezone(utc_tz)
        current_time = ticks[0].time.astimezone(utc_tz)
        
        # init trade info
        trade_info = pd.DataFrame(columns=['DateTime','SymbolBid','SymbolAsk'])
        counter = 0
        # init file name
        dt_name = dt.datetime.strftime(current_time,'%Y%m%d%H%M%S')
        filename = C.AllAssets[str(ass)]+'_'+dt_name
        dirfilename = ass_dir+filename+'.txt'
        print(filename)
        if not os.path.exists(dirfilename):
            # loop while day remains the same
            while current_time.day == new_time.day:
                # get ticks
                prev_ticks = ticks
                ticks = mt5.MT5CopyTicksFrom(C.AllAssets[str(ass)], new_time, samps, mt5.MT5_COPY_TICKS_ALL)
                # make sure new ticks are different
                print(counter)
                print(ticks[0].time.astimezone(utc_tz))
                print(ticks[-1].time.astimezone(utc_tz))
                assert(ticks[-1].time.astimezone(utc_tz)!=new_time)
                init_idx = 0
                while ticks[init_idx].time.astimezone(utc_tz) <= prev_ticks[-1].time.astimezone(utc_tz):
                    init_idx+=1
                new_time = ticks[-1].time.astimezone(utc_tz)
                # find last entry of current day
                
#                while ticks[diff_day_idx].time.astimezone(utc_tz).day != current_time.day:
#                    diff_day_idx-=1
#                # update current day
#                if diff_day_idx<len(ticks)-1:
#                    current_time = ticks[diff_day_idx+1].time.astimezone(utc_tz)
                # TODO: find entries already in trade info
                
                # get DT, B and A
                DateTimes = [dt.datetime.strftime(x.time.astimezone(utc_tz), '%y.%m.%d %H:%M:%S') for x in ticks[init_idx:]]
                bids = [int(np.round(x.bid*100000))/100000 for x in ticks[init_idx:]]
                asks = [int(np.round(x.ask*100000))/100000 for x in ticks[init_idx:]]
                # update trade info dataframe
                trade_info = trade_info.append(pd.DataFrame({'DateTime':DateTimes,'SymbolBid':bids,
                                                    'SymbolAsk':asks}), ignore_index=True)
                counter += 1
            # save trade info
            diff_day_idx = 1
            while ticks[-diff_day_idx].time.astimezone(utc_tz).day != current_time.day:
                diff_day_idx += 1
            trade_info = trade_info.iloc[:-diff_day_idx+1]
            trade_info.to_csv(dirfilename,sep=',',index=False)
            
        counter_days += 1
        current_time = init_day+dt.timedelta(days=counter_days)
        #a=p
        
    
    
    #ticks = mt5.MT5CopyTicksRange(C.AllAssets[str(ass)], dt.datetime(2019,2,4,2), dt.datetime(2019,2,8,2), mt5.MT5_COPY_TICKS_ALL)
     
    # shut down connection to MetaTrader 5
    
    #PLOTTING
    x_time = [x.time.astimezone(utc_tz) for x in ticks]
    # prepare Bid and Ask arrays
    bid = [y.bid for y in ticks]
    ask = [y.ask for y in ticks]
     
    # draw ticks on the chart
    plt.plot(x_time, ask,'r-', label='ask')
    plt.plot(x_time, bid,'g-', label='bid')
    # display legends 
    plt.legend(loc='upper left')
    # display header 
    plt.title(C.AllAssets[str(ass)]+' ticks')
    # display the chart
    plt.show()
    time.sleep(1)
mt5.MT5Shutdown()