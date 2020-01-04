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


def get_fileidxs(files_asset, init_date, end_date):
    """ get file idxs in files_asset list within init_date and end_date """
    init_date_dt = dt.datetime.strptime(init_date,'%y%m%d')
    end_date_dt = dt.datetime.strptime(end_date,'%y%m%d')
    init_idx = []
    while len(init_idx)<1 and init_date_dt<end_date_dt:
        init_idx = [f for f,file in enumerate(files_asset) if init_date in file[7:13]]
        init_date_dt = init_date_dt+dt.timedelta(days=1)
        init_date = dt.datetime.strftime(init_date_dt,'%y%m%d')
    end_idx = []
    while len(end_idx)<1 and init_date_dt<end_date_dt:
        end_idx = [f for f,file in enumerate(files_asset) if end_date in file[7:13]]
        end_date_dt = end_date_dt-dt.timedelta(days=1)
        end_date = dt.datetime.strftime(end_date_dt,'%y%m%d')
    #print(init_idx)
    #print(end_idx)
    assert(len(init_idx)==1)
    assert(len(end_idx)==1)
    return init_idx[0], end_idx[0]


 # connect to MetaTrader 5
mt5.MT5Initialize()
# wait till MetaTrader 5 establishes connection to the trade server and synchronizes the environment
mt5.MT5WaitForTerminal()
utc_tz = timezone('Etc/GMT-3')#

first_day = '20140102020000'
last_day = '2019.12.12'

#directory_format_destiny = 'D:/SDC/py/Data_PYF_G3F2/'
#if not os.path.exists(directory_format_destiny):
#    os.mkdir(directory_format_destiny)

#delta_dates = end_day-init_day
#dateTestDt = [init_day + dt.timedelta(i) for i in range(delta_dates.days + 1)]
#dateTest = [dt.date.strftime(d,'%Y.%m.%d') for d in dateTestDt]
samps = 1000
#assets = [1,2,3,4,7,8,10,11,12,13,14,16,17,19,27,28,29,30,31,32]
assets = [i for i in range(70,78)]
#assets = [2]
# request ticks from AUDUSD within 2019.04.01 13:00 - 2019.04.02 13:00 
for ass in assets:
    init_day = dt.datetime.strptime(first_day,'%Y%m%d%H%M%S')#.astimezone(utc_tz)#.date()
    end_day = dt.datetime.strptime(last_day,'%Y.%m.%d')#.astimezone(utc_tz)#.date()
    day_index = 0
    # init current time
    counter_days = 0
    ticks = mt5.MT5CopyTicksFrom(C.AllAssets[str(ass)], init_day, 1, mt5.MT5_COPY_TICKS_ALL)
    # gauge real init day time
    init_day = init_day.replace(day=ticks[0].time.day)#.astimezone(utc_tz)
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
        new_time = ticks[0].time#.astimezone(utc_tz)
        current_time = ticks[0].time#.astimezone(utc_tz)
        
        # init trade info
        trade_info = pd.DataFrame(columns=['DateTime','SymbolBid','SymbolAsk'])
        counter = 0
        # init file name
        dt_name = dt.datetime.strftime(current_time,'%Y%m%d%H%M%S')
        filename = C.AllAssets[str(ass)]+'_'+dt_name
        dirfilename = ass_dir+filename+'.txt'
        print(filename)
        init_date = dt_name[:8]
        files_asset = sorted(os.listdir(ass_dir))
        fileindir = [f for f,file in enumerate(files_asset) if init_date in file[7:15]]
        #print(fileindir)
        if len(fileindir)==0:#not os.path.exists(dirfilename):
            # loop while day remains the same
            while current_time.day == new_time.day:
                # get ticks
                samps = 100000
                prev_ticks = ticks
                ticks = mt5.MT5CopyTicksFrom(C.AllAssets[str(ass)], new_time, samps, mt5.MT5_COPY_TICKS_ALL)
                while ticks[-1].time <= prev_ticks[-1].time:#.astimezone(utc_tz)
                    samps += 10000
                    print(samps)
                    #time.sleep(.01)
                    ticks = mt5.MT5CopyTicksFrom(C.AllAssets[str(ass)], new_time, samps, mt5.MT5_COPY_TICKS_ALL)
                    #a=p
                # make sure new ticks are different
                if ticks[-1].time > prev_ticks[-1].time:#.astimezone(utc_tz)
                    #print(counter)
                    #print(ticks[0].time.astimezone(utc_tz))
                    #print(ticks[-1].time.astimezone(utc_tz))
                    assert(ticks[-1].time!=new_time)#.astimezone(utc_tz)
                    init_idx = 0
#                    while ticks[init_idx].time.astimezone(utc_tz) <= prev_ticks[-1].time.astimezone(utc_tz):
#                        init_idx+=1
                    new_time = ticks[-1].time#.astimezone(utc_tz)
                    # find last entry of current day
                    
    #                while ticks[diff_day_idx].time.astimezone(utc_tz).day != current_time.day:
    #                    diff_day_idx-=1
    #                # update current day
    #                if diff_day_idx<len(ticks)-1:
    #                    current_time = ticks[diff_day_idx+1].time.astimezone(utc_tz)
                    # TODO: find entries already in trade info
                    
                    # get DT, B and A
                    DateTimes = [x.time for x in ticks[init_idx:]]#[dt.datetime.strftime(x.time, '%Y.%m.%d %H:%M:%S') for x in ticks[init_idx:]]
                    bids = [int(np.round(x.bid*100000))/100000 for x in ticks[init_idx:]]
                    asks = [int(np.round(x.ask*100000))/100000 for x in ticks[init_idx:]]
                    # update trade info dataframe
                    trade_info = trade_info.append(pd.DataFrame({'DateTime':DateTimes,'SymbolBid':bids,
                                                        'SymbolAsk':asks}), ignore_index=True)
                    counter += 1
            # save trade info
            diff_day_idx = 1
            while ticks[-diff_day_idx].time.day != current_time.day:#.astimezone(utc_tz)
                diff_day_idx += 1
            trade_info = trade_info.iloc[:-diff_day_idx+1]
            
            # check for discontinuities
            check_disc = True
            error = False
            while check_disc:
                ser1=pd.Series(trade_info.iloc[1:]['DateTime'])
                ser1.index=range(0,trade_info.shape[0]-1)
                ser2=pd.Series(trade_info.iloc[:-1]['DateTime'])
                diffs_bool = ((ser1-ser2)<dt.timedelta(seconds=0))
                if diffs_bool.sum()>0:
                    print("\tDiscontinuity found")
                    idx_out = diffs_bool.idxmax()
                    date=trade_info.iloc[idx_out].DateTime
                    idx_last = idx_out-1
                    #print(info.iloc[idx_last])
                    idx_resume = ((ser1.iloc[idx_out:]-date)>dt.timedelta(seconds=0)).idxmax()
                    #print(info.iloc[idx_resume:])
                    trade_info = trade_info.iloc[:idx_out].append(trade_info.iloc[idx_resume:])
                    #print(info)
                    trade_info.index = range(trade_info.shape[0])
                    if idx_out==idx_resume:
                        error = True
                        check_disc = False
                else:
                    check_disc = False
            
            # save info
            if not error:
                # Bring back 
                first_datetime = trade_info['DateTime'].iloc[0].strftime('%Y%m%d%H%M%S')
                # update filename
                if first_datetime!=dt_name:
                    print("WARNING! "+first_datetime+" not in filename ")
                    filename = C.AllAssets[str(ass)]+'_'+first_datetime
                    dirfilename = ass_dir+filename+'.txt'
                trade_info['DateTime'] = trade_info['DateTime'].dt.strftime('%Y.%m.%d %H:%M:%S')
                if not os.path.exists(dirfilename):
                    trade_info.to_csv(dirfilename,sep=',',index=False)
                else:
                    print("WARNING! File already exists. Not saved")
            # format info's week
            # 1) Check if weekend day
            if current_time.weekday()==5:
                # 2) figure out week offset
                print("ERROR! current_time.weekday()==5")
#                raise ValueError("current_time.weekday()==5")
                print(dt.datetime.strftime(current_time,'%Y.%m.%d %H:%M:%S'))
#                offset = current_time-dt.datetime(current_time.year, current_time.month, current_time.day)+dt.timedelta(hours=1)
#                print(offset.hours)
            elif current_time.weekday()==6:
                print("ERROR! current_time.weekday()==6")
#                raise ValueError("current_time.weekday()==6")
                print(dt.datetime.strftime(current_time,'%Y.%m.%d %H:%M:%S'))
#                offset = current_time-dt.datetime(current_time.year, current_time.month, current_time.day)+dt.timedelta(hours=1)
#                print(offset.hours)
            
            
            
                
        counter_days += 1
        current_time = init_day+dt.timedelta(days=counter_days)
        #a=p
print("ALL DONE")
mt5.MT5Shutdown()