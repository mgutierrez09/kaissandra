# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 13:52:57 2018

@author: mgutierrez
Script to simulate trader

Advice from Brian Chesky: Get 100 people to love your product! Originaly from Paul Graham :)
Word of mouth
"""

import sys
import os
import time
#import pandas as pd
import datetime as dt
import numpy as np
import pickle
import math

if __name__ == '__main__':
    
    this_path = os.getcwd()
    path = '\\'.join(this_path.split('\\')[:-1])+'\\'
    if path not in sys.path:
        sys.path.insert(0, path)
        #sys.path.append(path)
        print(path+" added to python path")
        
    else:
        print(path+" already added to python path")
        
from kaissandra.local_config import local_vars
from kaissandra.config import Config as C
from kaissandra.updateRaw import load_in_memory
from kaissandra.results2 import load_spread_ranges

def round_num(number, prec):
    return round(prec*number)/prec

def build_spread_ranges_per_asset(baseName, extentionNameResults, extentionNameSpreads, 
                                  assets, thr_NSP, mar=(0.0,0.0), anchor=[]):
    """ Build spread range per asset """
    spread_ranges = []
    IDresults = []
    min_p_mcs = []
    min_p_mds = []
    for ass_id, ass_idx in enumerate(assets):
        thisAsset = C.AllAssets[str(ass_idx)]
#        print(thisAsset)
        IDspreads = baseName+thisAsset+extentionNameSpreads
        IDresult = baseName+thisAsset+extentionNameResults
        spreads_range, mim_pmc, mim_pmd = load_spread_ranges(local_vars.results_directory, IDspreads, value=thr_NSP)
        
        # add margin
        for idx in range(len(spreads_range['mar'])):
            spreads_range['mar'][idx] = mar
            # average with anchor spreads i given
            if len(anchor)>0:
#                print(idx)
#                print(spreads_range['th'][idx])
#                print(anchor[ass_id]['th'][idx])
                spreads_range['th'][idx] = (round_num((spreads_range['th'][idx][0]+anchor[ass_id]['th'][idx][0])/2, 1000), 
                                            round_num((spreads_range['th'][idx][1]+anchor[ass_id]['th'][idx][1])/2, 1000))
#                print(spreads_range['th'][idx])
        spread_ranges.append(spreads_range)
        IDresults.append(IDresult)
        min_p_mcs.append(mim_pmc)
        min_p_mds.append(mim_pmd)
            
    return spread_ranges, IDresults, min_p_mcs, min_p_mds

def delete_leftovers(List, null_entry=0):
    """  """
    try:
        first_lo = List.index(null_entry)
        return List[:first_lo]
    except ValueError:
        return List
    

if __name__ == '__main__':
        
    start_time = dt.datetime.strftime(dt.datetime.now(),'%y%m%d%H%M%S')
    numberNetwors = 2
    init_day_str = '20181112'
    end_day_str = '20200424'#'20200424'
    assets = [7]#[1,2,3,4,7,8,10,11,12,13,14,16,17,19,27,28,29,30,31,32]
    
    root_dir = local_vars.data_test_dir
    
    init_day = dt.datetime.strptime(init_day_str,'%Y%m%d').date()
    end_day = dt.datetime.strptime(end_day_str,'%Y%m%d').date()
    
    positions_file = start_time+'_F'+init_day_str+'T'+end_day_str+'.csv'
#    end_day = dt.datetime.strptime('2019.04.26','%Y.%m.%d').date()
#    delta_dates = dt.datetime.strptime('2018.11.09','%Y.%m.%d').date()-edges[-2]
#    dateTestDt = [edges[-2] + dt.timedelta(i) for i in range(delta_dates.days + 1)]
    delta_dates = end_day-init_day
    dateTestDt = [init_day + dt.timedelta(i) for i in range(delta_dates.days + 1)]
#    dateTestDt = [edges[-2] + dt.timedelta(i) for i in range(delta_dates.days + 1)]
    dateTest = []
    for d in dateTestDt:
        if d.weekday()<5:
            dateTest.append(dt.date.strftime(d,'%Y.%m.%d'))
    #dateTest = ['2018.11.15','2018.11.16']
    load_from_live = False
    # data structure
#    data=Data(movingWindow=100,nEventsPerStat=1000,
#     dateTest = dateTest)
    
#    first_day = '2018.11.12'
#    last_day = '2020.01.10'#'2019.08.22'
#    init_day = dt.datetime.strptime(first_day,'%Y.%m.%d').date()
#    end_day = dt.datetime.strptime(last_day,'%Y.%m.%d').date()
    delta_dates = end_day-init_day
    dateTestDt = [init_day + dt.timedelta(i) for i in range(delta_dates.days + 1)]
    dateTest = []
    for d in dateTestDt:
        if d.weekday()<5:
            dateTest.append(dt.date.strftime(d,'%Y.%m.%d'))
                    
    tic = time.time()
    # init positions vector
    # build asset to index mapping
    ass2index_mapping = {}
    ass_index = 0
    for ass in assets:
        ass2index_mapping[C.AllAssets[str(ass)]] = ass_index
        ass_index += 1
    
    
    directory = local_vars.live_results_dict+"simulate/trader/"
    log_file = directory+start_time+"trader_v30.log"
    summary_file = directory+start_time+"summary.log"
    
    day_index = 0
    t_journal_entries = 0
    change_dir = 0
    rewinded = 0
    week_counter = 0
    
    mean_volat_db = -31.031596804415166
    var_volat_db = 2.776629833274796
    mean_vol_db = 25.131909769520508
    var_vol_db = 5.52461170016939
    print(root_dir)
    
    null_entry = None
    av_vol = [null_entry for _ in range(10000)]
    av_volat = [null_entry for _ in range(10000)]
    
    time_stamps = [null_entry for _ in range(10000)]
    num_assets_vol = [null_entry for _ in range(10000)]
    num_assets_volat = [null_entry for _ in range(10000)]
    idx_stats = 0
    idx_stats_ass = [0 for _ in assets]
    w1 = 1-1/1
    w10 = 1-1/10
    w20 = 1-1/20
    w100 = 1-1/100
    w1000 = 1-1/1000
    w10000 = 1-1/10000
    ws = [w1]
    time_stamps_ass = [[null_entry for _ in range(10000)] for _ in assets]
    volat_per_ass = [[[0 for _ in ws] for _ in range(10000)] for _ in assets]
    
    max_vol_per_pos_ass = {}
    dt_max_vol_per_pos_ass = {}
#    max_diff_per_pos_ass = {}
#    dt_max_diff_per_pos_ass = {}
    max_vol_per_hour = {}
    mW = 50000
    track_last_dts = [[dt.datetime.strptime(init_day_str,'%Y%m%d') for _ in range(mW)] for i in assets]
    track_last_asks = [[0 for _ in range(mW)] for i in assets]
    track_idx = [0 for i in assets]
    
    total_tic = time.time()
    # loop over days 
    while day_index<len(dateTest):
        counter_back = 0
        init_list_index = day_index
        i = day_index
        # look for first non-continuous day entry
#        while day_index<len(dateTest)-1 and dt.datetime.strptime(dateTest[i+1],'%Y.%m.%d')-dt.datetime.strptime(dateTest[i],'%Y.%m.%d')==dt.timedelta(1):
#            i +=1
#            counter_back += 1
        counter_back += 1
        end_list_index = min(i+4, len(dateTest)-1)
        
        day_index += counter_back+4
        
        out = ("Week counter "+str(week_counter)+". From "+dateTest[init_list_index]+
               " to "+dateTest[end_list_index])
        week_counter += 1
        print(out)
        
        DateTimes, SymbolBids, SymbolAsks, Assets, nEvents = load_in_memory(assets, 
                                                                            C.AllAssets,
                                                                            dateTest, 
                                                                            init_list_index, 
                                                                            end_list_index,
                                                                            root_dir=root_dir)
    
        # set counters and flags
        event_idx = 0
        
        max_vols = [99999999 for _ in assets]# inf
        emas_volat = [[-1 for _ in ws] for _ in assets]
        means_volat = [0 for _ in ws]
        events_per_ass_counter = [-1 for _ in assets]
        margin = 0.0
#        last_dt_per_ass = [False for _ in assets]
        this_hour = None
        this_min = None
        this_sec = None
        this_hour_ass = [None for _ in assets]
        this_min_ass = [None for _ in assets]
        this_sec_ass = [None for _ in assets]
        every = "min"# "min", "hour", "sec"
        inter_tic = time.time()
        expected_ask = 0.8754 # EURGBP
        # get to 
        while event_idx<nEvents:
            rewind = 0
            no_budget = False
            # get time stamp
            DateTime = DateTimes[event_idx].decode("utf-8")
            time_stamp = dt.datetime.strptime(DateTime,
                                              '%Y.%m.%d %H:%M:%S')
            #print("\r"+DateTime, sep=' ', end='', flush=True)
            thisAsset = Assets[event_idx].decode("utf-8")
            if not math.isnan(SymbolBids[event_idx]) and not math.isnan(SymbolAsks[event_idx]):
                bid = int(np.round(SymbolBids[event_idx]*100000))/100000
                ask = int(np.round(SymbolAsks[event_idx]*100000))/100000
            else:
                # TODO: find previous entry and substitude
                out = " WARNING! NaN found. Skipping"
                print(out)
            e_spread = (ask-bid)/ask
            
            ass_idx = ass2index_mapping[thisAsset]
            # track timestamp for the last mW samps of each asset to 
            # calculate volume
            ass_id = ass_idx#assets.index(ass_idx)
            track_last_asks[ass_id][track_idx[ass_id]] = ask
            
            # init last dt of asset
            if (every=="hour" and this_hour_ass[ass_id]==None) or (every=="min" and this_min_ass[ass_id]==None) or \
                (every=="sec" and this_sec_ass[ass_id]==None):
                print("\r"+DateTime, sep=' ', end='', flush=True)
                this_hour_ass[ass_id] = time_stamp.hour
                this_min_ass[ass_id] = time_stamp.minute
                this_sec_ass[ass_id] = time_stamp.second
            if events_per_ass_counter[ass_id]>mW:
                #vol = (track_last_dts[ass_id][track_idx[ass_id]]-time_stamp).seconds
                window_asks = np.array(track_last_asks[ass_id])
                window_asks = window_asks[window_asks>0]
                volat = np.max(window_asks)/np.min(window_asks)-1#np.mean(window_asks)
#                if vol<=max_vols[ass_id]:
                # init volatility tracking
#                if max_vols[ass_id] == 99999999:
#                    max_vols[ass_id] = vol
#                # update volume tracking
#                max_vols[ass_id] = w20*max_vols[ass_id]+(1-w20)*vol
                # update max volume
#                if volat>max_volats[ass_id]:
                for i in range(len(emas_volat[ass_id])):
                    if emas_volat[ass_id][i] == -1:
                        emas_volat[ass_id][i] = volat
                    # update volatility tracking
                    emas_volat[ass_id][i] = ws[i]*emas_volat[ass_id][i]+(1-ws[i])*volat
                
                #array_vol = np.array(max_vols)[np.array(max_vols)!=99999999]
#                arrays_volat = [np.array(ema_volat)[np.array(ema_volat)!=-1] for ema_volat in emas_volat]
#                mean_vol = np.mean(array_vol)
#                means_volat = [np.mean(arr) for arr in arrays_volat]
                
                # update new hour
                if (every=="hour" and this_hour_ass[ass_id] != time_stamp.hour) or \
                   (every=="min" and this_min_ass[ass_id] != time_stamp.minute) or \
                   (every=="sec" and this_sec_ass[ass_id] != time_stamp.second):
                    volatility_idxs = [100*ema for ema in emas_volat[ass_id]]#(10*np.log10(ema)-mean_volat_db)/var_volat_db
                    #volume_idx = (10*np.log10(mean_vol)-mean_vol_db)/var_vol_dbe.time()-total_tic)/60), \
#                    print("\r"+DateTime+" "+thisAsset+" idx "+str(idx_stats)+\
#                          ": VI1 {0:.4f}".format(volatility_idxs[0])+\
#                          " Time {0:.2f} mins. Total time {1:.2f} mins. "\
#                          .format((time.time()-inter_tic)/60,(time.time()-total_tic)/60), sep=' ', end='', flush=True)
                    this_hour_ass[ass_id] = time_stamp.hour
                    this_min_ass[ass_id] = time_stamp.minute
                    this_sec_ass[ass_id] = time_stamp.second
                    
#                    if idx_stats >= len(time_stamps):
#                        av_vol = av_vol+[null_entry for _ in range(10000)]
#                        av_volat = av_volat+[null_entry for _ in range(10000)]
#                        num_assets_vol = num_assets_vol+[null_entry for _ in range(10000)]
#                        num_assets_volat = num_assets_volat+[null_entry for _ in range(10000)]
#                        time_stamps = time_stamps+[null_entry for _ in range(10000)]
#                    av_vol[idx_stats] = mean_vol#.append(mean_vol)
#                    av_volat[idx_stats] = means_volat#.append(mean_volat)
                    
                    if idx_stats_ass[ass_id] >= len(time_stamps_ass[ass_id]):
                        volat_per_ass[ass_id] = volat_per_ass[ass_id]+[[0 for _ in ws] for _ in range(10000)]
                        time_stamps_ass[ass_id] = time_stamps_ass[ass_id]+[null_entry for _ in range(10000)]
                    for i in range(len(emas_volat[ass_id])):
                        volat_per_ass[ass_id][idx_stats_ass[ass_id]][i] = emas_volat[ass_id][i]
                    time_stamps_ass[ass_id][idx_stats_ass[ass_id]] = time_stamp#.append(time_stamp)
                    
#                    num_assets_vol[idx_stats] = array_vol#.append(len(array_vol))
#                    num_assets_volat[idx_stats] = arrays_volat#.append(len(array_volat))
                    
                    idx_stats += 1  
                    idx_stats_ass[ass_id] += 1
            
            track_last_dts[ass_id][track_idx[ass_id]] = time_stamp
            
            track_idx[ass_id] = (track_idx[ass_id]+1) % mW
            events_per_ass_counter[ass_id] += 1
            event_idx += 1
            
#        av_vol = delete_leftovers(av_vol, null_entry=null_entry)
#        av_volat = delete_leftovers(av_volat, null_entry=null_entry)
#        num_assets_vol = delete_leftovers(num_assets_vol, null_entry=null_entry)
#        num_assets_volat = delete_leftovers(num_assets_volat, null_entry=null_entry)
#        time_stamps = delete_leftovers(time_stamps, null_entry=null_entry)
        
        for ass in range(len(assets)):
            volat_per_ass[ass] = delete_leftovers(volat_per_ass[ass], null_entry=[0 for _ in ws])
            time_stamps_ass[ass] = delete_leftovers(time_stamps_ass[ass], null_entry=null_entry)
        
        
        # save volumme structs
        pickle.dump( {
                    'volat_per_ass':volat_per_ass,
                    'time_stamps_ass':time_stamps_ass,
                    'assets':assets
                    }, open( directory+start_time+'_F'+init_day_str+'T'+end_day_str+"_"+every+"mW"+str(mW)+"_stats.p", "wb" ))

    # end of weeks
    