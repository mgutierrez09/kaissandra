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
        print("WARNING! ValueError. Returning the whole list.")
        return List
    

if __name__ == '__main__':
        
    start_time = dt.datetime.strftime(dt.datetime.now(),'%y%m%d%H%M%S')
    numberNetwors = 2
    init_day_str = '20200307'#'20190225'#'20181112'#'20191202'#
    end_day_str = '20200321'#'20191212'
    list_name = ['01050NYORPS2k12K5k12K2E1452ALSRNSP60', '01050NYORPS2k12K5k12K2E1453BSSRNSP60']
    list_epoch_journal = [0 for _ in range(numberNetwors)]
    list_t_index = [0 for _ in range(numberNetwors)]
    assets= [1,2,3,4,7,8,10,11,12,13,14,16,17,19,27,28,29,30,31,32]
    spreads_per_asset = False
    if not spreads_per_asset:
        list_IDresults = ['R01050NYORPS2CMF181112T200306ALk12K5K2E141452', 'R01050NYORPS2CMF181112T200306BSk12K5K2E141452']
        # size is N numberNetwors \times A assets. Eeach entry is a dict with 'sp', 'th', and 'mar' fields.
        list_spread_ranges = [[{'sp':[round_num(i,10) for i in np.linspace(.5,5,num=46)],
                               'th':[(0.5, 0.58), (0.5, 0.58), (0.5, 0.58), (0.5, 0.58), (0.54, 0.58), (0.55, 0.58), (0.58, 0.58), (0.58, 0.58), (0.55, 0.59), (0.54, 0.6), (0.54, 0.6), 
                                     (0.54, 0.6), (0.55, 0.6), (0.58, 0.6), (0.55, 0.61), (0.58, 0.61), (0.58, 0.61), (0.65, 0.6), (0.65, 0.6), (0.65, 0.6), (0.65, 0.6), (0.65, 0.6), 
                                     (0.65, 0.6), (0.65, 0.6), (0.65, 0.6), (0.64, 0.61), (0.65, 0.61), (0.65, 0.61), (0.67, 0.61), (0.67, 0.61), (0.69, 0.61), (0.69, 0.61), (0.69, 0.61), 
                                     (0.71, 0.61), (0.71, 0.61), (0.71, 0.61), (0.65, 0.63), (0.73, 0.61), (0.75, 0.61), (0.75, 0.61), (0.75, 0.61), (0.75, 0.61), (0.75, 0.61), (0.75, 0.61), 
                                     (0.75, 0.61), (0.75, 0.61)],
                               'mar':[(0,0.02) for _ in range(46)]} for _ in assets],
                              [{'sp':[round_num(i,10) for i in np.linspace(.5,5,num=46)],
                               'th':[(0.54, 0.57), (0.51, 0.58), (0.56, 0.57), (0.54, 0.58), (0.56, 0.58), (0.58, 0.58), (0.59, 0.58), (0.61, 0.58), (0.62, 0.58), (0.66, 0.57), (0.65, 0.58), 
                                     (0.66, 0.58), (0.66, 0.58), (0.67, 0.58), (0.67, 0.58), (0.67, 0.58), (0.68, 0.58), (0.68, 0.58), (0.71, 0.58), (0.71, 0.58), (0.71, 0.58), (0.71, 0.58), 
                                     (0.71, 0.58), (0.71, 0.58), (0.71, 0.58), (0.71, 0.58), (0.71, 0.58), (0.71, 0.58), (0.71, 0.58), (0.71, 0.58), (0.71, 0.58), (0.71, 0.58), (0.71, 0.58), 
                                     (0.72, 0.58), (0.72, 0.58), (0.72, 0.58), (0.73, 0.58), (0.74, 0.58), (0.74, 0.58), (0.74, 0.58), (0.74, 0.58), (0.74, 0.58), (0.74, 0.58), (0.74, 0.58), 
                                     (0.75, 0.58), (0.75, 0.58)],
                               'mar':[(0,0.02) for _ in range(46)]} for _ in assets]]
        list_lb_mc_ext = [.5, .51]
        list_lb_md_ext = [.58,.57]
    else:
        extentionNamesSpreads = ['CMF160101T181109AL', 'CMF160101T181109BS']#'CMF160101T181109BSk12K2K5E141453'
        extentionNamesResults = ['CMF181112T200306AL', 'CMF181112T200306BS']
        baseNames = ['R01050NYORPS2', 'R01050NYORPS2']
        list_anchor_spread_ranges = [[{'sp':[round_num(i,10) for i in np.linspace(.5,5,num=46)],
                               'th':[(0.5, 0.58), (0.5, 0.58), (0.5, 0.58), (0.5, 0.58), (0.54, 0.58), (0.55, 0.58), (0.58, 0.58), (0.58, 0.58), (0.55, 0.59), (0.54, 0.6), (0.54, 0.6), 
                                     (0.54, 0.6), (0.55, 0.6), (0.58, 0.6), (0.55, 0.61), (0.58, 0.61), (0.58, 0.61), (0.65, 0.6), (0.65, 0.6), (0.65, 0.6), (0.65, 0.6), (0.65, 0.6), 
                                     (0.65, 0.6), (0.65, 0.6), (0.65, 0.6), (0.64, 0.61), (0.65, 0.61), (0.65, 0.61), (0.67, 0.61), (0.67, 0.61), (0.69, 0.61), (0.69, 0.61), (0.69, 0.61), 
                                     (0.71, 0.61), (0.71, 0.61), (0.71, 0.61), (0.65, 0.63), (0.73, 0.61), (0.75, 0.61), (0.75, 0.61), (0.75, 0.61), (0.75, 0.61), (0.75, 0.61), (0.75, 0.61), 
                                     (0.75, 0.61), (0.75, 0.61)],
                               'mar':[(0,0.0) for _ in range(46)]} for _ in assets],
                              [{'sp':[round_num(i,10) for i in np.linspace(.5,5,num=46)],
                               'th':[(0.54, 0.57), (0.51, 0.58), (0.56, 0.57), (0.54, 0.58), (0.56, 0.58), (0.58, 0.58), (0.59, 0.58), (0.61, 0.58), (0.62, 0.58), (0.66, 0.57), (0.65, 0.58), 
                                     (0.66, 0.58), (0.66, 0.58), (0.67, 0.58), (0.67, 0.58), (0.67, 0.58), (0.68, 0.58), (0.68, 0.58), (0.71, 0.58), (0.71, 0.58), (0.71, 0.58), (0.71, 0.58), 
                                     (0.71, 0.58), (0.71, 0.58), (0.71, 0.58), (0.71, 0.58), (0.71, 0.58), (0.71, 0.58), (0.71, 0.58), (0.71, 0.58), (0.71, 0.58), (0.71, 0.58), (0.71, 0.58), 
                                     (0.72, 0.58), (0.72, 0.58), (0.72, 0.58), (0.73, 0.58), (0.74, 0.58), (0.74, 0.58), (0.74, 0.58), (0.74, 0.58), (0.74, 0.58), (0.74, 0.58), (0.74, 0.58), 
                                     (0.75, 0.58), (0.75, 0.58)],
                               'mar':[(0,0.0) for _ in range(46)]} for _ in assets]]
        mar = (0.0, 0.02)
        list_spread_ranges = []
        list_IDresults = []
        list_min_p_mcs = []
        list_min_p_mds = []
        for net in range(numberNetwors):
            spread_ranges, IDresults, min_p_mcs, min_p_mds = build_spread_ranges_per_asset(baseNames[net], 
                                                        extentionNamesResults[net], extentionNamesSpreads[net], assets, 60, 
                                                        mar=mar, anchor=list_anchor_spread_ranges[net])
            list_spread_ranges.append(spread_ranges)
            list_IDresults.append(IDresults)
            list_min_p_mcs.append(min_p_mcs)
            list_min_p_mds.append(min_p_mds)
    
    list_lim_groi_ext = [-10 for i in range(numberNetwors)] # in %
    list_max_lots_per_pos = [.1 for i in range(numberNetwors)]
    list_entry_strategy = ['spread_ranges' for i in range(numberNetwors)]#'fixed_thr','gre' or 'spread_ranges', 'gre_v2'
    list_IDgre = ['' for i in range(numberNetwors)]
    list_if_dir_change_close = [False for i in range(numberNetwors)]
    list_extend_for_any_thr = [True for i in range(numberNetwors)]
    list_thr_sl = [1000 for i in range(numberNetwors)]
    max_opened_positions = 20

    # depricated/not supported
    list_IDgre = ['' for i in range(numberNetwors)]
    list_epoch_gre = [None for i in range(numberNetwors)]
    list_weights = [np.array([0,1]) for i in range(numberNetwors)]
    list_w_str = ["" for i in range(numberNetwors)]
    #root_dir = local_vars.data_dir
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
    positions_dir = directory+"positions/"
    if not os.path.exists(positions_dir):
        os.makedirs(positions_dir)
    
    
    columns_positions = 'Asset\tDi\tTi\tDo\tTo\tGROI\tROI\tspread\tespread\text\tDir\tBi\tBo\tAi\tAo\tstrategy'
    file = open(positions_dir+positions_file,"a")
    file.write(columns_positions+"\n")
    file.close()
            #'_E'+str(epoch)+'TI'+str(t_index)+'MC'+str(thr_mc)+'MD'+str(thr_md)+'.csv'
    # save sorted journal
    #journal_all_days.drop('index',axis=1).to_csv(directory+start_time+'journal.log',sep='\t',float_format='%.3f',index_label='index')
    ##### loop over different day groups #####
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
    
    # tack volume
#    Xtr.resize((pointerTr+samps_tr, seq_len, nFeatures))
#    Ytr.resize((pointerTr+samps_tr, seq_len, size_output_layer))
#    
#    Itr.resize((pointerTr+samps_tr, seq_len, 2))
#    # update IO structures
#    Xtr[pointerTr:pointerTr+samps_tr,:,:] = X_i[i_tr:e_tr,:,:]
#    Ytr[pointerTr:pointerTr+samps_tr,:,:] = Y_i[i_tr:e_tr,:,:]
#    Itr[pointerTr:pointerTr+samps_tr,:,:] = I_i[i_tr:e_tr,:,:]
    null_entry = None
    av_vol = [null_entry for _ in range(10000)]
    av_volat = [null_entry for _ in range(10000)]
    time_stamps = [null_entry for _ in range(10000)]
    num_assets_vol = [null_entry for _ in range(10000)]
    num_assets_volat = [null_entry for _ in range(10000)]
    idx_stats = 0
    
    max_vol_per_pos_ass = {}
    dt_max_vol_per_pos_ass = {}
    max_diff_per_pos_ass = {}
    dt_max_diff_per_pos_ass = {}
    max_vol_per_hour = {}
    mW = 5000
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
        end_list_index = i+1
        
        day_index += counter_back+1
        
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
        n_pos_opened = 0
        secs_counter = 0
        approached = 0
        timeout = 0
        event_idx = 0
        EXIT = 0
        exit_pos = 0
        stoplosses = 0
        takeprofits = 0
        not_entered = 0
        not_entered_av_budget = 0
        not_entered_extention = 0
        not_entered_same_time = 0
        not_entered_secondary = 0
        close_dueto_dirchange = 0
        w = 1-1/20
        max_vols = [99999999 for _ in assets]# inf
        max_volats = [-1 for _ in assets]
        events_per_ass_counter = [-1 for _ in assets]
        margin = 0.0
#        last_dt_per_ass = [False for _ in assets]
        this_hour = None
        this_min = None
        this_sec = None
        every = "min"# "min", "hour", "sec"
        inter_tic = time.time()
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
            if (every=="hour" and this_hour==None) or (every=="min" and this_min==None) or \
                (every=="sec" and this_sec==None):
                print("\r"+DateTime, sep=' ', end='', flush=True)
                this_hour = time_stamp.hour
                this_min = time_stamp.minute
                this_sec = time_stamp.second
            if events_per_ass_counter[ass_id]>mW:
                vol = (track_last_dts[ass_id][track_idx[ass_id]]-time_stamp).seconds
                window_asks = np.array(track_last_asks[ass_id])
                window_asks = window_asks[window_asks>0]
                volat = (np.max(window_asks)-np.min(window_asks))/np.mean(window_asks)
#                if vol<=max_vols[ass_id]:
                # init volatility tracking
                if max_vols[ass_id] == 99999999:
                    max_vols[ass_id] = vol
                # update volume tracking
                max_vols[ass_id] = w*max_vols[ass_id]+(1-w)*vol
                # update max volume
#                if volat>max_volats[ass_id]:
                if max_volats[ass_id] == -1:
                    max_volats[ass_id] = volat
                # update volatility tracking
                max_volats[ass_id] = w*max_volats[ass_id]+(1-w)*volat
                
                array_vol = np.array(max_vols)[np.array(max_vols)!=99999999]
                array_volat = np.array(max_volats)[np.array(max_volats)!=-1]
                mean_vol = np.mean(array_vol)
                mean_volat = np.mean(array_volat)
                
                # update new hour
                if (every=="hour" and this_hour != time_stamp.hour) or \
                   (every=="min" and this_min != time_stamp.minute) or \
                   (every=="sec" and this_sec != time_stamp.second):
                    volatility_idx = (10*np.log10(mean_volat)-mean_volat_db)/var_volat_db
                    volume_idx = (10*np.log10(mean_vol)-mean_vol_db)/var_vol_db
                    print("\r"+DateTime+" idx "+str(idx_stats)+": Risk {0:.2f}".format(volatility_idx-volume_idx)+\
                          " Number assets {0:d}".format(len(array_vol))+\
                          " Time {0:.2f} mins. Total time {1:.2f} mins. "\
                          .format((time.time()-inter_tic)/60,(time.time()-total_tic)/60), \
                          sep=' ', end='', flush=True)
                    this_hour = time_stamp.hour
                    this_min = time_stamp.minute
                    this_second = time_stamp.second
                    
                    if idx_stats >= len(time_stamps):
                        av_vol = av_vol+[null_entry for _ in range(10000)]
                        av_volat = av_volat+[null_entry for _ in range(10000)]
                        num_assets_vol = num_assets_vol+[null_entry for _ in range(10000)]
                        num_assets_volat = num_assets_volat+[null_entry for _ in range(10000)]
                        time_stamps = time_stamps+[null_entry for _ in range(10000)]
                    av_vol[idx_stats] = mean_vol#.append(mean_vol)
                    av_volat[idx_stats] = mean_volat#.append(mean_volat)
                    num_assets_vol[idx_stats] = array_vol#.append(len(array_vol))
                    num_assets_volat[idx_stats] = array_volat#.append(len(array_volat))
                    time_stamps[idx_stats] = time_stamp#.append(time_stamp)
                    idx_stats += 1                    
            
            track_last_dts[ass_id][track_idx[ass_id]] = time_stamp
            
            track_idx[ass_id] = (track_idx[ass_id]+1) % mW
            events_per_ass_counter[ass_id] += 1
            event_idx += 1
            
        av_vol = delete_leftovers(av_vol, null_entry=null_entry)
        av_volat = delete_leftovers(av_volat, null_entry=null_entry)
        num_assets_vol = delete_leftovers(num_assets_vol, null_entry=null_entry)
        num_assets_volat = delete_leftovers(num_assets_volat, null_entry=null_entry)
        time_stamps = delete_leftovers(time_stamps, null_entry=null_entry)
        
        
        # save volumme structs
        pickle.dump( {
                    'av_vol':av_vol,
                    'av_volat':av_volat,
                    'time_stamps':time_stamps,
                    'len_array_vol':num_assets_vol,
                    'len_array_volat':num_assets_volat
                    }, open( directory+start_time+'_F'+init_day_str+'T'+end_day_str+"_"+every+"mW"+str(mW)+"_stats.p", "wb" ))

    # end of weeks
    