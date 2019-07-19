# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 11:13:31 2019

@author: mgutierrez
"""
import numpy as np
import time
import datetime as dt
import os
import pickle
import h5py

from kaissandra.config import configuration, retrieve_config, Config as C
from kaissandra.local_config import local_vars


def save_stats_func(config, all_stats, thisAsset, first_date, last_date, assdirname, outassdirname):
    """  """
    movingWindow = config['movingWindow']
    nEventsPerStat = config['nEventsPerStat']
    asset_relation = config['asset_relation']
    if config['feats_from_bids']:
        symbol_type = 'bid'
    else:
        symbol_type = 'ask'
    
    stats_dir = local_vars.stats_modular_directory+'mW'+str(movingWindow)+'nE'+str(nEventsPerStat)+'/'+asset_relation+'/'
    if not os.path.exists(stats_dir):
        os.makedirs(stats_dir)
        
    for key in all_stats.keys():
        filedirname = assdirname+'allstats_'+key+'_'+first_date+last_date+'.p'
        pickle.dump( all_stats[key], 
                    open( filedirname, "wb" ))
        # copy in stats directory
        filedirname = stats_dir+thisAsset+'_'+symbol_type+'_'+key+'_'+first_date+last_date+'.p'
        pickle.dump( all_stats[key], 
                    open( filedirname, "wb" ))
    filedirname = outassdirname+'allstats_out_'+first_date+last_date+'.p'
    pickle.dump( all_stats['out'], 
                    open( filedirname, "wb" ))
    filedirname = stats_dir+thisAsset+'_'+symbol_type+'_out_'+first_date+last_date+'.p'
    pickle.dump( all_stats[key], 
                    open( filedirname, "wb" ))
    
def track_stats(stats_in, stats_out, all_stats, m_in, m_out, last_iteration):
    """  """
    for key in stats_in.keys():
        if key not in all_stats:
            # init all stats
            all_stats[key] = {'mean':0.0, 'std':0.0, 'm':0}
        all_stats[key]['mean'] += m_in*stats_in[key]['mean']
        all_stats[key]['std'] += m_in*stats_in[key]['std']
        all_stats[key]['m'] += m_in
        if last_iteration:
            all_stats[key]['mean'] = all_stats[key]['mean']/all_stats[key]['m']
            all_stats[key]['std'] = all_stats[key]['std']/all_stats[key]['m']
            
    if 'out' not in all_stats:
        all_stats['out'] = {'mean_bid':0.0, 'std_bid':0.0, 
                            'mean_ask':0.0, 'std_ask':0.0,
                            'm':0}
    all_stats['out']['mean_bid'] += m_out*stats_out['mean_bid']
    all_stats['out']['std_bid'] += m_out*stats_out['std_bid']
    all_stats['out']['mean_ask'] += m_out*stats_out['mean_ask']
    all_stats['out']['std_ask'] += m_out*stats_out['std_ask']
    all_stats['out']['m'] += m_out
    if last_iteration:
        all_stats['out']['mean_bid'] = all_stats['out']['mean_bid']/all_stats['out']['m']
        all_stats['out']['std_bid'] = all_stats['out']['std_bid']/all_stats['out']['m']
        all_stats['out']['mean_ask'] = all_stats['out']['mean_ask']/all_stats['out']['m']
        all_stats['out']['std_ask'] = all_stats['out']['std_ask']/all_stats['out']['m']
    
    return all_stats

def get_stats_modular(config, groupdirname, groupoutdirname, feature_keys):
    """
    Function that calcultes mean and var of a group.
    Args:
        - data:
        - features:
        - returns:
        - nChannels:
    Returns:
        - means_in:
        - stds_in:
        - means_out:
        - stds_out:
    """
    tic = time.time()
    if 'nEventsPerStat' in config:
        nEventsPerStat = config['nEventsPerStat']
    else:
        nEventsPerStat = 5000
    if 'movingWindow' in config:
        movingWindow = config['movingWindow']
    else:
        movingWindow = 500
    
    nChannels = int(nEventsPerStat/movingWindow)
    stats_in = {}
#    print(feature_keys)
    if len(feature_keys)>0:
        print("\t getting variations")
        
        for i, f in enumerate(feature_keys):
            # create temp file for variations
            filename = local_vars.hdf5_directory+'temp'+str(np.random.randint(1000))+'.hdf5'
            ft = h5py.File(filename,'w')
            group_temp = ft.create_group('temp')
            # load features
            filedirname = groupdirname+C.PF[f]+'_0.hdf5'
            feature_file = h5py.File(filedirname,'r')
    #        groupname = '_'.join(groupdirname.split('/'))
            features = feature_file[C.PF[f]][C.PF[f]]
            # create variations vector
            variations = group_temp.create_dataset("variations", (features.shape[0],nChannels), dtype=float)
            for r in range(nChannels):
                if f not in C.non_var_features:
                    variations[r+1:,r] = features[r+1:,0]-features[:-(r+1),0]
                else:
                    variations[r+1:,r] = features[:-(r+1),0]
            
            
            means_in = np.zeros((nChannels,1))
            stds_in = np.zeros((nChannels,1))
            # loop over channels
            for r in range(nChannels):
                #nonZeros = variations[:,0,r]!=999
                #print(np.sum(nonZeros))
                means_in[r,0] = np.mean(variations[nChannels:,r],axis=0,keepdims=1)
                stds_in[r,0] = np.std(variations[nChannels:,r],axis=0,keepdims=1)
            # save input stats
            stats_in[C.PF[f]] = {'mean':means_in, 'std':stds_in}
            pickle.dump( stats_in[C.PF[f]], 
                        open( groupdirname+C.PF[f]+"_stats.p", "wb" ))
            
            ft.close()
            feature_file.close()
            os.remove(filename)
    # load returns
    filediroutname = groupoutdirname+'output_0.hdf5'
    file_out = h5py.File(filediroutname,'r')
    returns_bid = file_out['returns_bid']
    returns_ask = file_out['returns_ask']
    # get output stats
    stds_out_bid = np.std(returns_bid,axis=0)
    means_out_bid = np.mean(returns_bid,axis=0)
    stds_out_ask = np.std(returns_ask,axis=0)
    means_out_ask = np.mean(returns_ask,axis=0)
    stats_out = {'mean_bid':means_out_bid, 'std_bid':stds_out_bid,
                  'mean_ask':means_out_ask, 'std_ask':stds_out_ask}
    pickle.dump( stats_out, 
                open( groupoutdirname+"output_stats.p", "wb" ))
    file_out.close()
    
    print("\t Total time for stats: "+str(time.time()-tic))    
    # open temporal file for variations
#    try:
#        # create file
#        filename = local_vars.hdf5_directory+'temp'+str(np.random.randint(1000))+'.hdf5'
#        ft = h5py.File(filename,'w')
#        # create group
#        group_temp = ft.create_group('temp')
#        # reserve memory space for variations and normalized variations
#        variations = group_temp.create_dataset("variations", (features.shape[0],nF,nChannels), dtype=float)
#        # init variations and normalized variations to 999 (impossible value)
#        
#        #variations[:] = variations[:]+999
#        
#        # loop over channels
#        for r in range(nChannels):
#            variations[r+1:,:,r] = features[r+1:,:]-features[:-(r+1),:]
#            variations[r+1:,C.non_var_features,r] = features[:-(r+1),C.non_var_features]
#        print("\t time for variations: "+str(time.time()-tic))
#        # init stats    
#        means_in = np.zeros((nChannels,nF))
#        stds_in = np.zeros((nChannels,nF))
#        print("\t getting means and stds")
#        # loop over channels
#        for r in range(nChannels):
#            #nonZeros = variations[:,0,r]!=999
#            #print(np.sum(nonZeros))
#            means_in[r,:] = np.mean(variations[nChannels:,:,r],axis=0,keepdims=1)
#            stds_in[r,:] = np.std(variations[nChannels:,:,r],axis=0,keepdims=1)  
#    #       
#            # get output stats
#            stds_out = np.std(returns,axis=0)
#            means_out = np.mean(returns,axis=0)
#        print("\t Total time for stats: "+str(time.time()-tic))
#    except KeyboardInterrupt:
#        ft.close()
#        print("ERROR! Closing file and exiting.")
#        raise KeyboardInterrupt
#    ft.close()
    
    return stats_in, stats_out

def get_returns_modular(config, groupoutdirname, idx_init, DateTime, SymbolBid, SymbolAsk, m_out, shift):
    """
    Function that obtains the outputs from raw data.
    Args:
        - data:
        - DateTime:
        - SymbolBid:
        - SymbolAsk:
    Returns:
        - outputs:
        - ret_idx:
    """
    if 'nEventsPerStat' in config:
        nEventsPerStat = config['nEventsPerStat']
    else:
        nEventsPerStat = 5000
    if 'movingWindow' in config:
        movingWindow = config['movingWindow']
    else:
        movingWindow = 500
    if 'lookAheadVector' in config:
        lookAheadVector = config['lookAheadVector']
    else:
        lookAheadVector = [.1,.2,.5,1]
    if 'force_calulation_output' in config:
        force_calulation_output = config['force_calulation_output']
    else:
        force_calulation_output = False
    
    
    if not os.path.exists(groupoutdirname+'output_0.hdf5') or force_calulation_output:
        print("\tGetting output.")
        
        file = h5py.File(groupoutdirname+'output_'+str(shift)+'.hdf5','a')
        returns_bid = file.create_dataset("returns_bid", (m_out,len(lookAheadVector)),dtype=float)
        returns_ask = file.create_dataset("returns_ask", (m_out,len(lookAheadVector)),dtype=float)
        ret_idx = file.create_dataset("ret_idx", (m_out,len(lookAheadVector)+1),dtype=int)
        DT = file.create_dataset("DT", (m_out,len(lookAheadVector)+1),dtype='S19')
        B = file.create_dataset("B", (m_out,len(lookAheadVector)+1),dtype=float)
        A = file.create_dataset("A", (m_out,len(lookAheadVector)+1),dtype=float)
        
        
        nE = DateTime.shape[0]
        m = int(np.floor((nE/nEventsPerStat-1)*nEventsPerStat/movingWindow)+1)
        initRange = int(nEventsPerStat/movingWindow)
        
        np_00 = initRange*movingWindow-1
        np_e0 = m*movingWindow-1
        
        indexOrigins = [i for i in range(np_00,np_e0,movingWindow)]
        ret_idx[:,0] = indexOrigins+idx_init
        DT[:,0] = DateTime[indexOrigins]
        B[:,0] = SymbolBid[indexOrigins]
        A[:,0] = SymbolAsk[indexOrigins]
        #origins = np.array(tradeInfo.SymbolBid.iloc[np_00:np_e0:data.movingWindow])
        
        for nr in range(len(lookAheadVector)):
            #print("nr")
            #print(nr)
            unp_0e = int(initRange*movingWindow+np.floor(nEventsPerStat*lookAheadVector[nr])-2)
            unp_ee = int(np.min([m*movingWindow+np.floor(
                    nEventsPerStat*lookAheadVector[nr])-2,nE]))
    
            indexEnds = [i for i in range(unp_0e,unp_ee,movingWindow)]
            #fill ends wth last value
            for i in range(len(indexEnds),len(indexOrigins)):
                indexEnds.append(nE-1)
            returns_bid[:,nr] = SymbolBid[indexEnds]-SymbolBid[indexOrigins]
            returns_ask[:,nr] = SymbolAsk[indexEnds]-SymbolAsk[indexOrigins]
            ret_idx[:,nr+1] = indexEnds+idx_init
            DT[:,nr+1] = DateTime[indexEnds]
            B[:,nr+1] = SymbolBid[indexEnds]
            A[:,nr+1] = SymbolAsk[indexEnds]
        
        file.close()
    else:
        print("\tOutput already exists. Skipped")
    
    return None

def intit_em(lbd, nExS, mW, SymbolBid):
    """  """
    # init exponetial means
    em = np.zeros((lbd.shape))+SymbolBid[0]
    for i in range(nExS-mW):
        em = lbd*em+(1-lbd)*SymbolBid[i]
    return em

def init_sar(firstSymbol):
    """  """
    class SAR:
        oldSARhs = [firstSymbol for _ in range(C.n_sars)]
        oldSARls = [firstSymbol for _ in range(C.n_sars)]
        HPs = [0 for _ in range(C.n_sars)]
        LPs = [100000 for _ in range(C.n_sars)]
        AFHs = [C.stepAF for _ in range(C.n_sars)]
        AFLs = [C.stepAF for _ in range(C.n_sars)]
        maxAFs = [C.maxStepSars[i]*C.stepAF for i in range(C.n_sars)]
    
    return SAR

def get_sar(sar, maxValue, minValue, n_parsars):
    """  """
    parSARhigh = np.zeros((n_parsars))
    parSARlow = np.zeros((n_parsars))
    for i in range(n_parsars):
        sar.HPs[i] = np.max([maxValue,sar.HPs[i]])
        sar.LPs[i] = np.min([minValue,sar.LPs[i]])
        parSARhigh[i] = sar.oldSARhs[i]+sar.AFHs[i]*(sar.HPs[i]-sar.oldSARhs[i])
        parSARlow[i] = sar.oldSARls[i]-sar.AFLs[i]*(sar.oldSARls[i]-sar.LPs[i])
        if parSARhigh[i]<sar.HPs[i]:
            sar.AFHs[i] = np.min([sar.AFHs[i]+C.stepAF,sar.maxAFs[i]])
            sar.LPs[i] = minValue
        if parSARlow[i]>sar.LPs[i]:
            sar.AFLs[i] = np.min([sar.AFHs[i]+C.stepAF,sar.maxAFs[i]])
            sar.HPs[i] = maxValue
        sar.oldSARhs[i] = parSARhigh[i]
        sar.oldSARls[i] = parSARlow[i]
        
#    HP20 = np.max([maxValue[mm],HP20])
#    LP20 = np.min([minValue[mm],LP20])
#    parSARhigh20[mm] = oldSARh20+AFH20*(HP20-oldSARh20)
#    parSARlow20[mm] = oldSARl20-AFL20*(oldSARl20-LP20)
#    if parSARhigh20[mm]<HP20:
#        AFH20 = np.min([AFH20+stepAF,maxAF20])
#        LP20 = np.min(thisPeriodBids)
#    if parSARlow20[mm]>LP20:
#        AFL20 = np.min([AFH20+stepAF,maxAF20])
#        HP20 = np.max(thisPeriodBids)
#    oldSARh20 = parSARhigh20[mm]
#    oldSARl20 = parSARlow20[mm]
#    
#    HP2 = np.max([maxValue[mm],HP2])
#    LP2 = np.min([minValue[mm],LP2])
#    parSARhigh2[mm] = oldSARh2+AFH2*(HP2-oldSARh2)
#    parSARlow2[mm] = oldSARl2-AFL2*(oldSARl2-LP2)
#    if parSARhigh2[mm]<HP2:
#        AFH2 = np.min([AFH2+stepAF,maxAF2])
#        LP2 = np.min(thisPeriodBids)
#    if parSARlow2[mm]>LP2:
#        AFL2 = np.min([AFH2+stepAF,maxAF2])
#        HP2 = np.max(thisPeriodBids)
#    oldSARh2 = parSARhigh2[mm]
#    oldSARl2 = parSARlow2[mm]
    return [parSARhigh, parSARlow, sar]

def get_features_modular(config, groupdirname, DateTime, Symbol, m, shift):
    """
    Function that calculates features from raw data in per batches
    Args:
        - data
        - features
        - DateTime
        - SymbolBid
    Returns:
        - features
    """
    tic = time.time()
    if 'feature_keys' in config:
        feature_keys = config['feature_keys']
    else:
        feature_keys = [i for i in range(len(C.FI))]#[i for i in range(8,10)]+[12,15,16]#[i for i in range(8)]+[10,11,13,14]+[i for i in range(17,37)]#
    if 'movingWindow' in config:
        movingWindow = config['movingWindow']
    else:
        movingWindow = 500
    if 'nEventsPerStat' in config:
        nEventsPerStat = config['nEventsPerStat']
    else:
        nEventsPerStat = 5000
    if 'lbd' in config:
        lbd = config['lbd']
    else:
        lbd = 1-1/(nEventsPerStat*np.array([0.1, 0.5, 1, 5, 10, 50, 100]))
    if 'force_calculation_features' in config:
        force_calculation_features = config['force_calculation_features']
    else:
        force_calculation_features = [False for i in range(len(feature_keys))]
    # init scalars
    nExS = nEventsPerStat
    mW = movingWindow
    boolEmas = [C.FI['EMA'+i] in feature_keys for i in C.emas_ext]
    idxEmas = [i for i in range(len(boolEmas)) if boolEmas[i]]
    nEMAs = sum(boolEmas)
    boolSars = [C.FI['parSARhigh'+i] in feature_keys for i in C.sar_ext]
    idxSars = [i for i in range(len(boolSars)) if boolSars[i]]
    n_parsars = sum(boolSars)#int(C.FI['parSARhigh20'] in feature_keys) + int(C.FI['parSARhigh2'] in feature_keys)
    # init exponetial means
    if nEMAs>0:
        em = intit_em(lbd, nExS, mW, Symbol)
            
    if n_parsars>0:
        sar = init_sar(Symbol[0])
    
    batch_size = 10000000
    par_batches = int(np.ceil(m/batch_size))
    l_index = 0
    features_files = []
    features = []
    feature_keys_to_calc = []
    # init features
    for i,f in enumerate(feature_keys):
        featname = C.PF[f]+'_'+str(shift)
        filedirname = groupdirname+featname+'.hdf5'
        if not os.path.exists(filedirname) or force_calculation_features[i]:
            features_files.append(h5py.File(filedirname,'a') )
            features.append(features_files[-1].create_group(C.PF[f]).\
                        create_dataset(C.PF[f], (m,1),dtype=float))
            feature_keys_to_calc.append(f)
    feature_keys = feature_keys_to_calc
#    features_files = [h5py.File(groupdirname+C.PF[i]+'.hdf5','a') for i in feature_keys]
#    features = [features_files[i].create_group('_'.join(groupdirname.split('/'))).\
#                          create_dataset(C.PF[i], (m,1),dtype=float) for i in feature_keys]
#    f_prep_IO.create_group(group_name)
#    group.create_dataset("returns", (m_out,len(lookAheadVector)),dtype=float)
    # loop over batched
    if len(features_files) > 0:
        for b in range(par_batches):
            # get m
            m_i = np.min([batch_size, m-b*batch_size])
            
            # init structures
            if nEMAs>0:
                EMA = np.zeros((m_i,nEMAs))
            boolSymbol = C.FI['symbol'] in feature_keys
            if boolSymbol:
                symbol = np.zeros((m_i))
            boolVariance = C.FI['variance'] in feature_keys
            if boolVariance:
                variance = np.zeros((m_i))
            boolMaxValue = C.FI['maxValue'] in feature_keys
            if boolMaxValue:
                maxValue = np.zeros((m_i))
            boolMinValue = C.FI['minValue'] in feature_keys
            if boolMinValue:
                minValue = np.zeros((m_i))
            boolTimeInterval = C.FI['timeInterval'] in feature_keys
            if boolTimeInterval:
                timeInterval = np.zeros((m_i))
            boolTime = C.FI['time'] in feature_keys
            if boolTime:
                timeSecs = np.zeros((m_i))
            if n_parsars>0:
                parSARhigh = np.zeros((m_i, n_parsars))
                parSARlow = np.zeros((m_i, n_parsars))
            
            
            for mm in range(m_i):
                
                startIndex = l_index+mm*mW
                endIndex = startIndex+nExS
                thisPeriod = range(startIndex,endIndex)
                thisPeriodBids = Symbol[thisPeriod]
                if boolSymbol:
                    symbol[mm] = Symbol[thisPeriod[-1]]
                
                if nEMAs>0:
                    newBidsIndex = range(endIndex-mW,endIndex)
                    for i in newBidsIndex:
                        #a=data.lbd*em/(1-data.lbd**i)+(1-data.lbd)*tradeInfo.SymbolBid.loc[i]
                        em = lbd*em+(1-lbd)*Symbol[i]
                    EMA[mm,:] = em
                    
                if boolTimeInterval:
                    t0 = dt.datetime.strptime(DateTime[thisPeriod[0]].decode("utf-8"),'%Y.%m.%d %H:%M:%S')
                    te = dt.datetime.strptime(DateTime[thisPeriod[-1]].decode("utf-8"),'%Y.%m.%d %H:%M:%S')
                    timeInterval[mm] = (te-t0).seconds/nExS
    
                if boolVariance:
                    variance[mm] = np.var(thisPeriodBids)
                
                if boolMaxValue:
                    maxValue[mm] = np.max(thisPeriodBids)
                if boolMinValue:
                    minValue[mm] = np.min(thisPeriodBids)
                
                if boolTime:
                    timeSecs[mm] = (te.hour*60*60+te.minute*60+te.second)/C.secsInDay
                
                if n_parsars>0:
                    outsar = get_sar(sar, maxValue[mm], minValue[mm], n_parsars)
                    parSARhigh[mm,:] = outsar[0]
                    parSARlow[mm,:] = outsar[1]
                    sar = outsar[2]
                    
            # end of for mm in range(m_i):
            l_index = startIndex+mW
            #print(l_index)
            toc = time.time()
            print("\t\tmm="+str(b*batch_size+mm+1)+" of "+str(m)+". Total time: "+str(np.floor(toc-tic))+"s")
            # update features vector
            init_idx = b*batch_size
            end_idx = b*batch_size+m_i
    
            if boolSymbol:
                nF = feature_keys.index(C.FI['symbol'])
                features[nF][init_idx:end_idx,0] = symbol
    
            for e in range(nEMAs):
                if C.FI['EMA'+C.emas_ext[idxEmas[e]]] in feature_keys:
                    nF = feature_keys.index(C.FI['EMA'+C.emas_ext[idxEmas[e]]])
                    features[nF][init_idx:end_idx, 0] = EMA[:,e]
    
            if boolVariance:
                nF = feature_keys.index(C.FI['variance'])
                logVar = 10*np.log10(variance/C.std_var+1e-10)
                features[nF][init_idx:end_idx, 0] = logVar
    
            if boolTimeInterval:
                nF = feature_keys.index(C.FI['timeInterval'])
                logInt = 10*np.log10(timeInterval/C.std_time+0.01)
                features[nF][init_idx:end_idx, 0] = logInt
            
            for e in range(n_parsars):
                if C.FI['parSARhigh'+C.sar_ext[idxSars[e]]] in feature_keys:
                    nF = feature_keys.index(C.FI['parSARhigh'+C.sar_ext[idxSars[e]]])
                    features[nF][init_idx:end_idx, 0] = parSARhigh[:,e]
                #if C.FI['parSARlow'+C.sar_ext[idxSars[e]]] in feature_keys:
                    nF = feature_keys.index(C.FI['parSARlow'+C.sar_ext[idxSars[e]]])
                    features[nF][init_idx:end_idx, 0] = parSARlow[:,e]
            
            if boolTime:
                nF = feature_keys.index(C.FI['time'])
                features[nF][init_idx:end_idx, 0] = timeSecs
                
            if C.FI['difVariance'] in feature_keys:
                nF = feature_keys.index(C.FI['difVariance'])
                features[nF][init_idx:end_idx, 0] = logVar
                
            if C.FI['difTimeInterval'] in feature_keys:
                nF = feature_keys.index(C.FI['difTimeInterval'])
                features[nF][init_idx:end_idx, 0] = logInt
            
            if C.FI['maxValue'] in feature_keys:
                nF = feature_keys.index(C.FI['maxValue'])
                features[nF][init_idx:end_idx, 0] = maxValue-symbol
            
            if C.FI['minValue'] in feature_keys:
                nF = feature_keys.index(C.FI['minValue'])
                features[nF][init_idx:end_idx, 0] = symbol-minValue
            
            if C.FI['difMaxValue'] in feature_keys:
                nF = feature_keys.index(C.FI['difMaxValue'])
                features[nF][init_idx:end_idx, 0] = maxValue-symbol
                
            if C.FI['difMinValue'] in feature_keys:
                nF = feature_keys.index(C.FI['difMinValue'])
                features[nF][init_idx:end_idx, 0] = symbol-minValue
            
            if C.FI['minOmax'] in feature_keys:
                nF = feature_keys.index(C.FI['minOmax'])
                features[nF][init_idx:end_idx, 0] = minValue/maxValue
            
            if C.FI['difMinOmax'] in feature_keys:
                nF = feature_keys.index(C.FI['difMinOmax'])
                features[nF][init_idx:end_idx, 0] = minValue/maxValue
            
            for e in range(nEMAs):
                if C.FI['symbolOema'+C.emas_ext[idxEmas[e]]] in feature_keys:
                    nF = feature_keys.index(C.FI['symbolOema'+C.emas_ext[idxEmas[e]]])
                    features[nF][init_idx:end_idx, 0] = symbol/EMA[:,e]
            for e in range(nEMAs):
                if C.FI['difSymbolOema'+C.emas_ext[idxEmas[e]]] in feature_keys:
                    nF = feature_keys.index(C.FI['difSymbolOema'+C.emas_ext[idxEmas[e]]])
                    features[nF][init_idx:end_idx, 0] = symbol/EMA[:,e]
            
            # close file
            (file.close() for file in features_files)
    else:
        print("\tAll features already calculated. Skipped.")
    return feature_keys

def wrapper_get_features_modular(config, thisAsset, separators, assdirname, outassdirname, 
                                 group_raw, s, all_stats, last_iteration, 
                                 init_date, end_date, feats_from_bids=True):
    """
    Function that extracts features, results and normalization stats from raw data.
    Args:
        - config:
        - thisAsset
        - separators
        - group
        - stats
        - hdf5_directory
        - s
    Returns:
        - features 
        - returns
        - ret_idx
        - stats
    """
    nEventsPerStat = config['nEventsPerStat']
    movingWindow = config['movingWindow']
    save_stats = config['save_stats']
    asset_relation = config['asset_relation']
    phase_shift = config['phase_shift']
    
    # get trade info datasets
    DateTime = group_raw["DateTime"]
    SymbolBid = group_raw["SymbolBid"]
    SymbolAsk = group_raw["SymbolAsk"]
    if asset_relation == 'direct':
        Bids = SymbolBid
        Asks = SymbolAsk
    elif asset_relation == 'inverse':
        print("\tGetting inverse")
        Bids = 1/SymbolBid[:]
        Asks = 1/SymbolAsk[:]
    if feats_from_bids:
        Symbols = Bids
    else:
        Symbols = Asks
    # number of events
    nExS = nEventsPerStat
    mW = movingWindow
    phase_size = movingWindow/phase_shift
    shifts = [int(phase_size*phase) for phase in range(phase_shift)]
    for shift in shifts:
        nE = separators.index[s+1]-separators.index[s]+1-shift
        
        # check if number of events is not enough to build two features and one return
        if nE>=2*nEventsPerStat:
            
            # number of features and number of returns
            m_in = int(np.floor((nE/nExS-1)*nExS/mW)+1)
            m_out = int(m_in-nExS/mW)#len(range(int(nExS/mW)*mW-1,m_in*mW-1,mW))
    #        m = {'m_in':m_in,
    #             'm_out':m_out}
            groupdirname = assdirname+init_date+end_date+'/'
            groupoutdirname = outassdirname+init_date+end_date+'/'
            if not os.path.exists(groupdirname):
                os.makedirs(groupdirname)
                # save m
                pickle.dump( m_in, open( groupdirname+"m_in.p", "wb" ))
            if not os.path.exists(groupoutdirname):
                os.makedirs(groupoutdirname)
                pickle.dump( m_out, open( groupoutdirname+"m_out.p", "wb" ))
                    
            print("\tShift "+str(shift)+": getting features from raw data...")
            # get structures and save them in a hdf5 file
            feature_keys_to_calc = get_features_modular(config, groupdirname,
                                     DateTime[separators.index[s]+shift:separators.index[s+1]+1], 
                                     Symbols[separators.index[s]+shift:separators.index[s+1]+1], m_in, shift)
            get_returns_modular(config, groupoutdirname, separators.index[s],
                                    DateTime[separators.index[s]+shift:separators.index[s+1]+1], 
                                    Bids[separators.index[s]+shift:separators.index[s+1]+1],
                                    Asks[separators.index[s]+shift:separators.index[s+1]+1], m_out, shift)
            # only get stats for shift zero
            if shift==0:
                # get stats
                stats_in, stats_out = get_stats_modular(config, groupdirname, groupoutdirname, feature_keys_to_calc)
                if save_stats:
                    all_stats = track_stats(stats_in, stats_out, all_stats, m_in, m_out, last_iteration)
    
    return all_stats

def wrapper_wrapper_get_features_modular(config_entry, assets, seps):
    """  """
    import time
    from kaissandra.inputs import load_separators
    
    
    ticTotal = time.time()
    # init file directories
    hdf5_directory = local_vars.hdf5_directory
    if type(config_entry)==dict:
        config = configuration(config_entry)
        config_name = config['config_name']
    elif config_entry!='':
        config_name = config_entry
        config = retrieve_config(config_name)
    else:
        config_name = 'CFEATURESMODULAR'
        config = retrieve_config(config_name)
    if 'movingWindow' in config:
        movingWindow = config['movingWindow']
    else:
        movingWindow = 500
    if 'nEventsPerStat' in config:
        nEventsPerStat = config['nEventsPerStat']
    else:
        nEventsPerStat = 5000
    if 'feats_from_bids' in config:
        feats_from_bids = config['feats_from_bids']
    else:
        feats_from_bids = True
    if 'save_stats' in config:
        save_stats = config['save_stats']
    else:
        save_stats = False
    # define files and directories names
    if 'build_test_db' in config:
        build_test_db = config['build_test_db']
    else:
        build_test_db = False
    if build_test_db:
        test_flag = '_test'
    else:
        test_flag = ''
    if 'asset_relation' in config:
        asset_relation = config['asset_relation']
    else:
        asset_relation = 'direct' # direct, inverse
#        config['asset_relation'] = asset_relation
    
    if feats_from_bids:
        rootdirname = hdf5_directory+'mW'+str(movingWindow)+'_nE'+str(nEventsPerStat)+'/'+asset_relation+'/bid/'#test_flag+'_legacy_test.hdf5'
    else:
        rootdirname = hdf5_directory+'mW'+str(movingWindow)+'_nE'+str(nEventsPerStat)+'/'+asset_relation+'/ask/'
    outrdirname = hdf5_directory+'mW'+str(movingWindow)+'_nE'+str(nEventsPerStat)+'/'+asset_relation+'/out/'
    
    filename_raw = hdf5_directory+'tradeinfo'+test_flag+'.hdf5'
    separators_directory = hdf5_directory+'separators'+test_flag+'/'
    #assert(not (build_test_db and save_stats))
    # init hdf5 files
    #f_prep_IO = h5py.File(filename_prep_IO,'a')
    f_raw = h5py.File(filename_raw,'r')
    
    # loop over all assets
    for ass in assets:
        
        thisAsset = C.AllAssets[str(ass)]
        assdirname = rootdirname+thisAsset+'/'
        outassdirname = outrdirname+thisAsset+'/'
        print("Config "+config['config_name']+" "+str(ass)+". "+thisAsset)
        tic = time.time()
        # open file for read
        
        group_raw = f_raw[thisAsset]
        #bid_means[ass_idx] = np.mean(group_raw["SymbolBid"])
        # load separators
        separators = load_separators(thisAsset, separators_directory, from_txt=1)
        all_stats = {}
        last_iteration = False
        # loop over separators
        for s in seps:#range(0,len(separators)-1,2):
            # number of events within this separator chunk
            nE = separators.index[s+1]-separators.index[s]+1
            if len(seps) == 0:
                if s+1 == separators.shape[0]-1:
                    last_iteration = True
            else:
                if s+1 >= len(seps)-1:
                    last_iteration = True
            #print(nE)
            # check if number of events is not enough to build two features and one return
            if nE>=2*nEventsPerStat:
                print("\t"+"Config "+config['config_name']+" "+thisAsset+
                      " s {0:d} of {1:d}".format(int(s/2),int(len(separators)/2-1))+
                      ". From "+separators.DateTime.iloc[s]+" to "+separators.DateTime.iloc[s+1])
                init_date = dt.datetime.strftime(dt.datetime.strptime(
                        separators.DateTime.iloc[s],'%Y.%m.%d %H:%M:%S'),'%y%m%d%H%M%S')
                end_date = dt.datetime.strftime(dt.datetime.strptime(
                        separators.DateTime.iloc[s+1],'%Y.%m.%d %H:%M:%S'),'%y%m%d%H%M%S')
                if s == 0:
                    first_date = init_date
                elif last_iteration:
                    last_date = end_date
                # calculate features, returns and stats from raw data
                
                all_stats = wrapper_get_features_modular(
                        config, thisAsset, separators, assdirname, outassdirname, 
                        group_raw, s, all_stats, last_iteration, init_date, end_date, 
                        feats_from_bids=feats_from_bids)
                    
            else:
                print("\ts {0:d} of {1:d}. Not enough entries. Skipped.".format(int(s/2),int(len(separators)/2-1)))
        # end of for s in range(0,len(separators)-1,2):
        # save stats in attributes
            # update total number of samples
        
        print("\t"+"Config "+config['config_name']+
              " Time for "+thisAsset+":"+str(np.floor(time.time()-tic))+"s"+
              ". Total time:"+str(np.floor(time.time()-ticTotal))+"s")
    if save_stats:
        print("\tSaving stats")
        save_stats_func(config, all_stats, thisAsset, first_date, last_date, assdirname, outassdirname)
    f_raw.close()
    # release lock
    print("DONE")
    return None

def wrapper_get_stats_modular():
    """  """
    pass