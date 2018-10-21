# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 12:00:20 2018

@author: mgutierrez
Script to save features, results and sats into hdf5 files
"""
print("This script is depricated! Use get_features_hdf5.py insted")
error()
import numpy as np
import time
import pandas as pd
import h5py
import datetime as dt
import tensorflow as tf
from DataManager import Data, build_output_v11,save_as_matfile
from RNN import modelRNN

def get_features_from_raw_par(data, features, DateTime, SymbolBid):
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
    # init scalars
    nExS = data.nEventsPerStat
    mW = data.movingWindow
    nE = DateTime.shape[0]
    m = int(np.floor((nE/nExS-1)*nExS/mW)+1)
    secsInDay = 86400.0
    nEMAs = data.lbd.shape[0]
    
    # init exponetial means
    em = np.zeros((data.lbd.shape))+SymbolBid[0]
    for i in range(nExS-mW):
        em = data.lbd*em+(1-data.lbd)*SymbolBid[i]
    #/(1-np.maximum(data.lbd**i,1e-3))
    
    
    oldSARh20 = SymbolBid[0]
    oldSARh2 = SymbolBid[0]
    oldSARl20 = SymbolBid[0]
    oldSARl2 = SymbolBid[0]
    HP20 = 0
    HP2 = 0
    LP20 = 100000
    LP2 = 100000
    stepAF = 0.02
    AFH20 = stepAF
    AFH2 = stepAF
    AFL20 = stepAF
    AFL2 = stepAF
    maxAF20 = 20*stepAF    
    maxAF2 = 2*stepAF
    
    batch_size = 5000
    par_batches = int(np.ceil(m/batch_size))
    l_index = 0
    # loop over batched
    for b in range(par_batches):
        # get m
        m_i = np.min([batch_size, m-b*batch_size])
        
        # init structures
        EMA = np.zeros((m_i,nEMAs))
        bids = np.zeros((m_i))
        variance = np.zeros((m_i))
        maxValue = np.zeros((m_i))
        minValue = np.zeros((m_i))
        timeInterval = np.zeros((m_i))
        timeSecs = np.zeros((m_i))
        parSARhigh20 = np.zeros((m_i))
        parSARhigh2 = np.zeros((m_i))
        parSARlow20 = np.zeros((m_i))
        parSARlow2 = np.zeros((m_i))
        
        toc = time.time()
        
        for mm in range(m_i):
            
            startIndex = l_index+mm*mW
            endIndex = startIndex+nExS
            thisPeriod = range(startIndex,endIndex)
            thisPeriodBids = SymbolBid[thisPeriod]
            
            newBidsIndex = range(endIndex-mW,endIndex)
            for i in newBidsIndex:
                #a=data.lbd*em/(1-data.lbd**i)+(1-data.lbd)*tradeInfo.SymbolBid.loc[i]
                em = data.lbd*em+(1-data.lbd)*SymbolBid[i]
                
            t0 = dt.datetime.strptime(DateTime[thisPeriod[0]].decode("utf-8"),'%Y.%m.%d %H:%M:%S')
            te = dt.datetime.strptime(DateTime[thisPeriod[-1]].decode("utf-8"),'%Y.%m.%d %H:%M:%S')

            bids[mm] = SymbolBid[thisPeriod[-1]]
            EMA[mm,:] = em
            variance[mm] = np.var(thisPeriodBids)
            timeInterval[mm] = (te-t0).seconds/data.nEventsPerStat
            maxValue[mm] = np.max(thisPeriodBids)
            minValue[mm] = np.min(thisPeriodBids)
            timeSecs[mm] = (te.hour*60*60+te.minute*60+te.second)/secsInDay
            
            HP20 = np.max([maxValue[mm],HP20])
            LP20 = np.min([minValue[mm],LP20])
            parSARhigh20[mm] = oldSARh20+AFH20*(HP20-oldSARh20)
            parSARlow20[mm] = oldSARl20-AFL20*(oldSARl20-LP20)
            if parSARhigh20[mm]<HP20:
                AFH20 = np.min([AFH20+stepAF,maxAF20])
                LP20 = np.min(thisPeriodBids)
            if parSARlow20[mm]>LP20:
                AFL20 = np.min([AFH20+stepAF,maxAF20])
                HP20 = np.max(thisPeriodBids)
            oldSARh20 = parSARhigh20[mm]
            oldSARl20 = parSARlow20[mm]
            
            HP2 = np.max([maxValue[mm],HP2])
            LP2 = np.min([minValue[mm],LP2])
            parSARhigh2[mm] = oldSARh2+AFH2*(HP2-oldSARh2)
            parSARlow2[mm] = oldSARl2-AFL2*(oldSARl2-LP2)
            if parSARhigh2[mm]<HP2:
                AFH2 = np.min([AFH2+stepAF,maxAF2])
                LP2 = np.min(thisPeriodBids)
            if parSARlow2[mm]>LP2:
                AFL2 = np.min([AFH2+stepAF,maxAF2])
                HP2 = np.max(thisPeriodBids)
            oldSARh2 = parSARhigh2[mm]
            oldSARl2 = parSARlow2[mm]
        # end of for mm in range(m_i):
        l_index = startIndex
        #print(l_index)
        print("\t\tmm="+str(b*batch_size+mm)+" of "+str(m)+". Total time: "+str(np.floor(toc-tic))+"s")
        # update features vector
        init_idx = b*batch_size
        end_idx = b*batch_size+m_i

        nF = 0
        features[init_idx:end_idx,nF] = bids

        nF += 1
        features[init_idx:end_idx,nF:nF+data.lbd.shape[0]] = EMA
        
        nF += data.lbd.shape[0]
        logVar = 10*np.log10(variance/data.std_var+1e-10)
        features[init_idx:end_idx,nF] = logVar
        
        nF += 1
        logInt = 10*np.log10(timeInterval/data.std_time+0.01)
        features[init_idx:end_idx,nF] = logInt
        
        nF += 1
        features[init_idx:end_idx,nF] = parSARhigh20
        features[init_idx:end_idx,nF+1] = parSARlow20
        
        nF += 2
        features[init_idx:end_idx,nF] = timeSecs
        
        nF += 1
        features[init_idx:end_idx,nF] = parSARhigh2
        features[init_idx:end_idx,nF+1] = parSARlow2
        
        # repeat
        nF += 2
        features[init_idx:end_idx,nF] = logVar
        
        nF += 1
        features[init_idx:end_idx,nF] = logInt
        
        nF += 1
        features[init_idx:end_idx,nF] = maxValue-bids
        
        nF += 1
        features[init_idx:end_idx,nF] = bids-minValue
        
        nF += 1
        features[init_idx:end_idx,nF] = maxValue-bids
        
        nF += 1
        features[init_idx:end_idx,nF] = bids-minValue
        
        nF += 1
        features[init_idx:end_idx,nF] = minValue/maxValue
        
        nF += 1
        features[init_idx:end_idx,nF] = minValue/maxValue
        
        for i in range(data.lbd.shape[0]):          
            nF += 1        
            features[init_idx:end_idx,nF] = bids/EMA[:,i]
        
        for i in range(data.lbd.shape[0]):
            nF += 1        
            features[init_idx:end_idx,nF] = bids/EMA[:,i]
    # end of for b in range(par_batches):
#    save_as_matfile('features','features',features[:])
#    a=p
    return features

def get_features_from_raw(data, features, DateTime, SymbolBid):
    """
    Function that calculates features from raw data.
    Args:
        - data
        - features
        - DateTime
        - SymbolBid
    Returns:
        - features
    """
    
    tic = time.time()
    # init scalars
    nExS = data.nEventsPerStat
    mW = data.movingWindow
    nE = DateTime.shape[0]
    m = int(np.floor(nE/nExS-1)*nExS/mW+1)
    secsInDay = 86400.0

    # init exponetial means
    em = np.zeros((data.lbd.shape))+SymbolBid[0]
    for i in range(nExS-mW):
        em = data.lbd*em+(1-data.lbd)*SymbolBid[i]
    #/(1-np.maximum(data.lbd**i,1e-3))
    
    
    oldSARh20 = SymbolBid[0]
    oldSARh2 = SymbolBid[0]
    oldSARl20 = SymbolBid[0]
    oldSARl2 = SymbolBid[0]
    HP20 = 0
    HP2 = 0
    LP20 = 100000
    LP2 = 100000
    stepAF = 0.02
    AFH20 = stepAF
    AFH2 = stepAF
    AFL20 = stepAF
    AFL2 = stepAF
    maxAF20 = 20*stepAF    
    maxAF2 = 2*stepAF

    for mm in range(m):
        
        endIndex = mm*mW+nExS
        startIndex = mm*mW
        thisPeriod = range(startIndex,endIndex)
        thisPeriodBids = SymbolBid[thisPeriod]
        
        newBidsIndex = range(endIndex-mW,endIndex)
        for i in newBidsIndex:
            #a=data.lbd*em/(1-data.lbd**i)+(1-data.lbd)*tradeInfo.SymbolBid.loc[i]
            em = data.lbd*em+(1-data.lbd)*SymbolBid[i]

        if mm%5000==0:
            toc = time.time()
            print("\t\tmm="+str(mm)+" of "+str(m)+". Total time: "+str(np.floor(toc-tic))+"s")
            
        t0 = dt.datetime.strptime(DateTime[thisPeriod[0]].decode("utf-8"),'%Y.%m.%d %H:%M:%S')
        te = dt.datetime.strptime(DateTime[thisPeriod[-1]].decode("utf-8"),'%Y.%m.%d %H:%M:%S')
        
        maxValue = np.max(thisPeriodBids)
        minValue = np.min(thisPeriodBids)
        
        # update features vector
        nF = 0
        features[mm,nF] = SymbolBid[thisPeriod[-1]]
        
        nF += 1
        features[mm,nF:nF+data.lbd.shape[0]] = em
        
        nF += data.lbd.shape[0]
        features[mm,nF] = 10*np.log10(np.var(thisPeriodBids)/data.std_var+1e-10)
        
        nF += 1
        tInt = (te-t0).seconds/data.nEventsPerStat
        features[mm,nF] = 10*np.log10(tInt/data.std_time+0.01)
        
        nF += 1
        HP20 = np.max([maxValue,HP20])
        LP20 = np.min([minValue,LP20])
        features[mm,nF] = oldSARh20+AFH20*(HP20-oldSARh20)
        features[mm,nF+1] = oldSARl20-AFL20*(oldSARl20-LP20)
        if features[mm,nF]<HP20:
            AFH20 = np.min([AFH20+stepAF,maxAF20])
            LP20 = np.min(thisPeriodBids)
        if features[mm,nF+1]>LP20:
            AFL20 = np.min([AFH20+stepAF,maxAF20])
            HP20 = np.max(thisPeriodBids)
        oldSARh20 = features[mm,nF]
        oldSARl20 = features[mm,nF+1]
        
        nF += 2
        features[mm,nF] = (te.hour*60*60+te.minute*60+te.second)/secsInDay
        
        nF += 1
        HP2 = np.max([maxValue,HP2])
        LP2 = np.min([minValue,LP2])
        features[mm,nF] = oldSARh2+AFH2*(HP2-oldSARh2)
        features[mm,nF+1] = oldSARl2-AFL2*(oldSARl2-LP2)
        if features[mm,nF]<HP2:
            AFH2 = np.min([AFH2+stepAF,maxAF2])
            LP2 = np.min(thisPeriodBids)
        if features[mm,nF+1]>LP2:
            AFL2 = np.min([AFH2+stepAF,maxAF2])
            HP2 = np.max(thisPeriodBids)
        oldSARh2 = features[mm,nF]
        oldSARl2 = features[mm,nF+1]
        
        # repeat
        nF += 2
        features[mm,nF] = features[mm,1+data.lbd.shape[0]]
        
        nF += 1
        features[mm,nF] = features[mm,2+data.lbd.shape[0]]
        
        nF += 1
        features[mm,nF] = maxValue-features[mm,0]
        
        nF += 1
        features[mm,nF] = features[mm,0]-minValue
        
        nF += 1
        features[mm,nF] = maxValue-features[mm,0]
        
        nF += 1
        features[mm,nF] = features[mm,0]-minValue
        
        nF += 1
        features[mm,nF] = minValue/maxValue
        
        nF += 1
        features[mm,nF] = minValue/maxValue
        
        for i in range(data.lbd.shape[0]):          
            nF += 1        
            features[mm,nF] = features[mm,0]/features[mm,1+i]
        
        for i in range(data.lbd.shape[0]):
            nF += 1        
            features[mm,nF] = features[mm,0]/features[mm,1+i]
    
    
    return features

def get_returns_from_raw(data, returns, ret_idx, idx_init, DateTime, SymbolBid, SymbolAsk):
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
    nExS = data.nEventsPerStat
    mW = data.movingWindow
    nE = DateTime.shape[0]
    m = int(np.floor((nE/nExS-1)*nExS/mW)+1)
    initRange = int(nExS/mW)
    
    np_00 = initRange*data.movingWindow-1
    np_e0 = m*data.movingWindow-1
    
    indexOrigins = [i for i in range(np_00,np_e0,data.movingWindow)]
    ret_idx[:,0] = indexOrigins+idx_init
    #origins = np.array(tradeInfo.SymbolBid.iloc[np_00:np_e0:data.movingWindow])
    
    for nr in range(len(data.lookAheadVector)):
        #print("nr")
        #print(nr)
        unp_0e = int(initRange*data.movingWindow+np.floor(data.nEventsPerStat*data.lookAheadVector[nr])-2)
        unp_ee = int(np.min([m*data.movingWindow+np.floor(
                data.nEventsPerStat*data.lookAheadVector[nr])-2,nE]))

        indexEnds = [i for i in range(unp_0e,unp_ee,data.movingWindow)]
        #fill ends wth last value
        for i in range(len(indexEnds),len(indexOrigins)):
            indexEnds.append(nE-1)
        returns[:,nr] = SymbolBid[indexEnds]-SymbolBid[indexOrigins]
        ret_idx[:,nr+1] = indexEnds+idx_init
    
    return returns, ret_idx

def get_normalization_stats(data, features, returns, hdf5_directory):
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
    
    nChannels = int(data.nEventsPerStat/data.movingWindow)
    # open temporal file for variations
    try:
        # create file
        ft = h5py.File(hdf5_directory+'temp.hdf5','w')
        # create group
        group_temp = ft.create_group('temp')
        # reserve memory space for variations and normalized variations
        variations = group_temp.create_dataset("variations", (features.shape[0],data.nFeatures,nChannels), dtype=float)
        # init variations and normalized variations to 999 (impossible value)
        variations[:] = variations[:]+999
        # loop over channels
        for r in range(nChannels):
            variations[r+1:,:,r] = features[r+1:,:]-features[:-(r+1),:]
            variations[r+1:,data.noVarFeats,r] = features[:-(r+1),data.noVarFeats]
            
        # init stats    
        means_in = np.zeros((nChannels,data.nFeatures))
        stds_in = np.zeros((nChannels,data.nFeatures))
        # loop over channels
        for r in range(nChannels):
            nonZeros = variations[:,0,r]!=999
            #print(np.sum(nonZeros))
            means_in[r,:] = np.mean(variations[nonZeros,:,r],axis=0,keepdims=1)
            stds_in[r,:] = np.std(variations[nonZeros,:,r],axis=0,keepdims=1)  
    #       
            # get output stats
            stds_out = np.std(returns,axis=0)
            means_out = np.mean(returns,axis=0)
    except:
        ft.close()
        print("ERROR! Closing file and exiting.")
        error()
    ft.close()
    return [means_in, stds_in, means_out, stds_out]

def get_features_results_stats_from_raw(data, thisAsset, separators, f_prep_IO, group,
                               stats, hdf5_directory, s, save_stats):
    """
    Function that extracts features, results and normalization stats from raw data.
    Args:
        - data:
        - thisAsset
        - separators
        - f_prep_IO
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
    
    # get trade info datasets
    DateTime = group["DateTime"]
    SymbolBid = group["SymbolBid"]
    SymbolAsk = group["SymbolAsk"]
    # init structures
    features = []
    returns = []
    ret_idx = []
    # number of events
    nE = separators.index[s+1]-separators.index[s]+1
    # check if number of events is not enough to build two features and one return
    if nE>=2*data.nEventsPerStat:
#        print("\tSeparator batch {0:d} out of {1:d}".format(int(s/2),int(len(separators)/2-1)))
#        print("\t"+separators.DateTime.iloc[s]+" to "+separators.DateTime.iloc[s+1])
        # get init and end dates of these separators
        init_date = dt.datetime.strftime(dt.datetime.strptime(
                separators.DateTime.iloc[s],'%Y.%m.%d %H:%M:%S'),'%y%m%d%H%M%S')
        end_date = dt.datetime.strftime(dt.datetime.strptime(
                separators.DateTime.iloc[s+1],'%Y.%m.%d %H:%M:%S'),'%y%m%d%H%M%S')
        
        # group name of separator s
        group_name = thisAsset+'/'+init_date+end_date
        
        # create new gruop if not yet in file
        if group_name not in f_prep_IO:
            # create group, its attributes and its datasets
            group = f_prep_IO.create_group(group_name)
        else:
            # get group from file
            group = f_prep_IO[group_name]
        # get features, returns and stats if don't exist
        if group.attrs.get("means_in") is None:
            # number of samples
            nExS = data.nEventsPerStat
            mW = data.movingWindow
            
            # number of features and number of returns
            m_in = m = int(np.floor((nE/nExS-1)*nExS/mW)+1)
            m_out = len(range(int(nExS/mW)*mW-1,m_in*mW-1,mW))
            # save as attributes
            group.attrs.create("m_in", m_in, dtype=int)
            group.attrs.create("m_out", m_out, dtype=int)
            # create datasets
            try:
                features = group.create_dataset("features", (m_in,data.nFeatures),dtype=float)
                returns = group.create_dataset("returns", (m_out,len(data.lookAheadVector)),dtype=float)
                ret_idx = group.create_dataset("ret_idx", (m_out,len(data.lookAheadVector)+1),dtype=int)
            except ValueError:
                print("WARNING: RunTimeError. Trying to recover from it.")
                features = group['features']
                returns = group['returns']
                ret_idx = group['ret_idx']
                
            print("\tgetting IO from raw data...")
            # get structures and save them in a hdf5 file
            features = get_features_from_raw_par(data,features,
                                             DateTime[separators.index[s]:separators.index[s+1]+1], 
                                             SymbolBid[separators.index[s]:separators.index[s+1]+1])
            # get returns and their indexes associated to the raw data
            returns, ret_idx = get_returns_from_raw(data,returns,ret_idx,separators.index[s],
                                             DateTime[separators.index[s]:separators.index[s+1]+1], 
                                             SymbolBid[separators.index[s]:separators.index[s+1]+1],
                                             SymbolAsk[separators.index[s]:separators.index[s+1]+1])
            # get stats
            means_in, stds_in, means_out, stds_out = get_normalization_stats(data, features, returns, hdf5_directory)
            # save means and variances as atributes
            group.attrs.create("means_in", means_in, dtype=float)
            group.attrs.create("stds_in", stds_in, dtype=float)
            group.attrs.create("means_out", means_out, dtype=float)
            group.attrs.create("stds_out", stds_out, dtype=float)
            
        else:
            # get data sets
            features = group['features']
            returns = group['returns']
            ret_idx = group['ret_idx']
            # get attributes
            m_in = group.attrs.get("m_in")
            m_out = group.attrs.get("m_out")
            means_in = group.attrs.get("means_in")
            stds_in = group.attrs.get("stds_in")
            means_out = group.attrs.get("means_out")
            stds_out = group.attrs.get("stds_out")
            #print("\tIO loaded from HDF5 file.")
        
        if save_stats:
        # update combined stats of all data sets
            stats["means_t_in"] += m_in*means_in
            stats["stds_t_in"] += m_in*stds_in
            stats["means_t_out"] += m_out*means_out
            stats["stds_t_out"] += m_out*stds_out
            stats["m_t_in"] += m_in
            stats["m_t_out"] += m_out
    # end of if separators.index[s+1]-separators.index[s]>=2*data.nEventsPerStat:
#    else:
#        print("\tSeparator batch {0:d} out of {1:d} skkiped. Not enough entries".format(int(s/2),int(len(separators)/2-1)))
    # save results in a dictionary
    IO_prep = {}
    IO_prep['features'] = features
    IO_prep['returns'] = returns
    IO_prep['ret_idx'] = ret_idx
    
    return IO_prep, stats

def build_IO_from_hdf5(file_temp, data, model, IO_prep, stats, IO, nSampsPerLevel):
    """
    Function that builds X and Y from data contained in a HDF5 file.
    """
    # extract means and stats
    means_in = stats['means_t_in']
    stds_in = stats['stds_t_in']
    stds_out = stats['stds_t_out']
    # extract features and returns
    features = IO_prep['features']
    returns = IO_prep['returns']
    ret_idx = IO_prep['ret_idx']
    # extract IO structures
    X = IO['X']
    Y = IO['Y']
    I = IO['I']
    pointer = IO['pointer']
    # number of channels
    nChannels = int(data.nEventsPerStat/data.movingWindow)
    # sequence length
    seq_len = int((data.lB-data.nEventsPerStat)/data.movingWindow)
    # samples allocation per batch
    aloc = 2**17
    # number of features
    nF = data.nFeatures
    # create file
    
    # create group
    group_temp = file_temp.create_group('temp')
    # reserve memory space for variations and normalized variations
    variations = group_temp.create_dataset('variations', (features.shape[0],data.nFeatures,nChannels), dtype=float)
    variations_normed = group_temp.create_dataset('variations_normed', (features.shape[0],data.nFeatures,nChannels), dtype=float)
    # init variations and normalized variations to 999 (impossible value)
    variations[:] = variations[:]+999
    variations_normed[:] = variations[:]
    # loop over channels
    for r in range(nChannels):
        variations[r+1:,:,r] = features[r+1:,:]-features[:-(r+1),:]
        variations[r+1:,data.noVarFeats,r] = features[:-(r+1),data.noVarFeats]
        variations_normed[r+1:,:,r] = np.minimum(np.maximum((variations[r+1:,:,r]-means_in[r,:])/stds_in[r,:],-10),10)
    # remove the unaltered entries
    nonremoveEntries = variations_normed[:,0,-1]!=999
    # create new variations 
    variations_normed_new = group_temp.create_dataset('variations_normed_new', variations_normed[nonremoveEntries,:,:].shape, dtype=float)
    variations_normed_new[:] = variations_normed[nonremoveEntries,:,:]
    del group_temp['variations_normed']
    # get some scalas
    nSamps = variations_normed_new.shape[0]
    samp_remaining = nSamps-seq_len-nChannels+1
    chuncks = int(np.ceil(samp_remaining/aloc))
    # init counter of samps processed
    offset = 0
    # loop over chunks
    for i in range(chuncks):        
        # this batch length
        batch = np.min([samp_remaining,aloc])
        # create support numpy vectors to speed up iterations
        v_support = variations_normed_new[offset:offset+batch+seq_len, :,data.channels]
        r_support = returns[nChannels+offset:nChannels+offset+batch+seq_len, data.lookAheadIndex]
        i_support = ret_idx[nChannels+offset:nChannels+offset+batch+seq_len, data.lookAheadIndex]
        # update remaining samps to proceed
        samp_remaining = samp_remaining-batch
        # init formatted input and output
        X_i = np.zeros((batch, seq_len, nF))
        # real-valued output
        O_i = np.zeros((batch, seq_len, 1))
        # output index vector
        I_i = np.zeros((batch, seq_len, 1))
        
        for nI in range(batch):
            # init channels counter
            cc = 0
            for r in range(len(data.channels)):
                # get input
                X_i[nI,:,cc*nF:(cc+1)*nF] = v_support[nI:nI+seq_len, :, r]#variations_normed_new[offset+nI:offset+nI+seq_len, :,r]#
                cc += 1
            # due to substraction of features for variation, output gets the 
            # feature one entry later
            O_i[nI,:,0] = r_support[nI:nI+seq_len]#returns[nChannels+offset+nI:nChannels+offset+nI+seq_len, data.lookAheadIndex]
            I_i[nI,:,0] = i_support[nI:nI+seq_len]#ret_idx[nChannels+offset+nI:nChannels+offset+nI+seq_len, data.lookAheadIndex]
        
        
        # normalize output
        O_i = O_i/stds_out[0,data.lookAheadIndex]
        # update counters
        offset = offset+batch
        # get decimal and binary outputs
        Y_i, y_dec = build_output_v11(model, O_i, batch)
        # get samples per level
        for l in range(model.size_output_layer):
            nSampsPerLevel[l] = nSampsPerLevel[l]+np.sum(y_dec[:,-1,0]==l)
        # resize IO structures
        X.resize((pointer+batch, model.seq_len, model.nFeatures))
        Y.resize((pointer+batch, model.seq_len,model.commonY+model.size_output_layer))
        I.resize((pointer+batch, model.seq_len,1))
        # update IO structures
        X[pointer:pointer+batch,:,:] = X_i
        Y[pointer:pointer+batch,:,:] = Y_i
        I[pointer:pointer+batch,:,:] = I_i
        # uodate pointer
        pointer += batch
    # end of for i in range(chuncks):
    # update dictionary
    IO['X'] = X
    IO['Y'] = Y
    IO['I'] = I
    IO['pointer'] = pointer
    
    #print("\tXY built")
    
    return IO, nSampsPerLevel

def load_separators(thisAsset, separators_directory):
    """
    Function that loads and segments separators according to beginning and end dates.
    
    """
    # separators file name
    separators_filename = thisAsset+'_separators.txt'
    # load separators
    separators = pd.read_csv(separators_directory+separators_filename, index_col='Pointer')
    
    return separators

ticTotal = time.time()
# create data structure
data=Data(movingWindow=100,nEventsPerStat=1000,lB=1300,dateEnd='2018.03.08',comments="",
             dateTest = ['2017.11.27','2017.11.28','2017.11.29','2017.11.30','2017.12.01',
                         '2017.12.04','2017.12.05','2017.12.06','2017.12.07','2017.12.08'])
# get ID
ID = '000200'
# init booleans
save_stats = False
reset = False

build_IO = False
runRNN = True
# init file directories
hdf5_directory = '../HDF5/'
IO_directory = '../RNN/IO/'
filename_IO = IO_directory+'IO_'+ID+'.hdf5'
filename_prep_IO = hdf5_directory+'IO_mW'+str(data.movingWindow)+'_nE'+str(data.nEventsPerStat)+'.hdf5'
filename_raw = hdf5_directory+'tradeinfo.hdf5'
separators_directory = hdf5_directory+'separators/'
# make sure that IO is not built when IO_prep is processed
if (reset and build_IO) or (save_stats and build_IO):
    print("ERROR. Reset file or save stats and build IO cannot be active at the same time")
    error()
# reset file
if reset:
    f_w = h5py.File(filename_prep_IO,'w')
    f_w.close()
# init hdf5 file
f_prep_IO = h5py.File(filename_prep_IO,'a')
# init IO structures
if runRNN:
    # define if it is for training or for testing
    tOt = 'train'
    # create model
    model=modelRNN(data,
                   size_hidden_layer=100,
                   L=3,
                   size_output_layer=5,
                   keep_prob_dropout=1,
                   miniBatchSize=32,
                   outputGain=1,
                   commonY=3,
                   lR0=0.0001,
                   version="1.0")
    if build_IO:
        # open IO file for writting
        f_IO = h5py.File(filename_IO,'w')
        # init IO data sets
        X = f_IO.create_dataset('X', (0, model.seq_len, model.nFeatures), 
                                maxshape=(None,model.seq_len, model.nFeatures), dtype=float)
        Y = f_IO.create_dataset('Y', (0,model.seq_len,model.commonY+model.size_output_layer),
                                maxshape=(None,model.seq_len,model.commonY+model.size_output_layer),dtype=float)
        I = f_IO.create_dataset('I', (0,model.seq_len,1),maxshape=(None,model.seq_len,1),dtype=float)
        totalSampsPerLevel = np.zeros((model.size_output_layer))
        # save IO structures in dictionary
        IO = {}
        IO['X'] = X
        IO['Y'] = Y
        IO['I'] = I
        IO['pointer'] = 0

# init total number of samples
m = 0
aloc = 2**17
# max number of input channels
nChannels = int(data.nEventsPerStat/data.movingWindow)
# loop over all assets
for ass in data.assets:
    thisAsset = data.AllAssets[str(ass)]
    print(str(ass)+". "+thisAsset)
    tic = time.time()
    # open file for read
    f_raw = h5py.File(filename_raw,'r')
    group_raw = f_raw[thisAsset]
    # load separators
    separators = load_separators(thisAsset, separators_directory)
    # crate asset_group if does not exist
    if thisAsset not in f_prep_IO:
        # init total stats
        ass_group = f_prep_IO.create_group(thisAsset)
    else:
        # retrive ass group if exists
        ass_group = f_prep_IO[thisAsset]
    # init or load total stats
    stats = {}
    if save_stats:
        
        stats["means_t_in"] = np.zeros((nChannels,data.nFeatures))
        stats["stds_t_in"] = np.zeros((nChannels,data.nFeatures))
        stats["means_t_out"] = np.zeros((1,len(data.lookAheadVector)))
        stats["stds_t_out"] = np.zeros((1,len(data.lookAheadVector)))
        stats["m_t_in"] = 0
        stats["m_t_out"]  = 0
    else:
        stats["means_t_in"] = ass_group.attrs.get("means_t_in")
        stats["stds_t_in"] = ass_group.attrs.get("stds_t_in")
        stats["means_t_out"] = ass_group.attrs.get("means_t_out")
        stats["stds_t_out"] = ass_group.attrs.get("stds_t_out")
        stats["m_t_in"] = ass_group.attrs.get("m_t_in")
        stats["m_t_out"] = ass_group.attrs.get("m_t_out")
    
    # loop over separators
    for s in range(0,len(separators)-1,2):
        # number of events within this separator chunk
        nE = separators.index[s+1]-separators.index[s]+1
        # check if number of events is not enough to build two features and one return
        if nE>=2*data.nEventsPerStat:
            print("\ts {0:d} of {1:d}".format(int(s/2),int(len(separators)/2-1))+
                  ". From "+separators.DateTime.iloc[s]+" to "+separators.DateTime.iloc[s+1])
            #print("\t"+separators.DateTime.iloc[s]+" to "+separators.DateTime.iloc[s+1])
            # calculate features, returns and stats from raw data
            IO_prep, stats = get_features_results_stats_from_raw(
                                        data, thisAsset, separators, f_prep_IO, group_raw, 
                                        stats, hdf5_directory, s, save_stats)
            # build network input and output
            if build_IO:
                # get first day after separator
                day_s = separators.DateTime.iloc[s][0:10]
                # check if 
                if (tOt=='train' and day_s not in data.dateTest) or (tOt=='test' and day_s in data.dateTest):
                    try:
                        file_temp = h5py.File('../RNN/IO/temp_build.hdf5','w')
                        IO, totalSampsPerLevel = build_IO_from_hdf5(file_temp, data, model, IO_prep, stats, IO, totalSampsPerLevel)
                        # close temp file
                        file_temp.close()
                    except (KeyboardInterrupt,NameError):
                        print("KeyBoardInterrupt. Closing files and exiting program.")
                        f_prep_IO.close()
                        f_IO.close()
                        f_raw.close()
                        file_temp.close()
                        end()
                else:
                    print("\tNot in the set. Skipped.")
        else:
            print("\ts {0:d} of {1:d}. Not enough entries. Skipped.".format(int(s/2),int(len(separators)/2-1)))
    
    # save stats in attributes
    if save_stats:
        # normalize stats
        means_t_in = stats["means_t_in"]/stats["m_t_in"]
        stds_t_in = stats["stds_t_in"]/stats["m_t_in"]
        means_t_out = stats["means_t_out"]/stats["m_t_out"]
        stds_t_out = stats["stds_t_out"]/stats["m_t_out"]
        #save total stats as attributes
        ass_group.attrs.create("means_t_in", means_t_in, dtype=float)
        ass_group.attrs.create("stds_t_in", stds_t_in, dtype=float)
        ass_group.attrs.create("means_t_out", means_t_out, dtype=float)
        ass_group.attrs.create("stds_t_out", stds_t_out, dtype=float)
        ass_group.attrs.create("m_t_in", stats["m_t_in"], dtype=int)
        ass_group.attrs.create("m_t_out", stats["m_t_out"], dtype=int)
        # print number of IO samples
        print("\tStats saved. m_t_in="+str(stats["m_t_in"])+", m_t_out="+str(stats["m_t_out"]))
    
    # update total number of samples
    m += stats["m_t_out"]
    # flush content file
    f_prep_IO.flush()
    
    print("\tTime for "+thisAsset+":"+str(np.floor(time.time()-tic))+"s"+
          ". Total time:"+str(np.floor(time.time()-ticTotal))+"s")
    #print("Total time:"+str(np.floor(time.time()-ticTotal))+"s")
# end of for ass in data.assets:
# create number of samps attribute 
if save_stats:
    print("total number of samps m="+str(m))
    f_prep_IO.attrs.create('m', m, dtype=int)

f_prep_IO.close()
f_raw.close()
if build_IO:
    f_IO.close()

if runRNN:
    m_t = int(np.sum(totalSampsPerLevel))
    print("Samples to RNN: "+str(m_t)+".\nPercent per level:"+str(totalSampsPerLevel/m_t))
    tf.reset_default_graph()
    with tf.Session() as sess:
        model.train(sess, int(np.ceil(m/aloc)), ID=ID, IDIO=ID, data_format='hdf5')