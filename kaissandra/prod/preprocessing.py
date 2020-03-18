# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 15:01:21 2020

@author: mgutierrez
"""
import pickle
import numpy as np
import pandas as pd

def load_stats_input_live(feature_keys_manual, movingWindow, nEventsPerStat, 
                          thisAsset, ass_group, from_stats_file=False, 
               hdf5_directory='', save_pickle=False, tag='IOB'):
    """
    Function that loads stats
    """
#    if 'feature_keys_manual' in config:
#        feature_keys_manual = config['feature_keys_manual']
#    else:
#        feature_keys_manual = [i for i in range(37)]
#    if 'movingWindow' in config:
#        movingWindow = config['movingWindow']
#    else:
#        movingWindow = 50
#    if 'nEventsPerStat' in config:
#        nEventsPerStat = config['nEventsPerStat']
#    else:
#        nEventsPerStat = 500
    nF = len(feature_keys_manual)
    # init or load total stats
    stats = {}

    if not from_stats_file:
        stats["means_t_in"] = ass_group.attrs.get("means_t_in")
        stats["stds_t_in"] = ass_group.attrs.get("stds_t_in")
        stats["m_t_in"] = ass_group.attrs.get("m_t_in")
    
    elif from_stats_file:
        #try:
        stats = pickle.load( open( hdf5_directory+thisAsset+'_'+tag+'stats_mW'+
                                      str(movingWindow)+
                                     '_nE'+str(nEventsPerStat)+
                                     '_nF'+str(nF)+".p", "rb" ))
    else:
        print("EROR: Not a possible combination of input parameters")
        raise ValueError
        
    # save stats individually
    if save_pickle:
        raise ValueError("Depricated save_pickle=True in arguments")
        pickle.dump( stats, open( hdf5_directory+thisAsset+'_stats_mW'+
                                 str(movingWindow)+
                                 '_nE'+str(nEventsPerStat)+
                                 '_nF'+str(nF)+".p", "wb" ))
    return stats

def load_stats_output_live(config, hdf5_directory, thisAsset, tag='IOB'):
    """
    Load output stats
    """
    if 'feature_keys_manual' in config:
        feature_keys_manual = config['feature_keys_manual']
    else:
        feature_keys_manual = [i for i in range(37)]
    if 'movingWindow' in config:
        movingWindow = config['movingWindow']
    else:
        movingWindow = 50
    if 'nEventsPerStat' in config:
        nEventsPerStat = config['nEventsPerStat']
    else:
        nEventsPerStat = 500
    nF = len(feature_keys_manual)
    # TODO: pass output stats to their own container and load them from there
    stats = pickle.load( open( hdf5_directory+thisAsset+'_'+tag+'stats_mW'+
                                      str(movingWindow)+
                                     '_nE'+str(nEventsPerStat)+
                                     '_nF'+str(nF)+".p", "rb" ))
    stats_output = {'m_t_out':stats['m_t_out'],
                    'stds_t_out':stats['stds_t_out']}
    return stats_output

def init_features_live(config, tradeInfoLive):
    """
    Init features for online session
    """
    
    feature_keys_manual = config['feature_keys_manual']
    nEventsPerStat = config['nEventsPerStat']
    movingWindow = config['movingWindow']
    feats_from_bids = config['feats_from_bids']
    # TODO: TEMP
    average_over = np.array([0.1, 0.5, 1, 5, 10, 50, 100])
    lbd=1-1/(nEventsPerStat*average_over)
    
    if feats_from_bids:
        symbols = tradeInfoLive.SymbolBid
        
    else:
        symbols = tradeInfoLive.SymbolAsk
        
    class parSarInit:
        # old parsar=> 20
        #periodSAR = data.nEventsPerStat
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
    
    parSarStruct = parSarInit
    nF = len(feature_keys_manual)
    featuresLive = np.zeros((nF,1))
    
    initRange = int(nEventsPerStat/movingWindow)
    em = np.zeros((lbd.shape))+symbols.loc[symbols.index[0]]
    for i in range(initRange*movingWindow-movingWindow):
        em = lbd*em+(1-lbd)*symbols.loc[i+symbols.index[0]]
    
    
    if 1 in feature_keys_manual:
        featuresLive[1:1+lbd.shape[0],0] = symbols.iloc[0]
        for i in range(int(nEventsPerStat*(1-movingWindow/nEventsPerStat))):
            featuresLive[1:1+lbd.shape[0],0] = lbd*featuresLive[1:1+lbd.shape[0],0]+(
                    1-lbd)*symbols.iloc[i]
    #print(em-featuresLive[1:1+data.lbd.shape[0],0])
    if 10 in feature_keys_manual:
        featuresLive[10,0] = symbols.iloc[0]
        featuresLive[11,0] = symbols.iloc[0]
    
    if 13 in feature_keys_manual:
        featuresLive[13,0] = symbols.iloc[0]
        featuresLive[14,0] = symbols.iloc[0]
    
    return featuresLive,parSarStruct,em

def get_features_live(config, tradeInfoLive, featuresLive,parSarStruct,em):
    """
    Get features from raw inputs
    """
    nEvents = config['nEventsPerStat']
    movingWindow = config['movingWindow']
    feats_from_bids = config['feats_from_bids']
    # TODO: TEMP
    average_over = np.array([0.1, 0.5, 1, 5, 10, 50, 100])
    lbd=1-1/(nEvents*average_over)
    std_var = .1
    std_time = .1
    secsInDay = 86400.0
    
    if feats_from_bids:
        symbols = tradeInfoLive.SymbolBid
        
    else:
        symbols = tradeInfoLive.SymbolAsk
    initRange = int(nEvents/movingWindow)
    endIndex = initRange*movingWindow+symbols.index[0]
    newBidsIndex = range(endIndex-movingWindow,endIndex)
    for i in newBidsIndex:
        #print(tradeInfoLive.SymbolBid.loc[i])
        em = lbd*em+(1-lbd)*symbols.loc[i]
    
    
    newEventsRange = range(int(nEvents*(1-movingWindow/nEvents)),nEvents)
    eml = featuresLive[1:1+lbd.shape[0],0]
    for il in newEventsRange:
        eml = lbd*eml+(1-lbd)*symbols.iloc[il]
    featuresLive[1:1+lbd.shape[0],0] = eml
    
    parSarStruct.HP20 = np.max([np.max(symbols.iloc[:]),parSarStruct.HP20])
    parSarStruct.LP20 = np.min([np.min(symbols.iloc[:]),parSarStruct.LP20])
    featuresLive[10,0] = featuresLive[10,0]+parSarStruct.AFH20*(parSarStruct.HP20-featuresLive[10,0]) #parSar high
    featuresLive[11,0] = featuresLive[11,0]-parSarStruct.AFL20*(featuresLive[11,0]-parSarStruct.LP20) # parSar low
    if featuresLive[10,0]<parSarStruct.HP20:
        parSarStruct.AFH20 = np.min([parSarStruct.AFH20+parSarStruct.stepAF,parSarStruct.maxAF20])
        parSarStruct.LP20 = np.min(symbols.iloc[:])
    if featuresLive[11,0]>parSarStruct.LP20:
        parSarStruct.AFL20 = np.min([parSarStruct.AFH20+parSarStruct.stepAF,parSarStruct.maxAF20])
        parSarStruct.HP20 = np.max(symbols.iloc[:])
    
    parSarStruct.HP2 = np.max([np.max(symbols.iloc[:]),parSarStruct.HP2])
    parSarStruct.LP2 = np.min([np.min(symbols.iloc[:]),parSarStruct.LP2])
    featuresLive[13,0] = featuresLive[13,0]+parSarStruct.AFH2*(parSarStruct.HP2-featuresLive[13,0]) #parSar high
    featuresLive[14,0] = featuresLive[14,0]-parSarStruct.AFL2*(featuresLive[14,0]-parSarStruct.LP2) # parSar low
    if featuresLive[13,0]<parSarStruct.HP2:
        parSarStruct.AFH2 = np.min([parSarStruct.AFH2+parSarStruct.stepAF,parSarStruct.maxAF2])
        parSarStruct.LP2 = np.min(symbols.iloc[:])
    if featuresLive[14,0]>parSarStruct.LP2:
        parSarStruct.AFL2 = np.min([parSarStruct.AFH2+parSarStruct.stepAF,parSarStruct.maxAF2])
        parSarStruct.HP2 = np.max(symbols.iloc[:])

    featuresLive[0,0] = symbols.iloc[-1]
    
    featuresLive[8,0] = 10*np.log10(np.var(symbols.iloc[:])/std_var+1e-10)
    
    te = pd.to_datetime(tradeInfoLive.iloc[-1].DateTime)
    t0 = pd.to_datetime(tradeInfoLive.iloc[0].DateTime)
    timeInterval = (te-t0).seconds/nEvents
    featuresLive[9,0] = 10*np.log10(timeInterval/std_time+0.01)
    
    te = pd.to_datetime(tradeInfoLive.iloc[-1].DateTime)
    timeSec = (te.hour*60*60+te.minute*60+te.second)/secsInDay
    featuresLive[12,0] = timeSec
    
    # Repeat non-variation features to inclue variation betwen first and second input
    featuresLive[15,0] = featuresLive[8,0]
    featuresLive[16,0] = featuresLive[9,0]    
    featuresLive[17,0] = np.max(symbols.iloc[:])-featuresLive[0,0]
    featuresLive[18,0] = featuresLive[0,0]-np.min(symbols.iloc[:])
    featuresLive[19,0] = np.max(symbols.iloc[:])-featuresLive[0,0]    
    featuresLive[20,0] = featuresLive[0,0]-np.min(symbols.iloc[:])
    featuresLive[21,0] = np.min(symbols.iloc[:])/np.max(symbols.iloc[:])
    featuresLive[22,0] = np.min(symbols.iloc[:])/np.max(symbols.iloc[:])
    for i in range(lbd.shape[0]):              
        featuresLive[23+i,0] = featuresLive[0,0]/eml[i]
    for i in range(lbd.shape[0]):                
        featuresLive[30+i,0] = featuresLive[0,0]/eml[i]
    
    return featuresLive,parSarStruct,em