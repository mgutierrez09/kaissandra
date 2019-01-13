# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 12:41:50 2018

@author: mgutierrez
"""

import numpy as np
import time
import h5py
import os
import pickle
import pandas as pd
import datetime as dt
from scipy.stats import linregress
from scipy.signal import cwt, find_peaks_cwt, ricker, welch
from inputs import Data, load_separators, get_features_results_stats_from_raw
from config import configuration


def get_features(*ins):
    """  """
    ticTotal = time.time()
    if len(ins)>0:
        config = ins[0]
    else:    
        config = configuration('C01100')
    # create data structure
    data=Data(movingWindow=config['movingWindow'],
              nEventsPerStat=config['nEventsPerStat'],
              dateTest = config['dateTest'],
              feature_keys_manual=config['feature_keys_manual'],
              feature_keys_tsfresh=config['feature_keys_tsfresh'],
              assets=config['assets'])
    # init booleans
    save_stats = config['save_stats']  
    # init file directories
    hdf5_directory = config['hdf5_directory']#'../HDF5/'#
    # define files and directories names
    load_features_from = config['load_features_from']
    if load_features_from=='manual':
        filename_prep_IO = (hdf5_directory+'IO_mW'+str(data.movingWindow)+'_nE'+
                            str(data.nEventsPerStat)+'_nF'+str(data.nFeatures)+'.hdf5')
    elif load_features_from=='tsfresh':
        filename_prep_IO = (hdf5_directory+'feat_tsf_mW'+str(data.movingWindow)+'_nE_test'+
                            str(data.nEventsPerStat)+'.hdf5')
    else:
        #print("ERROR: load_features_from "+load_features_from+" not recognized")
        raise ValueError("Load_features_from "+load_features_from+" not recognized")
        
    if 'build_partial_raw' in config:
        build_partial_raw = config['build_partial_raw']
        
        
    else:
        build_partial_raw = False
        
    
    if build_partial_raw:
        # TODO: get init/end dates from dateTest in Data
        int_date = '180928'
        end_date = '181109'
        filename_raw = hdf5_directory+'tradeinfo_F'+int_date+'T'+end_date+'.hdf5'
        separators_directory = hdf5_directory+'separators_F'+int_date+'T'+end_date+'/'
        if save_stats:
            raise ValueError("save_stats must be False if building from partial raw")
    else:
        filename_raw = hdf5_directory+'tradeinfo.hdf5'
        separators_directory = hdf5_directory+'separators/'

    # reset file
    #reset = False
    #if reset:
    #    f_w = h5py.File(filename_prep_IO,'w')
    #    f_w.close()
    
    # reset only one asset
    reset_asset = ''
    
    if len(ins)>0:
        # wait while files are locked
        while os.path.exists(filename_raw+'.flag') or os.path.exists(filename_prep_IO+'.flag'):
            # sleep random time up to 10 seconds if any file is being used
            print(filename_raw+' or '+filename_prep_IO+' busy. Sleeping up to 10 secs')
            time.sleep(10*np.random.rand(1)[0])
        # lock HDF5 files from access
        fh = open(filename_raw+'.flag',"w")
        fh.close()
        fh = open(filename_prep_IO+'.flag',"w")
        fh.close()
    
    # init hdf5 files
    f_prep_IO = h5py.File(filename_prep_IO,'a')
    f_raw = h5py.File(filename_raw,'r')
    # init total number of samples
    m = 0
    # max number of input channels
    nChannels = int(data.nEventsPerStat/data.movingWindow)
    # index asset
    ass_idx = 0
    # loop over all assets
    for ass in data.assets:
        thisAsset = data.AllAssets[str(ass)]
        print("Config "+config['config_name']+" "+str(ass)+". "+thisAsset)
        tic = time.time()
        # open file for read
        
        group_raw = f_raw[thisAsset]
        #bid_means[ass_idx] = np.mean(group_raw["SymbolBid"])
        # load separators
        separators = load_separators(data, thisAsset, separators_directory, from_txt=1)
        
        if thisAsset==reset_asset:
            print(separators)
            del f_prep_IO[thisAsset]
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
            #print(nE)
            # check if number of events is not enough to build two features and one return
            if nE>=2*data.nEventsPerStat:
                print("\t"+"Config "+config['config_name']+
                      " s {0:d} of {1:d}".format(int(s/2),int(len(separators)/2-1))+
                      ". From "+separators.DateTime.iloc[s]+" to "+separators.DateTime.iloc[s+1])
                #print("\t"+separators.DateTime.iloc[s]+" to "+separators.DateTime.iloc[s+1])
                # calculate features, returns and stats from raw data
                IO_prep, stats = get_features_results_stats_from_raw(
                        data, thisAsset, separators, f_prep_IO, group_raw,
                        stats, hdf5_directory, s, save_stats)
                    
            else:
                print("\ts {0:d} of {1:d}. Not enough entries. Skipped.".format(int(s/2),int(len(separators)/2-1)))
        # end of for s in range(0,len(separators)-1,2):
        
        # update asset index
        ass_idx += 1
        # save stats in attributes
        if save_stats:
            # normalize stats
            stats["means_t_in"] = stats["means_t_in"]/stats["m_t_in"]
            stats["stds_t_in"] = stats["stds_t_in"]/stats["m_t_in"]
            stats["means_t_out"] = stats["means_t_out"]/stats["m_t_out"]
            stats["stds_t_out"] = stats["stds_t_out"]/stats["m_t_out"]
            means_t_in = stats["means_t_in"]
            stds_t_in = stats["stds_t_in"] 
            means_t_out = stats["means_t_out"]
            stds_t_out = stats["stds_t_out"]
            #save total stats as attributes
            ass_group.attrs.create("means_t_in", means_t_in, dtype=float)
            ass_group.attrs.create("stds_t_in", stds_t_in, dtype=float)
            ass_group.attrs.create("means_t_out", means_t_out, dtype=float)
            ass_group.attrs.create("stds_t_out", stds_t_out, dtype=float)
            ass_group.attrs.create("m_t_in", stats["m_t_in"], dtype=int)
            ass_group.attrs.create("m_t_out", stats["m_t_out"], dtype=int)
            # pickle them independently
            pickle.dump( stats, open( hdf5_directory+'/stats/'+thisAsset+'_stats_mW'+
                                     str(data.movingWindow)+'_nE'+
                                     str(data.nEventsPerStat)+'_nF'+
                                     str(data.nFeatures)+".p", "wb" ))
            # print number of IO samples
            print("\t"+"Config "+config['config_name']+
                  " Stats saved. m_t_in="+
                  str(stats["m_t_in"])+", m_t_out="+str(stats["m_t_out"]))
            
        # update total number of samples
        m += stats["m_t_out"]
        # flush content file
        f_prep_IO.flush()
        
        print("\t"+"Config "+config['config_name']+
              "Time for "+thisAsset+":"+str(np.floor(time.time()-tic))+"s"+
              ". Total time:"+str(np.floor(time.time()-ticTotal))+"s")
        
        #print("Total time:"+str(np.floor(time.time()-ticTotal))+"s")
    # end of for ass in data.assets:
    
    # create number of samps attribute 
    if save_stats:
        print("total number of samps m="+str(m))
        f_prep_IO.attrs.create('m', m, dtype=int)
    # close files
    f_prep_IO.close()
    f_raw.close()
    # release lock
    if len(ins)>0:
        os.remove(filename_raw+'.flag')
        os.remove(filename_prep_IO+'.flag')

def init_ema_variations(data, SymbolVar, nExS, mW):
    """ Init EMA vector for variation-based features.
    Args:
        - data (Data object): data info.
    Return:
        - em (np array): initialized EMA vector. """
    # init exponetial means
    em = np.zeros((data.lbd.shape))+SymbolVar[0]
    for i in range(nExS-mW):
        em = data.lbd*em+(1-data.lbd)*SymbolVar[i]
        
    return em

def init_parsar(data, firstSymbol):
    """ Init ParSar object.
    Args:
        - data (Data object): data-related parameters.
        - firstSymbol (float): first symbol value (bid, variation, ...).
    Return:
        - ParSar (ParSar object): initialized ParSar structure. """
    class ParSar:
        oldSARh = firstSymbol+np.zeros((len(data.parsars)))
        oldSARl = firstSymbol+np.zeros((len(data.parsars)))
        HP = np.zeros((len(data.parsars)))
        LP = 100000+np.zeros((len(data.parsars)))
        stepAF = 0.02
        AFH = stepAF+np.zeros((len(data.parsars)))
        AFL = stepAF+np.zeros((len(data.parsars)))
        maxAF = np.array(data.parsars)*stepAF
    
    return ParSar

def update_parsar(data, parsar, min_value, max_value, thisPeriodVariations):
    """ Update ParSar values and structure.
    Args:
        - parsar (ParSar object): structure with current values of Par Sar.
        - min_value (int): min value of current window.
        - max_value (int): max value of current window
        - thisPeriodVariations (np array): variation values of current period.
    Return:
        - parsar (ParSar object): updated structure with current values of Par Sar.
        - parSARhigh (np vector): current high par sar values.
        - parSARlow (np vector): current low par sar values."""
    thisParSARhigh = np.zeros((len(data.parsars)))
    thisParSARlow = np.zeros((len(data.parsars)))
    for ps in range(len(data.parsars)):
        parsar.HP[ps] = np.max([max_value,parsar.HP[ps]])
        parsar.LP[ps] = np.min([min_value,parsar.LP[ps]])
        thisParSARhigh[ps] = parsar.oldSARh[ps]+parsar.AFH[ps]*(parsar.HP[ps]-parsar.oldSARh[ps])
        thisParSARlow[ps] = parsar.oldSARl[ps]-parsar.AFL[ps]*(parsar.oldSARl[ps]-parsar.LP[ps])
        if thisParSARhigh[ps]<parsar.HP[ps]:
            parsar.AFH[ps] = np.min([parsar.AFH[ps]+parsar.stepAF,parsar.maxAF[ps]])
            parsar.LP[ps] = np.min(thisPeriodVariations)
        if thisParSARlow[ps]>parsar.LP[ps]:
            parsar.AFL[ps] = np.min([parsar.AFH[ps]+parsar.stepAF,parsar.maxAF[ps]])
            parsar.HP[ps] = np.max(thisPeriodVariations)
        parsar.oldSARh[ps] = thisParSARhigh[ps]
        parsar.oldSARl[ps] = thisParSARlow[ps]
            
    return parsar, thisParSARhigh, thisParSARlow

def get_features_from_var_raw(data, features, DateTime, SymbolVar, nExS, mW, nE, m, thisAsset):
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
    
    secsInDay = 86400.0
    nE = SymbolVar.shape[0]
    var_feat_keys_manual = data.var_feat_keys
    em = init_ema_variations(data, SymbolVar, nExS, mW)
    n_feats_tsfresh = data.n_feats_tsfresh
    features_tsfresh = data.feature_keys_tsfresh
    
    parSar = init_parsar(data, SymbolVar[0])
    
    batch_size = 10000000
    par_batches = int(np.ceil(m/batch_size))
    l_index = 0
    # loop over batched
    for b in range(par_batches):
        # get m
        m_i = np.min([batch_size, m-b*batch_size])
        
        # init structures
        tsf = np.zeros((m_i, n_feats_tsfresh))
        EMA = np.zeros((m_i, em.shape[0]))
        variations = np.zeros((m_i))
        variance = np.zeros((m_i))
        maxValue = np.zeros((m_i))
        minValue = np.zeros((m_i))
        timeInterval = np.zeros((m_i))
        timeSecs = np.zeros((m_i))
        parSARhigh = np.zeros((m_i, len(data.parsars)))
        parSARlow = np.zeros((m_i, len(data.parsars)))
        try:
            for mm in range(m_i):
                
                startIndex = l_index+mm*mW
                endIndex = startIndex+nExS
                if endIndex==nE:
                    print(thisAsset+"! endIndex==nE")
                thisPeriod = range(startIndex,endIndex)
                thisPeriodVariations = SymbolVar[thisPeriod]
                newBidsIndex = range(endIndex-mW,endIndex)
                
                # tsfresh features
                c = 0
                for f in features_tsfresh:
                    params = data.AllFeatures[str(f)]
                    n_new_feats = params[-1]
                    params = data.AllFeatures[str(f)]
                    feats = feval(params[0],thisPeriodVariations,params[1:])
                    n_new_feats = params[-1]
                    tsf[mm,c:c+n_new_feats] = feats
                    c += n_new_feats
                        
                
                condition_ema = 69 in var_feat_keys_manual or 70 in var_feat_keys_manual or 71 in var_feat_keys_manual\
                    or 72 in var_feat_keys_manual or 73 in var_feat_keys_manual or 74 in var_feat_keys_manual\
                    or 75 in var_feat_keys_manual
                if condition_ema:
                    for i in newBidsIndex:
                        em = data.lbd*em+(1-data.lbd)*SymbolVar[i]
                if 77 in var_feat_keys_manual:
                    t0 = dt.datetime.strptime(DateTime[thisPeriod[0]].decode("utf-8"),'%Y.%m.%d %H:%M:%S')
                    te = dt.datetime.strptime(DateTime[thisPeriod[-1]].decode("utf-8"),'%Y.%m.%d %H:%M:%S')
                    timeInterval[mm] = (te-t0).seconds/nExS
                if 68 in var_feat_keys_manual:
                    variations[mm] = SymbolVar[thisPeriod[-1]]
                if condition_ema:
                    EMA[mm,:] = em
                if 76 in var_feat_keys_manual:
                    variance[mm] = np.var(thisPeriodVariations)
                if 83 in var_feat_keys_manual:
                    maxValue[mm] = np.max(thisPeriodVariations)
                if 84 in var_feat_keys_manual:
                    minValue[mm] = np.min(thisPeriodVariations)
                if 82 in var_feat_keys_manual:
                    timeSecs[mm] = (te.hour*60*60+te.minute*60+te.second)/secsInDay
                condition_parsar = 78 in var_feat_keys_manual or 79 in var_feat_keys_manual \
                    or 80 in var_feat_keys_manual or 81 in var_feat_keys_manual
                if condition_parsar:
                    parSar, parSARhigh[mm,:], parSARlow[mm,:] = update_parsar\
                        (data, parSar, minValue[mm], maxValue[mm], thisPeriodVariations)
        except IndexError:
            print("IndexError @ asset "+thisAsset+" mm "+str(mm)+" thisPeriod[-1] "+str(thisPeriod[-1])+
                  " b "+str(b))
            raise IndexError
            
        # end of for mm in range(m_i):
        l_index = startIndex+mW
        #print(l_index)
        toc = time.time()
        print("\t\t"+thisAsset+" mm="+str(b*batch_size+mm+1)+" of "+str(m)+". Total time: "+str(np.floor(toc-tic))+"s")
        # update features vector
        init_idx = b*batch_size
        end_idx = b*batch_size+m_i
        
        epsilon = 1e-10
        nF = 0
        features[init_idx:end_idx,nF:n_feats_tsfresh] = tsf
        nF += n_feats_tsfresh
        if 68 in var_feat_keys_manual:
            features[init_idx:end_idx,nF] = variations
            nF += 1
        if condition_ema:
            features[init_idx:end_idx,nF:nF+data.lbd.shape[0]] = EMA
            nF += data.lbd.shape[0]
        if 76 in var_feat_keys_manual:
            logVar = 10*np.log10(variance/data.std_var+1e-10)
            features[init_idx:end_idx,nF] = logVar
            nF += 1
        if 77 in var_feat_keys_manual:
            logInt = 10*np.log10(timeInterval/data.std_time+0.01)
            features[init_idx:end_idx,nF] = logInt
            nF += 1
        if condition_parsar:
            for ps in range(len(data.parsars)):
                features[init_idx:end_idx,nF] = parSARhigh[:,ps]
                nF += 1
                features[init_idx:end_idx,nF] = parSARlow[:,ps]
                nF += 1
        if 82 in var_feat_keys_manual:
            features[init_idx:end_idx,nF] = timeSecs
            nF += 1
        if 83 in var_feat_keys_manual:
            features[init_idx:end_idx,nF] = maxValue
            nF += 1
        if 84 in var_feat_keys_manual:
            features[init_idx:end_idx,nF] = minValue
            nF += 1
        if 85 in var_feat_keys_manual:
            features[init_idx:end_idx,nF] = np.sign(minValue)*np.log(np.abs(minValue)+epsilon)-\
                    np.sign(maxValue)*np.log(np.abs(maxValue)+epsilon)
            nF += 1
        condition_varOema = 86 in var_feat_keys_manual or 87 in var_feat_keys_manual\
            or 88 in var_feat_keys_manual or 89 in var_feat_keys_manual or 90 in var_feat_keys_manual\
            or 91 in var_feat_keys_manual or 92 in var_feat_keys_manual
        if condition_varOema:
            for i in range(data.lbd.shape[0]):          
                features[init_idx:end_idx,nF] = np.sign(variations)*np.log(np.abs(variations)+epsilon)-\
                    np.sign(EMA[:,i])*np.log(np.abs(EMA[:,i])+epsilon)
                nF += 1
            
    return features

# Wrapper over features extraction for one asset
def wrapper(var_feat_keys, feature_keys_tsfresh, filename_raw, feats_var_directory, 
                     separators_directory, ass, save_stats, save_stats_in_stats):
    """  """
    data = Data(var_feat_keys=var_feat_keys, feature_keys_tsfresh=feature_keys_tsfresh)
    f_raw = h5py.File(filename_raw,'r')
    thisAsset = data.AllAssets[str(ass)]
    print(thisAsset)
    nExS = data.nEventsPerStat
    mW = data.movingWindow
    nF = len(data.var_feat_keys)+data.n_feats_tsfresh
    
    group_raw = f_raw[thisAsset]
    
    filename_features = (feats_var_directory+thisAsset+'_feats_var_mW'+str(data.movingWindow)+'_nE'+
                            str(data.nEventsPerStat)+'.hdf5')
    file_features = h5py.File(filename_features,'a')
    filename_returns = (feats_var_directory+thisAsset+'_rets_var_mW'+str(data.movingWindow)+'_nE'+
                            str(data.nEventsPerStat)+'.hdf5')
    file_returns = h5py.File(filename_returns,'a')
    filename_symbols = (feats_var_directory+thisAsset+'_symbols_mW'+str(data.movingWindow)+'_nE'+
                            str(data.nEventsPerStat)+'.hdf5')
    file_symbols = h5py.File(filename_symbols,'a')
    
    if save_stats:
        filename_stats = (feats_var_directory+thisAsset+'_stats_mW'+str(data.movingWindow)+'_nE'+
                                str(data.nEventsPerStat)+'.p')
        #file_stats = h5py.File(filename_stats,'a')
        stats = {"means_in":0.0,
                 "stds_in":0.0,
                 "means_out":0.0,
                 "stds_out":0.0,
                 "m_in":0,
                 "m_out":0}

    # load separators
    separators = load_separators(data, thisAsset, separators_directory, from_txt=1)
    
    for s in range(0,len(separators)-1,2):#len(separators)-1
        
        nE = separators.index[s+1]-separators.index[s]+1
        # check if number of events is not enough to build two features and one return
        if nE-data.nEventsPerStat>=2*data.nEventsPerStat:
            print("\t"+thisAsset+" s {0:d} of {1:d}".format(int(s/2),int(len(separators)/2-1))+
                          ". From "+separators.DateTime.iloc[s]+" to "+separators.DateTime.iloc[s+1])
            
            DateTime = group_raw["DateTime"][separators.index[s]:separators.index[s+1]+1]
            SymbolBid = group_raw["SymbolBid"][separators.index[s]:separators.index[s+1]+1]
            SymbolAsk = group_raw["SymbolAsk"][separators.index[s]:separators.index[s+1]+1]
            init_i = nExS-1
            end_i = -(nExS-1)
            SymbolVar = SymbolBid[init_i:]-SymbolBid[:end_i]
            
            nE = SymbolVar.shape[0]
            
            init_date, end_date = get_init_end_dates(separators, s)
            
            group_name = get_group_name(thisAsset, init_date, end_date)
            
            m_in, m_out = get_io_number_samples(nE, nExS, mW)
            
            features, exist_feats = retrieve_features_structure(file_features, group_name, m_in, nF)
    
            if not exist_feats:
                features = get_features_from_var_raw(data, features, DateTime[init_i:], 
                                                     SymbolVar, nExS, mW, 
                                                     nE, m_in, thisAsset)

                stats_feats = calculate_stats_from_var_feats(features)

                save_stats_fn(file_features, group_name, stats_feats)

            else:
                print("\tFeatures already exist")
                stats_feats = retrieve_stats(file_features, group_name)
                # check that stats are not noneType
                if type(stats_feats['means'])==type(None) or type(stats_feats['stds'])==type(None):
                    print("WARNING! Stats None. Running feats over")
                    features = get_features_from_var_raw(data, features, DateTime[init_i:], 
                                                     SymbolVar, nExS, mW, 
                                                     nE, m_in, thisAsset)

                    stats_feats = calculate_stats_from_var_feats(features)
                    save_stats_fn(file_features, group_name, stats_feats)
            
            returns, exist_rets = retrieve_returns_structure(file_returns, group_name, m_out, len(data.lookAheadVector))
            
            if not exist_rets:
                returns = get_returns(data, returns, SymbolBid, nE, nExS, mW, m_in)

                stats_rets = calculate_stats_from_returns(returns)

                save_stats_fn(file_returns, group_name, stats_rets)
            else:
                print("\tReturns already exist")
                stats_rets = retrieve_stats(file_returns, group_name)
            DT, B, A, exist_symbs = retrieve_symbols_structure(file_symbols, group_name, m_out, len(data.lookAheadVector))
    
            if not exist_symbs:
                DT, B, A = get_symbols(data, DT, B, A, DateTime, SymbolBid, SymbolAsk, nE, nExS, mW, m_in)

            else:
                print("\tSymbols already exist")
            
            if save_stats:
                stats["means_in"] += m_in*stats_feats['means']
                stats["stds_in"] += m_in*stats_feats['stds']
                stats["means_out"] += m_out*stats_rets['means']
                stats["stds_out"] += m_out*stats_rets['stds']
                stats["m_in"] += m_in
                stats["m_out"] += m_out

        else:
            print("\ts {0:d} of {1:d}. Not enough entries. Skipped.".format(int(s/2),int(len(separators)/2-1)))
    
    if save_stats:
        stats["means_in"] = stats["means_in"]/stats["m_in"]
        stats["stds_in"] = stats["stds_in"]/stats["m_in"]
        stats["means_out"] = stats["means_out"]/stats["m_out"]
        stats["stds_out"] = stats["stds_out"]/stats["m_out"]
        
        # group for stats in features file
        group_feats = file_features[thisAsset]
        group_feats.attrs.create("means_in", stats["means_in"], dtype=float)
        group_feats.attrs.create("stds_in", stats["stds_in"], dtype=float)
        group_feats.attrs.create("m_in", stats["m_in"], dtype=int)
        
        # group for stats in returns file
        group_returns = file_returns[thisAsset]
        group_returns.attrs.create("means_out", stats["means_out"], dtype=float)
        group_returns.attrs.create("stds_out", stats["stds_out"], dtype=float)
        group_returns.attrs.create("m_out", stats["m_out"], dtype=int)        
        print("\tStats saved in HDF5 file")
        
        # group for stats in stats file
        if save_stats_in_stats:
            pickle.dump( stats, open( filename_stats, "wb" ))     
            print("\tStats saved in pickle file")
        
        #file_stats.close()
    
    file_features.close()
    file_returns.close()
    file_symbols.close()
    f_raw.close()
    print(thisAsset+" DONE")
    
    return None

def calculate_stats_from_var_feats(features):
    """  """
    nF = features.shape[1]
    stats = {}
    stats["means"] = np.zeros((1,nF))
    stats["stds"] = np.zeros((1,nF))
    stats["m"] = features.shape[0]
    
    print("\t getting means and stds from features")
    # loop over channels
    stats["means"][0,:] = np.mean(features,axis=0,keepdims=1)
    stats["stds"][0,:] = np.std(features,axis=0,keepdims=1)
    
    return stats

def get_returns(data, returns, symbols, nE, nExS, mW, m_in):
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
    initRange = int(nExS/mW)
    
    np_00 = initRange*data.movingWindow-1
    np_e0 = m_in*data.movingWindow-1
    
    indexOrigins = [i for i in range(np_00,np_e0,data.movingWindow)]

    for nr in range(len(data.lookAheadVector)):
        unp_0e = int(initRange*data.movingWindow+np.floor(data.nEventsPerStat*data.lookAheadVector[nr])-2)
        unp_ee = int(np.min([m_in*data.movingWindow+np.floor(
                data.nEventsPerStat*data.lookAheadVector[nr])-2,nE]))
        indexEnds = [i for i in range(unp_0e,unp_ee,data.movingWindow)]
        #fill ends wth last value
        for i in range(len(indexEnds),len(indexOrigins)):
            indexEnds.append(nE-1)
        returns[:,nr] = symbols[indexEnds]-symbols[indexOrigins]
    
    return returns

def get_symbols(data, DT, B, A, DateTime, SymbolBid, SymbolAsk, nE, nExS, mW, m_in):
    """  """
    initRange = int(nExS/mW)
    
    np_00 = initRange*data.movingWindow-1
    np_e0 = m_in*data.movingWindow-1
    
    indexOrigins = [i for i in range(np_00,np_e0,data.movingWindow)]
    DT[:,0] = DateTime[indexOrigins]
    B[:,0] = SymbolBid[indexOrigins]
    A[:,0] = SymbolAsk[indexOrigins]
    for nr in range(len(data.lookAheadVector)):
        unp_0e = int(initRange*data.movingWindow+np.floor(data.nEventsPerStat*data.lookAheadVector[nr])-2)
        unp_ee = int(np.min([m_in*data.movingWindow+np.floor(
                data.nEventsPerStat*data.lookAheadVector[nr])-2,nE]))
        indexEnds = [i for i in range(unp_0e,unp_ee,data.movingWindow)]
        #fill ends wth last value
        for i in range(len(indexEnds),len(indexOrigins)):
            indexEnds.append(nE-1)
        DT[:,nr+1] = DateTime[indexEnds]
        B[:,nr+1] = SymbolBid[indexEnds]
        A[:,nr+1] = SymbolAsk[indexEnds]
        
    return DT, B, A
def calculate_stats_from_returns(returns):
    """  """
    nR = returns.shape[1]
    stats = {}
    stats["means"] = np.zeros((1,nR))
    stats["stds"] = np.zeros((1,nR))
    stats["m"] = returns.shape[0]
    
    print("\t getting means and stds from returns")
    # get output stats
    stats["stds"] = np.std(returns,axis=0)
    stats["means"] = np.mean(returns,axis=0)
    
    return stats

def retrieve_features_structure(file_features, group_name, m_in, nF):
    """ Retrieve features structure """
    
    if group_name not in file_features:
        group = file_features.create_group(group_name)
        features = group.create_dataset("features", (m_in,nF),dtype=float)
        exist = False
    else:
        features = file_features[group_name]["features"]
        exist = True
    
    return features, exist

def retrieve_returns_structure(file_returns, group_rets_name, m_out, nR):
    """  """
    if group_rets_name not in file_returns:
        group_rets = file_returns.create_group(group_rets_name)
        returns = group_rets.create_dataset("returns", (m_out,nR),dtype=float)
        exist = False
    else:
        returns = file_returns[group_rets_name]["returns"]
        exist = True
    
    return returns, exist
    
def retrieve_symbols_structure(file_symbs, group_symbs_name, m_out, nR):
    """  """
    if group_symbs_name not in file_symbs:
        group_symbs = file_symbs.create_group(group_symbs_name)
        DT = group_symbs.create_dataset("DT", (m_out,nR+1),dtype='S19')
        B = group_symbs.create_dataset("B", (m_out,nR+1),dtype=float)
        A = group_symbs.create_dataset("A", (m_out,nR+1),dtype=float)
        exist = False
    else:
        DT = file_symbs[group_symbs_name]["DT"]
        B = file_symbs[group_symbs_name]["B"]
        A = file_symbs[group_symbs_name]["A"]
        exist = True
    
    return DT, B, A, exist

def save_stats_fn(file, group_name, stats):
    """  """
    file[group_name].attrs.create("means", stats['means'], dtype=float)
    file[group_name].attrs.create("stds", stats['stds'], dtype=float)
    file[group_name].attrs.create("m", stats['m'], dtype=int)
    
def retrieve_stats(file, group_name):
    """  """
    stats = {}
    stats['means'] = file[group_name].attrs.get("means")
    stats['stds'] = file[group_name].attrs.get("stds")
    stats['m'] = file[group_name].attrs.get("m")
    return stats

# Helper functions
def get_init_end_dates(separators, s):
    """  """
    init_date = dt.datetime.strftime(dt.datetime.strptime(
                separators.DateTime.iloc[s],'%Y.%m.%d %H:%M:%S'),'%y%m%d%H%M%S')
    end_date = dt.datetime.strftime(dt.datetime.strptime(
               separators.DateTime.iloc[s+1],'%Y.%m.%d %H:%M:%S'),'%y%m%d%H%M%S')
    return init_date, end_date

def get_group_name(thisAsset, init_date, end_date):
    """ Geat group name in HDF5 file containing the features vector """
    return thisAsset+'/'+init_date+end_date+'/'

def get_io_number_samples(nE, nExS, mW):
    """  """
    # number of features and number of returns
    m_in = int(np.floor((nE/nExS-1)*nExS/mW)+1)
    m_out = int(m_in-nExS/mW)
    return m_in, m_out

def print_file_groups():
    """ Print groups of HDF5 file """
    data=Data(movingWindow=200,
                  nEventsPerStat=2000)
    hdf5_directory = 'D:/SDC/py/HDF5/'
    filename_prep_IO = (hdf5_directory+'IO_mW'+str(data.movingWindow)+'_nE'+
                            str(data.nEventsPerStat)+'_nF'+str(data.nFeatures)+'.hdf5')
    f_prep_IO = h5py.File(filename_prep_IO,'r')
    for ass in data.assets:
        thisAsset = data.AllAssets[str(ass)]
        # open file for read
       
        if thisAsset not in f_prep_IO:
            # init total stats
            raise ValueError
        else:
            # retrive ass group if exists
            ass_group = f_prep_IO[thisAsset]
            print(ass_group)
            for member in ass_group:        
                print(member)
    f_prep_IO.close()

def get_number_samples(window_size, sprite_length, n_events):
    """ 
    Get number of samples given window size, sprite length and number of events 
    Args:
        - (int) window size
        - (int) sprite length
        - (int) number of events
    Return:
        (int) number of samples
    """
    
    return int(np.floor((n_events/window_size-1)*window_size/sprite_length)+1)

def get_features_tsfresh():
    """
    Extract and save most common features based on TSFRESH tool
    """
    # config stuff
    hdf5_directory = 'D:/SDC/py/HDF5/'
    save_stats = True
    # init stuff
    filename_raw = hdf5_directory+'tradeinfo.hdf5'
    separators_directory = hdf5_directory+'separators/'
    
    f_raw = h5py.File(filename_raw,'r')
    
    data=Data(movingWindow=100,
              nEventsPerStat=1000,
              dateTest = [                                          '2018.03.09',
                '2018.03.12','2018.03.13','2018.03.14','2018.03.15','2018.03.16',
                '2018.03.19','2018.03.20','2018.03.21','2018.03.22','2018.03.23',
                '2018.03.26','2018.03.27','2018.03.28','2018.03.29','2018.03.30',
                '2018.04.02','2018.04.03','2018.04.04','2018.04.05','2018.04.06',
                '2018.04.09','2018.04.10','2018.04.11','2018.04.12','2018.04.13',
                '2018.04.16','2018.04.17','2018.04.18','2018.04.19','2018.04.20',
                '2018.04.23','2018.04.24','2018.04.25','2018.04.26','2018.04.27',
                '2018.04.30','2018.05.01','2018.05.02','2018.05.03','2018.05.04',
                '2018.05.07','2018.05.08','2018.05.09','2018.05.10','2018.05.11',
                '2018.05.14','2018.05.15','2018.05.16','2018.05.17','2018.05.18',
                '2018.05.21','2018.05.22','2018.05.23','2018.05.24','2018.05.25',
                '2018.05.28','2018.05.29','2018.05.30','2018.05.31','2018.06.01',
                '2018.06.04','2018.06.05','2018.06.06','2018.06.07','2018.06.08',
                '2018.06.11','2018.06.12','2018.06.13','2018.06.14','2018.06.15',
                '2018.06.18','2018.06.19','2018.06.20','2018.06.21','2018.06.22',
                '2018.06.25','2018.06.26','2018.06.27','2018.06.28','2018.06.29',
                '2018.07.02','2018.07.03','2018.07.04','2018.07.05','2018.07.06',
                '2018.07.09','2018.07.10','2018.07.11','2018.07.12','2018.07.13',
                '2018.07.30','2018.07.31','2018.08.01','2018.08.02','2018.08.03',
                '2018.08.06','2018.08.07','2018.08.08','2018.08.09','2018.08.10']+
               ['2018.08.13','2018.08.14','2018.08.15','2018.08.16','2018.08.17',
                '2018.08.20','2018.08.21','2018.08.22','2018.08.23','2018.08.24',
                '2018.08.27','2018.08.28','2018.08.29','2018.08.30','2018.08.31',
                '2018.09.03','2018.09.04','2018.09.05','2018.09.06','2018.09.07',
                '2018.09.10','2018.09.11','2018.09.12','2018.09.13','2018.09.14',
                '2018.09.17','2018.09.18','2018.09.19','2018.09.20','2018.09.21',
                '2018.09.24','2018.09.25','2018.09.26','2018.09.27'])
    filename_features_tsf = (hdf5_directory+'feats_tsf_mW'+str(data.movingWindow)+'_nE'+
                            str(data.nEventsPerStat)+'.hdf5')
    batch_size = 50000
    window_size = data.nEventsPerStat
    sprite_length = data.movingWindow
    nMaxChannels = int(window_size/sprite_length)
    features_tsfresh = data.feature_keys_manual
    n_feats_tsfresh = data.n_feats_tsfresh
    file_features_tsf = h5py.File(filename_features_tsf,'a')
    tic_t = time.time()
    # run over assets
    for ass in data.assets:
        tic_ass = time.time()
        # retrieve this asset's name
        thisAsset = data.AllAssets[str(ass)]
        print(str(ass)+". "+thisAsset)
        # load separators
        separators = load_separators(data, thisAsset, 
                                     separators_directory, 
                                     from_txt=1)
        # get raw data
        SymbolBid = f_raw[thisAsset]["SymbolBid"]
        # init normalization stats
        stats = {"means_t_in":np.zeros((nMaxChannels,n_feats_tsfresh)),
                "stds_t_in":np.zeros((nMaxChannels,n_feats_tsfresh)),
                "m_t_in":0}
        # run over separators
        for s in range(0,len(separators)-1,2):#
            tic = time.time()
            print("\t chuck {0:d} of {1:d}".format(int(s/2),int(len(separators)/2-1))+
                      ". From "+separators.DateTime.iloc[s]+" to "+separators.DateTime.iloc[s+1])
            
            # number of events in this chunck
            n_events = separators.index[s+1]-separators.index[s]+1
            if n_events>=2*window_size:
                print("\t Getting features from raw data...")
                tic = time.time()
                tic_chunck = tic
                # init and end dates in string format
                init_date = dt.datetime.strftime(dt.datetime.strptime(
                        separators.DateTime.iloc[s],'%Y.%m.%d %H:%M:%S'),'%y%m%d%H%M%S')
                end_date = dt.datetime.strftime(dt.datetime.strptime(
                        separators.DateTime.iloc[s+1],'%Y.%m.%d %H:%M:%S'),'%y%m%d%H%M%S')
                # hdf5 group name
                group_name_chunck = thisAsset+'/'+init_date+end_date
                # create or retrieve group
                # create new gruop if not yet in file
                # bids in this chunck
                events = SymbolBid[separators.index[s]:separators.index[s+1]+1]
                # number of samples
                m = int(np.floor((n_events/window_size-1)*window_size/sprite_length)+1)
                # group ts in m chunks
                # number of batches
                batches = int(np.ceil(m/batch_size))
                l_index = 0
                m_counter = 0
                features = np.zeros((m,n_feats_tsfresh))
                # loop over batched
                for b in range(batches):
                    print("\t batch "+str(b)+" out of "+str(batches-1))
                    # get batch size
                    m_i = np.min([batch_size, m-b*batch_size])
                    #print("Batch "+str(b)+" out of "+str(batches-1)+", m_i="+str(m_i))
                    # init and end of event index
                    i_event = l_index
                    e_event = i_event+(m_i-1)*sprite_length+window_size
                    # serial input
                    x_ser = events[i_event:e_event]
                    #print("x_ser length="+str(x_ser.shape[0]))
                    # loop over this batch samples
                    for mm in range(m_i):
                        # get sample indexes
                        i_sample = mm*sprite_length
                        e_sample = i_sample+window_size
                        #print("i_sample="+str(i_sample)+" e_sample="+str(e_sample))
                        x = x_ser[i_sample:e_sample]
                        c = 0
                        # extract features
                        for f in features_tsfresh:
                            group_name_feat = group_name_chunck+'/'+str(f)
                            if group_name_feat not in file_features_tsf:
                                params = data.AllFeatures[str(f)]
                                new_feats = feval(params[0],x,params[1:])
                                n_new_feats = params[-1]
                                features[m_counter,c:c+n_new_feats] = new_feats                                
                                c += n_new_feats
                        #assert(c==n_feats_tsfresh)
                        m_counter += 1
                    # serial loop
                    l_index = e_event-window_size+sprite_length
                # end of for b in range(batches):
                print("\t time for feature extraction: "+str(time.time()-tic))
                #print("\r"+"e_event="+str(e_event)+" events="+str(events.shape[0]))#, sep=' ', end='\n', flush=True
                
                c = 0
                # save features in HDF5 file
                for f in features_tsfresh:
                    group_name_feat = group_name_chunck+'/'+str(f)
                    params = data.AllFeatures[str(f)]
                    n_new_feats = params[-1]
                    #if group_name_feat in file_features_tsf:
                    #    del file_features_tsf[group_name_feat]
                    if group_name_feat not in file_features_tsf:
                        group_chunck = file_features_tsf.create_group(group_name_feat)
                        group_features = group_chunck.create_dataset("feature", (m,n_new_feats),dtype=float)
                        group_features[:,:] = features[:,c:c+n_new_feats]
                    else: # load features from HDF5 file if they are saved already
                        params = data.AllFeatures[str(f)]
                        n_new_feats = params[-1]
                        group_chunck = file_features_tsf[group_name_feat]
                        features[:,c:c+n_new_feats] = group_chunck["feature"]
                    c += n_new_feats
                # end of for f in features_tsfresh:
                # Calculate normalization stats
                tic = time.time()
                # open temporal file for variations
                try:
                    # create file
                    ft = h5py.File(hdf5_directory+'temp.hdf5','w')
                    # create group
                    group_temp = ft.create_group('temp')
                    # reserve memory space for variations and normalized variations
                    variations = group_temp.create_dataset("variations", (features.shape[0],features.shape[1],nMaxChannels), 
                                                           dtype=float)
                    print("\t getting variations")
                    # loop over channels
                    for r in range(nMaxChannels):
                        variations[r+1:,:,r] = features[:-(r+1),:]
                    print("\t time for variations: "+str(time.time()-tic))
                    # init stats    
                    means_in = np.zeros((nMaxChannels,features.shape[1]))
                    stds_in = np.zeros((nMaxChannels,features.shape[1]))
                    print("\t getting means and stds")
                    # loop over channels
                    for r in range(nMaxChannels):
                        #nonZeros = variations[:,0,r]!=999
                        #print(np.sum(nonZeros))
                        means_in[r,:] = np.mean(variations[nMaxChannels:,:,r],axis=0,keepdims=1)
                        stds_in[r,:] = np.std(variations[nMaxChannels:,:,r],axis=0,keepdims=1)  
                    print("\t time for stats: "+str(time.time()-tic))
                except KeyboardInterrupt:
                    ft.close()
                    print("KeyboardInterrupt! Closing file and exiting.")
                    raise KeyboardInterrupt
                ft.close()
                # end of normalization stats
                # save normalization stats
                for f in features_tsfresh:
                    group_name_feat = group_name_chunck+'/'+str(f)
                    n_new_feats = data.AllFeatures[str(f)][-1]
                    group_feature = file_features_tsf[group_name_feat]
                    # save means and variances as atributes
                    group_feature.attrs.create("means_in", means_in[:,c:c+n_new_feats], dtype=float)
                    group_feature.attrs.create("stds_in", stds_in[:,c:c+n_new_feats], dtype=float)
                    c += n_new_feats
                # update total stats
                stats["means_t_in"] += m*means_in
                stats["stds_t_in"] += m*stds_in
                stats["m_t_in"] += m
                means_in
                print("\t this chunck time: "+str(np.floor(time.time()-tic_chunck))+"s")
            else:
                print("\t chunck {0:d} of {1:d}. Not enough entries. Skipped.".format(int(s/2),int(len(separators)/2-1)))
            # end of if n_events>=2*window_size:
        # end of for s in range(0,len(separators)-1,2):
        means_t_in = stats["means_t_in"]/stats["m_t_in"]
        stds_t_in = stats["stds_t_in"]/stats["m_t_in"]
        if save_stats:
            for f in features_tsfresh:
                    group_name_feat = thisAsset
                    n_new_feats = data.AllFeatures[str(f)][-1]
                    group_feature = file_features_tsf[group_name_feat]
                    # save means and variances as atributes
                    group_feature.attrs.create("means_in"+str(f), means_t_in[:,c:c+n_new_feats], dtype=float)
                    group_feature.attrs.create("stds_in"+str(f), stds_t_in[:,c:c+n_new_feats], dtype=float)
                    c += n_new_feats
    print("\t this asset time: "+str(np.floor(time.time()-tic_ass)/60)+" minutes")
    # end of for ass in data.assets:
    file_features_tsf.close()
    print("DONE. Total time: "+str(np.floor(time.time()-tic_t)/60)+" minutes")
    
def feval(funcName, *args):
    return eval(funcName)(*args)

def complex_agg(x, agg):
    if agg == "real":
        return x.real
    elif agg == "imag":
        return x.imag
    elif agg == "abs":
        return np.abs(x)
    elif agg == "angle":
        return np.angle(x, deg=True)

def _aggregate_on_chunks(x, f_agg, chunk_len):
    """
    Takes the time series x and constructs a lower sampled version of it by applying the aggregation function f_agg on
    consecutive chunks of length chunk_len

    :param x: the time series to calculate the aggregation of
    :type x: pandas.Series
    :param f_agg: The name of the aggregation function that should be an attribute of the pandas.Series
    :type f_agg: str
    :param chunk_len: The size of the chunks where to aggregate the time series
    :type chunk_len: int
    :return: A list of the aggregation function over the chunks
    :return type: list
    """
    return [getattr(x[i * chunk_len: (i + 1) * chunk_len], f_agg)() for i in range(int(np.ceil(len(x) / chunk_len)))]    

def quantile(x, param):
    """
    Calculates the q quantile of x. This is the value of x greater than q% of the ordered values from x.

    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :param q: the quantile to calculate
    :type q: float
    :return: the value of this feature
    :return type: float
    """
    #x = pd.Series(x)
    return np.percentile(x, 100*param[0])

def fft_coefficient(x, param):
    """
    Calculates the fourier coefficients of the one-dimensional discrete Fourier Transform for real input by fast
    fourier transformation algorithm

    .. math::
        A_k =  \\sum_{m=0}^{n-1} a_m \\exp \\left \\{ -2 \\pi i \\frac{m k}{n} \\right \\}, \\qquad k = 0,
        \\ldots , n-1.

    The resulting coefficients will be complex, this feature calculator can return the real part (attr=="real"),
    the imaginary part (attr=="imag), the absolute value (attr=""abs) and the angle in degrees (attr=="angle).

    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :param param: contains dictionaries {"coeff": x, "attr": s} with x int and x >= 0, s str and in ["real", "imag",
        "abs", "angle"]
    :type param: list
    :return: the different feature values
    :return type: pandas.Series
    """

    fft = np.fft.rfft(x)
    res = complex_agg(fft[param[0]], param[1])
    return res

def linear_trend(x, param):
    """
    Calculate a linear least-squares regression for the values of the time series versus the sequence from 0 to
    length of the time series minus one.
    This feature assumes the signal to be uniformly sampled. It will not use the time stamps to fit the model.
    The parameters control which of the characteristics are returned.

    Possible extracted attributes are "pvalue", "rvalue", "intercept", "slope", "stderr", see the documentation of
    linregress for more information.

    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :param param: contains dictionaries {"attr": x} with x an string, the attribute name of the regression model
    :type param: list
    :return: the different feature values
    :return type: pandas.Series
    """
    # todo: we could use the index of the DataFrame here
    attr = param[0]
    linReg = linregress(range(len(x)), x)

    return getattr(linReg, attr)

def agg_linear_trend(x, param):
    """
    Calculates a linear least-squares regression for values of the time series that were aggregated over chunks versus
    the sequence from 0 up to the number of chunks minus one.

    This feature assumes the signal to be uniformly sampled. It will not use the time stamps to fit the model.

    The parameters attr controls which of the characteristics are returned. Possible extracted attributes are "pvalue",
    "rvalue", "intercept", "slope", "stderr", see the documentation of linregress for more information.

    The chunksize is regulated by "chunk_len". It specifies how many time series values are in each chunk.

    Further, the aggregation function is controlled by "f_agg", which can use "max", "min" or , "mean", "median"

    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :param param: contains dictionaries {"attr": x, "chunk_len": l, "f_agg": f} with x, f an string and l an int
    :type param: list
    :return: the different feature values
    :return type: pandas.Series
    """
    # todo: we could use the index of the DataFrame here
    calculated_agg = {}
    f_agg = param[0]
    chunk_len = param[1]
    attr = param[2]
    
    aggregate_result = _aggregate_on_chunks(x, f_agg, chunk_len)
    if f_agg not in calculated_agg or chunk_len not in calculated_agg[f_agg]:
        if chunk_len >= len(x):
            calculated_agg[f_agg] = {chunk_len: np.NaN}
        else:
            lin_reg_result = linregress(range(len(aggregate_result)), aggregate_result)
            calculated_agg[f_agg] = {chunk_len: lin_reg_result}

    if chunk_len >= len(x):
        res_data = np.NaN
    else:
        res_data = getattr(calculated_agg[f_agg][chunk_len], attr)
        
    return res_data

def first_location_of_minimum(x):
    """
    Returns the first location of the minimal value of x.
    The position is calculated relatively to the length of x.

    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :return: the value of this feature
    :return type: float
    """
    if not isinstance(x, (np.ndarray, pd.Series)):
        x = np.asarray(x)
    return np.argmin(x) / len(x) if len(x) > 0 else np.NaN

def energy_ratio_by_chunks(x, num_segments, segment_focus):
    """
    Calculates the sum of squares of chunk i out of N chunks expressed as a ratio with the sum of squares over the whole
    series.

    Takes as input parameters the number num_segments of segments to divide the series into and segment_focus
    which is the segment number (starting at zero) to return a feature on.

    If the length of the time series is not a multiple of the number of segments, the remaining data points are
    distributed on the bins starting from the first. For example, if your time series consists of 8 entries, the
    first two bins will contain 3 and the last two values, e.g. `[ 0.,  1.,  2.], [ 3.,  4.,  5.]` and `[ 6.,  7.]`.

    Note that the answer for `num_segments = 1` is a trivial "1" but we handle this scenario
    in case somebody calls it. Sum of the ratios should be 1.0.

    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :param param: contains dictionaries {"num_segments": N, "segment_focus": i} with N, i both ints
    :return: the feature values
    :return type: list of tuples (index, data)
    """
    full_series_energy = np.sum(x ** 2)

    assert segment_focus < num_segments
    assert num_segments > 0

    res_data = np.sum(np.array_split(x, num_segments)[segment_focus] ** 2.0)/full_series_energy

    return res_data

def change_quantiles(x, ql, qh, isabs, f_agg):
    """
    First fixes a corridor given by the quantiles ql and qh of the distribution of x.
    Then calculates the average, absolute value of consecutive changes of the series x inside this corridor.

    Think about selecting a corridor on the
    y-Axis and only calculating the mean of the absolute change of the time series inside this corridor.

    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :param ql: the lower quantile of the corridor
    :type ql: float
    :param qh: the higher quantile of the corridor
    :type qh: float
    :param isabs: should the absolute differences be taken?
    :type isabs: bool
    :param f_agg: the aggregator function that is applied to the differences in the bin
    :type f_agg: str, name of a numpy function (e.g. mean, var, std, median)

    :return: the value of this feature
    :return type: float
    """
    if ql >= qh:
        ValueError("ql={} should be lower than qh={}".format(ql, qh))

    div = np.diff(x)
    if isabs:
        div = np.abs(div)
    # All values that originate from the corridor between the quantiles ql and qh will have the category 0,
    # other will be np.NaN
    try:
        bin_cat = pd.qcut(x, [ql, qh], labels=False)
        bin_cat_0 = bin_cat == 0
    except ValueError:  # Occurs when ql are qh effectively equal, e.g. x is not long enough or is too categorical
        return 0
    # We only count changes that start and end inside the corridor
    ind = (bin_cat_0 & np.roll(bin_cat_0, 1))[1:]
    if sum(ind) == 0:
        return 0
    else:
        ind_inside_corridor = np.where(ind == 1)
        aggregator = getattr(np, f_agg)
        return aggregator(div[ind_inside_corridor])
    
def index_mass_quantile(x, q):
    """
    Those apply features calculate the relative index i where q% of the mass of the time series x lie left of i.
    For example for q = 50% this feature calculator will return the mass center of the time series

    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :param param: contains dictionaries {"q": x} with x float
    :type param: list
    :return: the different feature values
    :return type: pandas.Series
    """

    x = np.asarray(x)
    abs_x = np.abs(x)
    s = sum(abs_x)

    if s == 0:
        # all values in x are zero or it has length 0
        return np.NaN
    else:
        # at least one value is not zero
        mass_centralized = np.cumsum(abs_x) / s
        return (np.argmax(mass_centralized >= q)+1)/len(x)
    
def number_cwt_peaks(x, n):
    """
    This feature calculator searches for different peaks in x. To do so, x is smoothed by a ricker wavelet and for
    widths ranging from 1 to n. This feature calculator returns the number of peaks that occur at enough width scales
    and with sufficiently high Signal-to-Noise-Ratio (SNR)

    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :param n: maximum width to consider
    :type n: int
    :return: the value of this feature
    :return type: int
    """
    return len(find_peaks_cwt(vector=x, widths=np.array(list(range(1, n + 1))), wavelet=ricker))

def last_location_of_maximum(x):
    """
    Returns the relative last location of the maximum value of x.
    The position is calculated relatively to the length of x.

    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :return: the value of this feature
    :return type: float
    """
    x = np.asarray(x)
    return 1.0 - np.argmax(x[::-1]) / len(x) if len(x) > 0 else np.NaN

def first_location_of_maximum(x):
    """
    Returns the first location of the maximum value of x.
    The position is calculated relatively to the length of x.

    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :return: the value of this feature
    :return type: float
    """
    if not isinstance(x, (np.ndarray, pd.Series)):
        x = np.asarray(x)
    return np.argmax(x) / len(x) if len(x) > 0 else np.NaN

def last_location_of_minimum(x):
    """
    Returns the last location of the minimal value of x.
    The position is calculated relatively to the length of x.

    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :return: the value of this feature
    :return type: float
    """
    x = np.asarray(x)
    return 1.0 - np.argmin(x[::-1]) / len(x) if len(x) > 0 else np.NaN

def mean_change(x):
    """
    Returns the mean over the absolute differences between subsequent time series values which is

    .. math::

        \\frac{1}{n} \\sum_{i=1,\ldots, n-1}  x_{i+1} - x_{i}

    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :return: the value of this feature
    :return type: float
    """
    return np.mean(np.diff(x))

def abs_energy(x, param):
    """
    Returns the absolute energy of the time series which is the sum over the squared values

    .. math::

        E = \\sum_{i=1,\ldots, n} x_i^2

    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :return: the value of this feature
    :return type: float
    """
    if not isinstance(x, (np.ndarray, pd.Series)):
        x = np.asarray(x)
    return np.dot(x, x)

def sum_values(x, param):
    """
    Calculates the sum over the time series values

    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :return: the value of this feature
    :return type: bool
    """
    if len(x) == 0:
        return 0

    return np.sum(x)

def mean(x, param):
    """
    Returns the mean of x

    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :return: the value of this feature
    :return type: float
    """
    return np.mean(x)

def minimum(x, param):
    """
    Calculates the lowest value of the time series x.

    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :return: the value of this feature
    :return type: float
    """
    return np.min(x)

def median(x, param):
    """
    Returns the median of x

    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :return: the value of this feature
    :return type: float
    """
    return np.median(x)

def c3(x, param):
    """
    This function calculates the value of

    .. math::

        \\frac{1}{n-2lag} \sum_{i=0}^{n-2lag} x_{i + 2 \cdot lag}^2 \cdot x_{i + lag} \cdot x_{i}

    which is

    .. math::

        \\mathbb{E}[L^2(X)^2 \cdot L(X) \cdot X]

    where :math:`\\mathbb{E}` is the mean and :math:`L` is the lag operator. It was proposed in [1] as a measure of
    non linearity in the time series.

    .. rubric:: References

    |  [1] Schreiber, T. and Schmitz, A. (1997).
    |  Discrimination power of measures for nonlinearity in a time series
    |  PHYSICAL REVIEW E, VOLUME 55, NUMBER 5

    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :param lag: the lag that should be used in the calculation of the feature
    :type lag: int
    :return: the value of this feature
    :return type: float
    """
    if not isinstance(x, (np.ndarray, pd.Series)):
        x = np.asarray(x)
    n = x.size
    if 2 * param[0] >= n:
        return 0
    else:
        return np.mean((np.roll(x, 2 * -param[0]) * np.roll(x, -param[0]) * x)[0:(n - 2 * param[0])])
    
def cwt_coefficients(x, param):
    """
    Calculates a Continuous wavelet transform for the Ricker wavelet, also known as the "Mexican hat wavelet" which is
    defined by

    .. math::
        \\frac{2}{\\sqrt{3a} \\pi^{\\frac{1}{4}}} (1 - \\frac{x^2}{a^2}) exp(-\\frac{x^2}{2a^2})

    where :math:`a` is the width parameter of the wavelet function.

    This feature calculator takes three different parameter: widths, coeff and w. The feature calculater takes all the
    different widths arrays and then calculates the cwt one time for each different width array. Then the values for the
    different coefficient for coeff and width w are returned. (For each dic in param one feature is returned)

    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :param param: contains dictionaries {"widths":x, "coeff": y, "w": z} with x array of int and y,z int
    :type param: list
    :return: the different feature values
    :return type: pandas.Series
    """
    
    widths = param[0]
    coeff = param[1]
    w = param[2]
    res = np.zeros((len(w)))
    #print(coeff)
    calculated_cwt_for_widths = cwt(x, ricker, widths)
    #print(calculated_cwt_for_widths)
    for r in range(len(w)):
        i = widths.index(w[r])
        if calculated_cwt_for_widths.shape[1] <= coeff[r]:
            res[r] = np.NaN
        else:
            res[r] = calculated_cwt_for_widths[i, coeff[r]]

    return res

def maximum(x,param):
    """
    Calculates the highest value of the time series x.

    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :return: the value of this feature
    :return type: float
    """
    return np.max(x)

if __name__=='__main__':
    #get_features()
    pass