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
import re
import h5py
from scipy.stats import linregress
from scipy.signal import cwt, find_peaks_cwt, ricker
import pandas as pd

from kaissandra.config import retrieve_config, Config as C
from kaissandra.local_config import local_vars

import zipfile

def change_feature_name(oldfeatname, newfeatname, 
                        assets=[1,2,3,4,7,8,10,11,12,13,14,15,16,17,19,27,28,29,30,31,32], 
                        relations=['direct','inverse'], mW=500, nExS=5000, sources=['bid','ask']):
    """  """
    #from tqdm import tqdm
    
    rootdir = local_vars.hdf5_directory
    assetsdirs = ['mW'+str(mW)+'_nE'+str(nExS)+'/'+rel+'/'+s+'/'+C.AllAssets[str(ass)]+'/' for rel in relations for s in sources for ass in assets]
    statsdirs = [local_vars.stats_modular_directory+'mW'+str(mW)+'nE'+str(nExS)+'/'+rel+'/' for rel in relations]
    # change names from stats directory             
    for statdir in statsdirs:
        list_stats = sorted(os.listdir(statdir))
        for statfile in list_stats:
            if oldfeatname in statfile:
                print(statfile)
                list_name_splits = statfile.split(oldfeatname)
                newname = list_name_splits[0]+newfeatname+list_name_splits[1]
                print(newname)
                os.rename(statdir+statfile, statdir+newname)
    
    # change names from features directories
    for assetsdir in assetsdirs:#tqdm(assetsdirs, mininterval=1):#os.walk(path)
        list_all_dirs = sorted(os.listdir(rootdir+assetsdir))
        for file in list_all_dirs:
            if os.path.isdir(rootdir+assetsdir+file):
                list_feats = sorted(os.listdir(rootdir+assetsdir+file+'/'))
                for featfile in list_feats:
                    if oldfeatname in featfile:
                        print(featfile)
                        list_name_splits = featfile.split(oldfeatname)
                        newname = list_name_splits[0]+newfeatname+list_name_splits[1]
                        print(newname)
                        # modify HDF5 vector name
                        
                        os.rename(rootdir+assetsdir+file+'/'+featfile, 
                                  rootdir+assetsdir+file+'/'+newname)
                    elif newfeatname in featfile:
                        # make sure vector name is correc
                        filenamesplit = featfile.split('.')
                        if filenamesplit[-1] == 'hdf5':
                            groupname = (filenamesplit[0]).split('_')[0]
#                            print(featfile)
                            ft = h5py.File(rootdir+assetsdir+file+'/'+featfile,'a')
                            if groupname not in ft:
                                # do it manually
#                                print(groupname)
                                print(rootdir+assetsdir+file+'/'+featfile)
                                vector = ft['BOLDOWN10']['BOLDOWN10'][:]
                                newvectorfeat = ft.create_group(groupname).\
                                create_dataset(groupname, vector.shape, dtype=float)
                                newvectorfeat[:] = vector
                                #print(vector)
#                            elif groupname in ft and 'BOLDOWN10' in ft:
#                                print(rootdir+assetsdir+file+'/'+featfile)
#                                ft[groupname][groupname][:] = ft['BOLDOWN10']['BOLDOWN10'][:]
#                                print("ft[groupname][groupname][:] = ft['BOLDOWN10']['BOLDOWN10'][:]")
                                #a = p
                            ft.close()
                        
                        
#                        group_temp = ft.create_group('temp')
#                        # load features
#                        filedirname = groupdirname+C.PF[f][0]+'_0.hdf5'
#                        feature_file = h5py.File(filedirname,'r')
#                #        groupname = '_'.join(groupdirname.split('/'))
#                        features = feature_file[C.PF[f][0]][C.PF[f][0]]
#                        # create variations vector
#                        variations = group_temp.create_dataset("variations", (features.shape[0],nChannels), dtype=float)
#                        pass
            else:
                if oldfeatname in file:
                    print(file)
                    list_name_splits = file.split(oldfeatname)
                    newname = list_name_splits[0]+newfeatname+list_name_splits[1]
                    print(newname)
                    os.rename(rootdir+assetsdir+file, rootdir+assetsdir+newname)
    

def compress_zip(config, sources, features=[i for i in range(37)], 
                assets=[1,2,3,4,7,8,10,11,12,13,14,15,16,17,19,27,28,29,30,31,32], 
                relations=['direct','inverse'],dirname='',namefile='compressed.zip'):
    """  """
    from tqdm import tqdm
    if 'movingWindow' in config:
        mW = config['movingWindow']
    else:
        mW = 500
    if 'nEventsPerStat' in config:
        nExS = config['nEventsPerStat']
    else:
        nExS = 5000
    if 'phase_shift' in config:
        phase_shift = config['phase_shift']
    else:
        phase_shift = 2
        
    if len(features)==0:
        features = config['feature_keys']
    if len(assets)==0:
        assets = config['assets']
    if len(relations)==0:
        relations = config['build_asset_relations']
    if dirname=='':
        dirname = local_vars.hdf5_directory
    rootdir = local_vars.hdf5_directory
    assetsdirs = ['mW'+str(mW)+'_nE'+str(nExS)+'/'+rel+'/'+s+'/'+C.AllAssets[str(ass)]+'/' for rel in relations for s in sources for ass in assets]
    phase_size = mW/phase_shift
    shifts = [str(int(phase_size*phase)) for phase in range(phase_shift)]+['stats']
    ziph = zipfile.ZipFile(dirname+namefile, 'w', zipfile.ZIP_DEFLATED)
    #zipdir(path, zipf)#'tmp/'
    # ziph is zipfile handle
    stats_files = ['m_in','m_out','output']
    for assetsdir in tqdm(assetsdirs, mininterval=1):#os.walk(path)
        list_all_dirs = sorted(os.listdir(rootdir+assetsdir))
        for file in list_all_dirs:
            if os.path.isdir(rootdir+assetsdir+file):
                #print(rootdir+assetsdir+file)
                list_feats = sorted(os.listdir(rootdir+assetsdir+file+'/'))
                for featfile in list_feats:
                    featname = '_'.join('.'.join(featfile.split('.')[:-1]).split('_')[:-1])
                    try:
                        if C.FI[featname] in features:
                            if '.'.join(featfile.split('.')[:-1]).split('_')[-1] in shifts:
                                ziph.write(rootdir+assetsdir+file+'/'+featfile, assetsdir+file+'/'+featfile)
                    except KeyError:
                        if featname in stats_files:
                            ziph.write(rootdir+assetsdir+file+'/'+featfile, assetsdir+file+'/'+featfile)
                        #print(featfile+" not found in C.FI. Skipped")
                    # add all_stats
#                    all_stats_file = rootdir+assetsdir+file+'/allstats_'+C.FI[featname]+'_'
#                    m = re.search('^'+thisAsset+'_\d+'+'.txt$',file)
    ziph.close()
    
def decompress_zip(filepath, destiny=''):
    """  """
    from tqdm import tqdm
    if destiny=='':
        destiny = local_vars.hdf5_directory
    if not os.path.exists(destiny):
        os.makedirs(destiny)
    zfile = zipfile.ZipFile(filepath)
    for finfo in tqdm(zfile.infolist(), mininterval=1):
#        print(finfo.filename)
        zfile.extract(finfo.filename, destiny)

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
    pickle.dump( all_stats['out'], 
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
        print("First iteration form inside")
        all_stats['out'] = {'mean_bid':0.0, 'std_bid':0.0, 
                            'mean_ask':0.0, 'std_ask':0.0,
                            'm':0}
    all_stats['out']['mean_bid'] += m_out*stats_out['mean_bid']
    all_stats['out']['std_bid'] += m_out*stats_out['std_bid']
    all_stats['out']['mean_ask'] += m_out*stats_out['mean_ask']
    all_stats['out']['std_ask'] += m_out*stats_out['std_ask']
    all_stats['out']['m'] += m_out
    if last_iteration:
        print("Last iteration from inside")
        all_stats['out']['mean_bid'] = all_stats['out']['mean_bid']/all_stats['out']['m']
        all_stats['out']['std_bid'] = all_stats['out']['std_bid']/all_stats['out']['m']
        all_stats['out']['mean_ask'] = all_stats['out']['mean_ask']/all_stats['out']['m']
        all_stats['out']['std_ask'] = all_stats['out']['std_ask']/all_stats['out']['m']
    
    return all_stats

def get_stats_modular(config, groupdirname, groupoutdirname):
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
    if 'feature_keys' in config:
        feature_keys = config['feature_keys']
    else:
        feature_keys = [i for i in range(37)]
    
    nChannels = int(nEventsPerStat/movingWindow)
    stats_in = {}
#    print(feature_keys)
    if len(feature_keys)>0:
        print("\t getting variations")
        
        for i, f in enumerate(feature_keys):
            if not os.path.exists(groupdirname+C.PF[f][0]+"_stats.p"):
                # create temp file for variations
                filename = local_vars.hdf5_directory+'temp'+str(np.random.randint(1000))+'.hdf5'
                try:
                    ft = h5py.File(filename,'w')
                    group_temp = ft.create_group('temp')
                    # load features
                    filedirname = groupdirname+C.PF[f][0]+'_0.hdf5'
                    feature_file = h5py.File(filedirname,'r')
            #        groupname = '_'.join(groupdirname.split('/'))
                    features = feature_file[C.PF[f][0]][C.PF[f][0]]
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
                    stats_in[C.PF[f][0]] = {'mean':means_in, 'std':stds_in}
                    pickle.dump( stats_in[C.PF[f][0]], 
                                open( groupdirname+C.PF[f][0]+"_stats.p", "wb" ))
                    
                    ft.close()
                    feature_file.close()
                    try:
                        os.remove(filename)
                    except :
                        print("WARNING! file "+filename+" unable to be deleted. Skipped")
                except KeyboardInterrupt:
                    print("KeyboardInterrupt. Deleting files.")
                    ft.close()
                    feature_file.close()
                    os.remove(filename)
                    raise KeyboardInterrupt
            else:
                stats_in[C.PF[f][0]] = pickle.load(open( groupdirname+C.PF[f][0]+"_stats.p", "rb"))
    # load returns
    if not os.path.exists(groupoutdirname+"output_stats.p"):
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
    else:
        stats_out = pickle.load(open( groupoutdirname+"output_stats.p", "rb"))
    
    print("\t Total time for stats: "+str(time.time()-tic))
    return stats_in, stats_out

def get_returns_modular(config, groupoutdirname, idx_init, DateTime, SymbolBid, SymbolAsk, Symbol, m_out, shift):
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
    asset_relation = config['asset_relation']
    feats_from_bids = config['feats_from_bids']
    
    if not os.path.exists(groupoutdirname+'output_'+str(shift)+'.hdf5') or force_calulation_output:
        print("\tGetting output.")
        if asset_relation=='inverse' and feats_from_bids:
            SymbolBid = Symbol
            SymbolAsk = 1/SymbolAsk[:]
        elif asset_relation=='inverse' and not feats_from_bids:
            SymbolAsk = Symbol
            SymbolBid = 1/SymbolBid[:]
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
        
    return [parSARhigh, parSARlow, sar]

def init_sto(numer_periods, number_channels):
    """ Init Stochastic Oscilator signal """
    class STO:
        lows = np.zeros((numer_periods, number_channels)) # the lowest price traded in last numer_periods
        highs = np.zeros((numer_periods, number_channels)) # the highest price traded in last numer_periods
        value = 0
    return STO

def update_sto(STO, C, new_max, new_min, n_samples, lookback):
    """ Update STO """
    # get current channel
#    print("update_sto")
    n_channels = STO.lows.shape[1]
    chan = (n_samples-1)%n_channels
    # shift registers
    STO.lows[:-1,chan] = STO.lows[1:,chan]
    STO.highs[:-1,chan] = STO.highs[1:,chan]
    # update registers last positions
    STO.lows[-1,chan] = new_min
    STO.highs[-1,chan] = new_max
    # update value
    samples_chan = int(np.floor(n_samples/n_channels)+1)
#    print("samples_chan")
#    print(samples_chan)
#    print("lookback")
#    print(lookback)
    
    index = lookback-min(samples_chan, lookback)
#    print("index")
#    print(index)
#    print("chan")
#    print(chan)
#    print("STO.lows.shape")
#    print(STO.lows.shape)
#    print("STO.highs.shape")
#    print(STO.highs.shape)
#    print("STO.lows[index:,chan]")
#    print(STO.lows[index:,chan])
#    print("STO.highs[index:,chan]")
#    print(STO.highs[index:,chan])
    STO.value = (C-min(STO.lows[index:,chan]))/(max(STO.highs[index:,chan])-min(STO.lows[index:,chan]))
    
    return STO

def init_rsi(number_periods, number_channels):
    """  """
    class RSI:
        returns = np.zeros((number_periods, number_channels)) # returns for the last number_periods
        value = 0.0
    return RSI

def update_rsi(RSI, n_samples, last_return):
    """ Update Relative Strength Index (RSI) """
#    print("update_rsi")
    n_channels = RSI.returns.shape[1]
    chan = (n_samples-1)%n_channels
    samples_chan = int(np.floor(n_samples/n_channels)+1)
    lookback = RSI.returns.shape[0]
    n = min(samples_chan, lookback)
    #index = lookback-n
    last_gain = max(0, last_return)
    last_loss = max(0, (-1)*last_return)
#    print("samples_chan")
#    print(samples_chan)
    if samples_chan==1:
        gains = last_gain
        losses = last_loss
    else:
        gains = np.sum(RSI.returns[(RSI.returns[:,chan]>0),chan])/n
        losses = (-1)*np.sum(RSI.returns[(RSI.returns[:,chan]<0),chan])/n
    # update returns
    RSI.returns[:-1,chan] = RSI.returns[1:,chan]
    RSI.returns[-1,chan] = last_return
    # update registers last positions
    if losses*(n-1)+last_loss!=0: # avoid division by zero
        smoothened_gain = (gains*(n-1)+last_gain)/n
        smoothened_loss = (losses*(n-1)+last_loss)/n
        RSI.value = 1-1/(1+smoothened_gain/smoothened_loss)
    else:
        RSI.value = 1.0
    return RSI

def init_adx(number_periods, number_channels, first_symbols, mW):
    """ Init Average Directional Index """
    class ADX:
        lookback = number_periods
        DMplus = np.zeros((number_periods, number_channels))
        DMminus = np.zeros((number_periods, number_channels))
        TR = np.zeros((number_periods, number_channels))
        prev_close = first_symbols[::mW]
        prev_high = max(first_symbols)+np.zeros((number_channels))
        prev_low = min(first_symbols)+np.zeros((number_channels))
        
        value = 0.0
    return ADX

def update_adx(ADX, symbols, n_samples):
    """ Update ADX """
#    print("update_adx")
    n_channels = ADX.DMplus.shape[1]
    chan = (n_samples-1)%n_channels
    samples_chan = int(np.floor(n_samples/n_channels))
    lookback = ADX.lookback
    n = min(samples_chan, lookback)
    index = lookback-n
#    print("index")
#    print(index)
    # max and min
    max_symb = max(symbols)
    min_symb = min(symbols)
    # current values
    currTR = max(max_symb-min_symb, abs(max_symb-ADX.prev_close[chan]), abs(ADX.prev_close[chan]-min_symb))
    currDMp = max_symb-ADX.prev_high[chan] # DM+
    currDMm = ADX.prev_low[chan]-min_symb # DM-
    # smoothing values
    sDMp = wilder_smoothing_1(ADX.DMplus[index:,chan], currDMp)# smoothed DM+
    sDMm = wilder_smoothing_1(ADX.DMminus[index:,chan], currDMm)# smoothed DM-
    ATR = wilder_smoothing_1(ADX.TR[index:,chan], currTR)# smoothed DM-
    
    DIp = sDMp/ATR
    DIm = sDMm/ATR
    suma = abs(DIp+DIm)
    if suma>0:
        DX = abs(DIp-DIm)/suma # WARNING! Check division by zero
    else:
        DX = 1
    ADX.value = wilder_smoothing_2(ADX.value, DX, n+1)
    # update structure
    ADX.prev_close[chan] = symbols[-1]
    ADX.DMplus[:-1,chan] = ADX.DMplus[1:,chan]
    ADX.DMplus[-1] = currDMp
    ADX.DMminus[:-1,chan] = ADX.DMminus[1:,chan]
    ADX.DMminus[-1] = currDMm
    ADX.TR[:-1,chan] = ADX.TR[1:,chan]
    ADX.TR[-1,chan] = currTR
    ADX.prev_high[chan] = max_symb
    ADX.prev_low[chan] = min_symb
    
    return ADX

def init_will(number_periods, number_channels):
    """ Init Williams %R indicator """
    class WILL:
        lookback = number_periods
        lows = np.zeros((number_periods, number_channels)) # the lowest price traded in last numer_periods
        highs = np.zeros((number_periods, number_channels)) # the highest price traded in last numer_periods
        value = 0
        
        value = 0.0
    return WILL

def update_will(WILL, close_price, new_max, new_min, n_samples):
    """ Update Williams %R indicator """
    n_channels = WILL.lows.shape[1]
    chan = int((n_samples-1)%n_channels)
    lookback = WILL.lookback
    # shift registers
    WILL.lows[:-1,chan] = WILL.lows[1:,chan]
    WILL.highs[:-1,chan] = WILL.highs[1:,chan]
    # update registers last positions
    WILL.lows[-1,chan] = new_min
    WILL.highs[-1,chan] = new_max
    # update value
    samples_chan =int( np.floor(n_samples/n_channels)+1)
    index = lookback-min(samples_chan, lookback)
    highest_high = np.max(WILL.highs[index:,chan])
    WILL.value = (highest_high-close_price)/(highest_high-np.min(WILL.lows[index:,chan]))
    
    return WILL

def init_boll(number_periods, number_channels):
    """ Init Bollinger Bands """
    class BOLL:
        lookback = number_periods
        TPs = np.zeros((number_periods, number_channels)) # typical prices vector
        BOLU = 0.0
        BOLD = 0.0
        m = 1000
    return BOLL

def update_boll(BOLL, symbols, n_samples):
    """ Update Bollinger Bands """
    n_channels = BOLL.TPs.shape[1]
    chan = (n_samples-1)%n_channels
    
    samples_chan = int(np.floor(n_samples/n_channels))
    lookback = BOLL.lookback
    n = min(samples_chan, lookback)
    index = lookback-n
    
    TP = (max(symbols)+min(symbols)+symbols[-1])/3
    BOLL.TPs[:-1,chan] = BOLL.TPs[1:,chan]
    BOLL.TPs[-1,chan] = TP
    if index==0:
        sigma = np.var(BOLL.TPs[index:,chan])
    else:
        sigma = 0
    MA = TP#np.mean(BOLL.TPs[index:,chan])
    sigma_gain = BOLL.m*sigma#min(,1)
    lower_band = MA-sigma_gain
    
    BOLL.BOLU = MA+sigma_gain-symbols[-1]
    BOLL.BOLD = 2*sigma_gain
    if sigma_gain>0:
        BOLL.PERB = (symbols[-1]-lower_band)/(BOLL.BOLD)
    else:
        BOLL.PERB = 0
    
    return BOLL

def get_adi(symbols, seconds):
    """ Get Accumulation/Distribution Indicator """
    Pc = symbols[-1]
    Pl = np.min(symbols)
    Ph = np.max(symbols)
    V = 1/max(seconds,1)
    
    return ((Pc-Pl)-(Ph-Pc))/(Ph-Pl)*V

def init_aro(number_periods, number_channels):
    """  """
    class ARO:
        lookback = number_periods
        highs = np.zeros((number_periods, number_channels))
        lows = np.zeros((number_periods, number_channels))
        AROU = 0.0
        AROD = 0.0
    return ARO

def update_aro(ARO, new_max, new_min, n_samples):
    """  """
    n_channels = ARO.lows.shape[1]
    chan = int((n_samples-1)%n_channels)
    lookback = ARO.lookback
    # shift registers
    ARO.lows[:-1,chan] = ARO.lows[1:,chan]
    ARO.highs[:-1,chan] = ARO.highs[1:,chan]
    # update registers last positions
    ARO.lows[-1,chan] = new_min
    ARO.highs[-1,chan] = new_max
    # update value
    samples_chan =int( np.floor(n_samples/n_channels)+1)
    index = lookback-min(samples_chan, lookback)
    pos_hh = np.argmax(ARO.highs[index:,chan])
    pos_ll = np.argmin(ARO.lows[index:,chan])
    
    ARO.AROU = (lookback-(pos_hh+index))/lookback
    ARO.AROD = (lookback-(pos_ll+index))/lookback
    return ARO

def wilder_smoothing_1(historic, curr):
    """  """
    periods = historic.shape[0]
    if periods==0:
        return curr#sum(historic)
    else:
        return sum(historic)*(1-1/periods)+curr
    
def wilder_smoothing_2(prev, curr, period):
    """  """
    return (prev*(period-1)+curr)/period

#def get_features_modular(config, groupdirname, DateTime, Symbol, m, shift):
#    """
#    Function that calculates features from raw data in per batches
#    Args:
#        - data
#        - features
#        - DateTime
#        - SymbolBid
#    Returns:
#        - features
#    """
#    tic = time.time()
#    if 'feature_keys' in config:
#        feature_keys = config['feature_keys']
#    else:
#        feature_keys = [i for i in range(len(C.FI))]#[i for i in range(8,10)]+[12,15,16]#[i for i in range(8)]+[10,11,13,14]+[i for i in range(17,37)]#
#    if 'movingWindow' in config:
#        movingWindow = config['movingWindow']
#    else:
#        movingWindow = 500
#    if 'nEventsPerStat' in config:
#        nEventsPerStat = config['nEventsPerStat']
#    else:
#        nEventsPerStat = 5000
#    if 'lbd' in config:
#        lbd = config['lbd']
#    else:
#        lbd = 1-1/(nEventsPerStat*np.array([0.1, 0.5, 1, 5, 10, 50, 100]))
#    if 'force_calculation_features' in config:
#        force_calculation_features = config['force_calculation_features']
#    else:
#        force_calculation_features = [False for i in range(len(feature_keys))]
#    asset_relation = config['asset_relation']
#    # init scalars
#    nExS = nEventsPerStat
#    mW = movingWindow
#    number_channels = round(nExS/mW)
#    
#    features_files = []
#    features = []
#    feature_keys_to_calc = []
#    filedirnames = []
#    
#    # init features
#    for i,f in enumerate(feature_keys):
#        featname = C.PF[f][0]+'_'+str(shift)
#        filedirname = groupdirname+featname+'.hdf5'
#        
#        if not os.path.exists(filedirname) or force_calculation_features[i]:
#            filedirnames.append(filedirname)
#            features_files.append(h5py.File(filedirname,'a') )
#            features.append(features_files[-1].create_group(C.PF[f][0]).\
#                        create_dataset(C.PF[f][0], (m,1),dtype=float))
#            feature_keys_to_calc.append(f)
#    feature_keys = feature_keys_to_calc
#    feature_key_tsfresh = [f for f in feature_keys if f>=37 and f<2000]
#    n_feat_tf = len(feature_key_tsfresh)
#    
#    boolEmas = [C.FI['EMA'+i] in feature_keys for i in C.emas_ext]
#    idxEmas = [i for i in range(len(boolEmas)) if boolEmas[i]]
#    nEMAs = sum(boolEmas)
#    boolSars = [C.FI['parSARhigh'+i] in feature_keys for i in C.sar_ext]
#    idxSars = [i for i in range(len(boolSars)) if boolSars[i]]
#    n_parsars = sum(boolSars)#int(C.FI['parSARhigh20'] in feature_keys) + int(C.FI['parSARhigh2'] in feature_keys)
#    # Stochastic Oscillators
#    boolStos = [C.FI['STO'+i] in feature_keys for i in C.sto_ext]
#    idxStos = [i for i in range(len(boolStos)) if boolStos[i]]
#    n_stos = sum(boolStos)
#    # RSIs
#    boolRSIs = [C.FI['RSI'+i] in feature_keys for i in C.rsi_ext]
#    idxRSIs = [i for i in range(len(boolRSIs)) if boolRSIs[i]]
#    n_rsis = sum(boolRSIs)
#    # ADXs
#    boolADXs = [C.FI['ADX'+i] in feature_keys for i in C.adx_ext]
#    idxADXs = [i for i in range(len(boolADXs)) if boolADXs[i]]
#    n_adxs = sum(boolADXs)
#    
#    # loop over batched
#    if len(features_files) > 0:
#        if asset_relation=='inverse':
#            Symbol = 1/Symbol[:]
#        # init exponetial means
#        if nEMAs>0:
#            em = intit_em(lbd, nExS, mW, Symbol)
#        
#        if n_parsars>0:
#            sar = init_sar(Symbol[0])
#        # init STOs
#        sto_struct_list = [init_sto(int(C.sto_ext[st]), number_channels) for st in range(n_stos) if boolStos[st]]
#        
#        # init RSIs
#        rsi_struct_list = [init_rsi(int(C.rsi_ext[i]), number_channels) for i in range(n_rsis) if boolRSIs[i]]
#        
#        # init ADXs
#        adx_struct_list = [init_adx(int(C.adx_ext[i]), number_channels, Symbol[:nExS], mW) for i in range(n_adxs) if boolADXs[i]]
#        
#        batch_size = 10000000
#        par_batches = int(np.ceil(m/batch_size))
#        l_index = 0
#        
#        for b in range(par_batches):
#            # get m
#            m_i = np.min([batch_size, m-b*batch_size])
#            
#            # init structures
#            if nEMAs>0:
#                EMA = np.zeros((m_i,nEMAs))
#            boolSymbol = C.FI['symbol'] in feature_keys
#            if boolSymbol:
#                symbol = np.zeros((m_i))
#            boolVariance = C.FI['variance'] in feature_keys
#            if boolVariance:
#                variance = np.zeros((m_i))
#            boolMaxValue = C.FI['maxValue'] in feature_keys
#            if boolMaxValue:
#                maxValue = np.zeros((m_i))
#            boolMinValue = C.FI['minValue'] in feature_keys
#            if boolMinValue:
#                minValue = np.zeros((m_i))
#            boolTimeInterval = C.FI['timeInterval'] in feature_keys
#            if boolTimeInterval:
#                timeInterval = np.zeros((m_i))
#            boolTime = C.FI['time'] in feature_keys
#            if boolTime:
#                timeSecs = np.zeros((m_i))
#            if n_parsars>0:
#                parSARhigh = np.zeros((m_i, n_parsars))
#                parSARlow = np.zeros((m_i, n_parsars))
#            if n_stos>0:
#                STO = np.zeros((m_i, n_stos))
#            if n_rsis>0:
#                RSI = np.zeros((m_i, n_rsis))
#            if n_adxs>0:
#                ADX = np.zeros((m_i, n_adxs))
#            if n_feat_tf>0:
#                features_tsfresh = np.zeros((m_i,n_feat_tf))
#            
#            for mm in range(m_i):
#                
#                if mm%1000==0:
#                    toc = time.time()
#                    print("\t\tmm="+str(b*batch_size+mm)+" of "+str(m)+". Total time: "+str(np.floor(toc-tic))+"s")
#                startIndex = l_index+mm*mW
#                endIndex = startIndex+nExS
#                thisPeriod = range(startIndex,endIndex)
#                thisPeriodBids = Symbol[thisPeriod]
#                if boolSymbol:
#                    symbol[mm] = Symbol[thisPeriod[-1]]
#                
#                if nEMAs>0:
#                    newBidsIndex = range(endIndex-mW,endIndex)
#                    for i in newBidsIndex:
#                        #a=data.lbd*em/(1-data.lbd**i)+(1-data.lbd)*tradeInfo.SymbolBid.loc[i]
#                        em = lbd*em+(1-lbd)*Symbol[i]
#                    EMA[mm,:] = em
#                    
#                if boolTimeInterval:
#                    t0 = dt.datetime.strptime(DateTime[thisPeriod[0]].decode("utf-8"),'%Y.%m.%d %H:%M:%S')
#                    te = dt.datetime.strptime(DateTime[thisPeriod[-1]].decode("utf-8"),'%Y.%m.%d %H:%M:%S')
#                    timeInterval[mm] = (te-t0).seconds/nExS
#    
#                if boolVariance:
#                    variance[mm] = np.var(thisPeriodBids)
#                
#                if boolMaxValue:
#                    maxValue[mm] = np.max(thisPeriodBids)
#                if boolMinValue:
#                    minValue[mm] = np.min(thisPeriodBids)
#                
#                if boolTime:
#                    timeSecs[mm] = (te.hour*60*60+te.minute*60+te.second)/C.secsInDay
#                
#                if n_parsars>0:
#                    outsar = get_sar(sar, maxValue[mm], minValue[mm], n_parsars)
#                    parSARhigh[mm,:] = outsar[0]
#                    parSARlow[mm,:] = outsar[1]
#                    sar = outsar[2]
#                    
#                for st in range(n_stos):
#                    sto_struct_list[st] = update_sto(sto_struct_list[st], 
#                                   Symbol[thisPeriod[-1]], np.max(thisPeriodBids), 
#                                   np.min(thisPeriodBids), mm+1, int(C.sto_ext[idxStos[st]]))
#                    STO[mm, st] = sto_struct_list[st].value
#                    
#                for i in range(n_rsis):
#                    rsi_struct_list[i] = update_rsi(rsi_struct_list[i], 
#                                   mm+1, Symbol[thisPeriod[-1]]-Symbol[thisPeriod[0]])
#                    RSI[mm, i] = rsi_struct_list[i].value
#                    
#                for i in range(n_adxs):
#                    adx_struct_list[i] = update_adx(adx_struct_list[i], Symbol[thisPeriod], mm+1)
#                    ADX[mm, i] = adx_struct_list[i].value
#                
#                if n_feat_tf>0:
#                    for idx, feat in enumerate(feature_key_tsfresh):
#                        func_name = C.PF[feat][1]
#                        params = C.PF[feat][2:]
#                        features_tsfresh[mm, idx] = feval(func_name,thisPeriodBids,params)
#            # end of loop
#            l_index = startIndex+mW
#            #print(l_index)
#            toc = time.time()
#            print("\t\tmm="+str(b*batch_size+mm+1)+" of "+str(m)+". Total time: "+str(np.floor(toc-tic))+"s")
#            # update features vector
#            init_idx = b*batch_size
#            end_idx = b*batch_size+m_i
#    
#            if boolSymbol:
#                nF = feature_keys.index(C.FI['symbol'])
#                features[nF][init_idx:end_idx,0] = symbol
#    
#            for e in range(nEMAs):
#                if C.FI['EMA'+C.emas_ext[idxEmas[e]]] in feature_keys:
#                    nF = feature_keys.index(C.FI['EMA'+C.emas_ext[idxEmas[e]]])
#                    features[nF][init_idx:end_idx, 0] = EMA[:,e]
#    
#            if boolVariance:
#                nF = feature_keys.index(C.FI['variance'])
#                logVar = 10*np.log10(variance/C.std_var+1e-10)
#                features[nF][init_idx:end_idx, 0] = logVar
#    
#            if boolTimeInterval:
#                nF = feature_keys.index(C.FI['timeInterval'])
#                logInt = 10*np.log10(timeInterval/C.std_time+0.01)
#                features[nF][init_idx:end_idx, 0] = logInt
#            
#            for e in range(n_parsars):
#                if C.FI['parSARhigh'+C.sar_ext[idxSars[e]]] in feature_keys:
#                    nF = feature_keys.index(C.FI['parSARhigh'+C.sar_ext[idxSars[e]]])
#                    features[nF][init_idx:end_idx, 0] = parSARhigh[:,e]
#                    
#                    nF = feature_keys.index(C.FI['parSARlow'+C.sar_ext[idxSars[e]]])
#                    features[nF][init_idx:end_idx, 0] = parSARlow[:,e]
#            
#            if boolTime:
#                nF = feature_keys.index(C.FI['time'])
#                features[nF][init_idx:end_idx, 0] = timeSecs
#                
#            if C.FI['difVariance'] in feature_keys:
#                nF = feature_keys.index(C.FI['difVariance'])
#                features[nF][init_idx:end_idx, 0] = logVar
#                
#            if C.FI['difTimeInterval'] in feature_keys:
#                nF = feature_keys.index(C.FI['difTimeInterval'])
#                features[nF][init_idx:end_idx, 0] = logInt
#            
#            if C.FI['maxValue'] in feature_keys:
#                nF = feature_keys.index(C.FI['maxValue'])
#                features[nF][init_idx:end_idx, 0] = maxValue-symbol
#            
#            if C.FI['minValue'] in feature_keys:
#                nF = feature_keys.index(C.FI['minValue'])
#                features[nF][init_idx:end_idx, 0] = symbol-minValue
#            
#            if C.FI['difMaxValue'] in feature_keys:
#                nF = feature_keys.index(C.FI['difMaxValue'])
#                features[nF][init_idx:end_idx, 0] = maxValue-symbol
#                
#            if C.FI['difMinValue'] in feature_keys:
#                nF = feature_keys.index(C.FI['difMinValue'])
#                features[nF][init_idx:end_idx, 0] = symbol-minValue
#            
#            if C.FI['minOmax'] in feature_keys:
#                nF = feature_keys.index(C.FI['minOmax'])
#                features[nF][init_idx:end_idx, 0] = minValue/maxValue
#            
#            if C.FI['difMinOmax'] in feature_keys:
#                nF = feature_keys.index(C.FI['difMinOmax'])
#                features[nF][init_idx:end_idx, 0] = minValue/maxValue
#            
#            for e in range(nEMAs):
#                if C.FI['symbolOema'+C.emas_ext[idxEmas[e]]] in feature_keys:
#                    nF = feature_keys.index(C.FI['symbolOema'+C.emas_ext[idxEmas[e]]])
#                    features[nF][init_idx:end_idx, 0] = symbol/EMA[:,e]
#            for e in range(nEMAs):
#                if C.FI['difSymbolOema'+C.emas_ext[idxEmas[e]]] in feature_keys:
#                    nF = feature_keys.index(C.FI['difSymbolOema'+C.emas_ext[idxEmas[e]]])
#                    features[nF][init_idx:end_idx, 0] = symbol/EMA[:,e]
##            print("features_tsfresh")
##            print(features_tsfresh.shape)
##            print(features_tsfresh)   
#            for idx, feat in enumerate(feature_key_tsfresh):
#                feat_name = C.PF[feat][0]
#                if C.FI[feat_name] in feature_keys:
#                    nF = feature_keys.index(C.FI[feat_name])
#                    features[nF][init_idx:end_idx, 0] = features_tsfresh[:,idx]
#            
#            ## Trading features ##
#            # Stochastic oscilators
#            for st in range(n_stos):
#                if C.FI['STO'+C.sto_ext[idxStos[st]]] in feature_keys:
#                    nF = feature_keys.index(C.FI['STO'+C.sto_ext[idxStos[st]]])
#                    features[nF][init_idx:end_idx, 0] = STO[:,st]
#            # RSI
#            for e in range(n_rsis):
#                if C.FI['RSI'+C.rsi_ext[idxRSIs[e]]] in feature_keys:
#                    nF = feature_keys.index(C.FI['RSI'+C.rsi_ext[idxRSIs[e]]])
#                    features[nF][init_idx:end_idx, 0] = RSI[:,e]
#            # ADX
#            for e in range(n_adxs):
#                if C.FI['ADX'+C.adx_ext[idxADXs[e]]] in feature_keys:
#                    nF = feature_keys.index(C.FI['ADX'+C.adx_ext[idxADXs[e]]])
#                    features[nF][init_idx:end_idx, 0] = ADX[:,e]
#        # close file
#        (file.close() for file in features_files)
#    else:
#        print("\tAll features already calculated. Skipped.")
#    return feature_keys, Symbol

def get_features_modular_parallel(config, groupdirname, DateTime, Symbol, m, shift):
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
    asset_relation = config['asset_relation']
    # init scalars
    nExS = nEventsPerStat
    mW = movingWindow
    number_channels = round(nExS/mW)
    
    features_files = []
    features = []
    feature_keys_to_calc = []
    filedirnames = []
    
    # init features
    for i,f in enumerate(feature_keys):
        featname = C.PF[f][0]+'_'+str(shift)
        filedirname = groupdirname+featname+'.hdf5'
        if not os.path.exists(filedirname) or force_calculation_features[i]:
            filedirnames.append(filedirname)
            features_files.append(h5py.File(filedirname,'a') )
            features.append(features_files[-1].create_group(C.PF[f][0]).\
                        create_dataset(C.PF[f][0], (m,1),dtype=float))
            feature_keys_to_calc.append(f)
    feature_keys = feature_keys_to_calc
    feature_key_tsfresh = [f for f in feature_keys if f>=137 and f<2000]
    n_feat_tf = len(feature_key_tsfresh)
    
    boolEmas = [C.FI['EMA'+i] in feature_keys for i in C.emas_ext]
    idxEmas = [i for i in range(len(boolEmas)) if boolEmas[i]]
    nEMAs = sum(boolEmas)
    boolSars = [C.FI['parSARhigh'+i] in feature_keys for i in C.sar_ext]
    idxSars = [i for i in range(len(boolSars)) if boolSars[i]]
    n_parsars = sum(boolSars)#int(C.FI['parSARhigh20'] in feature_keys) + int(C.FI['parSARhigh2'] in feature_keys)
    # Stochastic Oscillators
    boolStos = [C.FI['STO'+i] in feature_keys for i in C.sto_ext]
    idxStos = [i for i in range(len(boolStos)) if boolStos[i]]
    n_stos = sum(boolStos)
    # RSIs
    boolRSIs = [C.FI['RSI'+i] in feature_keys for i in C.rsi_ext]
    idxRSIs = [i for i in range(len(boolRSIs)) if boolRSIs[i]]
    n_rsis = sum(boolRSIs)
    # ADXs
    boolADXs = [C.FI['ADX'+i] in feature_keys for i in C.adx_ext]
    idxADXs = [i for i in range(len(boolADXs)) if boolADXs[i]]
    n_adxs = sum(boolADXs)
    # Bollinger Bands
    boolBOLLUPs = [C.FI['BOLLUP'+i] in feature_keys for i in C.boll_ext]
    idxBOLLUPs = [i for i in range(len(boolBOLLUPs)) if boolBOLLUPs[i]]
    n_bolls = sum(boolBOLLUPs)
    boolBOLLDOWNs = [C.FI['BOLLDOWN'+i] in feature_keys for i in C.boll_ext]
    idxBOLLDOWNs = [i for i in range(len(boolBOLLDOWNs)) if boolBOLLDOWNs[i]]
    boolPERBOLLs = [C.FI['PERBOLL'+i] in feature_keys for i in C.boll_ext]
    idxPERBOLLs = [i for i in range(len(boolPERBOLLs)) if boolPERBOLLs[i]]
    # WILLs
    boolWILLs = [C.FI['WILL'+i] in feature_keys for i in C.will_ext]
    idxWILLs = [i for i in range(len(boolWILLs)) if boolWILLs[i]]
    n_wills = sum(boolWILLs)
    # ADIs
    boolADIs = [C.FI['ADI'+i] in feature_keys for i in C.adi_ext]
    idxADIs = [i for i in range(len(boolADIs)) if boolADIs[i]]
    n_adis = sum(boolADIs)
    # AROON index
    boolAROUPs = [C.FI['AROUP'+i] in feature_keys for i in C.aro_ext]
    idxAROUPs = [i for i in range(len(boolAROUPs)) if boolAROUPs[i]]
    n_aros = sum(boolAROUPs)
    boolARODOWNs = [C.FI['ARODOWN'+i] in feature_keys for i in C.aro_ext]
    idxARODOWNs = [i for i in range(len(boolARODOWNs)) if boolARODOWNs[i]]
    
    # loop over batched
    if len(features_files) > 0:
        try:
            if asset_relation=='inverse':
                Symbol = 1/Symbol[:]
            # init exponetial means
            if nEMAs>0:
                em = intit_em(lbd, nExS, mW, Symbol)
                    
            if n_parsars>0:
                sar = init_sar(Symbol[0])            
            # init STOs
            sto_struct_list = [init_sto(int(C.sto_ext[st]), number_channels) for st in range(n_stos) if boolStos[st]]            
            # init RSIs
            rsi_struct_list = [init_rsi(int(C.rsi_ext[i]), number_channels) for i in range(n_rsis) if boolRSIs[i]]            
            # init ADXs
            adx_struct_list = [init_adx(int(C.adx_ext[i]), number_channels, Symbol[:nExS], mW) for i in range(n_adxs) if boolADXs[i]]
            # init BOLLs
            boll_struct_list = [init_boll(int(C.boll_ext[i]), number_channels) for i in range(n_bolls) if boolBOLLUPs[i]]
            # init ADXs
            will_struct_list = [init_will(int(C.will_ext[i]), number_channels) for i in range(n_wills) if boolWILLs[i]]
            # init AROs
            aro_struct_list = [init_aro(int(C.aro_ext[i]), number_channels) for i in range(n_aros) if boolAROUPs[i]]
            
            batch_size = 10000000
            par_batches = int(np.ceil(m/batch_size))
            l_index = 0
            
            for b in range(par_batches):
                # get m
                m_i = np.min([batch_size, m-b*batch_size])
                parallel_symbs = np.zeros((nExS, m_i))
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
                boolVolume = C.FI['volume'] in feature_keys
                if boolVolume:
                    volume = np.zeros((m_i))
                if n_parsars>0:
                    parSARhigh = np.zeros((m_i, n_parsars))
                    parSARlow = np.zeros((m_i, n_parsars))
                if n_stos>0:
                    STO = np.zeros((m_i, n_stos))
                if n_rsis>0:
                    RSI = np.zeros((m_i, n_rsis))
                if n_adxs>0:
                    ADX = np.zeros((m_i, n_adxs))
                if n_bolls>0:
                    BOLLUP = np.zeros((m_i, n_bolls))
                    BOLLDOWN = np.zeros((m_i, n_bolls))
                    PERBOLL = np.zeros((m_i, n_bolls))
                if n_wills>0:
                    WILL = np.zeros((m_i, n_wills))
                if n_adis>0:
                    ADI = np.zeros((m_i, n_adis))
                if n_aros>0:
                    AROUP = np.zeros((m_i, n_aros))
                    ARODOWN = np.zeros((m_i, n_aros))
                
                if n_feat_tf>0:
                    features_tsfresh = np.zeros((m_i,n_feat_tf))
                
                for mm in range(m_i):
                    
                    if mm%1000==0:
                        toc = time.time()
                        print("\t\tmm="+str(b*batch_size+mm)+" of "+str(m)+". Total time: "+str(np.floor(toc-tic))+"s")
                    startIndex = l_index+mm*mW
                    endIndex = startIndex+nExS
                    thisPeriod = range(startIndex,endIndex)
                    thisPeriodBids = Symbol[thisPeriod]
                    parallel_symbs[:,mm] = thisPeriodBids
                    if boolSymbol:
                        symbol[mm] = Symbol[thisPeriod[-1]]
                    
                    if nEMAs>0:
                        newBidsIndex = range(endIndex-mW,endIndex)
                        for i in newBidsIndex:
                            #a=data.lbd*em/(1-data.lbd**i)+(1-data.lbd)*tradeInfo.SymbolBid.loc[i]
                            em = lbd*em+(1-lbd)*Symbol[i]
                        EMA[mm,:] = em
                        
                    if boolTimeInterval or n_adis>0 or boolVolume:
                        t0 = dt.datetime.strptime(DateTime[thisPeriod[0]].decode("utf-8"),'%Y.%m.%d %H:%M:%S')
                        te = dt.datetime.strptime(DateTime[thisPeriod[-1]].decode("utf-8"),'%Y.%m.%d %H:%M:%S')
                        seconds = (te-t0).seconds
                        if boolTimeInterval:
                            timeInterval[mm] = seconds/nExS
                        if boolVolume:
                            volume[mm] = 1/max(seconds, 1)
                    
                    if boolMaxValue or n_wills>0 or n_aros>0:
                        max_thisPeriod = np.max(thisPeriodBids)
                        if boolMaxValue:
                            maxValue[mm] = max_thisPeriod
                    if boolMinValue or n_wills>0 or n_aros>0:
                        min_thisPeriod = np.min(thisPeriodBids)
                        if boolMinValue:
                            minValue[mm] = min_thisPeriod
                    
                    if boolTime:
                        timeSecs[mm] = (te.hour*60*60+te.minute*60+te.second)/C.secsInDay
                    
                    if n_parsars>0:
                        outsar = get_sar(sar, maxValue[mm], minValue[mm], n_parsars)
                        parSARhigh[mm,:] = outsar[0]
                        parSARlow[mm,:] = outsar[1]
                        sar = outsar[2]
                        
                    for st in range(n_stos):
                        sto_struct_list[st] = update_sto(sto_struct_list[st], 
                                       Symbol[thisPeriod[-1]], np.max(thisPeriodBids), 
                                       np.min(thisPeriodBids), mm+1, int(C.sto_ext[idxStos[st]]))
                        STO[mm, st] = sto_struct_list[st].value
                        
                    for i in range(n_rsis):
                        rsi_struct_list[i] = update_rsi(rsi_struct_list[i], 
                                       mm+1, Symbol[thisPeriod[-1]]-Symbol[thisPeriod[0]])
                        RSI[mm, i] = rsi_struct_list[i].value
                        
                    for i in range(n_adxs):
                        adx_struct_list[i] = update_adx(adx_struct_list[i], Symbol[thisPeriod], mm+1)
                        ADX[mm, i] = adx_struct_list[i].value
                        
                    for i in range(n_bolls):
                        boll_struct_list[i] = update_boll(boll_struct_list[i], Symbol[thisPeriod], mm+1)
                        BOLLUP[mm, i] = boll_struct_list[i].BOLU
                        BOLLDOWN[mm, i] = boll_struct_list[i].BOLD
                        PERBOLL[mm, i] = boll_struct_list[i].PERB
                    
                    for i in range(n_wills):#(WILL, close_price, new_max, new_min, n_samples, lookback)
                        will_struct_list[i] = update_will(will_struct_list[i], Symbol[thisPeriod[-1]], max_thisPeriod, min_thisPeriod, mm+1)
                        WILL[mm, i] = will_struct_list[i].value
                        
                    for i in range(n_adis):#(WILL, close_price, new_max, new_min, n_samples, lookback)
                        ADI[mm, i] = get_adi(Symbol[thisPeriod], seconds)
                        
                    for i in range(n_aros):
                        aro_struct_list[i] = update_aro(aro_struct_list[i], max_thisPeriod, min_thisPeriod, mm+1)
                        AROUP[mm, i] = aro_struct_list[i].AROU
                        ARODOWN[mm, i] = aro_struct_list[i].AROD
                        
                if boolVariance:
                    variance = np.var(parallel_symbs, axis=0)
    #                print("variance")
    #                print(variance)
                if n_feat_tf>0:
                    for idx, feat in enumerate(feature_key_tsfresh):
    #                    print(C.PF[feat][1])
                        func_name = C.PF[feat][1]
                        params = C.PF[feat][2:]
                        ret = feval(func_name, parallel_symbs, params)                    
    #                    print(ret)
                        features_tsfresh[:, idx] = ret
    #                    print(features_tsfresh[:, idx])
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
                    
                if boolVolume:
                    nF = feature_keys.index(C.FI['volume'])
                    features[nF][init_idx:end_idx, 0] = volume
                
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
                
                for idx, feat in enumerate(feature_key_tsfresh):
                    feat_name = C.PF[feat][0]
                    if C.FI[feat_name] in feature_keys:
                        nF = feature_keys.index(C.FI[feat_name])
                        features[nF][init_idx:end_idx, 0] = features_tsfresh[:,idx]
                
                ## Trading features ##
                # Stochastic oscilators
                for st in range(n_stos):
                    if C.FI['STO'+C.sto_ext[idxStos[st]]] in feature_keys:
                        nF = feature_keys.index(C.FI['STO'+C.sto_ext[idxStos[st]]])
                        features[nF][init_idx:end_idx, 0] = STO[:,st]
                # RSI
                for e in range(n_rsis):
                    if C.FI['RSI'+C.rsi_ext[idxRSIs[e]]] in feature_keys:
                        nF = feature_keys.index(C.FI['RSI'+C.rsi_ext[idxRSIs[e]]])
                        features[nF][init_idx:end_idx, 0] = RSI[:,e]
                # ADX
                for e in range(n_adxs):
                    if C.FI['ADX'+C.adx_ext[idxADXs[e]]] in feature_keys:
                        nF = feature_keys.index(C.FI['ADX'+C.adx_ext[idxADXs[e]]])
                        features[nF][init_idx:end_idx, 0] = ADX[:,e]
                        
                # BOLL
                for e in range(n_bolls):
                    if C.FI['BOLLUP'+C.boll_ext[idxBOLLUPs[e]]] in feature_keys:
                        nF = feature_keys.index(C.FI['BOLLUP'+C.boll_ext[idxBOLLUPs[e]]])
                        features[nF][init_idx:end_idx, 0] = BOLLUP[:,e]
                    if C.FI['BOLLDOWN'+C.boll_ext[idxBOLLDOWNs[e]]] in feature_keys:
                        nF = feature_keys.index(C.FI['BOLLDOWN'+C.boll_ext[idxBOLLDOWNs[e]]])
                        features[nF][init_idx:end_idx, 0] = BOLLDOWN[:,e]
                    if C.FI['PERBOLL'+C.boll_ext[idxPERBOLLs[e]]] in feature_keys:
                        nF = feature_keys.index(C.FI['PERBOLL'+C.boll_ext[idxPERBOLLs[e]]])
                        features[nF][init_idx:end_idx, 0] = PERBOLL[:,e]
                        
                # WILL
                for e in range(n_wills):
                    if C.FI['WILL'+C.will_ext[idxWILLs[e]]] in feature_keys:
                        nF = feature_keys.index(C.FI['WILL'+C.will_ext[idxWILLs[e]]])
                        features[nF][init_idx:end_idx, 0] = WILL[:,e]
                        
                # ADI
                for e in range(n_adis):
                    if C.FI['ADI'+C.adi_ext[idxADIs[e]]] in feature_keys:
                        nF = feature_keys.index(C.FI['ADI'+C.adi_ext[idxADIs[e]]])
                        features[nF][init_idx:end_idx, 0] = ADI[:,e]
                
                # ARO
                for e in range(n_aros):
                    if C.FI['AROUP'+C.aro_ext[idxAROUPs[e]]] in feature_keys:
                        nF = feature_keys.index(C.FI['AROUP'+C.aro_ext[idxAROUPs[e]]])
                        features[nF][init_idx:end_idx, 0] = AROUP[:,e]
                    if C.FI['ARODOWN'+C.aro_ext[idxARODOWNs[e]]] in feature_keys:
                        nF = feature_keys.index(C.FI['ARODOWN'+C.aro_ext[idxARODOWNs[e]]])
                        features[nF][init_idx:end_idx, 0] = ARODOWN[:,e]
            # close file
            (file.close() for file in features_files)
        except KeyboardInterrupt:
            (file.close() for file in features_files)
            (os.remove(filedirname) for filediername in filedirnames)
            print("Interrupt in get_features_modular_parallel. Files closed and deleted")
            raise ValueError
    else:
        print("\tAll features already calculated. Skipped.")
    return feature_keys, Symbol

def wrapper_get_features_modular(config, thisAsset, separators, assdirname, outassdirname, 
                                 group_raw, s, all_stats, last_iteration, 
                                 init_date, end_date, get_feats=True):
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
    if 'feats_from_bids' in config:
        feats_from_bids = config['feats_from_bids']
    else:
        feats_from_bids = True
    nEventsPerStat = config['nEventsPerStat']
    movingWindow = config['movingWindow']
    save_stats = config['save_stats']
    #asset_relation = config['asset_relation']
    phase_shift = config['phase_shift']
    
    
#    if asset_relation == 'direct':
#        Bids = SymbolBid
#        Asks = SymbolAsk
#    elif asset_relation == 'inverse':
#        print("\tGetting inverse")
#        Bids = 1/SymbolBid[:]
#        Asks = 1/SymbolAsk[:]
    if get_feats:
        # get trade info datasets
        DateTime = group_raw["DateTime"]
        SymbolBid = group_raw["SymbolBid"]
        SymbolAsk = group_raw["SymbolAsk"]
        if feats_from_bids:
            Symbols = SymbolBid
        else:
            Symbols = SymbolAsk
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
            pickle.dump( m_in, open( groupdirname+"m_in_"+str(shift)+".p", "wb" ))
            if not os.path.exists(groupoutdirname):
                os.makedirs(groupoutdirname)
            pickle.dump( m_out, open( groupoutdirname+"m_out_"+str(shift)+".p", "wb" ))
            
            if get_feats:
                print("\tShift "+str(shift)+": getting features from raw data...")
                # get structures and save them in a hdf5 file
                feature_keys_to_calc, Symbol = get_features_modular_parallel(config, groupdirname,
                                         DateTime[separators.index[s]+shift:separators.index[s+1]+1], 
                                         Symbols[separators.index[s]+shift:separators.index[s+1]+1], m_in, shift)#
                get_returns_modular(config, groupoutdirname, separators.index[s],
                                        DateTime[separators.index[s]+shift:separators.index[s+1]+1], 
                                        SymbolBid[separators.index[s]+shift:separators.index[s+1]+1],
                                    SymbolAsk[separators.index[s]+shift:separators.index[s+1]+1], Symbol, m_out, shift)
            # only get stats for shift zero
            if shift==0:
                # get stats
                stats_in, stats_out = get_stats_modular(config, groupdirname, groupoutdirname)
                if save_stats:
                    all_stats = track_stats(stats_in, stats_out, all_stats, m_in, m_out, last_iteration)
    
    return all_stats

def wrapper_wrapper_get_features_modular(config_entry, assets=[], seps_input=[], get_feats=True):
    """  """
    import time
    from kaissandra.inputs import load_separators
    
    
    ticTotal = time.time()
    # init file directories
    hdf5_directory = local_vars.hdf5_directory
    if type(config_entry)==dict:
        config = config_entry
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
    if not save_stats and not get_feats:
        raise ValueError("get_feats and get_feats cannot both be false. Nothing is gonna be done.")
    if feats_from_bids:
        rootdirname = hdf5_directory+'mW'+str(movingWindow)+'_nE'+str(nEventsPerStat)+'/'+asset_relation+'/bid/'#test_flag+'_legacy_test.hdf5'
    else:
        rootdirname = hdf5_directory+'mW'+str(movingWindow)+'_nE'+str(nEventsPerStat)+'/'+asset_relation+'/ask/'
    outrdirname = hdf5_directory+'mW'+str(movingWindow)+'_nE'+str(nEventsPerStat)+'/'+asset_relation+'/out/'
    
    filename_raw = local_vars.data_dir+'tradeinfo'+test_flag+'.hdf5'
    separators_directory = local_vars.data_dir+'separators'+test_flag+'/'
    
    #assert(not (build_test_db and save_stats))
    # init hdf5 files
    #f_prep_IO = h5py.File(filename_prep_IO,'a')
    f_raw = h5py.File(filename_raw,'r')
    
    if len(assets)==0:
        assets = config['assets']
    
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
        if len(seps_input)==0:
            seps = range(0,len(separators)-1,2)
        else:
            seps = seps_input
        # loop over separators
        for s in seps:
            # number of events within this separator chunk
            nE = separators.index[s+1]-separators.index[s]+1
#            if len(seps) == 0:
#                if s+1 == separators.shape[0]-1:
#                    last_iteration = True
#            else:
            if s == seps[-1]:
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
                if s == seps[0]:
                    first_date = init_date
                if last_iteration:
                    last_date = end_date
                # calculate features, returns and stats from raw data
                
                all_stats = wrapper_get_features_modular(
                        config, thisAsset, separators, assdirname, outassdirname, 
                        group_raw, s, all_stats, last_iteration, init_date, end_date, 
                        get_feats=get_feats)
                    
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
    return np.percentile(x, 100*param[0], axis=0)

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

    fft = np.fft.rfft(x, axis=0)
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
    if len(x.shape)>1:
        linReg = np.array([getattr(linregress(range(x.shape[0]), x[:, i]), attr) for i in range(x.shape[1])])
    else:
        linReg = getattr(linregress(range(x.shape[0]), x), attr)

    return linReg#getattr(linReg, attr)

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
    
    if len(x.shape)>1:
        aggregate_results = [_aggregate_on_chunks(x[:,i], f_agg, chunk_len) for i in range(x.shape[1])]
    else:
        aggregate_results = _aggregate_on_chunks(x, f_agg, chunk_len)
    if f_agg not in calculated_agg or chunk_len not in calculated_agg[f_agg]:
        if chunk_len >= len(x):
            calculated_agg[f_agg] = {chunk_len: np.NaN}
        else:
            if len(x.shape)>1:
                lin_reg_results = [linregress(range(len(aggregate_result)), aggregate_result) for aggregate_result in aggregate_results]
                calculated_agg[f_agg] = [{chunk_len: lin_reg_result} for lin_reg_result in lin_reg_results]
            else:
                lin_reg_results = linregress(range(len(aggregate_results)), aggregate_results)
                calculated_agg[f_agg] = {chunk_len: lin_reg_results}

    if chunk_len >= len(x):
        res_data = np.NaN
    else:
        if len(x.shape)>1:
            res_data = np.array([getattr(calculated_agg[f_agg][i][chunk_len], attr) for i in range(x.shape[1])])
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
#    if not isinstance(x, (np.ndarray, pd.Series)):
#        x = np.asarray(x)
    return np.sum(x*x, axis=0)#np.dot(x, x)

def sum_values(x, param):
    """
    Calculates the sum over the time series values

    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :return: the value of this feature
    :return type: bool
    """
    if x.shape[0] == 0:
        return 0

    return np.sum(x, axis=0)

def mean(x, param):
    """
    Returns the mean of x

    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :return: the value of this feature
    :return type: float
    """
    return np.mean(x, axis=0)

def minimum(x, param):
    """
    Calculates the lowest value of the time series x.

    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :return: the value of this feature
    :return type: float
    """
    return np.min(x, axis=0)

def median(x, param):
    """
    Returns the median of x

    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :return: the value of this feature
    :return type: float
    """
    return np.median(x, axis=0)

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
    n = x.shape[0]
    if 2 * param[0] >= n:
        return 0
    else:
        if len(x.shape)==1:
            return np.mean((np.roll(x, 2 * -param[0], axis=0) * np.roll(x, -param[0], axis=0) * x)[0:(n - 2 * param[0])], axis=0)
        else:
            return np.mean((np.roll(x, 2 * -param[0], axis=0) * np.roll(x, -param[0], axis=0) * x)[0:(n - 2 * param[0]),:], axis=0)
    
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
    return np.max(x, axis=0)

def chi_square(X, y, num_feats):
    """  """
    from sklearn.feature_selection import SelectKBest
    from sklearn.feature_selection import chi2
    #from sklearn.preprocessing import MinMaxScaler
    
#    num_feats = X.shape[1]
    #X_norm = MinMaxScaler().fit_transform(X)
    chi_selector = SelectKBest(chi2, k=num_feats)
    chi_selector.fit(X, y)
    chi_support = chi_selector.get_support()
    return chi_support
#    chi_feature = X.loc[:,chi_support].columns.tolist()
#    print(str(len(chi_feature)), 'selected features')
    
def random_forest(X, y, num_feats=30, n_estimators=100):
    """  """
    print("Random Forest")
    from sklearn.feature_selection import SelectFromModel
    from sklearn.ensemble import RandomForestClassifier
    
    embeded_rf_selector = SelectFromModel(RandomForestClassifier(n_estimators=n_estimators))#, max_features=num_feats
    embeded_rf_selector.fit(X, y)
    
    return embeded_rf_selector
#    embeded_rf_feature = X.loc[:,embeded_rf_support].columns.tolist()
#    print(str(len(embeded_rf_feature)), 'selected features')
    
def light_gmb(X, y):
    print("light GMB")
    from sklearn.feature_selection import SelectFromModel
    from lightgbm import LGBMClassifier
    
    lgbc=LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=32, colsample_bytree=0.2,
                reg_alpha=3, reg_lambda=1, min_split_gain=0.01, min_child_weight=40)
    
    embeded_lgb_selector = SelectFromModel(lgbc)
    embeded_lgb_selector.fit(X, y)
    
    return embeded_lgb_selector

def xgboost(X, y):
    """  """
    print("XGBoost")
    from xgboost import XGBClassifier
    from sklearn.feature_selection import SelectFromModel
    
    lgbc = XGBClassifier(n_estimators=500, learning_rate=0.05, num_leaves=32, colsample_bytree=0.2,
                reg_alpha=3, reg_lambda=1, min_split_gain=0.01, min_child_weight=40)
    
    embeded_lgb_selector = SelectFromModel(lgbc)
    embeded_lgb_selector.fit(X, y)
    
    return embeded_lgb_selector
    
#    embeded_lgb_feature = X.loc[:,embeded_lgb_support].columns.tolist()
#    print(str(len(embeded_lgb_feature)), 'selected features')

def feature_analysis(config={}):
    """  """
    from kaissandra.preprocessing import (load_stats_modular, load_features_modular, 
                                          load_returns_modular, build_variations_modular, 
                                          load_separators)
    #ticTotal = time.time()
    # create data structure
    if config=={}:    
        config = retrieve_config('CRNN00000')
    # Feed retrocompatibility
    if 'nEventsPerStat' in config:
        nEventsPerStat = config['nEventsPerStat']
    else:
        nEventsPerStat = 5000
    if 'movingWindow' in config:
        movingWindow = config['movingWindow']
    else:
        movingWindow = 5000
    if 'assets' in config:
        assets = config['assets']
    else:
        assets = [1,2,3,4,7,8,10,11,12,13,14,15,16,17,19,27,28,29,30,31,32]
    if 'feature_keys_manual' not in config:
        feature_keys_manual = [i for i in range(37)]
    else:
        feature_keys_manual = config['feature_keys_manual']
    nFeatures = len(feature_keys_manual)
    if 'feats_from_bids' in config:
        feats_from_bids = config['feats_from_bids']
    else:
        feats_from_bids = True
    if 'seq_len' in config:
        seq_len = config['seq_len']
    else:
        seq_len = int((config['lB']-config['nEventsPerStat'])/config['movingWindow']+1)
#    if 'filetag' in config:
#        filetag = config['filetag']
#    else:
#        filetag = config['IO_results_name']
    size_output_layer = config['size_output_layer']
    if 'n_bits_outputs' in config:
        n_bits_outputs = config['n_bits_outputs']
    else:
        n_bits_outputs = [size_output_layer]
    outputGain = config['outputGain']
    if 'build_test_db' in config:
        build_test_db = config['build_test_db']
    else:
        build_test_db = False
    if 'build_asset_relations' in config:
        build_asset_relations = config['build_asset_relations']
    else:
        build_asset_relations = ['direct']
#    if 'asset_relation' in config:
#        asset_relation = config['asset_relation']
#    else:
#        asset_relation = 'direct'
    if 'phase_shift' in config:
        phase_shift = config['phase_shift']
    else:
        phase_shift = 1
    phase_size = movingWindow/phase_shift
    shifts = [int(phase_size*phase) for phase in range(phase_shift)]
    
    hdf5_directory = local_vars.hdf5_directory
    IO_directory = local_vars.IO_directory
    if not os.path.exists(IO_directory):
        os.mkdir(IO_directory)
    # init hdf5 files
    if feats_from_bids:
        symbol = 'bid'
    else:
        symbol = 'ask'
    featuredirnames = [hdf5_directory+'mW'+str(movingWindow)+'_nE'+str(nEventsPerStat)+'/'+bar+'/'+symbol+'/' for bar in build_asset_relations]
    outrdirnames = [hdf5_directory+'mW'+str(movingWindow)+'_nE'+str(nEventsPerStat)+'/'+bar+'/out/' for bar in build_asset_relations]
    if not build_test_db:
        
        separators_directory = hdf5_directory+'separators/'
    else:
        separators_directory = hdf5_directory+'separators_test/'
#    edges, edges_dt = get_edges_datasets_modular(folds, config, separators_directory, symbol)
    
    IO_tr_name = config['IO_results_name']
#    IO_cv_name = config['IO_cv_name']
    filename = IO_directory+'VAR'+IO_tr_name+'.hdf5'
#    filename_cv = IO_directory+'KFCv'+IO_cv_name+'.hdf5'
#    IO_results_name = IO_directory+'DTA_'+IO_cv_name+'.p'
    
#    print("Edges:")
#    print(edges)
#    print(filename_tr)
#    print(filename_cv)
#    print(IO_results_name)
    
#    if len(log)>0:
#        write_log(filename_tr)
#    if len(log)>0:
#        write_log(filename_cv)
    if os.path.exists(filename):
        if_build_IO = False
    else:
        if_build_IO = config['if_build_IO']
    #if_build_IO=True
    # create model
    # if IO structures have to be built 
    if if_build_IO:
        #print("Tag = "+str(tag))
        #IO = {}
        #attributes to track asset-IO belonging
        #ass_IO_ass_tr = np.zeros((len(assets))).astype(int)
        # structure that tracks the number of samples per level
#        IO['totalSampsPerLevel'] = np.zeros((n_bits_outputs[-1]))
        # open IO file for writting
        f_tr = h5py.File(filename,'w')
        # init IO data sets
        X = f_tr.create_dataset('X', 
                                (0, nFeatures), 
                                maxshape=(None, nFeatures), 
                                dtype=float)
        y = f_tr.create_dataset('y', 
                                (0, 1),
                                maxshape=(None, 1),
                                dtype=float)
        r = f_tr.create_dataset('r', 
                                (0, 1),
                                maxshape=(None, 1),
                                dtype=float)
#        
#            
#        Itr = f_tr.create_dataset('I', 
#                                (0, seq_len,2),maxshape=(None, seq_len, 2),
#                                dtype=int)
#        Rtr = f_tr.create_dataset('R', 
#                                (0,1),
#                                maxshape=(None,1),
#                                dtype=float)
#        
#        IO['Xtr'] = Xtr
#        IO['Ytr'] = Ytr
#        IO['Itr'] = Itr
#        IO['Rtr'] = Rtr # return
#        IO['pointerTr'] = 0
#        
#        ass_IO_ass_cv = np.zeros((len(assets))).astype(int)
#        f_cv = h5py.File(filename_cv,'w')
#        # init IO data sets
#        Xcv = f_cv.create_dataset('X', 
#                                (0, seq_len, nFeatures), 
#                                maxshape=(None,seq_len, nFeatures), 
#                                dtype=float)
#        Ycv = f_cv.create_dataset('Y', 
#                                (0, seq_len, size_output_layer),
#                                maxshape=(None, seq_len, size_output_layer),
#                                dtype=float)
#            
#        Icv = f_cv.create_dataset('I', 
#                                (0, seq_len,2),maxshape=(None, seq_len, 2),
#                                dtype=int)
#        Rcv = f_cv.create_dataset('R', 
#                                (0,1),
#                                maxshape=(None,1),
#                                dtype=float)
#        
#        
#        # save IO structures in dictionary
#        
#        Dcv = f_cv.create_dataset('D', (0,seq_len,2),
#                                maxshape=(None,seq_len,2),dtype='S19')
#        Bcv = f_cv.create_dataset('B', (0,seq_len,2),
#                                maxshape=(None,seq_len,2),dtype=float)
#        Acv = f_cv.create_dataset('A', (0,seq_len,2),
#                                maxshape=(None,seq_len,2),dtype=float)
#        IO['Xcv'] = Xcv
#        IO['Ycv'] = Ycv
#        IO['Icv'] = Icv
#        IO['Rcv'] = Rcv # return
#        IO['Dcv'] = Dcv
#        IO['Bcv'] = Bcv
#        IO['Acv'] = Acv
#        IO['pointerCv'] = 0
    
    
#    if len(log)>0:
#        write_log(IO_results_name)
    # index asset
#    ass_idx = 0
    # array containing bids means
    #bid_means = pickle.load( open( "../HDF5/bid_means.p", "rb" ))
    # loop over all assets
    pointer = 0
    nChannels = int(nEventsPerStat/movingWindow)
    init_idx_rets = nChannels+seq_len-1
    for ass in [1]:# 
        thisAsset = C.AllAssets[str(ass)]
        assdirnames = [featuredirname+thisAsset+'/' for featuredirname in featuredirnames]
        outassdirnames = [outrdirname+thisAsset+'/' for outrdirname in outrdirnames]
        
        #tic = time.time()
        # load separators
        separators = load_separators(thisAsset, 
                                     separators_directory, 
                                     from_txt=1)
        
        first_date = dt.datetime.strftime(dt.datetime.strptime(
                separators.DateTime.iloc[0],'%Y.%m.%d %H:%M:%S'),'%y%m%d%H%M%S')
        last_date = dt.datetime.strftime(dt.datetime.strptime(
                separators.DateTime.iloc[-1],'%Y.%m.%d %H:%M:%S'),'%y%m%d%H%M%S')
        
        list_stats_in = [None for _ in range(len(outassdirnames))]
        list_stats_out = [None for _ in range(len(outassdirnames))]
        for ind, assdirname in enumerate(assdirnames):
            list_stats_in[ind], list_stats_out[ind] = load_stats_modular(config, thisAsset, first_date, last_date, symbol)
#        stats_output = load_output_stats_modular(config, hdf5_directory+'stats/', 
#                                            thisAsset, tag=tag_stats)
        
        if if_build_IO:
            mess = str(ass)+". "+thisAsset
            print(mess)
#            if len(log)>0:
#                write_log(mess)
#            # loop over separators
            for s in [0]:#range(0,len(separators)-1,2):#
                mess = "\ts {0:d} of {1:d}".format(int(s/2),int(len(separators)/2-1))+\
                    ". From "+separators.DateTime.iloc[s]+" to "+\
                    separators.DateTime.iloc[s+1]
                #print(mess)
#                if len(log)>0:
#                    write_log(mess)
                # number of events within this separator chunk
                nE = separators.index[s+1]-separators.index[s]+1
                # get first day after separator
                #day_s = separators.DateTime.iloc[s][0:10]
                # check if number of events is not enough to build two features and one return
                if nE>=seq_len*(nEventsPerStat+movingWindow):
                    
                    init_date = dt.datetime.strftime(dt.datetime.strptime(
                            separators.DateTime.iloc[s],'%Y.%m.%d %H:%M:%S'),'%y%m%d%H%M%S')
                    end_date = dt.datetime.strftime(dt.datetime.strptime(
                            separators.DateTime.iloc[s+1],'%Y.%m.%d %H:%M:%S'),'%y%m%d%H%M%S')
                    
                    groupoutdirnames = [outassdirname+init_date+end_date+'/' for outassdirname in outassdirnames]
                    
                    list_features = [[load_features_modular(config, thisAsset, separators, assdirname, init_date, end_date, shift) \
                                     for shift in shifts] for assdirname in assdirnames]
                    # load returns
                    list_returns_struct = [[load_returns_modular(config, groupoutdirname, thisAsset, separators, symbol, init_date, end_date, shift) \
                                           for shift in shifts] for groupoutdirname in groupoutdirnames]

                    try:
                        
                        for ind in range(len(assdirnames)):
#                            prevPointerCv = IO['pointerCv']
                            for s, shift in enumerate(shifts):
                                file_temp_name = local_vars.IO_directory+\
                                    'temp_train_build'+\
                                    str(np.random.randint(10000))+'.hdf5'
                                while os.path.exists(file_temp_name):
                                    file_temp_name = IO_directory+'temp_train_build'\
                                        +str(np.random.randint(10000))+'.hdf5'
                                file_temp = h5py.File(file_temp_name,'w')
                                #Vars = build_variations(config, file_temp, list_features[features_counter], list_stats_in[ind], modular=True)
                                X_i = build_variations_modular(config, file_temp, list_features[ind][s], list_stats_in[ind])
                                
#                                print(list_returns_struct[ind][s]['returns'].shape)
#                                print(list_stats_out[ind]['stds_t_out'].shape)
                                r_i = list_returns_struct[ind][s]['returns'][:,-1]/list_stats_out[ind]['stds_t_out'][-1]
                                size_output_mg = n_bits_outputs[-1]
                                y_i = np.minimum(np.maximum(np.sign(r_i)*np.round(abs(r_i)*outputGain),-
                                                                                (size_output_mg-1)/2),(size_output_mg-1)/2).astype(int)
                                
#                                print("X_i")
#                                print(X_i)
#                                print("r_i")
#                                print(r_i)
#                                print("y_i")
#                                print(y_i)
                                
                                # resize structure
                                samps = r_i.shape[0]-init_idx_rets
                                X.resize((pointer+samps, nFeatures))
                                y.resize((pointer+samps, 1))
                                r.resize((pointer+samps, 1))
                                
                                # update structures
                                X[pointer:pointer+samps,:] = X_i[:samps,:,0]
                                y[pointer:pointer+samps,0] = y_i[init_idx_rets:]
                                r[pointer:pointer+samps,0] = r_i[init_idx_rets:]
                                
                                pointer = pointer + samps
                                
                                file_temp.close()
                                os.remove(file_temp_name)
                            
                    except (KeyboardInterrupt):
                        mess = "KeyBoardInterrupt. Closing files and exiting program."
                        print(mess)
#                        if len(log)>0:
#                            write_log(mess)
#                        f_tr.close()
#                        f_cv.close()
                        file_temp.close()
                        os.remove(file_temp_name)
#                        os.remove(filename_tr)
#                        os.remove(filename_cv)
                        raise KeyboardInterrupt
    else:
        f_tr = h5py.File(filename,'r')
        X = f_tr['X']
        y = f_tr['y']
        r = f_tr['r']
    # analyse feats
    print(X)
    print(r)
    
#    embeded_rf_selector = random_forest(X[:], y[:,0], num_feats=30, n_estimators=100)
#    embeded_rf_support = embeded_rf_selector.get_support()
#    print(embeded_rf_support)
#    
#    embeded_lgb_selector = light_gmb(X[:], y[:,0])
#    embeded_lgb_support = embeded_lgb_selector.get_support()
#    print(embeded_lgb_support)
    
    embeded_xgb_selector = xgboost(X[:], y[:,0])
    embeded_xgb_support = embeded_xgb_selector.get_support()
    print(embeded_xgb_support)
    
    f_tr.close()
    
    features = []
    for i,a in enumerate(embeded_xgb_support):
        if a or embeded_xgb_support[i]:
            features.append(i)
    print(features)
    
    return embeded_xgb_selector#embeded_rf_selector, embeded_lgb_selector

if __name__=='__main__':
    pass
#    feats_from_bids = False
#    movingWindow = 500
#    nEventsPerStat = 5000
#    build_test_db = False
#    save_stats = True
#    feature_keys = [i for i in range(2000,2012)]
#    config_name = 'CFEATTESTTRAIDFEATS500'
#    entries = {'config_name':config_name,
#               'feats_from_bids':feats_from_bids,'movingWindow':movingWindow,
#              'nEventsPerStat':nEventsPerStat,'feature_keys':feature_keys,
#              'build_test_db':build_test_db,'save_stats':save_stats}
#    
#    #config=configuration(entries)
#    
#    config=retrieve_config(config_name)
#    config['phase_shift'] = 1
#    config['feature_keys'] = [2008, 2009, 2010, 2011, 2012, 2013]
#    
#    list_feats_from_bids = [False]
#    list_asset_relation = ['direct']
#    for feats_from_bids in list_feats_from_bids:
#        config['feats_from_bids'] = feats_from_bids
#        for asset_relation in list_asset_relation:
#            config['asset_relation'] = asset_relation
#            wrapper_wrapper_get_features_modular(config, assets=[1], seps_input=[2])