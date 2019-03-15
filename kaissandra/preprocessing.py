# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 09:52:37 2019

@author: mgutierrez
"""

import numpy as np
import time
import pandas as pd
import h5py
import datetime as dt
import pickle
import scipy.io as sio
import os
import tensorflow as tf
from kaissandra.inputs import (Data, 
                               load_separators,
                               load_stats_manual,
                               load_stats_tsf,
                               load_stats_output,
                               load_returns,
                               load_manual_features,
                               load_tsf_features)
from kaissandra.config import retrieve_config
from kaissandra.local_config import local_vars

def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y

def build_bin_output(config, Output, batch_size):
    """
    Function that builds output binary Y based on real-valued returns Output vector.
    Args:
        - model: object containing model parameters
        - Output: real-valued returns vector
        - batch_size: scalar representing the batch size 
    Returns:
        - output binary matrix y_bin
    """
    size_output_layer = config['size_output_layer']
    seq_len = config['seq_len']
    outputGain = config['outputGain']
    # init y
    y = np.zeros((Output.shape))
    #print(y.shape)
    # quantize output to get y
    out_quantized = np.minimum(np.maximum(np.sign(Output)*np.round(abs(Output)*outputGain),-
        (size_output_layer-1)/2),(size_output_layer-1)/2)
    
    y = out_quantized+int((size_output_layer-1)/2)
    # conver y as integer
    y_dec=y.astype(int)
    # one hot output
    y_one_hot = convert_to_one_hot(y_dec, size_output_layer).T.reshape(
        batch_size,seq_len,size_output_layer)
    return y_one_hot, y_dec

def build_DTA(data, D, B, A, ass_IO_ass):
    """
    Function that builds structure based on IO to later get Journal and ROIs.
    Args:
        - data
        - I: structure containing indexes
        - ass_IO_ass: asset to IO assignment
    """
    # init columns
    columns = ["DT1","DT2","B1","B2","A1","A2","Asset"]
    # init DTA
    DTA = pd.DataFrame()
    # init hdf5 file with raw data
    

    ass_index = 0
    last_ass_IO_ass = 0
    # loop over assets
    for ass in data.assets:
        # get this asset's name
        thisAsset = data.AllAssets[str(ass)]
        print(thisAsset)
        # init DTA for this asset
        DTA_i = pd.DataFrame(columns = columns)
#        entry_idx = I[last_ass_IO_ass:ass_IO_ass[ass_index],:,0].reshape((-1))
#        exit_idx = I[last_ass_IO_ass:ass_IO_ass[ass_index],:,1].reshape((-1))
        # fill DTA_i up
        DTA_i['DT1'] = D[last_ass_IO_ass:ass_IO_ass[ass_index],:,0].reshape((-1))
        if DTA_i.shape[0]>0:
            DTA_i['DT1'] = DTA_i['DT1'].str.decode('utf-8')
            print(DTA_i['DT1'].iloc[0])
            print(DTA_i['DT1'].iloc[-1])
            if DTA_i['DT1'].iloc[0][:10] not in data.dateTest:
                print("WARNING!!! DTA_i['DT1'].iloc[0][:10] not in data.dateTest")
            #assert(DTA_i['DT1'].iloc[0][:10] in data.dateTest)
#            assert(DTA_i['DT1'].iloc[-1][:10] in data.dateTest)
            if DTA_i['DT1'].iloc[-1][:10] not in data.dateTest:
                print("WARNING!!! DTA_i['DT1'].iloc[-1][:10] not in data.dateTest")
            #DTA_i['DT1'] = DTA_i['DT1'].str.decode('utf-8')
            DTA_i['B1'] = B[last_ass_IO_ass:ass_IO_ass[ass_index],:,0].reshape((-1))
            DTA_i['A1'] = A[last_ass_IO_ass:ass_IO_ass[ass_index],:,0].reshape((-1))
            
            DTA_i['DT2'] = D[last_ass_IO_ass:ass_IO_ass[ass_index],:,1].reshape((-1))
            DTA_i['DT2'] = DTA_i['DT2'].str.decode('utf-8')
            DTA_i['B2'] = B[last_ass_IO_ass:ass_IO_ass[ass_index],:,1].reshape((-1))
            DTA_i['A2'] = A[last_ass_IO_ass:ass_IO_ass[ass_index],:,1].reshape((-1))
            DTA_i['Asset'] = thisAsset
    #        print(DTA_i['DT1'].iloc[0])
    #        print(DTA_i['DT1'].iloc[-1])
            # append DTA this asset to all DTAs
            DTA = DTA.append(DTA_i,ignore_index=True)
        last_ass_IO_ass = ass_IO_ass[ass_index]
        ass_index += 1
    # end of for ass in data.assets:
    return DTA

def build_variations(config, file_temp, data, features, stats):
    """
    Function that builds X and Y from data contained in a HDF5 file.
    """
    
    nEventsPerStat = config['nEventsPerStat']
    movingWindow = config['movingWindow']
    channels = config['channels']
    feature_keys_manual = config['feature_keys_manual']
    max_var = config['max_var']
    # total number of possible channels
    nChannels = int(nEventsPerStat/movingWindow)
    # extract means and stats
    stds_in = stats['stds_t_in']
    means_in = stats['means_t_in']
    
    
    # number of channels
    nC = len(channels)
    # samples allocation per batch
    
    # number of features
    nF = len(feature_keys_manual)
    # create group
    group_temp = file_temp.create_group('temp')
    # reserve memory space for variations and normalized variations
    variations = group_temp.create_dataset('variations', (features.shape[0],nF,nC), dtype=float)
    variations_normed = group_temp.create_dataset('variations_normed', (features.shape[0],nF,nC), dtype=float)
    # init variations and normalized variations to 999 (impossible value)
    variations[:] = variations[:]+999
    variations_normed[:] = variations[:]
    nonVarFeats = np.intersect1d(data.noVarFeats,feature_keys_manual)
    #non-variation idx for variations normed
    nonVarIdx = np.zeros((len(nonVarFeats))).astype(int)
    nv = 0
    for allnv in range(nF):
        if feature_keys_manual[allnv] in nonVarFeats:
            nonVarIdx[nv] = int(allnv)
            nv += 1
    # loop over channels
    for r in range(nC):
        variations[channels[r]+1:,:,r] = (features[channels[r]+1:,
                                               feature_keys_manual]-features[
                                                :-(channels[r]+1),
                                                feature_keys_manual])
        if nonVarFeats.shape[0]>0:
            variations[channels[r]+1:,nonVarIdx,r] = features[:-(channels[r]+1),nonVarFeats]
            
        variations_normed[channels[r]+1:,:,r] = np.minimum(np.maximum((variations[channels[r]+1:,
                          :,r]-means_in[r,feature_keys_manual])/stds_in[r,feature_keys_manual],-max_var),max_var)
    # remove the unaltered entries
    nonremoveEntries = range(nChannels,variations_normed.shape[0])#variations_normed[:,0,-1]!=999
    # create new variations 
    variations_normed_new = group_temp.create_dataset('variations_normed_new', variations_normed[nChannels:,:,:].shape, dtype=float)
    variations_normed_new[:] = variations_normed[nonremoveEntries,:,:]
    del group_temp['variations_normed']
    del group_temp['variations']
    return variations_normed_new

def build_XY(config, Vars, returns_struct, stats_output, IO, 
             last_day='', alloc=2**20, save_output=False):
    """  """
    nEventsPerStat = config['nEventsPerStat']
    movingWindow = config['movingWindow']
    if 'lookAheadIndex' in config:
        lookAheadIndex = config['lookAheadIndex']
    else:
        lookAheadIndex = 3
    seq_len = config['seq_len']
    feature_keys_manual = config['feature_keys_manual']
    nFeatures = len(feature_keys_manual)
    size_output_layer = config['size_output_layer']
    nChannels = int(nEventsPerStat/movingWindow)
    inverse_load = config['inverse_load']
    channels = config['channels']
    nC = len(channels)
    nF = len(feature_keys_manual)
    
    stds_out = stats_output['stds_t_out']
    # extract features and returns
    returns = returns_struct['returns']
    ret_idx = returns_struct['ret_idx']
    dts = returns_struct['DT']
    # add dateTimes, bids and asks if are included in file
    all_info = 0
    if 'D' in IO:
        all_info = 1
        
        bids = returns_struct['B']
        asks = returns_struct['A']
        
        D = IO['D']
        B = IO['B']
        A = IO['A']
        
        #XA = IO['XA']
        #YA = IO['YA']

    # extract IO structures
    X = IO['X']
    Y = IO['Y']
    I = IO['I']
    if save_output:
        R = IO['R']
    pointer = IO['pointer']
    
    # get some scalars
    nSamps = Vars.shape[0]
    samp_remaining = nSamps-nChannels-seq_len-1
    chunks = int(np.ceil(samp_remaining/alloc))
    assert(chunks==1)
    # init counter of samps processed
    offset = 0
    # loop over chunks
    if 1:#for i in range(chunks):
        # this batch length
        batch = samp_remaining#np.min([samp_remaining,alloc])
        init_idx_rets = nChannels+offset+seq_len-1
        end_idx_rets = nChannels+offset+batch+2*seq_len-1
        
        if dts[end_idx_rets, -1]>last_day:
            # look for last entry containing last day
            end_idx_rets = find_last_day_entry(dt_support)
            
        dt_support = dts[init_idx_rets:end_idx_rets, [0, lookAheadIndex+1]]
        # create support numpy vectors to speed up iterations
        v_support = Vars[offset:offset+batch+seq_len, :, :]
        r_support = returns[init_idx_rets:end_idx_rets, lookAheadIndex]
        # we only take here the entry time index, and later at DTA building the 
        # exit time index is derived from the entry time and the number of events to
        # look ahead
        i_support = ret_idx[init_idx_rets:end_idx_rets, [0, lookAheadIndex+1]]
        
        if all_info:
            b_support = bids[init_idx_rets:end_idx_rets, [0, lookAheadIndex+1]]
            a_support = asks[init_idx_rets:end_idx_rets, [0, lookAheadIndex+1]]
        # update remaining samps to proceed
        samp_remaining = samp_remaining-batch
        # init formatted input and output
        X_i = np.zeros((batch, seq_len, nFeatures))
        # real-valued output
        R_i = np.zeros((batch, seq_len, 1))
        # output index vector
        I_i = np.zeros((batch, seq_len, 2))
        if all_info:
            #XA_i = np.zeros((batch, seq_len, model.nFeatures))
            #OA_i = np.zeros((batch, seq_len, 1))
            
            D_i = np.chararray((batch, seq_len, 2),itemsize=19)
            B_i = np.zeros((batch, seq_len, 2))
            A_i = np.zeros((batch, seq_len, 2))
        
        for nI in range(batch):
            # init channels counter
            cc = 0
            for r in range(nC):
                # get input
                v_s_s = v_support[nI:nI+seq_len, :, r]
                if inverse_load:
                    X_i[nI,:,cc*nF:(cc+1)*nF] = v_s_s[::-1,:]
                else:
                    X_i[nI,:,cc*nF:(cc+1)*nF] = v_s_s
                cc += 1
            # due to substraction of features for variation, output gets the 
            # feature one entry later
            R_i[nI,:,0] = r_support[nI]
            I_i[nI,:,:] = i_support[nI,:]
            
            if all_info:
                D_i[nI,:,:] = dt_support[nI,:]
                B_i[nI,:,:] = b_support[nI,:]
                A_i[nI,:,:] = a_support[nI,:]
        
        
        # normalize output
        R_i = R_i/stds_out[0, lookAheadIndex]#stdO#
        #OA_i = OA_i/stds_out[0,data.lookAheadIndex]
        # update counters
        offset = offset+batch
        # get decimal and binary outputs
        Y_i, y_dec = build_bin_output(config, R_i, batch)
        #YA_i, ya_dec = build_bin_output(model, OA_i, batch)
        # get samples per level
        for l in range(size_output_layer):
            IO['totalSampsPerLevel'][l] = IO['totalSampsPerLevel'][l]+np.sum(y_dec[:,-1,0]==l)
        # resize IO structures
        X.resize((pointer+batch, seq_len, nFeatures))
        Y.resize((pointer+batch, seq_len, size_output_layer))
        
        I.resize((pointer+batch, seq_len, 2))
        # update IO structures
        X[pointer:pointer+batch,:,:] = X_i
        Y[pointer:pointer+batch,:,:] = Y_i
        I[pointer:pointer+batch,:,:] = I_i
        if save_output:
            R.resize((pointer+batch, 1))
            R[pointer:pointer+batch,0] = R_i[:,0,0]
        if all_info:
            # resize
            D.resize((pointer+batch, seq_len, 2))
            B.resize((pointer+batch, seq_len, 2))
            A.resize((pointer+batch, seq_len, 2))
            # update
            D[pointer:pointer+batch,:,:] = D_i
            B[pointer:pointer+batch,:,:] = B_i
            A[pointer:pointer+batch,:,:] = A_i
#        save_as_matfile('X_h_n_'+str(int(s/2)),'X_h_n'+str(int(s/2)),X_i)
#        save_as_matfile('O_h_n_'+str(int(s/2)),'O_h_n'+str(int(s/2)),O_i)
        
        # uodate pointer
        pointer += batch
        #print(pointer)
    # end of for i in range(chunks):
    # update dictionary
    IO['X'] = X
    IO['Y'] = Y
    IO['I'] = I
    IO['pointer'] = pointer
    if save_output:
        IO['R'] = R
    if all_info:
        #IO['XA'] = XA
        #IO['YA'] = YA
        
        IO['D'] = D
        IO['B'] = B
        IO['A'] = A
    
    return IO

def build_IO(config, file_temp, data, features_manual,features_tsf,returns_struct,
             stats_manual,stats_tsf,stats_output,IO,totalSampsPerLevel, 
             s, nE, thisAsset, inverse_load, save_output=False):
    """
    Function that builds X and Y from data contained in a HDF5 file.
    """
    seq_len = config['seq_len']
    feature_keys_manual = config['feature_keys_manual']
    nFeatures = len(feature_keys_manual)
    size_output_layer = config['size_output_layer']
    nEventsPerStat = config['nEventsPerStat']
    movingWindow = config['movingWindow']
    # total number of possible channels
    nChannels = int(nEventsPerStat/movingWindow)
    # extract means and stats
    if len(stats_manual)>0:
        means_in_manual = stats_manual['means_t_in']
        stds_in_manual = stats_manual['stds_t_in']
    else:
        means_in_manual = np.zeros((nChannels,0))
        stds_in_manual = np.zeros((nChannels,0))
        
    if len(stats_tsf)>0:
        means_in_tsf = stats_tsf['means_t_in']
        stds_in_tsf = stats_tsf['stds_t_in']
    else:
        means_in_tsf = np.zeros((nChannels,0))
        stds_in_tsf = np.zeros((nChannels,0))
    
    # concatenate features and stats
    features = np.append(features_manual, features_tsf, 1)
    stds_in = np.append(stds_in_manual, stds_in_tsf, 1)
    means_in = np.append(means_in_manual, means_in_tsf, 1)
    
    stds_out = stats_output['stds_t_out']
    # extract features and returns
    returns = returns_struct['returns']
    ret_idx = returns_struct['ret_idx']
    # add dateTimes, bids and asks if are included in file
    all_info = 0
    if 'D' in IO:
        all_info = 1
        dts = returns_struct['DT']
        bids = returns_struct['B']
        asks = returns_struct['A']
        
        D = IO['D']
        B = IO['B']
        A = IO['A']
        
        #XA = IO['XA']
        #YA = IO['YA']

    # extract IO structures
    X = IO['X']
    Y = IO['Y']
    I = IO['I']
    if save_output:
        R = IO['R']
    pointer = IO['pointer']
    # number of channels
    nC = len(data.channels)
    # samples allocation per batch
    aloc = 2**20
    # number of features
    nF = data.nFeatures
    # create group
    group_temp = file_temp.create_group('temp')
    # reserve memory space for variations and normalized variations
    variations = group_temp.create_dataset('variations', (features.shape[0],nF,nC), dtype=float)
    variations_normed = group_temp.create_dataset('variations_normed', (features.shape[0],nF,nC), dtype=float)
    # init variations and normalized variations to 999 (impossible value)
    variations[:] = variations[:]+999
    variations_normed[:] = variations[:]
    nonVarFeats = np.intersect1d(data.noVarFeats,data.feature_keys)
    #non-variation idx for variations normed
    nonVarIdx = np.zeros((len(nonVarFeats))).astype(int)
    nv = 0
    for allnv in range(nF):
        if data.feature_keys[allnv] in nonVarFeats:
            nonVarIdx[nv] = int(allnv)
            nv += 1
    # loop over channels
    for r in range(nC):
        variations[data.channels[r]+1:,:,r] = (features[data.channels[r]+1:,
                                               data.feature_keys]-features[
                                                :-(data.channels[r]+1),
                                                data.feature_keys])
        if nonVarFeats.shape[0]>0:
            variations[data.channels[r]+1:,nonVarIdx,r] = features[:-(data.channels[r]+1),nonVarFeats]
            
        variations_normed[data.channels[r]+1:,:,r] = np.minimum(np.maximum((variations[data.channels[r]+1:,
                          :,r]-means_in[r,data.feature_keys])/stds_in[r,data.feature_keys],-data.max_var),data.max_var)
    # remove the unaltered entries
    nonremoveEntries = range(nChannels,variations_normed.shape[0])#variations_normed[:,0,-1]!=999
    # create new variations 
    variations_normed_new = group_temp.create_dataset('variations_normed_new', variations_normed[nChannels:,:,:].shape, dtype=float)
    variations_normed_new[:] = variations_normed[nonremoveEntries,:,:]
    del group_temp['variations_normed']
    del group_temp['variations']
    # get some scalars
    nSamps = variations_normed_new.shape[0]
    samp_remaining = nSamps-nChannels-seq_len-1
    chunks = int(np.ceil(samp_remaining/aloc))
    # init counter of samps processed
    offset = 0
    # loop over chunks
    for i in range(chunks):
        # this batch length
        batch = np.min([samp_remaining,aloc])
        init_idx_rets = nChannels+offset+seq_len-1
        end_idx_rets = nChannels+offset+batch+2*seq_len-1
        # create support numpy vectors to speed up iterations
        v_support = variations_normed_new[offset:offset+batch+seq_len, :, :]
        r_support = returns[init_idx_rets:end_idx_rets, data.lookAheadIndex]
#        if len(stats_manual)>0:
#            tag = '_m_'
#        else:
#            tag = '_a_'
#        save_as_matfile(thisAsset+tag+str(int(s/2)),thisAsset+tag+str(int(s/2)),v_support)
#        raise KeyboardInterrupt
        # we only take here the entry time index, and later at DTA building the 
        # exit time index is derived from the entry time and the number of events to
        # look ahead
        i_support = ret_idx[init_idx_rets:end_idx_rets, [0,data.lookAheadIndex+1]]
        if all_info:
            dt_support = dts[init_idx_rets:end_idx_rets, [0,data.lookAheadIndex+1]]
            b_support = bids[init_idx_rets:end_idx_rets, [0,data.lookAheadIndex+1]]
            a_support = asks[init_idx_rets:end_idx_rets, [0,data.lookAheadIndex+1]]
        # update remaining samps to proceed
        samp_remaining = samp_remaining-batch
        # init formatted input and output
        X_i = np.zeros((batch, seq_len, nFeatures))
        # real-valued output
        R_i = np.zeros((batch, seq_len, 1))
        # output index vector
        I_i = np.zeros((batch, seq_len, 2))
        if all_info:
            #XA_i = np.zeros((batch, seq_len, model.nFeatures))
            #OA_i = np.zeros((batch, seq_len, 1))
            
            D_i = np.chararray((batch, seq_len, 2),itemsize=19)
            B_i = np.zeros((batch, seq_len, 2))
            A_i = np.zeros((batch, seq_len, 2))
        
        for nI in range(batch):
            # init channels counter
            cc = 0
            for r in range(nC):
                # get input
                v_s_s = v_support[nI:nI+seq_len, :, r]
                if inverse_load:
                    X_i[nI,:,cc*nF:(cc+1)*nF] = v_s_s[::-1,:]
                else:
                    X_i[nI,:,cc*nF:(cc+1)*nF] = v_s_s
                cc += 1
            # due to substraction of features for variation, output gets the 
            # feature one entry later
            R_i[nI,:,0] = r_support[nI]
            I_i[nI,:,:] = i_support[nI,:]
            if all_info:
                #OA_i[nI,:,0] = ra_support[nI]
                D_i[nI,:,:] = dt_support[nI,:]
                B_i[nI,:,:] = b_support[nI,:]
                A_i[nI,:,:] = a_support[nI,:]
        
        
        # normalize output
        R_i = R_i/stds_out[0,data.lookAheadIndex]#stdO#
        #OA_i = OA_i/stds_out[0,data.lookAheadIndex]
        # update counters
        offset = offset+batch
        # get decimal and binary outputs
        Y_i, y_dec = build_bin_output(config, R_i, batch)
        #YA_i, ya_dec = build_bin_output(model, OA_i, batch)
        # get samples per level
        for l in range(size_output_layer):
            totalSampsPerLevel[l] = totalSampsPerLevel[l]+np.sum(y_dec[:,-1,0]==l)
        # resize IO structures
        X.resize((pointer+batch, seq_len, nFeatures))
        Y.resize((pointer+batch, seq_len, size_output_layer))
        
        I.resize((pointer+batch, seq_len, 2))
        # update IO structures
        X[pointer:pointer+batch,:,:] = X_i
        Y[pointer:pointer+batch,:,:] = Y_i
        I[pointer:pointer+batch,:,:] = I_i
        if save_output:
            R.resize((pointer+batch, 1))
            R[pointer:pointer+batch,0] = R_i[:,0,0]
        if all_info:
            # resize
            D.resize((pointer+batch, seq_len, 2))
            B.resize((pointer+batch, seq_len, 2))
            A.resize((pointer+batch, seq_len, 2))
            # update
            D[pointer:pointer+batch,:,:] = D_i
            B[pointer:pointer+batch,:,:] = B_i
            A[pointer:pointer+batch,:,:] = A_i
#        save_as_matfile('X_h_n_'+str(int(s/2)),'X_h_n'+str(int(s/2)),X_i)
#        save_as_matfile('O_h_n_'+str(int(s/2)),'O_h_n'+str(int(s/2)),O_i)
        
        # uodate pointer
        pointer += batch
        #print(pointer)
    # end of for i in range(chunks):
    # update dictionary
    IO['X'] = X
    IO['Y'] = Y
    IO['I'] = I
    IO['pointer'] = pointer
    if save_output:
        IO['R'] = R
    if all_info:
        #IO['XA'] = XA
        #IO['YA'] = YA
        
        IO['D'] = D
        IO['B'] = B
        IO['A'] = A
    
    return IO, totalSampsPerLevel

def build_IO_wrapper(*ins):
    """  """
    ticTotal = time.time()
    # create data structure
    if len(ins)>0:
        tOt = ins[0]
        config = ins[1]
    else:    
        config = retrieve_config('CTESTNR')
    # Feed retrocompatibility
    if 'feature_keys_manual' not in config:
        feature_keys_manual = [i for i in range(37)]
    else:
        feature_keys_manual = config['feature_keys_manual']
    nFeatures = len(feature_keys_manual)
    if 'feature_keys_tsfresh' not in config:
        feature_keys_tsfresh = []
    else:
        feature_keys_tsfresh = config['feature_keys_tsfresh']
    if 'from_stats_file' in config:
        from_stats_file = config['from_stats_file']
    else:
        from_stats_file = True
    if 'inverse_load' in config:
        inverse_load = config['inverse_load']
    else:
        inverse_load = True
    if 'feats_from_bids' in config:
        feats_from_bids = config['feats_from_bids']
    else:
        feats_from_bids = True
    if 'seq_len' in config:
        seq_len = config['seq_len']
    else:
        seq_len = int((config['lB']-config['nEventsPerStat'])/config['movingWindow']+1)
    
    size_output_layer = config['size_output_layer']
    
    data = Data(movingWindow=config['movingWindow'],
              nEventsPerStat=config['nEventsPerStat'],
              lB=config['lB'], 
              dateTest=config['dateTest'],
              assets=config['assets'],
              channels=config['channels'],
              max_var=config['max_var'],
              feature_keys_manual=feature_keys_manual,
              feature_keys_tsfresh=feature_keys_tsfresh)
    # init structures
    if tOt == 'tr':
        filetag = config['IDweights']
    elif tOt == 'te':
        filetag = config['IDresults']
    else:
        raise ValueError("tOt not known")
    hdf5_directory = local_vars.hdf5_directory
    IO_directory = local_vars.IO_directory
    if not os.path.exists(IO_directory):
        os.mkdir(IO_directory)
    # init hdf5 files
    if type(feats_from_bids)==bool:
        if feats_from_bids:
            # only get short bets (negative directions)
            tag = 'IO_mW'
            tag_stats = 'IOB'
        else:
            # only get long bets (positive directions)
            tag = 'IOA_mW'
            tag_stats = 'IOA'
    else:
        raise ValueError("feats_from_bids must be a bool")
            
    filename_prep_IO = (hdf5_directory+tag+str(data.movingWindow)+'_nE'+
                        str(data.nEventsPerStat)+'_nF'+str(data.n_feats_manual)+'.hdf5')
    filename_features_tsf = (hdf5_directory+'feats_tsf_mW'+str(data.movingWindow)+'_nE'+
                         str(data.nEventsPerStat)+'_2.hdf5')
    
    separators_directory = hdf5_directory+'separators/'
    filename_IO = IO_directory+'IOE_'+filetag+'.hdf5'
    if data.n_feats_manual>0:
        f_prep_IO = h5py.File(filename_prep_IO,'r')
    else:
        f_prep_IO = None
    if data.n_feats_tsfresh>0:
        f_feats_tsf = h5py.File(filename_features_tsf,'r')
    else:
        f_feats_tsf = None
        
    if os.path.exists(filename_IO) and len(ins)>0:
        if_build_IO = False
    else:
        if_build_IO = config['if_build_IO']
    # create model
    # if IO structures have to be built 
    if if_build_IO:
        print("Tag = "+str(tag))
        # open IO file for writting
        f_IO = h5py.File(filename_IO,'w')
        # init IO data sets
        X = f_IO.create_dataset('X', 
                                (0, seq_len, nFeatures), 
                                maxshape=(None,seq_len, nFeatures), 
                                dtype=float)
        Y = f_IO.create_dataset('Y', 
                                (0, seq_len, size_output_layer),
                                maxshape=(None, seq_len, size_output_layer),
                                dtype=float)
            
        I = f_IO.create_dataset('I', 
                                (0, seq_len,2),maxshape=(None, seq_len, 2),
                                dtype=int)
        R = f_IO.create_dataset('R', 
                                (0,1),
                                maxshape=(None,1),
                                dtype=float)
        
            # attributes to track asset-IO belonging
        ass_IO_ass = np.zeros((len(data.assets))).astype(int)
        # structure that tracks the number of samples per level
        totalSampsPerLevel = np.zeros((size_output_layer))
        # save IO structures in dictionary
        IO = {}
        IO['X'] = X
        IO['Y'] = Y
        IO['I'] = I
        IO['R'] = R # return
        IO['pointer'] = 0
        if tOt=='te':
            D = f_IO.create_dataset('D', (0,seq_len,2),
                                    maxshape=(None,seq_len,2),dtype='S19')
            B = f_IO.create_dataset('B', (0,seq_len,2),
                                    maxshape=(None,seq_len,2),dtype=float)
            A = f_IO.create_dataset('A', (0,seq_len,2),
                                    maxshape=(None,seq_len,2),dtype=float)
            IO['D'] = D
            IO['B'] = B
            IO['A'] = A
            IO_results_name = IO_directory+'DTA_'+filetag+'.p'
        
    # index asset
    ass_idx = 0
    # array containing bids means
    #bid_means = pickle.load( open( "../HDF5/bid_means.p", "rb" ))
    # loop over all assets
    for ass in data.assets:
        thisAsset = data.AllAssets[str(ass)]
        
        tic = time.time()
        # load separators
        separators = load_separators(thisAsset, 
                                     separators_directory, 
                                     from_txt=1)
        # retrive asset group
        if f_prep_IO != None:
            ass_group = f_prep_IO[thisAsset]
            stats_manual = load_stats_manual(data, 
                               thisAsset, 
                               ass_group,
                               from_stats_file=from_stats_file, 
                               hdf5_directory=hdf5_directory+'stats/',tag=tag_stats)
        else:
            stats_manual = []
        
        stats_output = load_stats_output(data, hdf5_directory+'stats/', thisAsset, tag=tag_stats)
        
        if f_feats_tsf != None:
            stats_tsf = load_stats_tsf(data, thisAsset, hdf5_directory, f_feats_tsf,
                                       load_from_stats_file=True)
            #print(stats_tsf)
        else:
            stats_tsf = []
        if if_build_IO:
            print(str(ass)+". "+thisAsset)
            # loop over separators
            for s in range(0,len(separators)-1,2):
                print("\ts {0:d} of {1:d}".format(int(s/2),int(len(separators)/2-1))+
                              ". From "+separators.DateTime.iloc[s]+" to "+
                              separators.DateTime.iloc[s+1])
                # number of events within this separator chunk
                nE = separators.index[s+1]-separators.index[s]+1
                # get first day after separator
                day_s = separators.DateTime.iloc[s][0:10]
                # check if number of events is not enough to build two features and one return
                if nE>=2*data.nEventsPerStat:
                    if (tOt == 'tr' and (day_s not in data.dateTest and day_s<=data.dateTest[-1])) or \
                        (tOt == 'te' and (day_s in data.dateTest and day_s<=data.dateTest[-1])):  
                    #if day_s not in data.dateTest and day_s<=data.dateTest[-1]:
                        
                        # load features, returns and stats from HDF files
                        if f_prep_IO != None: 
                            features_manual = load_manual_features(data, 
                                                                   thisAsset, 
                                                                   separators, 
                                                                   f_prep_IO, 
                                                                   s)
                        else:
                            features_manual = np.array([])
                        
                        if f_feats_tsf != None:
                            features_tsf = load_tsf_features(data, thisAsset, separators, f_feats_tsf, s)
                        else:
                            features_tsf = np.array([])
                        # redefine features tsf or features manual in case they are
                        # None to fit to the concatenation
                        if features_tsf.shape[0]==0:
                            features_tsf = np.zeros((features_manual.shape[0],0))
                        if features_manual.shape[0]==0:
                            features_manual = np.zeros((features_tsf.shape[0],0))
                        # load returns
                        returns_struct = load_returns(data, hdf5_directory, thisAsset, separators, filename_prep_IO, s)
                        # build network inputs and outputs
                        # check if the separator chuck belongs to the training/test set
                        if 1:
                            
                            try:
                                file_temp_name = local_vars.IO_directory+'temp_train_build'+str(np.random.randint(10000))+'.hdf5'
                                while os.path.exists(file_temp_name):
                                    file_temp_name = IO_directory+'temp_train_build'+str(np.random.randint(10000))+'.hdf5'
                                file_temp = h5py.File(file_temp_name,'w')
                                IO, totalSampsPerLevel = build_IO(config,
                                                                  file_temp, 
                                                                      data, 
                                                                      features_manual,
                                                                      features_tsf,
                                                                      returns_struct,
                                                                      stats_manual,
                                                                      stats_tsf,
                                                                      stats_output,
                                                                      IO, 
                                                                      totalSampsPerLevel, 
                                                                      s, nE, thisAsset, 
                                                                      inverse_load,
                                                                      save_output=True)
                                # close temp file
                                file_temp.close()
                                os.remove(file_temp_name)
                            except (KeyboardInterrupt):
                                print("KeyBoardInterrupt. Closing files and exiting program.")
                                f_IO.close()
                                file_temp.close()
                                os.remove(file_temp_name)
                                if f_prep_IO != None:
                                    f_prep_IO.close()
                                if f_feats_tsf != None:
                                    f_feats_tsf.close()
                                raise KeyboardInterrupt
                    else:
                        print("\tNot in the set. Skipped.")
                        # end of if (tOt=='train' and day_s not in data.dateTest) ...
                    
                else:
                    print("\ts {0:d} of {1:d}. Not enough entries. Skipped.".format(
                            int(s/2),int(len(separators)/2-1)))
        # end of for s in range(0,len(separators)-1,2):
        # add pointer index for later separating assets
        if if_build_IO:
            ass_IO_ass[ass_idx] = IO['pointer']
            print("\tTime for "+thisAsset+":"+str(np.floor(time.time()-tic))+"s"+
              ". Total time:"+str(np.floor(time.time()-ticTotal))+"s")
            
        # update asset index
        ass_idx += 1
        
        # update total number of samples
        #m += stats_manual["m_t_out"]
        # flush content file
        if f_prep_IO != None:
            f_prep_IO.flush()
        if f_feats_tsf != None:
            f_feats_tsf.flush()
        
        
    
        #print("Total time:"+str(np.floor(time.time()-ticTotal))+"s")
    # end of for ass in data.assets:
    
    # close files
    if f_prep_IO != None:
        f_prep_IO.close()
    if f_feats_tsf != None:
        f_feats_tsf.close()
    
    if if_build_IO:
        if tOt=='te':
            print("Building DTA...")
            DTA = build_DTA(data, IO['D'], IO['B'], IO['A'], ass_IO_ass)
            pickle.dump( DTA, open( IO_results_name, "wb" ))
        f_IO.attrs.create('ass_IO_ass', ass_IO_ass, dtype=int)
        f_IO.close()
    # print percent of samps per level
    else:
        # get ass_IO_ass from disk
        f_IO = h5py.File(filename_IO,'r')
        ass_IO_ass = f_IO.attrs.get("ass_IO_ass")
        f_IO.close()
    # get total number of samps
    m_t = ass_IO_ass[-1]
    print("Samples to RNN: "+str(m_t))
    if if_build_IO:
        print("Percent per level:"+str(totalSampsPerLevel/m_t))
    print("DONE building IO")
    return filename_IO

def get_list_unique_days(thisAsset):
    """  """
    import re
    dir_ass = local_vars.data_dir+thisAsset
    list_dir = sorted(os.listdir(dir_ass))
    list_regs = [re.search('^'+thisAsset+'_\d+'+'.txt$',f) for f in list_dir]
    list_unique_days = [re.search('\d+',m.group()).group()[:8] for m in list_regs]
    return list(set(list_unique_days))

def get_day_indicator(list_unique_days, first_day=dt.date(2016, 1, 1), last_day=dt.date(2018, 11, 9)):
    """  """
    max_days = (last_day-first_day).days+1
    x = np.zeros((max_days))
    days_idx = [(dt.datetime.strptime(day,'%Y%m%d').date()-first_day).days for day in list_unique_days]
    x[days_idx] = 1
    return x
    
def get_numer_days(thisAsset):
    """  """
    import re
    dir_ass = local_vars.data_dir+thisAsset
    list_dir = sorted(os.listdir(dir_ass))
    list_regs = [re.search('^'+thisAsset+'_\d+'+'.txt$',f) for f in list_dir]
    #list_days = [re.search('\d+',m.group()).group()[:8] for m in list_regs]
    
    return len(set([re.search('\d+',m.group()).group()[:8] for m in list_regs]))

def get_edges_datasets(K, config={}):
    """ Get the edges representing days that split the dataset in K-1/K ratio of
    samples for training and 1/K ratio for cross-validation """
    if 'nEventsPerStat' in config:
        nEventsPerStat = config['nEventsPerStat']
    else:
        nEventsPerStat = 10000
    if 'movingWindow' in config:
        movingWindow = config['movingWindow']
    else:
        movingWindow = 1000
    if 'feature_keys_manual' in config:
        feature_keys_manual = config['feature_keys_manual']
    else:
        feature_keys_manual = [i for i in range(37)]
    if 'assets' in config:
        assets = config['assets']
    else:
        assets = [1,2,3,4,7,8,10,11,12,13,14,15,16,17,19,27,28,29,30,31,32]
    n_feats_manual = len(feature_keys_manual)
    dataset_dirfilename = local_vars.hdf5_directory+'IOA_mW'+str(movingWindow)+'_nE'+\
                        str(nEventsPerStat)+'_nF'+str(n_feats_manual)+'.hdf5'
    dataset_file = h5py.File(dataset_dirfilename,'r')
    weights_ass = np.zeros((len(assets),1))
    samps_ass = np.zeros((len(assets)))
    n_days_ass = np.zeros((len(assets)))
    
    first_day=dt.date(2016, 1, 1)
    last_day=dt.date(2018, 11, 9)
    max_days = (last_day-first_day).days+1
    A = np.zeros((len(assets), max_days))
    AllAssets = Data().AllAssets
    for a, ass in enumerate(assets):
        thisAsset = AllAssets[str(ass)]
        #print(thisAsset)
        samps_ass[a] = dataset_file[thisAsset].attrs.get("m_t_out")
        list_unique_days = get_list_unique_days(thisAsset)
        n_days_ass[a] = len(list_unique_days)
        A[a,:] = get_day_indicator(list_unique_days)
    weights_ass[:,0] = n_days_ass/samps_ass
    weights_ass[:,0] = weights_ass[:,0]/sum(weights_ass)
    weights_day = np.sum(A*weights_ass,0)
    weights_day = weights_day/sum(weights_day)
    
    edges_loc = np.array([k/K for k in range(1,K)])
    edges_idx = np.zeros((K-1)).astype(int)
    for e, edge in enumerate(edges_loc):
        edges_idx[e] = np.argmin(abs(np.cumsum(weights_day)-edge))
    edges = [first_day+dt.timedelta(days=int(d)) for d in edges_idx]
    return edges

def K_fold(folds=3, fold_idx=0, config={}):
    """  """
    ticTotal = time.time()
    # create data structure
    if config=={}:    
        config = retrieve_config('CTESTNR')
    # Feed retrocompatibility
    if 'nEventsPerStat' in config:
        nEventsPerStat = config['nEventsPerStat']
    else:
        nEventsPerStat = 10000
    if 'movingWindow' in config:
        movingWindow = config['movingWindow']
    else:
        movingWindow = 1000
    if 'assets' in config:
        assets = config['assets']
    else:
        assets = [1,2,3,4,7,8,10,11,12,13,14,15,16,17,19,27,28,29,30,31,32]
    if 'feature_keys_manual' not in config:
        feature_keys_manual = [i for i in range(37)]
    else:
        feature_keys_manual = config['feature_keys_manual']
    nFeatures = len(feature_keys_manual)
    if 'from_stats_file' in config:
        from_stats_file = config['from_stats_file']
    else:
        from_stats_file = True
    if 'inverse_load' in config:
        inverse_load = config['inverse_load']
    else:
        inverse_load = True
    if 'feats_from_bids' in config:
        feats_from_bids = config['feats_from_bids']
    else:
        feats_from_bids = True
    if 'seq_len' in config:
        seq_len = config['seq_len']
    else:
        seq_len = int((config['lB']-config['nEventsPerStat'])/config['movingWindow']+1)
    size_output_layer = config['size_output_layer']
    edges = get_edges_datasets(folds)

    hdf5_directory = local_vars.hdf5_directory
    IO_directory = local_vars.IO_directory
    if not os.path.exists(IO_directory):
        os.mkdir(IO_directory)
    # init hdf5 files
    if type(feats_from_bids)==bool:
        if feats_from_bids:
            # only get short bets (negative directions)
            tag = 'IO_mW'
            tag_stats = 'IOB'
        else:
            # only get long bets (positive directions)
            tag = 'IOA_mW'
            tag_stats = 'IOA'
    else:
        raise ValueError("feats_from_bids must be a bool")
            
    filename_prep_IO = (hdf5_directory+tag+str(movingWindow)+'_nE'+
                        str(nEventsPerStat)+'_nF'+str(nFeatures)+'.hdf5')
    
    separators_directory = hdf5_directory+'separators/'
    filename_IO = IO_directory+'IOKF'+filetag+str(fold_idx+1)+'OF'+str(folds)+'.hdf5'
    f_prep_IO = h5py.File(filename_prep_IO,'r')
        
    if os.path.exists(filename_IO):
        if_build_IO = False
    else:
        if_build_IO = config['if_build_IO']
    # create model
    # if IO structures have to be built 
    if if_build_IO:
        print("Tag = "+str(tag))
        # open IO file for writting
        f_IO = h5py.File(filename_IO,'w')
        # init IO data sets
        X = f_IO.create_dataset('X', 
                                (0, seq_len, nFeatures), 
                                maxshape=(None,seq_len, nFeatures), 
                                dtype=float)
        Y = f_IO.create_dataset('Y', 
                                (0, seq_len, size_output_layer),
                                maxshape=(None, seq_len, size_output_layer),
                                dtype=float)
            
        I = f_IO.create_dataset('I', 
                                (0, seq_len,2),maxshape=(None, seq_len, 2),
                                dtype=int)
        R = f_IO.create_dataset('R', 
                                (0,1),
                                maxshape=(None,1),
                                dtype=float)
        
            # attributes to track asset-IO belonging
        ass_IO_ass = np.zeros((len(assets))).astype(int)
        # structure that tracks the number of samples per level
        totalSampsPerLevel = np.zeros((size_output_layer))
        # save IO structures in dictionary
        IO = {}
        IO['X'] = X
        IO['Y'] = Y
        IO['I'] = I
        IO['R'] = R # return
        IO['pointer'] = 0
        if tOt=='te':
            D = f_IO.create_dataset('D', (0,seq_len,2),
                                    maxshape=(None,seq_len,2),dtype='S19')
            B = f_IO.create_dataset('B', (0,seq_len,2),
                                    maxshape=(None,seq_len,2),dtype=float)
            A = f_IO.create_dataset('A', (0,seq_len,2),
                                    maxshape=(None,seq_len,2),dtype=float)
            IO['D'] = D
            IO['B'] = B
            IO['A'] = A
            IO_results_name = IO_directory+'DTA_'+filetag+'.p'
        
    # index asset
    ass_idx = 0
    AllAssets = Data().AllAssets
    # array containing bids means
    #bid_means = pickle.load( open( "../HDF5/bid_means.p", "rb" ))
    # loop over all assets
    for ass in assets:
        thisAsset = AllAssets[str(ass)]
        
        tic = time.time()
        # load separators
        separators = load_separators(thisAsset, 
                                     separators_directory, 
                                     tOt='tr', 
                                     from_txt=1)
        # retrive asset group
        ass_group = f_prep_IO[thisAsset]
        stats_manual = load_stats_manual(data, 
                               thisAsset, 
                               ass_group,
                               from_stats_file=from_stats_file, 
                               hdf5_directory=hdf5_directory+'stats/',tag=tag_stats)
        
        stats_output = load_stats_output(data, hdf5_directory+'stats/', thisAsset, tag=tag_stats)
        
        if if_build_IO:
            print(str(ass)+". "+thisAsset)
            # loop over separators
            for s in range(0,len(separators)-1,2):
                print("\ts {0:d} of {1:d}".format(int(s/2),int(len(separators)/2-1))+
                              ". From "+separators.DateTime.iloc[s]+" to "+
                              separators.DateTime.iloc[s+1])
                # number of events within this separator chunk
                nE = separators.index[s+1]-separators.index[s]+1
                # get first day after separator
                day_s = separators.DateTime.iloc[s][0:10]
                # check if number of events is not enough to build two features and one return
                if nE>=2*nEventsPerStat:
                    if 1:  # TODO! check if s_init in set
                            #if day_s not in data.dateTest and day_s<=data.dateTest[-1]:
                        
                        # load features, returns and stats from HDF files
                        features_manual = load_manual_features(data, 
                                                               thisAsset, 
                                                               separators, 
                                                               f_prep_IO, 
                                                               s)

                        # load returns
                        returns_struct = load_returns(data, hdf5_directory, 
                                                      thisAsset, separators, 
                                                      filename_prep_IO, s)
                        try:
                            file_temp_name = local_vars.IO_directory+'temp_train_build'+str(np.random.randint(10000))+'.hdf5'
                            while os.path.exists(file_temp_name):
                                file_temp_name = IO_directory+'temp_train_build'+str(np.random.randint(10000))+'.hdf5'
                            file_temp = h5py.File(file_temp_name,'w')
                            Vars = build_variations(config, file_temp, data, features, stats)
                            IO = build_XY(config, Vars, returns_struct, stats_output, IO, 
                                     last_day='', save_output=False)
                            IO, totalSampsPerLevel = build_IO(config,
                                                              file_temp, 
                                                              data, 
                                                              features_manual,
                                                              features_tsf,
                                                              returns_struct,
                                                              stats_manual,
                                                              stats_tsf,
                                                              stats_output,
                                                              IO, 
                                                              totalSampsPerLevel, 
                                                              s, nE, thisAsset, 
                                                              inverse_load,
                                                              save_output=True)
                            # close temp file
                            file_temp.close()
                            os.remove(file_temp_name)
                        except (KeyboardInterrupt):
                            print("KeyBoardInterrupt. Closing files and exiting program.")
                            f_IO.close()
                            file_temp.close()
                            os.remove(file_temp_name)
                            if f_prep_IO != None:
                                f_prep_IO.close()
                            raise KeyboardInterrupt
                    else:
                        print("\tNot in the set. Skipped.")
                        # end of if (tOt=='train' and day_s not in data.dateTest) ...
                    
                else:
                    print("\ts {0:d} of {1:d}. Not enough entries. Skipped.".format(
                            int(s/2),int(len(separators)/2-1)))
        # end of for s in range(0,len(separators)-1,2):
        # add pointer index for later separating assets
        if if_build_IO:
            ass_IO_ass[ass_idx] = IO['pointer']
            print("\tTime for "+thisAsset+":"+str(np.floor(time.time()-tic))+"s"+
              ". Total time:"+str(np.floor(time.time()-ticTotal))+"s")
            
        # update asset index
        ass_idx += 1
        
        # update total number of samples
        #m += stats_manual["m_t_out"]
        # flush content file
        if f_prep_IO != None:
            f_prep_IO.flush()
        #print("Total time:"+str(np.floor(time.time()-ticTotal))+"s")
    # end of for ass in data.assets:
    
    # close files
    if f_prep_IO != None:
        f_prep_IO.close()
    if if_build_IO:
        if tOt=='te':
            print("Building DTA...")
            DTA = build_DTA(data, IO['D'], IO['B'], IO['A'], ass_IO_ass)
            pickle.dump( DTA, open( IO_results_name, "wb" ))
        f_IO.attrs.create('ass_IO_ass', ass_IO_ass, dtype=int)
        
    # print percent of samps per level
    else:
        # get ass_IO_ass from disk
        f_IO = h5py.File(filename_IO,'r')
        ass_IO_ass = f_IO.attrs.get("ass_IO_ass")
    # get total number of samps
    m_t = ass_IO_ass[-1]
    print("Samples to RNN: "+str(m_t))
    if if_build_IO:
        print("Percent per level:"+str(IO['totalSampsPerLevel']/m_t))
    f_IO.close()
    print("DONE building IO")
    return filename_IO