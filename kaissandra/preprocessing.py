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
from kaissandra.inputs import (Data, 
                               load_separators,
                               load_stats_manual,
                               load_stats_tsf,
                               load_stats_output,
                               load_returns,
                               load_manual_features,
                               load_tsf_features)
from kaissandra.config import retrieve_config, write_log, Config as C
from kaissandra.local_config import local_vars

def convert_to_one_hot(Y, c):
    Y = np.eye(c)[Y.reshape(-1)].T
    return Y

def build_bin_output_mcmdmg(config, Output, batch_size):
    """
    Function that builds output binary Y based on real-valued returns Output vector.
    Args:
        - model: object containing model parameters
        - Output: real-valued returns vector
        - batch_size: scalar representing the batch size 
    Returns:
        - output binary matrix y_bin
    """
    size_output_mg = config['n_bits_outputs'][-1]
    seq_len = config['seq_len']
    outputGain = config['outputGain']
    # init y
    y = np.zeros((Output.shape))
    #print(y.shape)
    # quantize output to get y
    out_quantized = np.minimum(np.maximum(np.sign(Output)*np.round(abs(Output)*outputGain),-
        (size_output_mg-1)/2),(size_output_mg-1)/2)
    
    y = out_quantized+int((size_output_mg-1)/2)
    # conver y as integer
    y_dec=y.astype(int)
    # one hot output
    y_one_hot = convert_to_one_hot(y_dec, size_output_mg).T.reshape(
        batch_size,seq_len, size_output_mg)
    #print("y_one_hot.shape")
    #print(y_one_hot.shape)

    # add y_c bits if proceed
    y_c = np.zeros((y_one_hot.shape[0],y_one_hot.shape[1],0))
    
    y_c0 = np.zeros((y_one_hot.shape[0],y_one_hot.shape[1],1))
    # find up/down outputs (=> those with a zero in the middle bit of one-hot vector)
    nonZeroYs = y_one_hot[:,:,int((size_output_mg-1)/2)]!=1
    # set 1s in y_c0 vector at non-zero entries
    y_c0[nonZeroYs,0] = 1
    y_c = np.append(y_c,y_c0,axis=2)
    y_c1 = np.zeros((y_one_hot.shape[0],y_one_hot.shape[1],1))
    y_c2 = np.zeros((y_one_hot.shape[0],y_one_hot.shape[1],1))
    # find negative (positive) returns
#    negativeYs = out_quantized<0
#    positiveYs = out_quantized>0
    negativeYs = Output<0
    positiveYs = Output>0
    # set to 1 the corresponding entries
    y_c1[np.squeeze(negativeYs,axis=2),0] = 1
    y_c2[np.squeeze(positiveYs,axis=2),0] = 1
    y_c = np.append(y_c,y_c1,axis=2)
    y_c = np.append(y_c,y_c2,axis=2)
    
    y_bin = np.append(y_c,y_one_hot,axis=2)
#    print(y_bin)
    # build output vector
    return y_bin, y_dec

def build_bin_output_mg(config, Output, batch_size):
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

def build_variations(config, file_temp, features, stats, modular=False):
    """
    Function that builds X and Y from data contained in a HDF5 file.
    """
    
    nEventsPerStat = config['nEventsPerStat']
    movingWindow = config['movingWindow']
    channels = config['channels']
    if not modular:
        feature_keys_manual = config['feature_keys_manual']
    else:
        feature_keys_manual = config['feature_keys']
    max_var = config['max_var']
    if 'noVarFeatsManual' in config:
        noVarFeatsManual = config['noVarFeatsManual']
    else:
        noVarFeatsManual = [8,9,12,17,18,21,23,24,25,26,27,28,29]
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
    nonVarFeats = np.intersect1d(noVarFeatsManual, feature_keys_manual)
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
            
        variations_normed[channels[r]+1:,:,r] = np.minimum(np.maximum((\
                         variations[channels[r]+1:,
                         :,r]-means_in[r,feature_keys_manual])/\
                         stds_in[r,feature_keys_manual],-max_var),max_var)
    # remove the unaltered entries
    nonremoveEntries = range(nChannels,variations_normed.shape[0])#variations_normed[:,0,-1]!=999
    # create new variations 
    variations_normed_new = group_temp.create_dataset('variations_normed_new', 
                                                      variations_normed[nChannels:,:,:].shape, 
                                                      dtype=float)
    variations_normed_new[:] = variations_normed[nonremoveEntries,:,:]
    del group_temp['variations_normed']
    del group_temp['variations']
    return variations_normed_new

def build_variations_modular(config, file_temp, features, stats):
    """
    Function that builds X and Y from data contained in a HDF5 file.
    """
    
    nEventsPerStat = config['nEventsPerStat']
    movingWindow = config['movingWindow']
    channels = config['channels']
    feature_keys = config['feature_keys']
    max_var = config['max_var']
    if 'noVarFeatsManual' in config:
        noVarFeatsManual = config['noVarFeatsManual']
    else:
        noVarFeatsManual = [8,9,12,17,18,21,23,24,25,26,27,28,29]
    # total number of possible channels
    nChannels = int(nEventsPerStat/movingWindow)
    # extract means and stats
    stds_in = stats['stds_t_in']
    means_in = stats['means_t_in']
    
    
    # number of channels
    nC = len(channels)
    # samples allocation per batch
    
    # number of features
    nF = len(feature_keys)
    # create group
    group_temp = file_temp.create_group('temp')
    # reserve memory space for variations and normalized variations
    variations = group_temp.create_dataset('variations', (features.shape[0],nF,nC), dtype=float)
    variations_normed = group_temp.create_dataset('variations_normed', (features.shape[0],nF,nC), dtype=float)
    # init variations and normalized variations to 999 (impossible value)
    variations[:] = variations[:]+999
    variations_normed[:] = variations[:]
    nonVarFeats = np.intersect1d(noVarFeatsManual, feature_keys)
    #non-variation idx for variations normed
    nonVarIdx = np.zeros((len(nonVarFeats))).astype(int)
    VarIdx = np.zeros((nF-len(nonVarFeats))).astype(int)
    nv = 0
    v = 0
    for allnv in range(nF):
        if feature_keys[allnv] in nonVarFeats:
            nonVarIdx[nv] = int(allnv)
            nv += 1
        else:
            VarIdx[v] = int(allnv)
            v += 1
#    print(features.shape)
#    print(nonVarIdx)
#    print(VarIdx)
#    print(variations.shape)
    # loop over channels
    for r in range(nC):
        variations[channels[r]+1:,VarIdx,r] = (features[channels[r]+1:,
                                               VarIdx]-features[
                                                :-(channels[r]+1),
                                                VarIdx])
        if nonVarFeats.shape[0]>0:
            variations[channels[r]+1:,nonVarIdx,r] = features[:-(channels[r]+1), nonVarIdx]
            
        variations_normed[channels[r]+1:,:,r] = np.minimum(np.maximum((\
                         variations[channels[r]+1:,
                         :,r]-means_in[r,:])/\
                         stds_in[r,:],-max_var),max_var)
    # remove the unaltered entries
    nonremoveEntries = range(nChannels,variations_normed.shape[0])#variations_normed[:,0,-1]!=999
    # create new variations 
    variations_normed_new = group_temp.create_dataset('variations_normed_new', 
                                                      variations_normed[nChannels:,:,:].shape, 
                                                      dtype=float)
    variations_normed_new[:] = variations_normed[nonremoveEntries,:,:]
    del group_temp['variations_normed']
    del group_temp['variations']
    return variations_normed_new

def map_index2sets(K, fold_idx):
    """  """
    assert(fold_idx<K)
    assert(fold_idx>=0)
    sets_list = ['Tr' for _ in range(K)]
    sets_list[K-(fold_idx+1)] = 'Cv'
    return sets_list

def find_edge_indexes(dts, edges_dt, group_name, fold_idx, sets_list, 
                      first_day=dt.datetime(2016, 1, 1, 0, 0), 
                      last_day=dt.datetime(2018, 11, 9, 0 ,0)):
    """ Find entries matching the edges """
    # find starting/end dates of the chunck
    
    
#    print(group_name)
    initenddates = group_name.split('/')[-1]
#    ic_str = dts[0]
    init_chunk = dt.datetime.strptime(initenddates[:12],'%y%m%d%H%M%S')
#    print("init_chunk")
#    print(init_chunk)
    end_chunk = dt.datetime.strptime(initenddates[12:],'%y%m%d%H%M%S')
#    print("end_chunk")
#    print(end_chunk)
    sets_edge = [first_day]+edges_dt+[last_day]
#    print("sets_edge")
#    print(sets_edge)
    T_idx = []
    for i, s in enumerate(sets_edge[:-1]):
        if (init_chunk>=sets_edge[i] and init_chunk<sets_edge[i+1]) or\
           (end_chunk>=sets_edge[i] and end_chunk<sets_edge[i+1]) or\
           (init_chunk<sets_edge[i] and end_chunk>=sets_edge[i+1]):
            T_idx.append(i)
#    print("T_idx")
#    print(T_idx)
    # find intersection between edge sets and this chunk set
    E = []
    for edge in edges_dt:
        # check edged belongs to Tr/Cv set
        if edge>=init_chunk and edge<end_chunk:
            E.append(edge)
#    print("E")
#    print(E)
    edges_idx_tr = np.zeros((0,2)).astype(int)
    edges_idx_cv = np.zeros((0,2)).astype(int)
    # alg to extract Tr/Cv sets
    p = init_chunk
    #Ti = T_idx[0]
    i = -1
    #e = end_chunk
    for i,e in enumerate(E):
        Ti = T_idx[i]
        p_idx = np.argmax(dts[:,0]>=dt.datetime.strftime(p,'%Y.%m.%d %H:%M:%S').encode('utf-8'))
        e_idx = dts.shape[0]-np.argmax(dts[::-1,0]<dt.datetime.strftime(e,'%Y.%m.%d %H:%M:%S').encode('utf-8'))
        set_edges_idx = [p_idx, e_idx]
        if sets_list[Ti]=='Tr':
            edges_idx_tr.resize((edges_idx_tr.shape[0]+1, 2))
            edges_idx_tr[-1,:] = set_edges_idx
        elif sets_list[Ti]=='Cv':
            edges_idx_cv.resize((edges_idx_cv.shape[0]+1, 2))
            edges_idx_cv[-1,:] = set_edges_idx
        p = e
    # last
    p_idx = np.argmax(dts[:,0]>=dt.datetime.strftime(p,'%Y.%m.%d %H:%M:%S').encode('utf-8'))
    e_idx = dts.shape[0]-np.argmax(dts[::-1,0]<dt.datetime.strftime(end_chunk,'%Y.%m.%d %H:%M:%S').encode('utf-8'))
#    print("p_idx")
#    print(p_idx)
#    print("e_idx")
#    print(e_idx)
    set_edges_idx = [p_idx, e_idx]
#    print("set_edges_idx")
#    print(set_edges_idx)
#    print("i")
#    print(i)
#    print("T_idx[i+1]")
#    print(T_idx[i+1])
#    print("sets_list")
#    print(sets_list)
#    print("sets_list[T_idx[i+1]]")
#    print(sets_list[T_idx[i+1]])
#    print("sets_list")
#    print(sets_list)
#    print("T_idx")
#    print(T_idx)
#    print("i")
#    print(i)
    if sets_list[T_idx[i+1]]=='Tr':
        edges_idx_tr.resize((edges_idx_tr.shape[0]+1, 2))
        edges_idx_tr[-1,:] = set_edges_idx
    elif sets_list[T_idx[i+1]]=='Cv':
        edges_idx_cv.resize((edges_idx_cv.shape[0]+1, 2))
        edges_idx_cv[-1,:] = set_edges_idx
    if edges_idx_tr.shape[0]==0:
        edges_idx_tr = np.array([[0,0]])
    if edges_idx_cv.shape[0]==0:
        edges_idx_cv = np.array([[0,0]])
#    print("edges_idx_tr")
#    print(edges_idx_tr)
#    print("edges_idx_cv")
#    print(edges_idx_cv)
    return edges_idx_tr, edges_idx_cv#np.array([[0,dts.shape[0]]]), np.array([[0,0]])
    
def build_XY(config, Vars, returns_struct, stats_output, IO, edges_dt, 
             K, fold_idx, alloc=2**30, save_output=False, modular=False, skip_cv=False, skip_tr=False):
    """  """
    nEventsPerStat = config['nEventsPerStat']
    movingWindow = config['movingWindow']
    if 'lookAheadIndex' in config:
        lookAheadIndex = config['lookAheadIndex']
    else:
        lookAheadIndex = 3
    if 'build_XY_mode' in config:
        build_XY_mode = config['build_XY_mode']
    else:
        build_XY_mode = 'K_fold'
    if 'first_day' in config:
        first_day = dt.datetime.strptime(config['first_day'],'%Y.%m.%d')
    else:
        first_day = dt.datetime.strptime('2016.01.01','%Y.%m.%d')
    if 'last_day' in config:
        last_day = dt.datetime.strptime(config['last_day'],'%Y.%m.%d')
    else:
        last_day = dt.datetime.strptime('2018.11.09','%Y.%m.%d')
    seq_len = config['seq_len']
    if not modular:
        feature_keys_manual = config['feature_keys_manual']
    else: 
        feature_keys_manual = config['feature_keys']
    nFeatures = len(feature_keys_manual)
    size_output_layer = config['size_output_layer']
    nChannels = int(nEventsPerStat/movingWindow)
    inverse_load = config['inverse_load']
    channels = config['channels']
    nC = len(channels)
    nF = len(feature_keys_manual)
    n_bits_outputs = config['n_bits_outputs']
    
    stds_out = stats_output['stds_t_out']
    # extract features and returns
    returns = returns_struct['returns']
    ret_idx = returns_struct['ret_idx']
    dts = returns_struct['DT']
    # add dateTimes, bids and asks if are included in file
    bids = returns_struct['B']
    asks = returns_struct['A']
    initenddates = returns_struct['group_name']
    D = IO['Dcv']
    B = IO['Bcv']
    A = IO['Acv']
        
    # extract IO structures
    Xtr = IO['Xtr']
    Ytr = IO['Ytr']
    Itr = IO['Itr']
    Xcv = IO['Xcv']
    Ycv = IO['Ycv']
    Icv = IO['Icv']
    if save_output:
        Rtr = IO['Rtr']
        Rcv = IO['Rcv']
    pointerTr = IO['pointerTr']
    pointerCv = IO['pointerCv']
    
    # get some scalars
    #nSamps = Vars.shape[0]
    #samp_remaining = nSamps-nChannels-seq_len+1
    #assert(chunks==1)
    # init counter of samps processed
    # loop over chunks
    if not skip_tr or not skip_cv:#for i in range(chunks):
        # this batch length
        #batch = samp_remaining#np.min([samp_remaining,alloc])
        init_idx_rets = nChannels+seq_len-1
        #end_idx_rets = nChannels+batch+2*seq_len-1
        
        dt_support = dts[init_idx_rets:, [0, lookAheadIndex+1]]
        batch = dt_support.shape[0]
#        print("dts.shape")
#        print(dts.shape)
#        print("dts[init_idx_rets:end_idx_rets, [0, lookAheadIndex+1]].shape")
#        print(dts[init_idx_rets:, [0, lookAheadIndex+1]].shape)
        #print("end_idx_rets-init_idx_rets")
        #print(end_idx_rets-init_idx_rets)
#        print("init_idx_rets")
#        print(init_idx_rets)
        #print("end_idx_rets")
        #print(end_idx_rets)
#        print("batch")
#        print(batch)
#        print("dt_support.shape")
#        print(dt_support.shape)
        # create support numpy vectors to speed up iterations
        v_support = Vars[:batch+seq_len, :, :]
        r_support = returns[init_idx_rets:, lookAheadIndex]
#        print(init_idx_rets)
#        print(returns.shape)
#        print(r_support.shape)
#        print("returns.shape")
#        print(returns.shape)
#        print("r_support.shape")
#        print(r_support.shape)
        # we only take here the entry time index, and later at DTA building the 
        # exit time index is derived from the entry time and the number of events to
        # look ahead
        i_support = ret_idx[init_idx_rets:, [0, lookAheadIndex+1]]
        
        b_support = bids[init_idx_rets:, [0, lookAheadIndex+1]]
        a_support = asks[init_idx_rets:, [0, lookAheadIndex+1]]
        # init formatted input and output
        X_i = np.zeros((batch, seq_len, nFeatures))
        # real-valued output
        R_i = np.zeros((batch, seq_len, 1))
        # output index vector
        I_i = np.zeros((batch, seq_len, 2))
        
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
            
            D_i[nI,:,:] = dt_support[nI,:]
            B_i[nI,:,:] = b_support[nI,:]
            A_i[nI,:,:] = a_support[nI,:]
        
        
        # normalize output
        if len(stds_out.shape)==2:
            R_i = R_i/stds_out[0, lookAheadIndex]#stdO#
        elif len(stds_out.shape)==1:
            R_i = R_i/stds_out[lookAheadIndex]#stdO#
        #OA_i = OA_i/stds_out[0,data.lookAheadIndex]
        # get decimal and binary outputs
        # TODO: generalize for the number of outputs
        if len(n_bits_outputs)==1:
            Y_i, y_dec = build_bin_output_mg(config, R_i, batch)
        else:
            Y_i, y_dec = build_bin_output_mcmdmg(config, R_i, batch)
        #YA_i, ya_dec = build_bin_output(model, OA_i, batch)
        # get samples per level
        for l in range(n_bits_outputs[-1]):
            IO['totalSampsPerLevel'][l] = IO['totalSampsPerLevel'][l]+np.sum(y_dec[:,-1,0]==l)
        #if dts[end_idx_rets, -1]>last_day:
        #pickle.dump( dt_support, open( '.\dts', "wb" ))
        # look for last entry containing last day
        if build_XY_mode=='K_fold':
            list_index2sets = map_index2sets(K, fold_idx)
        elif build_XY_mode=='manual':
            if 'list_index2sets' in config:
                list_index2sets = config['list_index2sets']
            else:
                list_index2sets = ['Tr','Cv']
        else:
            raise ValueError("build_XY_mode not known")
#        print("print(list_index2sets)")
#        print(list_index2sets)
        edges_tr_idx, edges_cv_idx = find_edge_indexes(dt_support, edges_dt, 
                                                       initenddates, fold_idx, 
                                                       list_index2sets,
                                                       first_day=first_day,
                                                       last_day=last_day)
        
        #samps_tr = 0
        if not skip_tr:
            for e in range(edges_tr_idx.shape[0]):
                # resize IO structures
                i_tr = edges_tr_idx[e, 0]
                e_tr = edges_tr_idx[e, 1]
                samps_tr = e_tr-i_tr
    #            print("samps_tr")
    #            print(samps_tr)
    #            print("X_i.shape")
    #            print(X_i.shape)
                Xtr.resize((pointerTr+samps_tr, seq_len, nFeatures))
                Ytr.resize((pointerTr+samps_tr, seq_len, size_output_layer))
                
                Itr.resize((pointerTr+samps_tr, seq_len, 2))
                # update IO structures
                Xtr[pointerTr:pointerTr+samps_tr,:,:] = X_i[i_tr:e_tr,:,:]
                Ytr[pointerTr:pointerTr+samps_tr,:,:] = Y_i[i_tr:e_tr,:,:]
                Itr[pointerTr:pointerTr+samps_tr,:,:] = I_i[i_tr:e_tr,:,:]
                if save_output:
                    Rtr.resize((pointerTr+batch, 1))
                    Rtr[pointerTr:pointerTr+batch,0] = R_i[i_tr:e_tr,0,0]
                # uodate pointer
                pointerTr += samps_tr
#        print("pointerTr")
#        print(pointerTr)
        if not skip_cv:
            for e in range(edges_cv_idx.shape[0]):
                # resize IO structures
                i_cv = edges_cv_idx[e, 0]
                e_cv = edges_cv_idx[e, 1]
                samps_cv = e_cv-i_cv
    #            print("samps_cv")
    #            print(samps_cv)
                Xcv.resize((pointerCv+samps_cv, seq_len, nFeatures))
                Ycv.resize((pointerCv+samps_cv, seq_len, size_output_layer))
                
                Icv.resize((pointerCv+samps_cv, seq_len, 2))
                # update IO structures
                Xcv[pointerCv:pointerCv+samps_cv,:,:] = X_i[i_cv:e_cv,:,:]
                Ycv[pointerCv:pointerCv+samps_cv,:,:] = Y_i[i_cv:e_cv,:,:]
                Icv[pointerCv:pointerCv+samps_cv,:,:] = I_i[i_cv:e_cv,:,:]
                if save_output:
                    Rcv.resize((pointerCv+samps_cv, 1))
                    Rcv[pointerTr:pointerCv+samps_cv,0] = R_i[i_cv:e_cv,0,0]
                # resize
                D.resize((pointerCv+samps_cv, seq_len, 2))
                B.resize((pointerCv+samps_cv, seq_len, 2))
                A.resize((pointerCv+samps_cv, seq_len, 2))
                # update
                D[pointerCv:pointerCv+samps_cv,:,:] = D_i[i_cv:e_cv,:,:]
                B[pointerCv:pointerCv+samps_cv,:,:] = B_i[i_cv:e_cv,:,:]
                A[pointerCv:pointerCv+samps_cv,:,:] = A_i[i_cv:e_cv,:,:]
                pointerCv += samps_cv
#        print("pointerCv")
#        print(pointerCv)
        #print(pointer)
    # end of for i in range(chunks):
    # update dictionary
    IO['Xtr'] = Xtr
    IO['Ytr'] = Ytr
    IO['Itr'] = Itr
    IO['pointerTr'] = pointerTr
    if save_output:
        IO['Rtr'] = Rtr
    
    IO['Xcv'] = Xcv
    IO['Ycv'] = Ycv
    IO['Icv'] = Icv
    IO['pointerCv'] = pointerCv
    if save_output:
        IO['Rcv'] = Rcv
    IO['Dcv'] = D
    IO['Bcv'] = B
    IO['Acv'] = A
    
    return IO

def get_list_unique_days(thisAsset):
    """  """
    import re
    dir_ass = local_vars.data_dir+thisAsset
    try:
        list_dir = sorted(os.listdir(dir_ass))
        list_regs = [re.search('^'+thisAsset+'_\d+'+'.txt$',f) for f in list_dir]
        list_unique_days = list(set([re.search('\d+',m.group()).group()[:8] for m in list_regs]))
        
        pickle.dump( list_unique_days, open( local_vars.hdf5_directory+"list_unique_days_"+thisAsset+".p", "wb" ))
    except FileNotFoundError:
        print("WARNING! dir_ass not found. Loading list_unique_days from pickle")
        list_unique_days = pickle.load( open( local_vars.hdf5_directory+"list_unique_days_"+thisAsset+".p", "rb" ))
    return list_unique_days

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

def get_edges_datasets(K, config, dataset_dirfilename=''):
    """ Get the edges representing days that split the dataset in K-1/K ratio of
    samples for training and 1/K ratio for cross-validation """
    print("Getting dataset edges...")
    if 'nEventsPerStat' in config:
        nEventsPerStat = config['nEventsPerStat']
    else:
        nEventsPerStat = 5000
    if 'movingWindow' in config:
        movingWindow = config['movingWindow']
    else:
        movingWindow = 500
    if 'feature_keys_manual' in config:
        feature_keys_manual = config['feature_keys_manual']
    else:
        feature_keys_manual = [i for i in range(37)]
    n_feats_manual = len(feature_keys_manual)
    if 'assets' in config:
        assets = config['assets']
    else:
        assets = [1,2,3,4,7,8,10,11,12,13,14,15,16,17,19,27,28,29,30,31,32]
    if 'build_XY_mode' in config:
        build_XY_mode = config['build_XY_mode']
    else:
        build_XY_mode = 'K_fold'
    if 'edge_dates' in config:
        edge_dates = config['edge_dates']
    else:
        edge_dates = ['2018.03.09']
    if dataset_dirfilename=='':
        dataset_dirfilename = local_vars.hdf5_directory+'IOA_mW'+str(movingWindow)+'_nE'+\
                            str(nEventsPerStat)+'_nF'+str(n_feats_manual)+'.hdf5'
    if build_XY_mode=='K_fold':
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
        edges_dt = [dt.datetime.fromordinal((first_day+dt.timedelta(days=int(d))).toordinal())  for d in edges_idx]
    elif build_XY_mode=='manual':
        edges_dt = [dt.datetime.strptime(edge_date,'%Y.%m.%d') for edge_date in edge_dates]
        edges = [edge_dt.date() for edge_dt in edges_dt]
    else:
        raise ValueError("build_XY_mode not recognized")
    return edges, edges_dt

def load_stats_manual_v2(config, thisAsset, ass_group, from_stats_file=False, 
               hdf5_directory='', save_pickle=False, tag='IOB'):
    """
    Function that loads stats
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
    # init or load total stats
    stats = {}

    if not from_stats_file:
        stats["means_t_in"] = ass_group.attrs.get("means_t_in")
        stats["stds_t_in"] = ass_group.attrs.get("stds_t_in")
        stats["m_t_in"] = ass_group.attrs.get("m_t_in")
    
    elif from_stats_file:
        try:
            stats = pickle.load( open( local_vars.stats_directory+thisAsset+'_'+tag+'stats_mW'+
                                      str(movingWindow)+
                                     '_nE'+str(nEventsPerStat)+
                                     '_nF'+str(nF)+".p", "rb" ))
        except FileNotFoundError:
            print("WARNING FileNotFoundError: "+hdf5_directory+thisAsset+'_'+tag+'stats_mW'+
                                      str(movingWindow)+
                                     '_nE'+str(nEventsPerStat)+
                                     '_nF'+str(nF)+".p. Getting stats from features file")
            stats["means_t_in"] = ass_group.attrs.get("means_t_in")
            stats["stds_t_in"] = ass_group.attrs.get("stds_t_in")
            stats["m_t_in"] = ass_group.attrs.get("m_t_in")
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

def load_stats_output_v2(config, hdf5_directory, thisAsset, tag='IOB'):
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
    stats = pickle.load( open( local_vars.stats_directory+thisAsset+'_'+tag+'stats_mW'+
                                      str(movingWindow)+
                                     '_nE'+str(nEventsPerStat)+
                                     '_nF'+str(nF)+".p", "rb" ))
    stats_output = {'m_t_out':stats['m_t_out'],
                    'stds_t_out':stats['stds_t_out']}
    return stats_output

def load_manual_features_v2(config, thisAsset, separators, f_prep_IO, s):
    """
    Function that extracts features from previously saved structures.
    Args:
        - data:
        - thisAsset
        - separators
        - f_prep_IO
        - group
        - hdf5_directory
        - s
    Returns:
        - features
    """
    if 'nEventsPerStat' in config:
        nEventsPerStat = config['nEventsPerStat']
    else:
        nEventsPerStat = 1000
    # init structures
    features = []
    # number of events
    nE = separators.index[s+1]-separators.index[s]+1
    # check if number of events is not enough to build two features and one return
    if nE>=2*nEventsPerStat:
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
            raise ValueError("Group should already exist. Run get_features_results_stats_from_raw first to create it.")
        else:
            # get group from file
            group = f_prep_IO[group_name]
        # get features, returns and stats if don't exist
        if group.attrs.get("means_in") is None:
            raise ValueError("Group should already exist. Run get_features_results_stats_from_raw first to create it.")
        else:
            # get data sets
            features = group['features']
            
    # end of if separators.index[s+1]-separators.index[s]>=2*data.nEventsPerStat:
    # save results in a dictionary
#    features_struct = {}
#    features_struct['features'] = features
    
    return features

def load_returns_v2(config, hdf5_directory, thisAsset, separators, filename_prep_IO, s):
    """
    Function that extracts features, results and normalization stats from already saved
    structures.
    Args:
        - data:
        - thisAsset
        - separators
        - f_prep_IO
        - group
        - hdf5_directory
        - s
    Returns:
        - features 
        - returns
        - ret_idx
    """
    if 'nEventsPerStat' in config:
        nEventsPerStat = config['nEventsPerStat']
    else:
        nEventsPerStat = 1000
    f_prep_IO = h5py.File(filename_prep_IO,'r')
    # init structures
    returns = []
    ret_idx = []
    # number of events
    nE = separators.index[s+1]-separators.index[s]+1
    # check if number of events is not enough to build two features and one return
    if nE>=2*nEventsPerStat:
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
            print("Group should already exist. Run get_features_results_stats_from_raw first to create it.")
            raise ValueError
        else:
            # get group from file
            group = f_prep_IO[group_name]
        # get features, returns and stats if don't exist
        if group.attrs.get("means_in") is None:
            print("Group should already exist. Run get_features_results_stats_from_raw first to create it.")
            raise ValueError
        else:
            # get data sets
            returns = group['returns']
            ret_idx = group['ret_idx']
            if 'DT' in group:
                DT = group['DT']
                B = group['B']
                A = group['A']
    # end of if separators.index[s+1]-separators.index[s]>=2*data.nEventsPerStat:
    # save results in a dictionary
    returns_struct = {}
    returns_struct['returns'] = returns
    returns_struct['ret_idx'] = ret_idx
    returns_struct['group_name'] = group_name
    if 'DT' in group:
        returns_struct['DT'] = DT
        returns_struct['B'] = B
        returns_struct['A'] = A
    return returns_struct

def build_DTA_v2(config, AllAssets, D, B, A, ass_IO_ass):
    """
    Function that builds structure based on IO to later get Journal and ROIs.
    Args:
        - data
        - I: structure containing indexes
        - ass_IO_ass: asset to IO assignment
    """
    if 'assets' in config:
        assets = config['assets']
    else:
        assets = [1,2,3,4,7,8,10,11,12,13,14,15,16,17,19,27,28,29,30,31,32]
    # init columns
    columns = ["DT1","DT2","B1","B2","A1","A2","Asset"]
    # init DTA
    DTA = pd.DataFrame()
    # init hdf5 file with raw data
    ass_index = 0
    last_ass_IO_ass = 0
    # loop over assets
    for ass in assets:
        # get this asset's name
        thisAsset = AllAssets[str(ass)]
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
            # TODO: Check that DTA set belongs to cros-val set
#            if DTA_i['DT1'].iloc[0][:10] not in data.dateTest:
#                print("WARNING!!! DTA_i['DT1'].iloc[0][:10] not in data.dateTest")
            #assert(DTA_i['DT1'].iloc[0][:10] in data.dateTest)
#            assert(DTA_i['DT1'].iloc[-1][:10] in data.dateTest)
#            if DTA_i['DT1'].iloc[-1][:10] not in data.dateTest:
#                print("WARNING!!! DTA_i['DT1'].iloc[-1][:10] not in data.dateTest")
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

def build_datasets(folds=3, fold_idx=0, config={}, log=''):
    """  """
    ticTotal = time.time()
    # create data structure
    if config=={}:    
        config = retrieve_config('CRNN00000')
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
    if 'feats_from_bids' in config:
        feats_from_bids = config['feats_from_bids']
    else:
        feats_from_bids = True
    if 'seq_len' in config:
        seq_len = config['seq_len']
    else:
        seq_len = int((config['lB']-config['nEventsPerStat'])/config['movingWindow']+1)
    if 'filetag' in config:
        filetag = config['filetag']
    else:
        filetag = config['IO_results_name']
    size_output_layer = config['size_output_layer']
    if 'n_bits_outputs' in config:
        n_bits_outputs = config['n_bits_outputs']
    else:
        n_bits_outputs = [size_output_layer]
    if 'build_test_db' in config:
        build_test_db = config['build_test_db']
    else:
        build_test_db = False
    
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
    
    if not build_test_db:
        filename_prep_IO = (hdf5_directory+tag+str(movingWindow)+'_nE'+
                            str(nEventsPerStat)+'_nF'+str(nFeatures)+'.hdf5')
        separators_directory = hdf5_directory+'separators/'
    else:
        filename_prep_IO = (hdf5_directory+tag+str(movingWindow)+'_nE'+
                            str(nEventsPerStat)+'_nF'+str(nFeatures)+'_test.hdf5')
        separators_directory = hdf5_directory+'separators_test/'
    
    edges, edges_dt = get_edges_datasets(folds, config, dataset_dirfilename=filename_prep_IO)
    
    filename_tr = IO_directory+'IOKFW'+filetag[1:]+'.hdf5'
    filename_cv = IO_directory+'IOKF'+filetag+'.hdf5'
    
    if len(log)>0:
        write_log(filename_tr)
    if len(log)>0:
        write_log(filename_cv)
    f_prep_IO = h5py.File(filename_prep_IO,'r')
    print("filename_prep_IO")
    print(filename_prep_IO)
        
    if os.path.exists(filename_tr) and os.path.exists(filename_cv):
        if_build_IO = False
    else:
        if_build_IO = config['if_build_IO']
    #if_build_IO=True
    # create model
    # if IO structures have to be built 
    if if_build_IO:
        print("Tag = "+str(tag))
        IO = {}
        #attributes to track asset-IO belonging
        ass_IO_ass_tr = np.zeros((len(assets))).astype(int)
        # structure that tracks the number of samples per level
        IO['totalSampsPerLevel'] = np.zeros((n_bits_outputs[-1]))
        # open IO file for writting
        f_tr = h5py.File(filename_tr,'w')
        # init IO data sets
        Xtr = f_tr.create_dataset('X', 
                                (0, seq_len, nFeatures), 
                                maxshape=(None,seq_len, nFeatures), 
                                dtype=float)
        Ytr = f_tr.create_dataset('Y', 
                                (0, seq_len, size_output_layer),
                                maxshape=(None, seq_len, size_output_layer),
                                dtype=float)
        
            
        Itr = f_tr.create_dataset('I', 
                                (0, seq_len,2),maxshape=(None, seq_len, 2),
                                dtype=int)
        Rtr = f_tr.create_dataset('R', 
                                (0,1),
                                maxshape=(None,1),
                                dtype=float)
        
        IO['Xtr'] = Xtr
        IO['Ytr'] = Ytr
        IO['Itr'] = Itr
        IO['Rtr'] = Rtr # return
        IO['pointerTr'] = 0
        
        ass_IO_ass_cv = np.zeros((len(assets))).astype(int)
        f_cv = h5py.File(filename_cv,'w')
        # init IO data sets
        Xcv = f_cv.create_dataset('X', 
                                (0, seq_len, nFeatures), 
                                maxshape=(None,seq_len, nFeatures), 
                                dtype=float)
        Ycv = f_cv.create_dataset('Y', 
                                (0, seq_len, size_output_layer),
                                maxshape=(None, seq_len, size_output_layer),
                                dtype=float)
            
        Icv = f_cv.create_dataset('I', 
                                (0, seq_len,2),maxshape=(None, seq_len, 2),
                                dtype=int)
        Rcv = f_cv.create_dataset('R', 
                                (0,1),
                                maxshape=(None,1),
                                dtype=float)
        
        
        # save IO structures in dictionary
        
        Dcv = f_cv.create_dataset('D', (0,seq_len,2),
                                maxshape=(None,seq_len,2),dtype='S19')
        Bcv = f_cv.create_dataset('B', (0,seq_len,2),
                                maxshape=(None,seq_len,2),dtype=float)
        Acv = f_cv.create_dataset('A', (0,seq_len,2),
                                maxshape=(None,seq_len,2),dtype=float)
        IO['Xcv'] = Xcv
        IO['Ycv'] = Ycv
        IO['Icv'] = Icv
        IO['Rcv'] = Rcv # return
        IO['Dcv'] = Dcv
        IO['Bcv'] = Bcv
        IO['Acv'] = Acv
        IO['pointerCv'] = 0
        
    IO_results_name = IO_directory+'DTA_'+filetag+'.p'
    print(IO_results_name)
    if len(log)>0:
        write_log(IO_results_name)
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
                                     from_txt=1)
        # retrive asset group
        ass_group = f_prep_IO[thisAsset]
        stats_manual = load_stats_manual_v2(config, 
                                            thisAsset, 
                                            ass_group,
                                            from_stats_file=from_stats_file, 
                                            hdf5_directory=hdf5_directory+'stats/', 
                                            tag=tag_stats)
        
        stats_output = load_stats_output_v2(config, hdf5_directory+'stats/', 
                                            thisAsset, tag=tag_stats)
        
        if if_build_IO:
            mess = str(ass)+". "+thisAsset
            print(mess)
            if len(log)>0:
                write_log(mess)
            # loop over separators
            for s in range(0,len(separators)-1,2):
                mess = "\ts {0:d} of {1:d}".format(int(s/2),int(len(separators)/2-1))+\
                    ". From "+separators.DateTime.iloc[s]+" to "+\
                    separators.DateTime.iloc[s+1]
                print(mess)
                if len(log)>0:
                    write_log(mess)
                # number of events within this separator chunk
                nE = separators.index[s+1]-separators.index[s]+1
                # get first day after separator
                #day_s = separators.DateTime.iloc[s][0:10]
                # check if number of events is not enough to build two features and one return
                if nE>=seq_len*(nEventsPerStat+movingWindow):
#                    print("nE")
#                    print(nE)
                    if 1:
                            #if day_s not in data.dateTest and day_s<=data.dateTest[-1]:
                        
                        # load features, returns and stats from HDF files
                        features_manual = load_manual_features_v2(config, 
                                                               thisAsset, 
                                                               separators, 
                                                               f_prep_IO, 
                                                               s)

                        # load returns
                        returns_struct = load_returns_v2(config, hdf5_directory, 
                                                         thisAsset, separators, 
                                                         filename_prep_IO, s)
                        try:
                            file_temp_name = local_vars.IO_directory+\
                                'temp_train_build'+\
                                str(np.random.randint(10000))+'.hdf5'
                            while os.path.exists(file_temp_name):
                                file_temp_name = IO_directory+'temp_train_build'\
                                    +str(np.random.randint(10000))+'.hdf5'
                            file_temp = h5py.File(file_temp_name,'w')
                            Vars = build_variations(config, file_temp, 
                                                    features_manual, 
                                                    stats_manual)
                            IO = build_XY(config, Vars, returns_struct, 
                                          stats_output, IO, edges_dt,
                                          folds, fold_idx, save_output=False)
                            # close temp file
                            file_temp.close()
                            os.remove(file_temp_name)
                        except (KeyboardInterrupt):
                            mess = "KeyBoardInterrupt. Closing files and exiting program."
                            print(mess)
                            if len(log)>0:
                                write_log(mess)
                            f_tr.close()
                            f_cv.close()
                            file_temp.close()
                            os.remove(file_temp_name)
                            if f_prep_IO != None:
                                f_prep_IO.close()
                            raise KeyboardInterrupt
                    else:
                        mess = "\tNot in the set. Skipped."
                        print(mess)
                        if len(log)>0:
                            write_log(mess)
                        # end of if (tOt=='train' and day_s not in data.dateTest) ...
                    
                else:
                    print("\ts {0:d} of {1:d}. Not enough entries. Skipped.".format(
                            int(s/2),int(len(separators)/2-1)))
        # end of for s in range(0,len(separators)-1,2):
        # add pointer index for later separating assets
        if if_build_IO:
            ass_IO_ass_tr[ass_idx] = IO['pointerTr']
            ass_IO_ass_cv[ass_idx] = IO['pointerCv']
            mess = "\tTime for "+thisAsset+":"+str(np.floor(time.time()-tic))+"s"+\
              ". Total time:"+str(np.floor(time.time()-ticTotal))+"s"
            print(mess)
            if len(log)>0:
                write_log(mess)
            
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
        mess = "Building DTA..."
        print(mess)
        if len(log)>0:
            write_log(mess)
        DTA = build_DTA_v2(config, AllAssets, IO['Dcv'], 
                           IO['Bcv'], IO['Acv'], ass_IO_ass_cv)
        pickle.dump( DTA, open( IO_results_name, "wb" ))
        f_cv.attrs.create('ass_IO_ass', ass_IO_ass_cv, dtype=int)
        f_tr.attrs.create('ass_IO_ass', ass_IO_ass_tr, dtype=int)
        f_cv.attrs.create('totalSampsPerLevel', IO['totalSampsPerLevel'], dtype=int)
        f_tr.attrs.create('totalSampsPerLevel', IO['totalSampsPerLevel'], dtype=int)
        totalSampsPerLevel = IO['totalSampsPerLevel']
    # print percent of samps per level
    else:
        # get ass_IO_ass from disk
        f_cv = h5py.File(filename_cv,'r')
        f_tr = h5py.File(filename_tr,'r')
        ass_IO_ass_cv = f_cv.attrs.get("ass_IO_ass")
        ass_IO_ass_tr = f_tr.attrs.get("ass_IO_ass")
        if 'totalSampsPerLevel' in f_tr:
            totalSampsPerLevel = f_tr.attrs.get("totalSampsPerLevel")
        elif 'totalSampsPerLevel' in f_cv:
            totalSampsPerLevel = f_cv.attrs.get("totalSampsPerLevel")
        else: 
            totalSampsPerLevel = [-1]
    # get total number of samps
    m_tr = ass_IO_ass_tr[-1]
    m_cv = ass_IO_ass_cv[-1]
    m_t = m_tr+m_cv
    print("Edges:")
    print(edges)
    print(filename_tr)
    print(filename_cv)
    print(IO_results_name)
    mess = "Samples for fitting: "+str(m_tr)+"\n"+"Samples for cross-validation: "+\
        str(m_cv)+"\n"+"Total samples: "+str(m_t)
    print(mess)
    if len(log)>0:
        write_log(mess)
    if sum(totalSampsPerLevel)>0:
        mess = "Percent per level:"+str(IO['totalSampsPerLevel']/m_t)
        print(mess)
        if len(log)>0:
            write_log(mess)
    else:
        print("totalSampsPerLevel not in IO :(")
    f_tr.close()
    f_cv.close()
    mess = "DONE building IO"
    print(mess)
    if len(log)>0:
        write_log(mess)
    return filename_tr, filename_cv, IO_results_name

def get_edges_datasets_modular(K, config, separators_directory, symbol='bid'):
    """ Get the edges representing days that split the dataset in K-1/K ratio of
    samples for training and 1/K ratio for cross-validation """
    print("Getting dataset edges...")
    if 'nEventsPerStat' in config:
        nEventsPerStat = config['nEventsPerStat']
    else:
        nEventsPerStat = 5000
    if 'movingWindow' in config:
        movingWindow = config['movingWindow']
    else:
        movingWindow = 500
    if 'assets' in config:
        assets = config['assets']
    else:
        assets = [1,2,3,4,7,8,10,11,12,13,14,15,16,17,19,27,28,29,30,31,32]
    if 'build_XY_mode' in config:
        build_XY_mode = config['build_XY_mode']
    else:
        build_XY_mode = 'K_fold'
    if 'edge_dates' in config:
        edge_dates = config['edge_dates']
    else:
        edge_dates = ['2018.03.09']
    if 'asset_relation' in config:
        asset_relation = config['asset_relation']
    else:
        asset_relation = 'direct'
    
    if build_XY_mode=='K_fold':
        weights_ass = np.zeros((len(assets),1))
        samps_ass = np.zeros((len(assets)))
        n_days_ass = np.zeros((len(assets)))
        
        first_day_str = config['first_day']
        last_day_str = config['last_day']
        first_day = dt.date(int(first_day_str[:4]), int(first_day_str[5:7]), int(first_day_str[8:]))
        last_day = dt.date(int(last_day_str[:4]), int(last_day_str[5:7]), int(last_day_str[8:]))
        max_days = (last_day-first_day).days+1
        A = np.zeros((len(assets), max_days))
        AllAssets = Data().AllAssets
        for a, ass in enumerate(assets):
            thisAsset = AllAssets[str(ass)]
            
            separators = load_separators(thisAsset, 
                                     separators_directory, 
                                     from_txt=1)
            first_date = dt.datetime.strftime(dt.datetime.strptime(
                    separators.DateTime.iloc[0],'%Y.%m.%d %H:%M:%S'),'%y%m%d%H%M%S')
            last_date = dt.datetime.strftime(dt.datetime.strptime(
                    separators.DateTime.iloc[-1],'%Y.%m.%d %H:%M:%S'),'%y%m%d%H%M%S')
            
            samps_ass[a] = pickle.load(open( local_vars.hdf5_directory+'stats_modular/mW'+str(movingWindow)+
                                     'nE'+str(nEventsPerStat)+'/'+asset_relation+'/'+thisAsset+'_'+symbol+'_out_'+
                                     first_date+last_date+'.p', "rb"))['m']
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
        edges_dt = [dt.datetime.fromordinal((first_day+dt.timedelta(days=int(d))).toordinal())  for d in edges_idx]
    elif build_XY_mode=='manual':
        edges_dt = [dt.datetime.strptime(edge_date,'%Y.%m.%d') for edge_date in edge_dates]
        edges = [edge_dt.date() for edge_dt in edges_dt]
    else:
        raise ValueError("build_XY_mode not recognized")
    return edges, edges_dt

def load_features_modular(config, thisAsset, separators, assdirname, init_date, end_date, shift):
    """
    Function that extracts features from previously saved structures.
    """
    feature_keys = config['feature_keys']
    groupdirname = assdirname+init_date+end_date+'/'
    m_in = pickle.load( open( groupdirname+"m_in_"+str(shift)+".p", "rb" ))
    features = np.zeros((m_in, len(feature_keys)))
    for i,f in enumerate(feature_keys):
        filedirname = groupdirname+C.PF[f][0]+'_'+str(shift)+'.hdf5'
        feature_file = h5py.File(filedirname,'r')
        features[:,i] = feature_file[C.PF[f][0]][C.PF[f][0]][:,0]
            
    return features

def load_stats_modular(config, thisAsset, first_date, last_date, symbol, ass_rel):
    """  """
    movingWindow = config['movingWindow']
    nEventsPerStat = config['nEventsPerStat']
    if config['feats_from_bids']:
        symbol_type = 'bid'
    else:
        symbol_type = 'ask'
    feature_keys = config['feature_keys']
    asset_relation = ass_rel#config['asset_relation']#
    nChannels = int(nEventsPerStat/movingWindow)
    stats_dir = local_vars.stats_modular_directory+'mW'+str(movingWindow)+'nE'+str(nEventsPerStat)+'/'+asset_relation+'/'
    means_in = np.zeros((nChannels, len(feature_keys)))
    stds_in = np.zeros((nChannels, len(feature_keys)))
    for i, feat in enumerate(feature_keys):
        key = C.PF[feat][0]
        # copy in stats directory
        filedirname = stats_dir+thisAsset+'_'+symbol_type+'_'+key+'_'+first_date+last_date+'.p'
        stats = pickle.load(open( filedirname, "rb"))
        means_in[:,i] = stats['mean'][:,0]
        stds_in[:,i] = stats['std'][:,0]
    stats_in = {'means_t_in':means_in,
                'stds_t_in':stds_in}
    filedirname = stats_dir+thisAsset+'_'+symbol_type+'_out_'+first_date+last_date+'.p'
    out = pickle.load( open( filedirname, "rb" ))
    stats_out = {'stds_t_out':out['std_'+symbol],
                 'means_t_out':out['mean_'+symbol],
                 'm_t_out':out['m']}
    return stats_in, stats_out

def load_stats_modular_oneNet(config, thisAsset, first_date, last_date, symbol, asset_relation):
    """  """
    movingWindow = config['movingWindow']
    nEventsPerStat = config['nEventsPerStat']
    feature_keys = config['feature_keys']
    nChannels = int(nEventsPerStat/movingWindow)
    stats_dir = local_vars.stats_modular_directory+'mW'+str(movingWindow)+'nE'+str(nEventsPerStat)+'/'+asset_relation+'/'
    means_in = np.zeros((nChannels, len(feature_keys)))
    stds_in = np.zeros((nChannels, len(feature_keys)))
    for i, feat in enumerate(feature_keys):
        key = C.PF[feat][0]
        # copy in stats directory
        filedirname = stats_dir+thisAsset+'_'+symbol+'_'+key+'_'+first_date+last_date+'.p'
        stats = pickle.load(open( filedirname, "rb"))
        means_in[:,i] = stats['mean'][:,0]
        stds_in[:,i] = stats['std'][:,0]
    stats_in = {'means_t_in':means_in,
                'stds_t_in':stds_in}
    filedirname = stats_dir+thisAsset+'_'+symbol+'_out_'+first_date+last_date+'.p'
    out = pickle.load( open( filedirname, "rb" ))
    stats_out = {'stds_t_out':out['std_'+symbol],
                 'means_t_out':out['mean_'+symbol],
                 'm_t_out':out['m']}
    return stats_in, stats_out

def load_returns_modular(config, groupoutdirname, thisAsset, separators, symbol, init_date, end_date, shift):
    """
    Function that extracts results from previously saved structures.
    
    """
    
    # init structures
    returns = []
    ret_idx = []
    
    file = h5py.File(groupoutdirname+'output_'+str(shift)+'.hdf5','r')
    returns = file['returns_'+symbol]
    ret_idx = file['ret_idx']
    DT = file['DT']
    B = file['B']
    A = file['A']
    # group name of separator s
    group_name = thisAsset+'/'+init_date+end_date
    # save results in a dictionary
    returns_struct = {}
    returns_struct['returns'] = returns
    returns_struct['ret_idx'] = ret_idx
    returns_struct['group_name'] = group_name
    returns_struct['DT'] = DT
    returns_struct['B'] = B
    returns_struct['A'] = A
    
    return returns_struct

def load_output_stats_modular(config, hdf5_directory, thisAsset, tag='IOB'):
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
    stats = pickle.load( open( local_vars.stats_directory+thisAsset+'_'+tag+'stats_mW'+
                                      str(movingWindow)+
                                     '_nE'+str(nEventsPerStat)+
                                     '_nF'+str(nF)+".p", "rb" ))
    stats_output = {'m_t_out':stats['m_t_out'],
                    'stds_t_out':stats['stds_t_out']}
    return stats_output

def build_DTA_shift(config, AllAssets, D, B, A, ass_IO_ass):
    """
    Function that builds structure based on IO to later get Journal and ROIs.
    Args:
        - data
        - I: structure containing indexes
        - ass_IO_ass: asset to IO assignment
    """
    if 'movingWindow' in config:
        movingWindow = config['movingWindow']
    else:
        movingWindow = 500
    if 'assets' in config:
        assets = config['assets']
    else:
        assets = [1,2,3,4,7,8,10,11,12,13,14,15,16,17,19,27,28,29,30,31,32]
    if 'phase_shift' in config:
        phase_shift = config['phase_shift']
    else:
        phase_shift = 1
    phase_size = movingWindow/phase_shift
    shifts = [int(phase_size*phase) for phase in range(phase_shift)]
    # init columns
    columns = ["DT1","DT2","B1","B2","A1","A2","Asset"]
    # init DTA
    DTA = pd.DataFrame()
    # init hdf5 file with raw data
    ass_index = 0
    last_ass_IO_ass = 0
    # loop over assets
    for ass in assets:
        # get this asset's name
        thisAsset = AllAssets[str(ass)]
        print(thisAsset)
        for shift in shifts:
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

def sort_input(array, sorted_idx, prevPointerCv, char=False):
    """  """
    if char:
        temp = np.chararray(array[prevPointerCv:,:,:].shape, itemsize=19)
    else:
        temp = np.zeros(array[prevPointerCv:,:,:].shape)
    temp[:,:,:] = array[prevPointerCv:,:,:]
    temp[:,:,:] = temp[sorted_idx,:,:] 
#    IO['Bcv'][prevPointerCv:,:,:] = temp
    return temp

def build_datasets_modular(folds=3, fold_idx=0, config={}, log=''):
    """  """
    ticTotal = time.time()
    # create data structure
    if config=={}:    
        config = retrieve_config('CRNN00000')
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
    if 'feature_keys' not in config:
        feature_keys = [i for i in range(37)]
    else:
        feature_keys = config['feature_keys']
    nFeatures = len(feature_keys)
    if 'feats_from_bids' in config:
        feats_from_bids = config['feats_from_bids']
    else:
        feats_from_bids = True
    if 'seq_len' in config:
        seq_len = config['seq_len']
    else:
        seq_len = int((config['lB']-config['nEventsPerStat'])/config['movingWindow']+1)
    size_output_layer = config['size_output_layer']
    if 'n_bits_outputs' in config:
        n_bits_outputs = config['n_bits_outputs']
    else:
        n_bits_outputs = [size_output_layer]
    if 'build_test_db' in config:
        build_test_db = config['build_test_db']
    else:
        build_test_db = False
    if 'build_asset_relations' in config:
        build_asset_relations = config['build_asset_relations']
    else:
        build_asset_relations = ['direct']
    print("build_asset_relations")
    print(config['build_asset_relations'])
    if 'asset_relation' in config:
        asset_relation = config['asset_relation']
    else:
        asset_relation = 'direct'
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
    
    
    IO_tr_name = config['IO_tr_name']
    IO_cv_name = config['IO_cv_name']
    filename_tr = IO_directory+'KFTr'+IO_tr_name+'.hdf5'
    filename_cv = IO_directory+'KFCv'+IO_cv_name+'.hdf5'
    IO_results_name = IO_directory+'DTA_'+IO_cv_name+'.p'
    print(filename_tr)
    print(filename_cv)
    print(IO_results_name)
    
    
    edges, edges_dt = get_edges_datasets_modular(folds, config, separators_directory, symbol=symbol)
    print("Edges:")
    print(edges)
    
    if len(log)>0:
        write_log(filename_tr)
    if len(log)>0:
        write_log(filename_cv)
        
    if os.path.exists(filename_tr) and os.path.exists(filename_cv):
        if_build_IO = False
    else:
        if_build_IO = config['if_build_IO']
    # create model
    # if IO structures have to be built 
    if if_build_IO:
        #print("Tag = "+str(tag))
        IO = {}
        #attributes to track asset-IO belonging
        ass_IO_ass_tr = np.zeros((len(assets))).astype(int)
        # structure that tracks the number of samples per level
        IO['totalSampsPerLevel'] = np.zeros((n_bits_outputs[-1]))
        # open IO file for writting
        f_tr = h5py.File(filename_tr,'w')
        # init IO data sets
        Xtr = f_tr.create_dataset('X', 
                                (0, seq_len, nFeatures), 
                                maxshape=(None,seq_len, nFeatures), 
                                dtype=float)
        Ytr = f_tr.create_dataset('Y', 
                                (0, seq_len, size_output_layer),
                                maxshape=(None, seq_len, size_output_layer),
                                dtype=float)
        
            
        Itr = f_tr.create_dataset('I', 
                                (0, seq_len,2),maxshape=(None, seq_len, 2),
                                dtype=int)
        Rtr = f_tr.create_dataset('R', 
                                (0,1),
                                maxshape=(None,1),
                                dtype=float)
        
        IO['Xtr'] = Xtr
        IO['Ytr'] = Ytr
        IO['Itr'] = Itr
        IO['Rtr'] = Rtr # return
        IO['pointerTr'] = 0
        
        ass_IO_ass_cv = np.zeros((len(assets))).astype(int)
        f_cv = h5py.File(filename_cv,'w')
        # init IO data sets
        Xcv = f_cv.create_dataset('X', 
                                (0, seq_len, nFeatures), 
                                maxshape=(None,seq_len, nFeatures), 
                                dtype=float)
        Ycv = f_cv.create_dataset('Y', 
                                (0, seq_len, size_output_layer),
                                maxshape=(None, seq_len, size_output_layer),
                                dtype=float)
            
        Icv = f_cv.create_dataset('I', 
                                (0, seq_len,2),maxshape=(None, seq_len, 2),
                                dtype=int)
        Rcv = f_cv.create_dataset('R', 
                                (0,1),
                                maxshape=(None,1),
                                dtype=float)
        
        
        # save IO structures in dictionary
        
        Dcv = f_cv.create_dataset('D', (0,seq_len,2),
                                maxshape=(None,seq_len,2),dtype='S19')
        Bcv = f_cv.create_dataset('B', (0,seq_len,2),
                                maxshape=(None,seq_len,2),dtype=float)
        Acv = f_cv.create_dataset('A', (0,seq_len,2),
                                maxshape=(None,seq_len,2),dtype=float)
        IO['Xcv'] = Xcv
        IO['Ycv'] = Ycv
        IO['Icv'] = Icv
        IO['Rcv'] = Rcv # return
        IO['Dcv'] = Dcv
        IO['Bcv'] = Bcv
        IO['Acv'] = Acv
        IO['pointerCv'] = 0
    
    
    if len(log)>0:
        write_log(IO_results_name)
    # index asset
    ass_idx = 0
    # array containing bids means
    #bid_means = pickle.load( open( "../HDF5/bid_means.p", "rb" ))
    # loop over all assets
    for ass in assets:
        if if_build_IO:
            thisAsset = C.AllAssets[str(ass)]
            assdirnames = [featuredirname+thisAsset+'/' for featuredirname in featuredirnames]
            outassdirnames = [outrdirname+thisAsset+'/' for outrdirname in outrdirnames]
            
            tic = time.time()
            # load separators
            separators = load_separators(thisAsset, 
                                         separators_directory, 
                                         from_txt=1)
            if not build_test_db:
                sep_for_stats = separators
            else:
                sep_for_stats = load_separators(thisAsset, 
                                         hdf5_directory+'separators/', 
                                         from_txt=1)

            first_date = dt.datetime.strftime(dt.datetime.strptime(
                    sep_for_stats.DateTime.iloc[0],'%Y.%m.%d %H:%M:%S'),'%y%m%d%H%M%S')
            last_date = dt.datetime.strftime(dt.datetime.strptime(
                    sep_for_stats.DateTime.iloc[-1],'%Y.%m.%d %H:%M:%S'),'%y%m%d%H%M%S')
            
            list_stats_in = [None for _ in range(len(outassdirnames))]
            list_stats_out = [None for _ in range(len(outassdirnames))]
            for ind, assdirname in enumerate(assdirnames):
                if 'direct' in assdirname:
                    ass_rel = 'direct'
                elif 'inverse' in assdirname:
                    ass_rel = 'inverse'
                else:
                    raise ValueError
                list_stats_in[ind], list_stats_out[ind] = load_stats_modular(config, thisAsset, first_date, last_date, symbol, ass_rel)
    #        stats_output = load_output_stats_modular(config, hdf5_directory+'stats/', 
    #                                            thisAsset, tag=tag_stats)
        
        
            mess = str(ass)+". "+thisAsset
            print(mess)
            if len(log)>0:
                write_log(mess)
            # loop over separators
            for s in range(0,len(separators)-1,2):
                mess = "\ts {0:d} of {1:d}".format(int(s/2),int(len(separators)/2-1))+\
                    ". From "+separators.DateTime.iloc[s]+" to "+\
                    separators.DateTime.iloc[s+1]
                print(mess)
                if len(log)>0:
                    write_log(mess)
                # number of events within this separator chunk
                nE = separators.index[s+1]-separators.index[s]+1
                # get first day after separator
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
                        features_counter = 0
                        
                        for ind in range(len(assdirnames)):
                            prevPointerCv = IO['pointerCv']
                            for s, shift in enumerate(shifts):
                                file_temp_name = local_vars.IO_directory+\
                                    'temp_train_build'+\
                                    str(np.random.randint(10000))+'.hdf5'
                                while os.path.exists(file_temp_name):
                                    file_temp_name = IO_directory+'temp_train_build'\
                                        +str(np.random.randint(10000))+'.hdf5'
                                file_temp = h5py.File(file_temp_name,'w')
                                #Vars = build_variations(config, file_temp, list_features[features_counter], list_stats_in[ind], modular=True)
                                Vars = build_variations_modular(config, file_temp, list_features[ind][s], list_stats_in[ind])
                                
                                if build_asset_relations[ind]==asset_relation:
                                    skip_cv = False
                                else:
                                    skip_cv = True
                                IO = build_XY(config, Vars, list_returns_struct[ind][s], 
                                              list_stats_out[ind], IO, edges_dt,
                                              folds, fold_idx, save_output=False, 
                                              modular=True, skip_cv=skip_cv)
                                # close temp file
                                file_temp.close()
                                os.remove(file_temp_name)
                                features_counter += 1
                            
                            if build_asset_relations[ind]==asset_relation and phase_shift>1 and prevPointerCv<IO['pointerCv']:
                                # rearrange IO in chronological order for Cv
                                sorted_idx = np.argsort(IO['Dcv'][prevPointerCv:,0,0],kind='mergesort')
                                IO['Dcv'][prevPointerCv:,:,:] = sort_input(IO['Dcv'], sorted_idx, prevPointerCv, char=True)
                                IO['Bcv'][prevPointerCv:,:,:] = sort_input(IO['Bcv'], sorted_idx, prevPointerCv, char=False)
                                IO['Acv'][prevPointerCv:,:,:] = sort_input(IO['Acv'], sorted_idx, prevPointerCv, char=False)
                                IO['Xcv'][prevPointerCv:,:,:] = sort_input(IO['Xcv'], sorted_idx, prevPointerCv, char=False)
                                IO['Ycv'][prevPointerCv:,:,:] = sort_input(IO['Ycv'], sorted_idx, prevPointerCv, char=False)
                            
                    except (KeyboardInterrupt):
                        mess = "KeyBoardInterrupt. Closing files and exiting program."
                        print(mess)
                        if len(log)>0:
                            write_log(mess)
                        f_tr.close()
                        f_cv.close()
                        file_temp.close()
                        os.remove(file_temp_name)
                        os.remove(filename_tr)
                        os.remove(filename_cv)
                        raise KeyboardInterrupt
                else:
                    pass
#                    print("\ts {0:d} of {1:d}. Not enough entries. Skipped.".format(
#                            int(s/2),int(len(separators)/2-1)))
        # end of for s in range(0,len(separators)-1,2):
        # add pointer index for later separating assets
        if if_build_IO:
            ass_IO_ass_tr[ass_idx] = IO['pointerTr']
            ass_IO_ass_cv[ass_idx] = IO['pointerCv']
            mess = "\tTime for "+thisAsset+":"+str(np.floor(time.time()-tic))+"s"+\
              ". Total time:"+str(np.floor(time.time()-ticTotal))+"s"
            print(mess)
            if len(log)>0:
                write_log(mess)
        # update asset index
        ass_idx += 1
        
    # end of for ass in data.assets:
    if if_build_IO:
        mess = "Building DTA..."
        print(mess)
        if len(log)>0:
            write_log(mess)
        DTA = build_DTA_v2(config, C.AllAssets, IO['Dcv'], 
                           IO['Bcv'], IO['Acv'], ass_IO_ass_cv)
        pickle.dump( DTA, open( IO_results_name, "wb" ))
        f_cv.attrs.create('ass_IO_ass', ass_IO_ass_cv, dtype=int)
        f_tr.attrs.create('ass_IO_ass', ass_IO_ass_tr, dtype=int)
        f_cv.attrs.create('totalSampsPerLevel', IO['totalSampsPerLevel'], dtype=int)
        f_tr.attrs.create('totalSampsPerLevel', IO['totalSampsPerLevel'], dtype=int)
        totalSampsPerLevel = IO['totalSampsPerLevel']
    # print percent of samps per level
    else:
        # get ass_IO_ass from disk
        f_cv = h5py.File(filename_cv,'r')
        f_tr = h5py.File(filename_tr,'r')
        ass_IO_ass_cv = f_cv.attrs.get("ass_IO_ass")
        ass_IO_ass_tr = f_tr.attrs.get("ass_IO_ass")
        if 'totalSampsPerLevel' in f_tr:
            totalSampsPerLevel = f_tr.attrs.get("totalSampsPerLevel")
        elif 'totalSampsPerLevel' in f_cv:
            totalSampsPerLevel = f_cv.attrs.get("totalSampsPerLevel")
        else: 
            totalSampsPerLevel = [-1]
    # get total number of samps
    m_tr = ass_IO_ass_tr[-1]
    m_cv = ass_IO_ass_cv[-1]
    m_t = m_tr+m_cv
    print("Edges:")
    print(edges)
    print(filename_tr)
    print(filename_cv)
    print(IO_results_name)
    mess = "Samples for fitting: "+str(m_tr)+"\n"+"Samples for cross-validation: "+\
        str(m_cv)+"\n"+"Total samples: "+str(m_t)
    print(mess)
    if len(log)>0:
        write_log(mess)
    if sum(totalSampsPerLevel)>0:
        mess = "Percent per level:"+str(IO['totalSampsPerLevel']/m_t)
        print(mess)
        if len(log)>0:
            write_log(mess)
    else:
        print("totalSampsPerLevel not in IO :(")
    f_tr.close()
    f_cv.close()
    mess = "DONE building IO"
    print(mess)
    if len(log)>0:
        write_log(mess)
    return filename_tr, filename_cv, IO_results_name

def build_datasets_modular_oneNet(folds=3, fold_idx=0, config={}, log=''):
    """  """
    ticTotal = time.time()
    # create data structure
    if config=={}:    
        config = retrieve_config('CRNN00000')
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
    if 'feature_keys' not in config:
        feature_keys = [i for i in range(37)]
    else:
        feature_keys = config['feature_keys']
    nFeatures = len(feature_keys)
    if 'feats_from_bids' in config:
        feats_from_bids = config['feats_from_bids']
    else:
        feats_from_bids = True
    if 'feats_from_all' in config:
        feats_from_all = config['feats_from_all']
    else:
        feats_from_all = False
    if 'seq_len' in config:
        seq_len = config['seq_len']
    else:
        seq_len = int((config['lB']-config['nEventsPerStat'])/config['movingWindow']+1)
    size_output_layer = config['size_output_layer']
    if 'n_bits_outputs' in config:
        n_bits_outputs = config['n_bits_outputs']
    else:
        n_bits_outputs = [size_output_layer]
    if 'build_test_db' in config:
        build_test_db = config['build_test_db']
    else:
        build_test_db = False
    if 'build_asset_relations' in config:
        build_asset_relations = config['build_asset_relations']
    else:
        build_asset_relations = ['direct']
    print("build_asset_relations")
    print(config['build_asset_relations'])
    if 'asset_relation' in config:
        asset_relation = config['asset_relation']
    else:
        asset_relation = 'direct'
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
    if feats_from_all:
        symbols = ['bid', 'ask']
    else:
        if feats_from_bids:
            symbols = ['bid']
        else:
            symbols = ['ask']
#    if feats_from_bids:
#        symbol = 'bid'
#    else:
#        symbol = 'ask'
    featuredirnames = [hdf5_directory+'mW'+str(movingWindow)+'_nE'+str(nEventsPerStat)+'/'+bar+'/'+symbol+'/' 
                       for bar in build_asset_relations for symbol in symbols]
    outrdirnames = [hdf5_directory+'mW'+str(movingWindow)+'_nE'+str(nEventsPerStat)+'/'+bar+'/out/' 
                    for bar in build_asset_relations]
    if not build_test_db:
        separators_directory = hdf5_directory+'separators/'
    else:
        separators_directory = hdf5_directory+'separators_test/'
    
    
    IO_tr_name = config['IO_tr_name']
    IO_cv_name = config['IO_cv_name']
    filename_tr = IO_directory+'KFTr'+IO_tr_name+'.hdf5'
    filename_cv = IO_directory+'KFCv'+IO_cv_name+'.hdf5'
    IO_results_name = IO_directory+'DTA_'+IO_cv_name+'.p'
    print(filename_tr)
    print(filename_cv)
    print(IO_results_name)
    
    
    edges, edges_dt = get_edges_datasets_modular(folds, config, separators_directory)
    print("Edges:")
    print(edges)
    
    if len(log)>0:
        write_log(filename_tr)
    if len(log)>0:
        write_log(filename_cv)
        
    if os.path.exists(filename_tr):
        if_build_IOtr = False
    else:
        if_build_IOtr = True
    if os.path.exists(filename_cv):
        if_build_IOcv = False
    else:
        if_build_IOcv = True
    # create model
    # if IO structures have to be built 
    if if_build_IOtr:
        #print("Tag = "+str(tag))
        IO = {}
        #attributes to track asset-IO belonging
        ass_IO_ass_tr = np.zeros((len(assets))).astype(int)
        # structure that tracks the number of samples per level
        IO['totalSampsPerLevel'] = np.zeros((n_bits_outputs[-1]))
        # open IO file for writting
        f_tr = h5py.File(filename_tr,'w')
        # init IO data sets
        Xtr = f_tr.create_dataset('X', 
                                (0, seq_len, nFeatures), 
                                maxshape=(None,seq_len, nFeatures), 
                                dtype=float)
        Ytr = f_tr.create_dataset('Y', 
                                (0, seq_len, size_output_layer),
                                maxshape=(None, seq_len, size_output_layer),
                                dtype=float)
        
            
        Itr = f_tr.create_dataset('I', 
                                (0, seq_len,2),maxshape=(None, seq_len, 2),
                                dtype=int)
        Rtr = f_tr.create_dataset('R', 
                                (0,1),
                                maxshape=(None,1),
                                dtype=float)
        
        IO['Xtr'] = Xtr
        IO['Ytr'] = Ytr
        IO['Itr'] = Itr
        IO['Rtr'] = Rtr # return
        IO['pointerTr'] = 0
    else:
        IO = {}
        IO['Xtr'] = []
        IO['Ytr'] = []
        IO['Itr'] = []
        IO['Rtr'] = [] # return
        IO['pointerTr'] = 0
        
    if if_build_IOcv:
        ass_IO_ass_cv = np.zeros((len(assets))).astype(int)
        f_cv = h5py.File(filename_cv,'w')
        # init IO data sets
        Xcv = f_cv.create_dataset('X', 
                                (0, seq_len, nFeatures), 
                                maxshape=(None,seq_len, nFeatures), 
                                dtype=float)
        Ycv = f_cv.create_dataset('Y', 
                                (0, seq_len, size_output_layer),
                                maxshape=(None, seq_len, size_output_layer),
                                dtype=float)
            
        Icv = f_cv.create_dataset('I', 
                                (0, seq_len,2),maxshape=(None, seq_len, 2),
                                dtype=int)
        Rcv = f_cv.create_dataset('R', 
                                (0,1),
                                maxshape=(None,1),
                                dtype=float)
        
        
        # save IO structures in dictionary
        
        Dcv = f_cv.create_dataset('D', (0,seq_len,2),
                                maxshape=(None,seq_len,2),dtype='S19')
        Bcv = f_cv.create_dataset('B', (0,seq_len,2),
                                maxshape=(None,seq_len,2),dtype=float)
        Acv = f_cv.create_dataset('A', (0,seq_len,2),
                                maxshape=(None,seq_len,2),dtype=float)
        IO['Xcv'] = Xcv
        IO['Ycv'] = Ycv
        IO['Icv'] = Icv
        IO['Rcv'] = Rcv # return
        IO['Dcv'] = Dcv
        IO['Bcv'] = Bcv
        IO['Acv'] = Acv
        IO['pointerCv'] = 0
    else:
        IO['Xcv'] = []
        IO['Ycv'] = []
        IO['Icv'] = []
        IO['Rcv'] = [] # return
        IO['Dcv'] = []
        IO['Bcv'] = []
        IO['Acv'] = []
        IO['pointerCv'] = 0
    
    if len(log)>0:
        write_log(IO_results_name)
    # index asset
    ass_idx = 0
    # array containing bids means
    #bid_means = pickle.load( open( "../HDF5/bid_means.p", "rb" ))
    # loop over all assets
    for ass in assets:
        
        if if_build_IOtr or if_build_IOcv:
            thisAsset = C.AllAssets[str(ass)]
            assdirnames = [featuredirname+thisAsset+'/' for featuredirname in featuredirnames]
            outassdirnames = [outrdirname+thisAsset+'/' for outrdirname in outrdirnames]
            
            tic = time.time()
            # load separators
            separators = load_separators(thisAsset, 
                                         separators_directory, 
                                         from_txt=1)
            if not build_test_db:
                sep_for_stats = separators
            else:
                sep_for_stats = load_separators(thisAsset, 
                                         hdf5_directory+'separators/', 
                                         from_txt=1)

            first_date = dt.datetime.strftime(dt.datetime.strptime(
                    sep_for_stats.DateTime.iloc[0],'%Y.%m.%d %H:%M:%S'),'%y%m%d%H%M%S')
            last_date = dt.datetime.strftime(dt.datetime.strptime(
                    sep_for_stats.DateTime.iloc[-1],'%Y.%m.%d %H:%M:%S'),'%y%m%d%H%M%S')
            
            list_stats_in = [None for _ in range(len(assdirnames))]
            list_stats_out = [None for _ in range(len(assdirnames))]
            for ind, assdirname in enumerate(assdirnames):
                if 'bid' in assdirname:
                    sym = 'bid'
                elif 'ask' in assdirname:
                    sym = 'ask'
                else:
                    raise ValueError
                if 'direct' in assdirname:
                    ass_rel = 'direct'
                elif 'inverse' in assdirname:
                    ass_rel = 'inverse'
                else:
                    raise ValueError
                list_stats_in[ind], list_stats_out[ind] = load_stats_modular_oneNet(config, thisAsset, first_date, last_date, sym, ass_rel)
                
            mess = str(ass)+". "+thisAsset
            print(mess)
            if len(log)>0:
                write_log(mess)
            # loop over separators
            for s in range(0,len(separators)-1,2):
                mess = "\ts {0:d} of {1:d}".format(int(s/2),int(len(separators)/2-1))+\
                    ". From "+separators.DateTime.iloc[s]+" to "+\
                    separators.DateTime.iloc[s+1]
                print(mess)
                if len(log)>0:
                    write_log(mess)
                # number of events within this separator chunk
                nE = separators.index[s+1]-separators.index[s]+1
                # get first day after separator
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
                                           for shift in shifts] for groupoutdirname in groupoutdirnames for symbol in symbols]
                    try:
                        features_counter = 0
                        
                        for ind, assdirname in enumerate(assdirnames):
                            prevPointerCv = IO['pointerCv']
                            if ('bid' in assdirname and feats_from_bids) or ('ask' in assdirname and not feats_from_bids):
                                cv_dir =True
                            else:
                                cv_dir = False
                                
                            for s, shift in enumerate(shifts):
                                file_temp_name = local_vars.IO_directory+\
                                    'temp_train_build'+\
                                    str(np.random.randint(10000))+'.hdf5'
                                while os.path.exists(file_temp_name):
                                    file_temp_name = IO_directory+'temp_train_build'\
                                        +str(np.random.randint(10000))+'.hdf5'
                                file_temp = h5py.File(file_temp_name,'w')
                                #Vars = build_variations(config, file_temp, list_features[features_counter], list_stats_in[ind], modular=True)
                                Vars = build_variations_modular(config, file_temp, list_features[ind][s], list_stats_in[ind])
                                
                                if build_asset_relations[ind%len(build_asset_relations)]!=asset_relation and not cv_dir and not if_build_IOcv:
                                    skip_cv = True
                                else:
                                    skip_cv = False
                                IO = build_XY(config, Vars, list_returns_struct[ind][s], 
                                              list_stats_out[ind], IO, edges_dt,
                                              folds, fold_idx, save_output=False, 
                                              modular=True, skip_cv=skip_cv, skip_tr=if_build_IOtr)
                                # close temp file
                                file_temp.close()
                                os.remove(file_temp_name)
                                features_counter += 1
                            
                            if build_asset_relations[ind%len(build_asset_relations)]==asset_relation and phase_shift>1 and prevPointerCv<IO['pointerCv'] and if_build_IOcv:
                                # rearrange IO in chronological order for Cv
                                sorted_idx = np.argsort(IO['Dcv'][prevPointerCv:,0,0],kind='mergesort')
                                IO['Dcv'][prevPointerCv:,:,:] = sort_input(IO['Dcv'], sorted_idx, prevPointerCv, char=True)
                                IO['Bcv'][prevPointerCv:,:,:] = sort_input(IO['Bcv'], sorted_idx, prevPointerCv, char=False)
                                IO['Acv'][prevPointerCv:,:,:] = sort_input(IO['Acv'], sorted_idx, prevPointerCv, char=False)
                                IO['Xcv'][prevPointerCv:,:,:] = sort_input(IO['Xcv'], sorted_idx, prevPointerCv, char=False)
                                IO['Ycv'][prevPointerCv:,:,:] = sort_input(IO['Ycv'], sorted_idx, prevPointerCv, char=False)
                            
                    except (KeyboardInterrupt):
                        mess = "KeyBoardInterrupt. Closing files and exiting program."
                        print(mess)
                        if len(log)>0:
                            write_log(mess)
                        f_tr.close()
                        f_cv.close()
                        file_temp.close()
                        os.remove(file_temp_name)
                        os.remove(filename_tr)
                        os.remove(filename_cv)
                        raise KeyboardInterrupt
                else:
                    pass
#                    print("\ts {0:d} of {1:d}. Not enough entries. Skipped.".format(
#                            int(s/2),int(len(separators)/2-1)))
        # end of for s in range(0,len(separators)-1,2):
        # add pointer index for later separating assets
        if if_build_IOtr:
            ass_IO_ass_tr[ass_idx] = IO['pointerTr']
        if if_build_IOcv:
            ass_IO_ass_cv[ass_idx] = IO['pointerCv']
        if if_build_IOtr or if_build_IOcv:
            mess = "\tTime for "+thisAsset+":"+str(np.floor(time.time()-tic))+"s"+\
              ". Total time:"+str(np.floor(time.time()-ticTotal))+"s"
            print(mess)
            if len(log)>0:
                write_log(mess)
        # update asset index
        ass_idx += 1
        
    # end of for ass in data.assets:
    if if_build_IOcv:
        mess = "Building DTA..."
        print(mess)
        if len(log)>0:
            write_log(mess)
        DTA = build_DTA_v2(config, C.AllAssets, IO['Dcv'], 
                           IO['Bcv'], IO['Acv'], ass_IO_ass_cv)
        pickle.dump( DTA, open( IO_results_name, "wb" ))
        f_cv.attrs.create('ass_IO_ass', ass_IO_ass_cv, dtype=int)
        f_cv.attrs.create('totalSampsPerLevel', IO['totalSampsPerLevel'], dtype=int)
        
    # print percent of samps per level
    else:
        # get ass_IO_ass from disk
        f_cv = h5py.File(filename_cv,'r')
        ass_IO_ass_cv = f_cv.attrs.get("ass_IO_ass")
        
        
    if if_build_IOtr:
        f_tr.attrs.create('ass_IO_ass', ass_IO_ass_tr, dtype=int)
        
        f_tr.attrs.create('totalSampsPerLevel', IO['totalSampsPerLevel'], dtype=int)
        totalSampsPerLevel = IO['totalSampsPerLevel']
    else:
        f_tr = h5py.File(filename_tr,'r')
        ass_IO_ass_tr = f_tr.attrs.get("ass_IO_ass")
        if 'totalSampsPerLevel' in f_tr:
            totalSampsPerLevel = f_tr.attrs.get("totalSampsPerLevel")
        else: 
            totalSampsPerLevel = [-1]
    # get total number of samps
    m_tr = ass_IO_ass_tr[-1]
    m_cv = ass_IO_ass_cv[-1]
    m_t = m_tr+m_cv
    print("Edges:")
    print(edges)
    print(filename_tr)
    print(filename_cv)
    print(IO_results_name)
    mess = "Samples for fitting: "+str(m_tr)+"\n"+"Samples for cross-validation: "+\
        str(m_cv)+"\n"+"Total samples: "+str(m_t)
    print(mess)
    if len(log)>0:
        write_log(mess)
    if sum(totalSampsPerLevel)>0:
        mess = "Percent per level:"+str(IO['totalSampsPerLevel']/m_t)
        print(mess)
        if len(log)>0:
            write_log(mess)
    else:
        print("totalSampsPerLevel not in IO :(")
    f_tr.close()
    f_cv.close()
    mess = "DONE building IO"
    print(mess)
    if len(log)>0:
        write_log(mess)
    return filename_tr, filename_cv, IO_results_name