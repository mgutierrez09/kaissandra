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
    """
    
    """
    ticTotal = time.time()
    if len(ins)>0:
        config = ins[0]
    else:    
        config = configuration('C0000')
    # create data structure
    data=Data(movingWindow=config['movingWindow'],
              nEventsPerStat=config['nEventsPerStat'],
              dateTest = config['dateTest'])
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
        filename_prep_IO = (hdf5_directory+'feat_tsf_mW'+str(data.movingWindow)+'_nE'+
                            str(data.nEventsPerStat)+'.hdf5')
    else:
        #print("ERROR: load_features_from "+load_features_from+" not recognized")
        raise ValueError("Load_features_from "+load_features_from+" not recognized")
        
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
    get_features_tsfresh()