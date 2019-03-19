# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 12:43:11 2018

@author: mgutierrez
Script to perform the training of the RNN
"""
import time
import h5py
import numpy as np
import os
import tensorflow as tf

from kaissandra.RNN import modelRNN
from kaissandra.inputs import (Data, 
                    load_separators, 
                    build_IO, 
                    load_stats_manual,
                    load_stats_tsf,
                    load_stats_output,
                    load_returns,
                    load_manual_features,
                    load_tsf_features)
from kaissandra.config import retrieve_config
from kaissandra.local_config import local_vars

def evaluateRNN(*ins):
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
    
    data=Data(movingWindow=config['movingWindow'],
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
    if 0:#len(ins)>0:
        # wait while files are locked
        while os.path.exists(filename_prep_IO+'.flag'):
            # sleep random time up to 10 seconds if any file is being used
            print(filename_prep_IO+' busy. Sleeping up to 10 secs')
            time.sleep(10*np.random.rand(1)[0])
        # lock HDF5 file from access
        fh = open(filename_prep_IO+'.flag',"w")
        fh.close()
    if data.n_feats_manual>0:
        f_prep_IO = h5py.File(filename_prep_IO,'r')
    else:
        f_prep_IO = None
    if data.n_feats_tsfresh>0:
        f_feats_tsf = h5py.File(filename_features_tsf,'r')
        #file_features_tsf = h5py.File(filename_features_tsf,'r')
#        for ass in f_feats_tsf.keys():
#            print(f_feats_tsf[ass])
#            for chunck in f_feats_tsf[ass].keys():
#                print(f_feats_tsf[ass][chunck])
#                for feat in f_feats_tsf[ass][chunck].keys():
#                    print(f_feats_tsf[ass][chunck][feat])
#                    for v in f_feats_tsf[ass][chunck][feat].keys():
#                        print(f_feats_tsf[ass][chunck][feat][v].shape)
            #a=p
    else:
        f_feats_tsf = None
        
    if os.path.exists(filename_IO) and len(ins)>0:
        if_build_IO = False
    else:
        if_build_IO = config['if_build_IO']
    # create model
    model=modelRNN(data,
                       size_hidden_layer=config['size_hidden_layer'],
                       L=config['L'],
                       size_output_layer=config['size_output_layer'],
                       keep_prob_dropout=config['keep_prob_dropout'],
                       miniBatchSize=config['miniBatchSize'],
                       outputGain=config['outputGain'],
                       commonY=config['commonY'],
                       lR0=config['lR0'],
                       num_epochs=config['num_epochs'])
    # if IO structures have to be built 
    if if_build_IO:
        print("Tag = "+str(tag))
        # open IO file for writting
        f_IO = h5py.File(filename_IO,'w')
        # init IO data sets
        X = f_IO.create_dataset('X', 
                                (0, model.seq_len, model.nFeatures), 
                                maxshape=(None,model.seq_len, model.nFeatures), 
                                dtype=float)
        Y = f_IO.create_dataset('Y', 
                                (0,model.seq_len,model.commonY+model.size_output_layer),
                                maxshape=(None,model.seq_len,model.commonY+
                                          model.size_output_layer),
                                dtype=float)
            
        I = f_IO.create_dataset('I', 
                                (0,model.seq_len,2),maxshape=(None,model.seq_len,2),
                                dtype=int)
        R = f_IO.create_dataset('R', 
                                (0,1),
                                maxshape=(None,1),
                                dtype=float)
            # attributes to track asset-IO belonging
        ass_IO_ass = np.zeros((len(data.assets))).astype(int)
        # structure that tracks the number of samples per level
        totalSampsPerLevel = np.zeros((model.size_output_layer))
        # save IO structures in dictionary
        IO = {}
        IO['X'] = X
        IO['Y'] = Y
        IO['I'] = I
        IO['R'] = R # return
        IO['pointer'] = 0
        
    # index asset
    ass_idx = 0
    # array containing bids means
    #bid_means = pickle.load( open( "../HDF5/bid_means.p", "rb" ))
    # loop over all assets
    for ass in data.assets:
        thisAsset = data.AllAssets[str(ass)]
        
        tic = time.time()
        # load separators
        separators = load_separators(data, 
                                     thisAsset, 
                                     separators_directory, 
                                     tOt='tr', 
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
                                IO, totalSampsPerLevel = build_IO(file_temp, 
                                                                      data, 
                                                                      model, 
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
    # release lock
    if 0:#len(ins)>0:
        os.remove(filename_prep_IO+'.flag')
    
    if if_build_IO:
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
    output_filename = IO_directory+'NNIO'+filetag
    if not os.path.exists(output_filename):
        # reset graph
        tf.reset_default_graph()
        # start session
        with tf.Session() as sess:
            f_IO = h5py.File(filename_IO,'r')
            X = f_IO['X']
            Y = f_IO['Y']
            model.evaluate(sess, X, Y, tOt=tOt)
            f_IO.close()
    else:
        print("Output file "+output_filename+" already exists. Delete it to run evaluate")
        
if __name__ == "__main__":
    pass