# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 12:43:11 2018

@author: mgutierrez
Script to perform the training of the RNN
"""
import time
import numpy as np
import tensorflow as tf
import h5py
import pickle
import os

from kaissandra.RNN import modelRNN
from kaissandra.local_config import local_vars
from kaissandra.inputs import (Data, 
                    load_separators, 
                    build_DTA, 
                    build_IO,
                    load_stats_manual,
                    load_stats_output,
                    load_stats_tsf,
                    load_manual_features,
                    load_tsf_features,
                    load_returns)
from kaissandra.config import retrieve_config

def test_RNN(*ins):
    """
    <DocString>
    """
    ticTotal = time.time()
    if len(ins)>0:
        config = ins[0]
    else:
        # test reset git
        config = retrieve_config('C0288INVO')
    # add compatibility
    if 'feature_keys_manual' not in config:
        feature_keys_manual = [i for i in range(37)]
    else:
        feature_keys_manual = config['feature_keys_manual']
    if 'feature_keys_tsfresh' not in config:
        feature_keys_tsfresh = []
    else:
        feature_keys_tsfresh = config['feature_keys_tsfresh']
    if 'inverse_load' in config:
        inverse_load = config['inverse_load']
    else:
        inverse_load = False
    if 'feats_from_bids' in config:
        feats_from_bids = config['feats_from_bids']
    else:
        feats_from_bids = True
    # create data structure
    data=Data(movingWindow=config['movingWindow'],
                  nEventsPerStat=config['nEventsPerStat'],
                  lB=config['lB'], 
                  dateTest=config['dateTest'],
                  assets=config['assets'],
                  channels=config['channels'],
                  max_var=config['max_var'],
                  feature_keys_manual=feature_keys_manual,
                  feature_keys_tsfresh=feature_keys_tsfresh)
    
    #if_build_IO = config['if_build_IO']
    IDresults = config['IDresults']
    if 'IO_results_name' not in config:
        IO_results_name = IDresults
    else:
        IO_results_name = config['IO_results_name']
    #IO_results_name = config['IO_results_name']
    hdf5_directory = local_vars.hdf5_directory
    IO_directory = local_vars.IO_directory
    if not os.path.exists(IO_directory):
        os.mkdir(IO_directory)
    
    if type(feats_from_bids)==bool:
        if feats_from_bids:
            # only get short bets (negative directions)
            tag = 'IO_mW'
        else:
            # only get long bets (positive directions)
            tag = 'IOA_mW'
    elif type(feats_from_bids)==str:
        if feats_from_bids=='SHORT' or feats_from_bids=='COMBS':
            # only get short bets (negative directions)
            tag = 'IO_mW'
        elif feats_from_bids=='LONG' or feats_from_bids=='COMBL':
            # only get long bets (positive directions)
            tag = 'IOA_mW'
        else:
            raise ValueError("Entry "+feats_from_bids+" not known. Options are SHORT/LONG/COMB")
    else:
        raise ValueError("feats_from_bids must be a bool or str with options SHORT/LONG/COMB")
    
    filename_prep_IO = (hdf5_directory+tag+str(data.movingWindow)+'_nE'+
                        str(data.nEventsPerStat)+'_nF'+str(data.n_feats_manual)+'.hdf5')
    filename_features_tsf = (hdf5_directory+'feats_tsf_mW'+str(data.movingWindow)+
                             '_nE'+str(data.nEventsPerStat)+'_2.hdf5')
    
    if 'build_partial_raw' in config:
        build_partial_raw = config['build_partial_raw']
        
        
    else:
        build_partial_raw = False
        
    if build_partial_raw:
        # TODO: get init/end dates from dateTest in Data
        int_date = '180928'
        end_date = '181109'
        separators_directory = hdf5_directory+'separators_F'+int_date+'T'+end_date+'/'
    else:
        separators_directory = hdf5_directory+'separators/'
        
    if 'from_stats_file' in config:
        from_stats_file = config['from_stats_file']
    else:
        from_stats_file = True
        
    filename_IO = IO_directory+'IO_'+IO_results_name+'.hdf5'
    # check if file locked
    if 0:#len(ins)>0:
        # wait while files are locked
        while os.path.exists(filename_prep_IO+'.flag'):
            # sleep random time up to 10 seconds if any file is being used
            print(filename_prep_IO+' busy. Sleeping up to 10 secs')
            time.sleep(10*np.random.rand(1)[0])
        # lock HDF5 file from access
        fh = open(filename_prep_IO+'.flag',"w")
        fh.close()
    # init hdf5 files
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
    model=modelRNN(data,
                   size_hidden_layer=config['size_hidden_layer'],
                   L=config['L'],
                   size_output_layer=config['size_output_layer'],
                   keep_prob_dropout=1,
                   miniBatchSize=config['miniBatchSize'],
                   outputGain=config['outputGain'],
                   commonY=config['commonY'],
                   lR0=config['lR0'])
    
    if if_build_IO:
        print("Tag = "+str(tag))
        # open IO file for writting
        f_IO = h5py.File(filename_IO,'w')
        # init IO data sets
        X = f_IO.create_dataset('X', (0, model.seq_len, model.nFeatures), 
                                maxshape=(None,model.seq_len, model.nFeatures), 
                                dtype=float)
        XA = f_IO.create_dataset('XA', (0, model.seq_len, model.nFeatures), 
                                maxshape=(None,model.seq_len, model.nFeatures), 
                                dtype=float)
        Y = f_IO.create_dataset('Y', (0,model.seq_len,
                                      model.commonY+model.size_output_layer),
                                      maxshape=(None,model.seq_len,
                                      model.commonY+model.size_output_layer),
                                      dtype=float)
        YA = f_IO.create_dataset('YA', (0,model.seq_len,
                                      model.commonY+model.size_output_layer),
                                      maxshape=(None,model.seq_len,
                                      model.commonY+model.size_output_layer),
                                      dtype=float)
        I = f_IO.create_dataset('I', (0,model.seq_len,2),
                                maxshape=(None,model.seq_len,2),dtype=int)
        D = f_IO.create_dataset('D', (0,model.seq_len,2),
                                maxshape=(None,model.seq_len,2),dtype='S19')
        B = f_IO.create_dataset('B', (0,model.seq_len,2),
                                maxshape=(None,model.seq_len,2),dtype=float)
        A = f_IO.create_dataset('A', (0,model.seq_len,2),
                                maxshape=(None,model.seq_len,2),dtype=float)
        # attributes to track asset-IO belonging
        ass_IO_ass = np.zeros((len(data.assets))).astype(int)
        # structure that tracks the number of samples per level
        totalSampsPerLevel = np.zeros((model.size_output_layer))
        # save IO structures in dictionary
        IO = {}
        IO['X'] = X
        IO['Y'] = Y
        IO['XA'] = XA
        IO['YA'] = YA
        IO['I'] = I
        IO['D'] = D
        IO['B'] = B
        IO['A'] = A
        IO['pointer'] = 0
    
    alloc = 50000
    # max number of input channels
    # index asset
    ass_idx = 0
    # loop over all assets
    for ass in data.assets:
        thisAsset = data.AllAssets[str(ass)]
        
        tic = time.time()
        # load separators
        separators = load_separators(data, thisAsset, separators_directory,
                                     tOt='te', from_txt=1)

        # retrive asset group
        if f_prep_IO != None:
            ass_group = f_prep_IO[thisAsset]
            stats_manual = load_stats_manual(data, 
                               thisAsset, 
                               ass_group,
                               from_stats_file=from_stats_file, 
                               hdf5_directory=hdf5_directory+'stats/')
        else:
            stats_manual = []
        
        stats_output = load_stats_output(data, hdf5_directory, thisAsset)
        
        if f_feats_tsf != None:
            stats_tsf = load_stats_tsf(data, thisAsset, hdf5_directory, f_feats_tsf,
                                       load_from_stats_file=True)
        else:
            stats_tsf = []
            
        if if_build_IO:
            print(str(ass)+". "+thisAsset)
            # loop over separators
            for s in range(0,len(separators)-1,2):
                # number of events within this separator chunk
                nE = separators.index[s+1]-separators.index[s]+1
                # check if number of events is not enough to build 
                #two features and one return
                if nE>=2*data.nEventsPerStat:
                    print("\ts {0:d} of {1:d}".format(int(s/2),
                          int(len(separators)/2-1))+
                          ". From "+separators.DateTime.iloc[s]+
                          " to "+separators.DateTime.iloc[s+1])
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
                        
                    returns_struct = load_returns(data, hdf5_directory, thisAsset, separators, filename_prep_IO, s)
                    # build network input and output
                    # get first day after separator
                    day_s = separators.DateTime.iloc[s][0:10]
                    day_e = separators.DateTime.iloc[s+1][0:10]
                    # check if the separator chuck belongs to the training/test set
                    if day_s in data.dateTest:
                        try:
                            file_temp_name = (local_vars.IO_directory+'temp_test_build'+
                                              str(np.random.randint(10000))+
                                              '.hdf5')
                            while os.path.exists(file_temp_name):
                                file_temp_name = (local_vars.IO_directory+'temp_test_build'+
                                                  str(np.random.randint(10000))+
                                                  '.hdf5')
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
                                                                  inverse_load)
                            # close temp file
                            file_temp.close()
                            os.remove(file_temp_name)
                        except (KeyboardInterrupt,NameError):
                            print("KeyBoardInterrupt. Closing files and"+
                                  " exiting program.")
                            f_prep_IO.close()
                            f_IO.close()
                            file_temp.close()
                            os.remove(file_temp_name)
                            raise KeyboardInterrupt
                        
                    else:
                        print("\tNot in the set. Skipped.")
                    # end of if (tOt=='train' and day_s not in data.dateTest) ...
                else:
                    print("\ts {0:d} of {1:d}..".format(
                            int(s/2),int(len(separators)/2-1))+
                          " Not enough entries. Skipped")
        # end of for s in range(0,len(separators)-1,2):
        # add pointer index for later separating assets
        if if_build_IO:
            ass_IO_ass[ass_idx] = IO['pointer']
            print("\tTime for "+thisAsset+":"+str(np.floor(time.time()-tic))+"s"+
              ". Total time:"+str(np.floor(time.time()-ticTotal))+"s")
            
        # update asset index
        ass_idx += 1
        
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
    # Build DTA
    if if_build_IO:
        print("Building DTA...")
        DTA = build_DTA(data, IO['D'], IO['B'], IO['A'], ass_IO_ass)
        pickle.dump( DTA, open( IO_directory+"DTA"+"_"+IO_results_name+".p", "wb" ))
        f_IO.attrs.create('ass_IO_ass', ass_IO_ass, dtype=int)
        f_IO.close()
    else:
        # get ass_IO_ass from disk
        DTA = pickle.load( open( IO_directory+"DTA"+"_"+IO_results_name+".p", "rb" ))
        f_IO = h5py.File(filename_IO,'r')
        ass_IO_ass = f_IO.attrs.get("ass_IO_ass")
        f_IO.close()
    # get total number of samps
    m_t = ass_IO_ass[-1]
    # print percent of samps per level
    if if_build_IO:
        print("Samples to RNN: "+str(m_t)+".\nPercent per level:"+
              str(totalSampsPerLevel/m_t))
    # reset graph
    tf.reset_default_graph()
    # start session
    with tf.Session() as sess:
        # run test RNN
        print("IDresults: "+IDresults)
        model.test2(sess, config, alloc, filename_IO, DTA=DTA)
        # TEMP: GRE calculation not implemented in test2 yet. Use old test
#        if save_journal:
#            pass
##            model.test(sess, data, IDresults, IDweights, 
##                       alloc, True, 
##                       'test', filename_IO, startFrom=startFrom,
##                       IDIO=IO_results_name, data_format='hdf5', DTA=DTA, 
##                       save_journal=save_journal, endAt=endAt)
#            
#        else:
#            model.test2(sess, dateTest, IDresults, IDweights, alloc,
#                         weights_directory, filename_IO,
#                       startFrom=startFrom, data_format='hdf5', DTA=DTA, 
#                       save_journal=save_journal, endAt=endAt,resolution=resolution)
        
if __name__=='__main__':
    pass
    #test_RNN()
    