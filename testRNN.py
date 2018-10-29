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
from RNN import modelRNN
from inputs import Data, load_separators, load_stats, load_features_results, build_DTA_v20, build_IO
from config import configuration


def test_RNN(*ins):
    """
    <DocString>
    """
    ticTotal = time.time()
    if len(ins)>0:
        config = ins[0]
    else:    
        config = configuration('C0004')

    # create data structure
    data=Data(movingWindow=config['movingWindow'],
                  nEventsPerStat=config['nEventsPerStat'],
                  lB=config['lB'], 
                  dateTest=config['dateTest'],
                  assets=config['assets'])
    
    #if_build_IO = config['if_build_IO']
    startFrom = config['startFrom']
    endAt = config['endAt']
    save_journal = config['save_journal']
    
    IDweights = config['IDweights']
    IDresults = config['IDresults']
    hdf5_directory = config['hdf5_directory']
    IO_directory = config['IO_directory']
    
    filename_prep_IO = (hdf5_directory+'IO_mW'+str(data.movingWindow)+'_nE'+
                        str(data.nEventsPerStat)+'_nF'+str(data.nFeatures)+
                        '.hdf5')
    
    separators_directory = hdf5_directory+'separators/'
    filename_IO = IO_directory+'IO_'+IDresults+'.hdf5'
    # init hdf5 files
    f_prep_IO = h5py.File(filename_prep_IO,'r')
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
        
        # open IO file for writting
        f_IO = h5py.File(filename_IO,'w')
        # init IO data sets
        X = f_IO.create_dataset('X', (0, model.seq_len, model.nFeatures), 
                                maxshape=(None,model.seq_len, model.nFeatures), 
                                dtype=float)
        Y = f_IO.create_dataset('Y', (0,model.seq_len,
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
        IO['I'] = I
        IO['D'] = D
        IO['B'] = B
        IO['A'] = A
        IO['pointer'] = 0
    
    aloc = 2**17
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
        ass_group = f_prep_IO[thisAsset]
        # load stats
        stats = load_stats(data, 
                           thisAsset, 
                           ass_group, 
                           0, 
                           from_stats_file=True, 
                           hdf5_directory=hdf5_directory+'stats/',
                           save_pickle=False)
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
                    # load features, returns and stats from file
                    IO_prep = load_features_results(data, thisAsset, separators,
                                                    f_prep_IO, s)
                    # build network input and output
                    # get first day after separator
                    day_s = separators.DateTime.iloc[s][0:10]
                    # check if the separator chuck belongs to the training/test set
                    if day_s in data.dateTest:
                        try:
                            file_temp_name = ('../RNN/IO/temp_test_build'+
                                              str(np.random.randint(10000))+
                                              '.hdf5')
                            while os.path.exists(file_temp_name):
                                file_temp_name = ('../RNN/IO/temp_test_build'+
                                                  str(np.random.randint(10000))+
                                                  '.hdf5')
                            file_temp = h5py.File(file_temp_name,'w')
                            IO, totalSampsPerLevel = build_IO(file_temp, data, 
                                                              model, IO_prep, 
                                                              stats,IO, 
                                                              totalSampsPerLevel, 
                                                              s, nE, thisAsset)
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
                            end()
                        
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
        f_prep_IO.flush()
        
        
    
        #print("Total time:"+str(np.floor(time.time()-ticTotal))+"s")
    # end of for ass in data.assets:
    
    # close files
    f_prep_IO.close()
    if if_build_IO:
        print("Building DTA...")
        DTA = build_DTA_v20(data, IO['D'], IO['B'], IO['A'], ass_IO_ass)
        pickle.dump( DTA, open( "../RNN/IO/DTA"+"_"+IDresults+".p", "wb" ))
        f_IO.attrs.create('ass_IO_ass', ass_IO_ass, dtype=int)
        f_IO.close()
    else:
        # get ass_IO_ass from disk
        DTA = pickle.load( open( "../RNN/IO/DTA"+"_"+IDresults+".p", "rb" ))
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
        model.test(sess, data, IDresults, IDweights, 
                   int(np.ceil(m_t/aloc)), 1, 'test', startFrom=startFrom,
                   IDIO=IDresults, data_format='hdf5', DTA=DTA, 
                   save_journal=save_journal, endAt=endAt)
        
if __name__=='__main__':
    test_RNN()
    