# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 12:43:11 2018

@author: mgutierrez
Script to perform the training of the RNN
"""
import time
import h5py
import pickle
import numpy as np
import tensorflow as tf
from RNN import modelRNN
from inputs import Data, load_separators, load_stats, load_features_results, build_IO
from config import config


if __name__ == "__main__":
    
    ticTotal = time.time()
    # create data structure
    
    data=Data(movingWindow=100,nEventsPerStat=1000,lB=1200,dateStart='2016.01.01',dateEnd='2018.08.10',
              channels=[0,9],   
              dateTest = [                                                       '2018.03.09',
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
                             '2018.08.06','2018.08.07','2018.08.08','2018.08.09','2018.08.10'])
    # init structures
    if_build_IO = True
    IDweights = '000270'
    hdf5_directory = 'D:/SDC/py/HDF5/'#'../HDF5/'#
    IO_directory = '../RNN/IO/'
    filename_prep_IO = hdf5_directory+'IO_mW'+str(data.movingWindow)+'_nE'+str(data.nEventsPerStat)+'_nF'+str(data.nFeatures)+'.hdf5'
    separators_directory = hdf5_directory+'separators/'
    filename_IO = IO_directory+'IO_'+IDweights+'.hdf5'
    
    # init hdf5 files
    f_prep_IO = h5py.File(filename_prep_IO,'r')
    # init IO structures
    # define if it is for training or for testing
        
    # create model
    model=modelRNN(data,
                       size_hidden_layer=100,
                       L=3,
                       size_output_layer=5,
                       keep_prob_dropout=1,
                       miniBatchSize=32,
                       outputGain=.6,
                       commonY=3,
                       lR0=0.0001,
                       version="1.0")
        
    
    
    if if_build_IO:
        
        # open IO file for writting
        f_IO = h5py.File(filename_IO,'w')
        # init IO data sets
        X = f_IO.create_dataset('X', (0, model.seq_len, model.nFeatures), 
                                maxshape=(None,model.seq_len, model.nFeatures), dtype=float)
        Y = f_IO.create_dataset('Y', (0,model.seq_len,model.commonY+model.size_output_layer),
                                maxshape=(None,model.seq_len,model.commonY+model.size_output_layer),dtype=float)
            
        I = f_IO.create_dataset('I', (0,model.seq_len,2),maxshape=(None,model.seq_len,2),dtype=int)
    #       D = f_IO.create_dataset('D', (0,model.seq_len,2),maxshape=(None,model.seq_len,2),dtype='S19')
    #        B = f_IO.create_dataset('B', (0,model.seq_len,2),maxshape=(None,model.seq_len,2),dtype=float)
    #        A = f_IO.create_dataset('A', (0,model.seq_len,2),maxshape=(None,model.seq_len,2),dtype=float)
            # attributes to track asset-IO belonging
        ass_IO_ass = np.zeros((len(data.assets))).astype(int)
        # structure that tracks the number of samples per level
        totalSampsPerLevel = np.zeros((model.size_output_layer))
        # save IO structures in dictionary
        IO = {}
        IO['X'] = X
        IO['Y'] = Y
        IO['I'] = I
        IO['pointer'] = 0
        
    # init total number of samples
    m = 0
    aloc = 2**20
    # max number of input channels
    nChannels = int(data.nEventsPerStat/data.movingWindow)
    # all output gains
    #outputGains = np.zeros((len(data.assets)))
    # index asset
    ass_idx = 0
    # array containing bids means
    #bid_means = pickle.load( open( "../HDF5/bid_means.p", "rb" ))
    # loop over all assets
    for ass in data.assets:
        thisAsset = data.AllAssets[str(ass)]
        print(str(ass)+". "+thisAsset)
        tic = time.time()
        # load separators
        separators = load_separators(data, thisAsset, separators_directory, tOt='tr', from_txt=1)
        # retrive asset group
        ass_group = f_prep_IO[thisAsset]
        # load stats
    #    stats = load_stats(data, thisAsset, ass_group, 0)
        stats = load_stats(data, thisAsset, ass_group, 0, from_stats_file=True, hdf5_directory=hdf5_directory+'stats/')
        # get variable output gain
        #outputGains[ass_idx] = get_outputGain(stats['stds_t_out'][0,data.lookAheadIndex], bid_means[ass_idx], fixed_spread_ratio)
        #model.outputGain = outputGains[ass_idx]
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
                IO_prep = load_features_results(data, thisAsset, separators, f_prep_IO, s)
                # build network input and output
                if build_IO:
                    # get first day after separator
                    day_s = separators.DateTime.iloc[s][0:10]
                    # check if the separator chuck belongs to the training/test set
                    if day_s not in data.dateTest:
                        
                        try:
                            #print("\tBuilding IO")
                            file_temp = h5py.File('../RNN/IO/temp_build.hdf5','w')
                            IO, totalSampsPerLevel = build_IO(file_temp, data, model, IO_prep, stats, 
                                                                        IO, totalSampsPerLevel, s, nE, thisAsset)
                            # close temp file
                            file_temp.close()
                        except (KeyboardInterrupt):
                            print("KeyBoardInterrupt. Closing files and exiting program.")
                            f_prep_IO.close()
                            f_IO.close()
                            file_temp.close()
                            end()
                    else:
                        print("\tNot in the set. Skipped.")
                    # end of if (tOt=='train' and day_s not in data.dateTest) ...
                else:
                    pass
                # end of if build_IO:
            else:
                print("\ts {0:d} of {1:d}. Not enough entries. Skipped.".format(int(s/2),int(len(separators)/2-1)))
        # end of for s in range(0,len(separators)-1,2):
        # add pointer index for later separating assets
        if build_IO:
            ass_IO_ass[ass_idx] = IO['pointer']
            #print(ass_IO_ass)
            
        # update asset index
        ass_idx += 1
        
        # update total number of samples
        m += stats["m_t_out"]
        # flush content file
        f_prep_IO.flush()
        
        print("\tTime for "+thisAsset+":"+str(np.floor(time.time()-tic))+"s"+
              ". Total time:"+str(np.floor(time.time()-ticTotal))+"s")
    
        #print("Total time:"+str(np.floor(time.time()-ticTotal))+"s")
    # end of for ass in data.assets:
    
    # close files
    f_prep_IO.close()
    
    if build_IO:
        f_IO.attrs.create('ass_IO_ass', ass_IO_ass, dtype=int)
        f_IO.close()
    
    # print percent of samps per level
    if not build_IO:
        # get ass_IO_ass from disk
        f_IO = h5py.File(filename_IO,'r')
        ass_IO_ass = f_IO.attrs.get("ass_IO_ass")
        f_IO.close()
    # get total number of samps
    m_t = ass_IO_ass[-1]
    if build_IO:
        print("Samples to RNN: "+str(m_t)+".\nPercent per level:"+str(totalSampsPerLevel/m_t))
    # reset graph
    tf.reset_default_graph()
    # start session
    with tf.Session() as sess:    
        model.train(sess, int(np.ceil(m_t/aloc)), ID=IDweights, IDIO=IDweights, data_format='hdf5', filename_IO=filename_IO, aloc=aloc)