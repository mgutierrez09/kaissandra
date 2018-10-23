# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 17:03:11 2018

@author: mgutierrez
"""

def configuration():
    """
    <DocString>
    """
    # data parameters
    dateTest = [                                                      '2018.03.09',
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
                  '2018.08.06','2018.08.07','2018.08.08','2018.08.09','2018.08.10']
    movingWindow = 100
    nEventsPerStat = 1000
    lB = 1200
    assets = [1, 2, 3, 4, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 19, 27, 28, 29, 30, 31, 32]
    # general parameters
    if_build_IO = True
    IDweights = '000272'
    hdf5_directory = 'D:/SDC/py/HDF5/'
    IO_directory = '../RNN/IO/'
    
    # model parameters
    size_hidden_layer=100
    L=3
    size_output_layer=5
    keep_prob_dropout=1
    miniBatchSize=32
    outputGain=.5
    commonY=3
    lR0=0.0001
    num_epochs=1000
    
    # test-specific parameters
    IDresults = '100272'
    startFrom = -1
    endAt = -1
    save_journal = False
    
    # getFeatures
    save_stats = True
    # add parameters to config dictionary
    config = {'dateTest':dateTest,
            'movingWindow':movingWindow,
           'nEventsPerStat':nEventsPerStat,
           'lB':lB,
           'assets':assets,
           
           'size_hidden_layer':size_hidden_layer,
           'L':L,
           'size_output_layer':size_output_layer,
           'keep_prob_dropout':keep_prob_dropout,
           'miniBatchSize':miniBatchSize,
           'outputGain':outputGain,
           'commonY':commonY,
           'lR0':lR0,
           'num_epochs':num_epochs,
           
           'if_build_IO':if_build_IO,
           'IDweights':IDweights,
           'hdf5_directory':hdf5_directory,
           'IO_directory':IO_directory,
           
           'IDresults':IDresults,
           'startFrom':startFrom,
           'endAt':endAt,
           'save_journal':save_journal,
           
           'save_stats':save_stats}
    
    
    return config