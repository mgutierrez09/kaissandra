# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 17:03:11 2018

@author: mgutierrez
"""

import pickle
import os

config_directory = '../config/'
config_extension = ".config"


def configuration(*ins):
    """
    <DocString>
    """
    if len(ins)==0:
        config_name = 'C01100'
        config_filename = config_directory+config_name+config_extension
        
        # data parameters
        dateTest = ([                                               '2018.03.09',
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
                '2018.09.24','2018.09.25','2018.09.26','2018.09.27']+['2018.09.28',
                '2018.10.01','2018.10.02','2018.10.03','2018.10.04','2018.10.05',
                '2018.10.08','2018.10.09','2018.10.10','2018.10.11','2018.10.12',
                '2018.10.15','2018.10.16','2018.10.17','2018.10.18','2018.10.19',
                '2018.10.22','2018.10.23','2018.10.24','2018.10.25','2018.10.26',
                '2018.10.29','2018.10.30','2018.10.31','2018.11.01','2018.11.02',
                '2018.11.05','2018.11.06','2018.11.07','2018.11.08','2018.11.09'])
        
#        dateTest = [                                               '2018.03.09',
#                '2018.03.12','2018.03.13','2018.03.14','2018.03.15','2018.03.16',
#                '2018.03.19','2018.03.20']
        
        
        movingWindow = 110
        nEventsPerStat = 1100
        lB = int(nEventsPerStat+movingWindow*3)
        assets = [1,2,3,4,7,8,10,11,12,13,14,15,16,17,19,27,28,29,30,31,32]
        channels = [0]
        max_var = 10
        feature_keys_manual = [i for i in range(37)]
        feature_keys_tsfresh = []#[37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67]#[i for i in range(37,68)]
        
        # general parameters
        if_build_IO = True
        from_stats_file = True
        IDweights = '001100'
        
        hdf5_directory = 'D:/SDC/py/HDF5/'
        IO_directory = '../RNN/IO/'
        
        # model parameters
        size_hidden_layer=100
        L=3
        size_output_layer=5
        keep_prob_dropout=1
        miniBatchSize=512
        outputGain=.4
        commonY=3
        lR0=0.0001
        num_epochs=1
        
        # test-specific parameters
        IDresults = '101100'
        IO_results_name = IDresults
        startFrom = -1
        endAt = -1
        save_journal = True
        
        # feature-specific configuration
        save_stats = True
        load_features_from = 'manual' # {manual, tsfresh}
        build_partial_raw = False
        
        # add parameters to config dictionary
        config = {'config_name':config_name,
                
                'dateTest':dateTest,
                'movingWindow':movingWindow,
               'nEventsPerStat':nEventsPerStat,
               'lB':lB,
               'assets':assets,
               'channels':channels,
               'max_var':max_var,
               'feature_keys_manual':feature_keys_manual,
               'feature_keys_tsfresh':feature_keys_tsfresh,
               
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
               'from_stats_file':from_stats_file,
               'IDweights':IDweights,
               'IO_results_name':IO_results_name,
               'hdf5_directory':hdf5_directory,
               'IO_directory':IO_directory,
               
               'IDresults':IDresults,
               'startFrom':startFrom,
               'endAt':endAt,
               'save_journal':save_journal,
               
               'save_stats':save_stats,
               'load_features_from':load_features_from,
               'build_partial_raw':build_partial_raw}
        
        # save config file for later use
        if not os.path.exists(config_filename):
            pickle.dump( config, open( config_filename, "wb" ))
            print(config)
            print("Config file "+config_filename+" saved")
        else:
            raise OSError("ERROR config file "+config_name+" already exists. "+
                  "Pass name as an arg if you want to load it")
    else:
        config_filename = config_directory+ins[0]+".config"
        if os.path.exists(config_filename):
            config = pickle.load( open( config_filename, "rb" ))
            print("Config file "+ins[0]+" loaded from disk")
        else:
            raise OSError("ERROR config file "+ins[0]+" does not exist")
        
    return config

def get_config(config_name):
    """
    
    """
    config_filename = config_directory+config_name+config_extension
    if os.path.exists(config_filename):
        config = pickle.load( open( config_filename, "rb" ))
        return config
    else:
        print("Config name "+config_name+" does not exist")
        return None

def save_config(config):
    """  """
    config_filename = config_directory+config['config_name']+config_extension
    if not os.path.exists(config_filename):
        pickle.dump( config, open( config_filename, "wb" ))
        print("Config file "+config_filename+" saved")
        return True
    else:
        return False

def print_config(config_name):
    """
    
    """
    config_filename = config_directory+config_name+config_extension
    if os.path.exists(config_filename):
        config = pickle.load( open( config_filename, "rb" ))
        print(config)
        return True
    else:
        print("Config name "+config_name+" does not exist")
        return False

def delete_config(config_name):
    """
    
    """
    config_filename = config_directory+config_name+config_extension
    os.remove(config_filename)
    print("Config file "+config_name+" deleted")
    
def modify_config(config_name,key,value):
    """
    
    """
    config_filename = config_directory+config_name+config_extension
    if os.path.exists(config_filename):
        config = pickle.load( open( config_filename, "rb" ))
        config[key] = value
        pickle.dump( config, open( config_filename, "wb" ))
        print("Config file "+config_filename+" saved")
        return True
    else:
        print("Config name "+config_name+" does not exist")
        config = None
        return False
    
        
if __name__=='__main__':
    #configuration()
    pass