# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 17:03:11 2018

@author: mgutierrez
"""

import pickle
import os
from local_config import local_vars

config_directory = '../config/'
config_extension = ".config"


def configuration(*ins):
    """
    <DocString>
    """
    if len(ins)>0:
        entries = ins[0]
    else:
        entries = {}
    if 'config_name' in entries:
        config_name = entries['config_name']
    else:
        config_name = 'C0289STRO'
    
    config_filename = config_directory+config_name+config_extension
    
    if not os.path.exists(config_filename):
        if 'dateTest' in entries:
            dateTest = entries['dateTest']
        else:
            # data parameters
            dateTest = ([                                                   '2018.03.09',
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
            
        if 'movingWindow' in entries:
            movingWindow = entries['movingWindow']
        else:
            movingWindow = 100
        if 'nEventsPerStat' in entries:
            nEventsPerStat = entries['nEventsPerStat']
        else:
            nEventsPerStat = 1000
        if 'lB' in entries:
            lB = entries['lB']
        else:
            lB = int(nEventsPerStat+movingWindow*3)
        if 'assets' in entries:
            assets = entries['assets']
        else:
            assets = [1,2,3,4,7,8,10,11,12,13,14,15,16,17,19,27,28,29,30,31,32]
        if 'channels' in entries:
            channels = entries['channels']
        else:
            channels = [0]
        if 'max_var' in entries:
            max_var = entries['max_var']
        else:
            max_var = 10
        if 'feature_keys_manual' in entries:
            feature_keys_manual = entries['feature_keys_manual']
        else:
            feature_keys_manual = [i for i in range(37)]
        if 'feature_keys_tsfresh' in entries:
            feature_keys_tsfresh = entries['feature_keys_tsfresh']
        else:
            feature_keys_tsfresh = []#[i for i in range(37,68)]#[37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67]#
        if 'var_feat_keys' in entries:
            var_feat_keys = entries['var_feat_keys']
        else:
            var_feat_keys = []#[i for i in range(68,93)]
        
        # general parameters
        if 'if_build_IO' in entries:
            if_build_IO = entries['if_build_IO']
        else:
            if_build_IO = True
        if 'from_stats_file' in entries:
            from_stats_file = entries['from_stats_file']
        else:
            from_stats_file = True
        if 'IDweights' in entries:
            IDweights = entries['IDweights']
        else:
            IDweights = '00'+config_name[1:]
        if 'inverse_load' in entries:
            inverse_load = entries['inverse_load']
        else:
            inverse_load = False
        if 'weights_directory' in entries:
            weights_directory = entries['weights_directory']
        else:
            weights_directory = local_vars.weights_directory 
        if 'results_directory' in entries:
            results_directory = entries['results_directory']
        else:
            results_directory = local_vars.results_directory 
        
        if 'hdf5_directory' in entries:
            hdf5_directory = entries['hdf5_directory']
        else:
            hdf5_directory = local_vars.hdf5_directory#'D:/SDC/py/HDF5/'
        if 'IO_directory' in entries:
            IO_directory = entries['IO_directory']
        else:
            IO_directory = local_vars.IO_directory
        
        # model parameters
        if 'size_hidden_layer' in entries:
            size_hidden_layer = entries['size_hidden_layer']
        else:
            size_hidden_layer=100
        if 'L' in entries:
            L = entries['L']
        else:
            L=3
        if 'size_output_layer' in entries:
            size_output_layer = entries['size_output_layer']
        else:
            size_output_layer=5
        if 'keep_prob_dropout' in entries:
            keep_prob_dropout = entries['keep_prob_dropout']
        else:
            keep_prob_dropout=1
        if 'miniBatchSize' in entries:
            miniBatchSize = entries['miniBatchSize']
        else:
            miniBatchSize=32
        if 'outputGain' in entries:
            outputGain = entries['outputGain']
        else:
            outputGain=.6
        if 'commonY' in entries:
            commonY = entries['commonY']
        else:
            commonY=3
        if 'lR0' in entries:
            lR0 = entries['lR0']
        else:
            lR0=0.0001
        if 'num_epochs' in entries:
            num_epochs = entries['num_epochs']
        else:
            num_epochs=20
        
        # test-specific parameters
        if 'IDresults' in entries:
            IDresults = entries['IDresults']
        else:
            IDresults = '10'+config_name[1:]#'100310INVO'
        if 'IO_results_name' in entries:
            IO_results_name = entries['IO_results_name']
        else:
            IO_results_name = IDresults
        if 'startFrom' in entries:
            startFrom = entries['startFrom']
        else:
            startFrom = -1
        if 'endAt' in entries:
            endAt = entries['endAt']
        else:
            endAt = -1
        if 'save_journal' in entries:
            save_journal = entries['save_journal']
        else:
            save_journal = False
        
        # feature-specific configuration
        if 'save_stats' in entries:
            save_stats = entries['save_stats']
        else:
            save_stats = True
        if 'load_features_from' in entries:
            load_features_from = entries['load_features_from']
        else:
            load_features_from = 'manual' # {manual, tsfresh}
        if 'build_partial_raw' in entries:
            build_partial_raw = entries['build_partial_raw']
        else:
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
                  'var_feat_keys':var_feat_keys,
                  
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
                  'inverse_load':inverse_load,
                  'weights_directory':weights_directory,
                  'results_directory':results_directory,
                  
                  'IDresults':IDresults,
                  'startFrom':startFrom,
                  'endAt':endAt,
                  'save_journal':save_journal,
                  
                  'save_stats':save_stats,
                  'load_features_from':load_features_from,
                  'build_partial_raw':build_partial_raw}
    
        # save config file for later use
    
        if not os.path.exists(config_directory):
            os.mkdir(config_directory)
        pickle.dump( config, open( config_filename, "wb" ))
        print(config)
        print("Config file "+config_filename+" saved")
    else:
        config = pickle.load( open( config_filename, "rb" ))
        if len(ins)>0:
            print("WARNING! Arguments not taken into consideration")
        print("Config file "+config_filename+" exists. Loaded from disk")
    return config

def retrieve_config(config_name):
    """  """
    config_filename = '../config/'+config_name+".config"
        
    if os.path.exists(config_filename):
        config = pickle.load( open( config_filename, "rb" ))
        print("Config file "+config_filename+" loaded from disk")
    else:
        raise OSError("ERROR config file "+config_filename+" does not exist")
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
    """ Print configuration file """
    config_filename = config_directory+config_name+config_extension
    if os.path.exists(config_filename):
        config = pickle.load( open( config_filename, "rb" ))
        print(config)
        return True
    else:
        print("Config name "+config_name+" does not exist")
        return False

def delete_config(config_name):
    """ Delete configuration file from disk """
    config_filename = config_directory+config_name+config_extension
    os.remove(config_filename)
    print("Config file "+config_name+" deleted")
    
def modify_config(config_name,key,value):
    """
    
    """
    config_filename = config_directory+config_name+config_extension
    if os.path.exists(config_filename):
        config = pickle.load( open( config_filename, "rb" ))
        if key in config:
            config[key] = value
            pickle.dump( config, open( config_filename, "wb" ))
            print("Config file "+config_filename+" saved")
            return True
        else:
            raise ValueError(key+" not in "+config_name)
    else:
        print("Config name "+config_name+" does not exist")
        config = None
        return False
    
        
if __name__=='__main__':
    #configuration()
    pass