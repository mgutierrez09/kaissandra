# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 17:03:11 2018

@author: mgutierrez
"""

import pickle
import os
import numpy as np

from kaissandra.local_config import local_vars

config_extension = ".config"

def write_log(log_message, log_file):
        """
        Write in log file
        """
        file = open(local_vars.log_directory+log_file,"a")
        file.write(log_message+"\n")
        file.close()
        return None

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
        config_name = 'C0317INVO'
    
    config_filename = local_vars.config_directory+config_name+config_extension
    
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
                        
#            dateTest = ['2018.11.12','2018.11.13','2018.11.14','2018.11.15','2018.11.16',
#                        '2018.11.19','2018.11.20','2018.11.21','2018.11.22','2018.11.23',
#                        '2018.11.26','2018.11.27','2018.11.28','2018.11.29','2018.11.30',
#                        '2018.12.03','2018.12.04','2018.12.05','2018.12.06','2018.12.07',
#                        '2018.12.10','2018.12.11','2018.12.12','2018.12.13','2018.12.14',
#                        '2018.12.17','2018.12.18','2018.12.19','2018.12.20','2018.12.21',
#                        '2018.12.24','2018.12.25','2018.12.26','2018.12.27','2018.12.28',
#                        '2018.12.31','2019.01.01','2019.01.02','2019.01.03','2019.01.04',
#                        '2019.01.07','2019.01.08','2019.01.09','2019.01.10','2019.01.11',
#                        '2019.01.14','2019.01.15','2019.01.16','2019.01.17','2019.01.18',
#                        '2019.01.21','2019.01.22','2019.01.23','2019.01.24','2019.01.25',
#                        '2019.01.28','2019.01.29','2019.01.30','2019.01.31','2019.02.01',
#                        '2019.02.04','2019.02.05','2019.02.06','2019.02.07','2019.02.08',
#                        '2019.02.11','2019.02.12','2019.02.13','2019.02.14','2019.02.15',
#                        '2019.02.18','2019.02.19','2019.02.20','2019.02.21','2019.02.22']
            
    #        dateTest = [                                               '2018.03.09',
    #                '2018.03.12','2018.03.13','2018.03.14','2018.03.15','2018.03.16',
    #                '2018.03.19','2018.03.20']
            
        if 'movingWindow' in entries:
            movingWindow = entries['movingWindow']
        else:
            movingWindow = 1000
        if 'nEventsPerStat' in entries:
            nEventsPerStat = entries['nEventsPerStat']
        else:
            nEventsPerStat = 10000
        if 'lB' in entries:
            lB = entries['lB']
        else:
            lB = int(nEventsPerStat+movingWindow*3)
        seq_len = int((lB-nEventsPerStat)/movingWindow+1)
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
        if 'noVarFeatsManual' in entries:
            noVarFeatsManual = entries['noVarFeatsManual']
        else:
            noVarFeatsManual = [8,9,12,17,18,21,23,24,25,26,27,28,29]
        if 'feature_keys_tsfresh' in entries:
            feature_keys_tsfresh = entries['feature_keys_tsfresh']
        else:
            feature_keys_tsfresh = []#[i for i in range(37,68)]#[37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67]#
        if 'var_feat_keys' in entries:
            var_feat_keys = entries['var_feat_keys']
        else:
            var_feat_keys = []#[i for i in range(68,93)]
        if 'lookAheadIndex' in entries:
            lookAheadIndex = entries['lookAheadIndex']
        else:
            lookAheadIndex = 3
        if 'build_XY_mode' in entries:
            build_XY_mode = entries['build_XY_mode']
        else:
            build_XY_mode = 'K_fold'#'datebased'
        
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
            inverse_load = True
        if 'feats_from_bids' in entries:
            feats_from_bids = entries['feats_from_bids']
        else:
            feats_from_bids = False
        
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
            size_output_layer = 5
        if 'keep_prob_dropout' in entries:
            keep_prob_dropout = entries['keep_prob_dropout']
        else:
            keep_prob_dropout = 1
        if 'miniBatchSize' in entries:
            miniBatchSize = entries['miniBatchSize']
        else:
            miniBatchSize = 256
        if 'outputGain' in entries:
            outputGain = entries['outputGain']
        else:
            outputGain = 1
        if 'commonY' in entries:
            commonY = entries['commonY']
        else:
            commonY = 3
        if 'lR0' in entries:
            lR0 = entries['lR0']
        else:
            lR0=0.0001
        if 'num_epochs' in entries:
            num_epochs = entries['num_epochs']
        else:
            num_epochs = 100
        if 'rand_mB' in entries:
            rand_mB = entries['rand_mB']
        else:
            rand_mB = True
        if 'loss_funcs' in entries:
            loss_funcs = entries['loss_funcs']
        else:
            loss_funcs = ['cross_entropy']
        if 'n_bits_outputs' in entries:
            n_bits_outputs = entries['n_bits_outputs']
        else:
            n_bits_outputs = [size_output_layer]
        assert(sum(n_bits_outputs)==size_output_layer)
        # test-specific parameters
        if 'IDresults' in entries:
            IDresults = entries['IDresults']
        else:
            IDresults = '10'+config_name[1:]#'100310INVO'
        if 'IO_results_name' in entries:
            IO_results_name = entries['IO_results_name']
        else:
            # TODO: change to [:-1]
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
#        if 'save_log' in entries:
#            save_log = entries['save_log']
#        else:
#            save_log = False
        if 't_indexes' not in entries:
            t_indexes = [l for l in range(int((lB-nEventsPerStat)/movingWindow+1))]
        else:
            t_indexes = entries['t_indexes']
        if 'resolution' not in entries:
            resolution = 10
        else:
            resolution = entries['resolution']
        if 'resolution' not in entries:
            resolution = 10
        else:
            resolution = entries['resolution']
        if 'thresholds_mc' not in entries:
            thresholds_mc = [.5+i/resolution for i in range(int(resolution/2))]
        else:
            thresholds_mc = entries['thresholds_mc']
        if 'thresholds_md' not in entries:
            thresholds_md = [.5+i/resolution for i in range(int(resolution/2))]
        else:
            thresholds_md = entries['thresholds_md']
        if 'thresholds_mg' not in entries:
            thresholds_mg = [int(np.round((.5+i/resolution)*100))/100 for i in range(int(resolution/2))]
        else:
            thresholds_mg = entries['thresholds_mg']
        if 'results_from' in entries:
            results_from = entries['results_from']
        else:
            results_from = 'COMB' # {'BIDS','ASKS','COMB'}
        if 'combine_ts' in entries:
            combine_ts = entries['combine_ts']
        else:
            combine_ts = {'if_combine':False,'params_combine':[{'alg':'mean'}]}# 'adc': AD-combine
                                                           # 'mean': mean
        
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
        if 'build_test_db' in entries:
            build_test_db = entries['build_test_db']
        else:
            build_test_db = False
        assert(not (build_test_db and save_stats))
        
        # add parameters to config dictionary
        config = {'config_name':config_name,
                  
                  'dateTest':dateTest,
                  'movingWindow':movingWindow,
                  'nEventsPerStat':nEventsPerStat,
                  'lB':lB,
                  'seq_len':seq_len,
                  'assets':assets,
                  'channels':channels,
                  'max_var':max_var,
                  'feature_keys_manual':feature_keys_manual,
                  'noVarFeatsManual':noVarFeatsManual,
                  'feature_keys_tsfresh':feature_keys_tsfresh,
                  'var_feat_keys':var_feat_keys,
                  'lookAheadIndex':lookAheadIndex,
                  'build_XY_mode':build_XY_mode,
                  
                  'size_hidden_layer':size_hidden_layer,
                  'L':L,
                  'size_output_layer':size_output_layer,
                  'keep_prob_dropout':keep_prob_dropout,
                  'miniBatchSize':miniBatchSize,
                  'outputGain':outputGain,
                  'commonY':commonY,
                  'lR0':lR0,
                  'num_epochs':num_epochs,
                  'rand_mB':rand_mB,
                  'loss_funcs':loss_funcs,
                  'n_bits_outputs':n_bits_outputs,
                  
                  'if_build_IO':if_build_IO,
                  'from_stats_file':from_stats_file,
                  'IDweights':IDweights,
                  'IO_results_name':IO_results_name,
                  'inverse_load':inverse_load,
                  'results_from':results_from,
                  'combine_ts':combine_ts,
                  
                  'IDresults':IDresults,
                  'startFrom':startFrom,
                  'endAt':endAt,
                  'save_journal':save_journal,
                  't_indexes':t_indexes,
                  'resolution':resolution,
                  'thresholds_mc':thresholds_mc,
                  'thresholds_md':thresholds_md,
                  'thresholds_mg':thresholds_mg,
                  
                  'save_stats':save_stats,
                  'load_features_from':load_features_from,
                  'build_partial_raw':build_partial_raw,
                  'feats_from_bids':feats_from_bids,
                  'build_test_db':build_test_db}
    
        # save config file for later use
    
        if not os.path.exists(local_vars.config_directory):
            os.mkdir(local_vars.config_directory)
        pickle.dump( config, open( config_filename, "wb" ))
        print(config)
        print("Config file "+config_filename+" saved")
    else:
        config = pickle.load( open( config_filename, "rb" ))
        for key, val in entries.items():
            config[key] = val
        if len(entries)>0:
            print("WARNING! Config values have been overwritten with entries values")
        print("Config file "+config_filename+" exists. Loaded from disk")
    return config

def retrieve_config(config_name):
    """  """
    config_dir_filename = local_vars.config_directory+config_name+'.config'#
        
    if os.path.exists(config_dir_filename):
        config = pickle.load( open( config_dir_filename, "rb" ))
        print("Config file "+config_dir_filename+" loaded from disk")
    else:
        raise OSError("ERROR config file "+config_dir_filename+" does not exist")
    return config

def get_config(config_name):
    """
    
    """
    config_filename = local_vars.config_directory+config_name+config_extension
    if os.path.exists(config_filename):
        config = pickle.load( open( config_filename, "rb" ))
        return config
    else:
        print("Config name "+config_name+" does not exist")
        return None

def save_config(config):
    """  """
    config_filename = local_vars.config_directory+config['config_name']+config_extension
    if not os.path.exists(config_filename):
        pickle.dump( config, open( config_filename, "wb" ))
        print("Config file "+config_filename+" saved")
        return True
    else:
        return False

def print_config(config_name):
    """ Print configuration file """
    config_filename = local_vars.config_directory+config_name+config_extension
    if os.path.exists(config_filename):
        config = pickle.load( open( config_filename, "rb" ))
        print(config)
        return True
    else:
        print("Config name "+config_name+" does not exist")
        return False

def delete_config(config_name):
    """ Delete configuration file from disk """
    config_filename = local_vars.config_directory+config_name+config_extension
    os.remove(config_filename)
    print("Config file "+config_name+" deleted")
    
def modify_config(config_name,key,value):
    """
    
    """
    config_filename = local_vars.config_directory+config_name+config_extension
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
    
def add_to_config(config_name,key,value):
    """
    
    """
    config_filename = local_vars.config_directory+config_name+config_extension
    if os.path.exists(config_filename):
        config = pickle.load( open( config_filename, "rb" ))
        if key not in config:
            config[key] = value
            pickle.dump( config, open( config_filename, "wb" ))
            print("Config file "+config_filename+" saved")
            return True
        else:
            raise ValueError(key+" in "+config_name+". Use modify_config instead")
    else:
        print("Config name "+config_name+" does not exist")
        config = None
        return False

def configuration_trader(*ins):
    """ Function to generate a trader config file """
    
    config_name = 'TTEST'
    config_filename = local_vars.config_directory+config_name+config_extension
    
    if not os.path.exists(config_filename):
        dateTest = ['2018.11.12','2018.11.13','2018.11.14','2018.11.15','2018.11.16',
                '2018.11.19','2018.11.20','2018.11.21','2018.11.22','2018.11.23',
                '2018.11.26','2018.11.27','2018.11.28','2018.11.29','2018.11.30',
                '2018.12.03','2018.12.04','2018.12.05','2018.12.06','2018.12.07',
                '2018.12.10','2018.12.11','2018.12.12','2018.12.13','2018.12.14',
                '2018.12.17','2018.12.18','2018.12.19','2018.12.20','2018.12.21',
                '2018.12.24','2018.12.25','2018.12.26','2018.12.27','2018.12.28',
                '2018.12.31','2019.01.01','2019.01.02','2019.01.03','2019.01.04',
                '2019.01.07','2019.01.08','2019.01.09','2019.01.10','2019.01.11',
                '2019.01.14','2019.01.15','2019.01.16','2019.01.17','2019.01.18',
                '2019.01.21','2019.01.22','2019.01.23','2019.01.24','2019.01.25',
                '2019.01.28','2019.01.29','2019.01.30','2019.01.31','2019.02.01',
                '2019.02.04','2019.02.05','2019.02.06','2019.02.07','2019.02.08',
                '2019.02.11','2019.02.12','2019.02.13','2019.02.14','2019.02.15',
                '2019.02.18','2019.02.19','2019.02.20','2019.02.21','2019.02.22']
        numberNetworks = 1
        IDweights = ['000318INVO']
        IDresults = ['100318INVO']
        lIDs = [len(IDweights[i]) for i in range(numberNetworks)]
        list_name = ['15e_1t_77m_2p']
        IDepoch = ['15']
        netNames = ['31815']#['350E13T3S', '350E6T2L', '327T21E0S', '500E29T3L']
        list_t_indexs = [[1]]
        list_inv_out = [True for i in range(numberNetworks)]
        list_feats_from = ['B']#['B','B','B','A']# {B: from bid symbols, A: from ask symbols}
        list_entry_strategy = ['spread_ranges' for i in range(numberNetworks)] #'fixed_thr','gre' or 'spread_ranges'
        # {'S': short, 'L':long, 'C':combine} TODO: combine not supported yet
        list_spread_ranges = [{'sp': [2], 'th': [(0.7, 0.7)],'dir':'C'}]
        list_priorities = [[0]]#[[3],[2],[1],[0]]
        phase_shifts = [1 for i in range(numberNetworks)]
        list_thr_sl = [1000]
        list_thr_tp = [1000 for i in range(numberNetworks)]
        delays = [0 for i in range(numberNetworks)]
        mWs = [100]#[500,500,200,200]
        nExSs = [1000]#[5000,5000,2000,2000]
        lBs = [1300]#[6500,6500,2600,2600]#[1300]
        list_lim_groi_ext = [-100 for i in range(numberNetworks)]
        list_w_str = ['55' for i in range(numberNetworks)]
        
        model_dict = {'size_hidden_layer':[100 for i in range(numberNetworks)],
                      'L':[3 for i in range(numberNetworks)],
                      'size_output_layer':[5 for i in range(numberNetworks)],
                      'outputGain':[1 for i in range(numberNetworks)]}
        
        list_weights = [np.array([.5,.5]) for i in range(numberNetworks)]
        list_lb_mc_op = [.5 for i in range(numberNetworks)]
        list_lb_md_op = [.8 for i in range(numberNetworks)]
        list_lb_mc_ext = [.5 for i in range(numberNetworks)]
        list_lb_md_ext = [.6 for i in range(numberNetworks)]
        list_ub_mc_op = [1 for i in range(numberNetworks)]
        list_ub_md_op = [1 for i in range(numberNetworks)]
        list_ub_mc_ext = [1 for i in range(numberNetworks)]
        list_ub_md_ext = [1 for i in range(numberNetworks)]
        list_fix_spread = [False for i in range(numberNetworks)]
        list_fixed_spread_pips = [4 for i in range(numberNetworks)]
        list_max_lots_per_pos = [.1 for i in range(numberNetworks)]
        list_flexible_lot_ratio = [False for i in range(numberNetworks)]
        list_if_dir_change_close = [False for i in range(numberNetworks)]
        list_if_dir_change_extend = [False for i in range(numberNetworks)]
        
        config = {'config_name':config_name,
                  'dateTest':dateTest,
                  'numberNetworks':numberNetworks,
                  'IDweights':IDweights,
                  'IDresults':IDresults,
                  'lIDs':lIDs,
                  'list_name':list_name,
                  'IDepoch':IDepoch,
                  'netNames':netNames,
                  'list_t_indexs':list_t_indexs,
                  'list_inv_out':list_inv_out,
                  'list_entry_strategy':list_entry_strategy,
                  'list_spread_ranges':list_spread_ranges,
                  'list_priorities':list_priorities,
                  'list_feats_from':list_feats_from,
                  'phase_shifts':phase_shifts,
                  'list_thr_sl':list_thr_sl,
                  'list_thr_tp':list_thr_tp,
                  'delays':delays,
                  'mWs':mWs,
                  'nExSs':nExSs,
                  'lBs':lBs,
                  'list_lim_groi_ext':list_lim_groi_ext,
                  'list_w_str':list_w_str,
                  'model_dict':model_dict,
                  'list_weights':list_weights,
                  'list_lb_mc_op':list_lb_mc_op,
                  'list_lb_md_op':list_lb_md_op,
                  'list_lb_mc_ext':list_lb_mc_ext,
                  'list_lb_md_ext':list_lb_md_ext,
                  'list_ub_mc_op':list_ub_mc_op,
                  'list_ub_md_op':list_ub_md_op,
                  'list_ub_mc_ext':list_ub_mc_ext,
                  'list_ub_md_ext':list_ub_md_ext,
                  'list_fix_spread':list_fix_spread,
                  'list_fixed_spread_pips':list_fixed_spread_pips,
                  'list_max_lots_per_pos':list_max_lots_per_pos,
                  'list_flexible_lot_ratio':list_flexible_lot_ratio,
                  'list_if_dir_change_close':list_if_dir_change_close,
                  'list_if_dir_change_extend':list_if_dir_change_extend
                }
        if not os.path.exists(local_vars.config_directory):
            os.mkdir(local_vars.config_directory)
        pickle.dump( config, open( config_filename, "wb" ))
        #print(config)
        print("Config file "+config_filename+" saved")
    else:
        config = pickle.load( open( config_filename, "rb" ))
        if len(ins)>0:
            print("WARNING! Arguments not taken into consideration")
        print("Config file "+config_filename+" exists. Loaded from disk")
    return config
        
if __name__=='__main__':
    #configuration()
    pass