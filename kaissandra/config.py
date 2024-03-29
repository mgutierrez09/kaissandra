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

class Config():
    """ Config class containing general information """
    
    AllAssets = {"0":"[USDX]",
             "1":'AUDCAD',# yes
             "2":'EURAUD', # yes
             "3":'EURCAD', # yes
             "4":'EURCHF', # yes
             "5":'EURCZK',
             "6":'EURDKK',
             "7":'EURGBP',
             "8":'EURNZD',
             "9":'EURPLN',
             "10":'EURUSD',
             "11":'GBPAUD', # yes but as AUDGBP
             "12":'GBPCAD',
             "13":'GBPCHF',
             "14":'GBPUSD',
             "15":'GOLD',
             "16":'USDCAD',
             "17":'USDCHF',
             "18":'USDHKD', # yes
             "19":'USDJPY',
             "20":'USDMXN', # no
             "21":'USDNOK',
             "22":'USDPLN',
             "23":'USDRUB',
             "24":'USDSGD',
             "25":'XAGUSD',
             "26":'XAUUSD',
             "27":"CADJPY",
             "28":"EURJPY",
             "29":"AUDJPY", # yes
             "30":"CHFJPY",
             "31":"GBPJPY",
             "32":"NZDUSD",
             # new assets
             "33":"AUDUSD",
             "34":"AUDCHF",
             "35":"AUDNZD",
             "36":"CADCHF",
             "37":"EURPLN",
             "38":"EURHUF",
             "39":"USDCNH",
             "40":"USDRON",
             "41":"USDPLN",
             "42":"EURRUB",
             "43":"EURCZK",
             "44":"USDCZK",
             "45":"USDRUB",
             # Crypto
             "46":"BCHEUR",
             "47":"BCHUSD",
             "48":"BTCEUR",
             "49":"BTCUSD",
             "50":"DSHBTC",
             "51":"DSHEUR",
             "52":"DSHUSD",
             "53":"EOSBTC",
             "54":"EOSEUR",
             "55":"EOSUSD",
             "56":"ETCBTC",
             "57":"ETCEUR",
             "58":"ETCUSD",
             "59":"ETHBTC",
             "60":"ETHEUR",
             "61":"BCHBTC",
             "62":"ETHUSD",
             "63":"ZECUSD",
             "64":"LTCBTC",
             "65":"LTCEUR",
             "66":"LTCUSD",
             "67":"XLMBTC",
             "68":"XLMEUR",
             "69":"XLMUSD",
             "70":"XMRBTC",
             "71":"XMREUR",
             "72":"XMRUSD",
             "73":"XRPBTC",
             "74":"XRPEUR",
             "75":"XRPUSD",
             "76":"ZECBTC",
             "77":"ZECUSD"}
    # list of currencies
    AllCurrencies = ['AUD', 'CAD', 'CHF', 'EUR', 'GBP', 'JPY', 'NZD', 'USD']
    
    # feature indexes
    FI = {"symbol":0,
          "EMA01":1,
          "EMA05":2,
          "EMA1":3,
          "EMA5":4,
          "EMA10":5,
          "EMA50":6,
          "EMA100":7,
          "variance":8,
          "timeInterval":9,
          "parSARhigh20":10,
          "parSARlow20":11,
          "time":12,
          "parSARhigh2":13,
          "parSARlow2":14,
          "difVariance":15,
          "difTimeInterval":16,
          "maxValue":17,
          "minValue":18,
          "difMaxValue":19,
          "difMinValue":20,
          "minOmax":21,
          "difMinOmax":22,
          "symbolOema01":23,
          "symbolOema05":24,
          "symbolOema1":25,
          "symbolOema5":26,
          "symbolOema10":27,
          "symbolOema50":28,
          "symbolOema100":29,
          "difSymbolOema01":30,
          "difSymbolOema05":31,
          "difSymbolOema1":32,
          "difSymbolOema5":33,
          "difSymbolOema10":34,
          "difSymbolOema50":35,
          "difSymbolOema100":36,
          "volume":37,
          "mean":38,
          "median":39,
          # canonical inputs
          "EUR":40,
          "GBP":41,
          "USD":42,
          "JPY":43,
          "CHF":44,
          "AUD":45,
          "CAD":46,
          "NZD":47,
          # TSFRESH
          "quantile0.3":137,
          "quantile0.4":138,
          "quantile0.2":139,
          "quantile0.1":140,
          "quantile0.5":141,
          "quantile0.8":142,
          "quantile0.9":143,
          "quantile0.6":144,
          "quantile0.7":145,
          "fft_coefficient0real":146,
          #"fft_coefficient0abs":47,
          "sum_values":147,
          "median_":148,
          "c3_3":149,
          "c3_2":150,
          "c3_1":151,
          "mean_":152,
          "abs_energy":153,
          "linear_trend":154,
          "agg_min50inter":155,
          "agg_min5inter":156,
          "agg_max10inter":157,
          "agg_mean50inter":158,
          "agg_max50inter":159,
          "agg_mean5inter":160,
          "agg_min10inter":161,
          "agg_mean10inter":162,
          "agg_max5inter":163,
          # Trading Signals
          "STO14":2000, # Stochastic Oscilator
          "STO3":2001,
          "RSI14":2002, # Relative Strength Index
          "RSI3":2003,
          "ADX14":2004, # Average Directional Index
          "ADX3":2005,
          "WILL14":2006, # Williams %R indicator
          "WILL3":2007,
          "BOLLUP20":2008, # Bollinger Bands
          "BOLLUP10":2009,
          "BOLLDOWN20":2010,
          "BOLLDOWN10":2011,
          "PERBOLL20":2012, # %B
          "PERBOLL10":2013,
          "ADI":2014, # Accumulation/Distribution Indicator
          "AROUP25":2015, # Aroon up/down indicators
          "ARODOWN25":2016,
          "AROUP10":2017,
          "ARODOWN10":2018}
    
    # Primary features
    PF = {FI["symbol"]:["symbol"],
          FI["EMA01"]:["EMA01"],
          FI["EMA05"]:["EMA05"],
          FI["EMA1"]:["EMA1"],
          FI["EMA5"]:["EMA5"],
          FI["EMA10"]:["EMA10"],
          FI["EMA50"]:["EMA50"],
          FI["EMA100"]:["EMA100"],
          FI["variance"]:["variance"],
          FI["timeInterval"]:["timeInterval"],
          FI["parSARhigh20"]:["parSARhigh20"],
          FI["parSARlow20"]:["parSARlow20"],
          FI["time"]: ["time"],
          FI["parSARhigh2"]:["parSARhigh2"],
          FI["parSARlow2"]:["parSARlow2"],
          FI["difVariance"]:["difVariance"],
          FI["difTimeInterval"]:["difTimeInterval"],
          FI["maxValue"]:["maxValue"],
          FI["minValue"]:["minValue"],
          FI["difMaxValue"]:["difMaxValue"],
          FI["difMinValue"]:["difMinValue"],
          FI["minOmax"]:["minOmax"],
          FI["difMinOmax"]:["difMinOmax"],
          FI["symbolOema01"]:["symbolOema01"],
          FI["symbolOema05"]:["symbolOema05"],
          FI["symbolOema1"]:["symbolOema1"],
          FI["symbolOema5"]:["symbolOema5"],
          FI["symbolOema10"]:["symbolOema10"],
          FI["symbolOema50"]:["symbolOema50"],
          FI["symbolOema100"]:["symbolOema100"],
          FI["difSymbolOema01"]:["difSymbolOema01"],
          FI["difSymbolOema05"]:["difSymbolOema05"],
          FI["difSymbolOema1"]:["difSymbolOema1"],
          FI["difSymbolOema5"]:["difSymbolOema5"],
          FI["difSymbolOema10"]:["difSymbolOema10"],
          FI["difSymbolOema50"]:["difSymbolOema50"],
          FI["difSymbolOema100"]:["difSymbolOema100"],
          FI["volume"]:["volume"],
          FI["mean"]:["mean"],
          FI["median"]:["median"],
          # Canonical features
          FI["EUR"]:["EUR"],
          FI["GBP"]:["GBP"],
          FI["USD"]:["USD"],
          FI["JPY"]:["JPY"],
          FI["CHF"]:["CHF"],
          FI["AUD"]:["AUD"],
          FI["CAD"]:["CAD"],
          FI["NZD"]:["NZD"],
          # automate features
          FI["quantile0.3"]:["quantile0.3","quantile",0.3],
          FI["quantile0.4"]:["quantile0.4","quantile",0.4],
          FI["quantile0.2"]:["quantile0.2","quantile",0.2],
          FI["quantile0.1"]:["quantile0.1","quantile",0.1],
          FI["quantile0.5"]:["quantile0.5","quantile",0.5],
          FI["quantile0.8"]:["quantile0.8","quantile",0.8],
          FI["quantile0.9"]:["quantile0.9","quantile",0.9],
          FI["quantile0.6"]:["quantile0.6","quantile",0.6],
          FI["quantile0.7"]:["quantile0.7","quantile",0.7],
          FI["fft_coefficient0real"]:["fft_coefficient0real","fft_coefficient",0,"real"],
          FI["sum_values"]:["sum_values","sum_values"],
          FI["median"]:["median","median"],
          FI["c3_3"]:["c3_3","c3",3],
          FI["c3_2"]:["c3_2","c3",2],
          FI["c3_1"]:["c3_1","c3",1],
          FI["mean"]:["mean","mean"],
          FI["abs_energy"]:["abs_energy","abs_energy"],
          FI["linear_trend"]:["linear_trend","linear_trend","intercept"],
          FI["agg_min50inter"]:["agg_min50inter","agg_linear_trend","min",50,"intercept"],
          FI["agg_min5inter"]:["agg_min5inter","agg_linear_trend","min",5,"intercept"],
          FI["agg_max10inter"]:["agg_max10inter","agg_linear_trend","max",10,"intercept"],
          FI["agg_mean50inter"]:["agg_mean50inter","agg_linear_trend","mean",50,"intercept"],
          FI["agg_max50inter"]:["agg_max50inter","agg_linear_trend","max",50,"intercept"],
          FI["agg_mean5inter"]:["agg_mean5inter","agg_linear_trend","mean",5,"intercept"],
          FI["agg_min10inter"]:["agg_min10inter","agg_linear_trend","min",10,"intercept"],
          FI["agg_mean10inter"]:["agg_mean10inter","agg_linear_trend","mean",10,"intercept"],
          FI["agg_max5inter"]:["agg_max5inter","agg_linear_trend","max",5,"intercept"],
          # Trading signals
          FI["STO14"]:["STO14", [14]], # Stochastic Oscilator
          FI["STO3"]:["STO3", [3]],
          FI["RSI14"]:["RSI14", [14]], # Relative Strength Index
          FI["RSI3"]:["RSI3", [3]],
          FI["ADX14"]:["ADX14", [14]], # Average Directional Index
          FI["ADX3"]:["ADX3", [3]], 
          FI["WILL14"]:["WILL14", [14]],
          FI["WILL3"]:["WILL3", [3]],
          FI["BOLLUP20"]:["BOLLUP20",[20]],
          FI["BOLLUP10"]:["BOLLUP10",[10]],
          FI["BOLLDOWN20"]:["BOLLDOWN20",[20]],
          FI["BOLLDOWN10"]:["BOLLDOWN10",[10]],
          FI["PERBOLL20"]:["PERBOLL20",[20]],
          FI["PERBOLL10"]:["PERBOLL10",[10]],
          FI["ADI"]:["ADI"],
          FI["AROUP25"]:["AROUP25",[25]],
          FI["ARODOWN25"]:["ARODOWN25",[25]],
          FI["AROUP10"]:["AROUP10",[10]],
          FI["ARODOWN10"]:["ARODOWN10",[10]]}
    
    secsInDay = 86400.0
    
    emas_ext = ['01','05','1','5','10','50','100']
    
    n_sars = 2
    maxStepSars = [20, 2]
    stepAF = 0.02
    sar_ext = ['20','2']
    sto_ext = ['14','3']
    rsi_ext = ['14','3']
    adx_ext = ['14','3']
    will_ext = ['14','3']
    boll_ext = ['20','10']
    adi_ext = ['']
    aro_ext = ['25','10']
    
    std_var = 0.1
    std_time = 0.1
    
    non_var_features = [8,9,12,17,18,21,23,24,25,26,27,28,29,37]+[i for i in range(40,48)]+[i for i in range(2000,2019)]
    canonical_features = [i for i in range(40,48)]

def write_log(log_message, log_file):
        """
        Write in log file
        """
        file = open(local_vars.log_directory+log_file,"a")
        file.write(log_message+"\n")
        file.close()
        return None

def configuration(entries, save=True):
    """
    <DocString>
    """
#    if len(ins)>0:
#        entries = ins[0]
#    else:
#        entries = {}
    if 'config_name' in entries:
        config_name = entries['config_name']
    else:
        config_name = 'CFEATSMOD400'
    
    config_filename = local_vars.config_directory+config_name+config_extension
    
    if not os.path.exists(config_filename):
        # if build_XY_mode=manual, this dates are used as edges
        if 'first_day' in entries:
            first_day = entries['first_day']
        else:
            first_day = '2016.01.01'
        if 'last_day' in entries:
            last_day = entries['last_day']
        else:
            last_day = '2018.11.09'
#        if 'dateTest' in entries:
#            dateTest = entries['dateTest']
#        else:
#            init_day = dt.datetime.strptime(first_day,'%Y.%m.%d').date()
#            end_day = dt.datetime.strptime(last_day,'%Y.%m.%d').date()
#            delta_dates = end_day-init_day
#            dateTestDt = [init_day + dt.timedelta(i) for i in range(delta_dates.days + 1)]
#            dateTest = []
#            for d in dateTestDt:
#                if d.weekday()<5:
#                    dateTest.append(dt.date.strftime(d,'%Y.%m.%d'))
        if 'edge_dates' in entries:
            edge_dates = entries['edge_dates']
        else:
            edge_dates = ['2018.03.09']
        if 'list_index2sets' in entries:
            list_index2sets = entries['list_index2sets']
        else:
            list_index2sets = ['Tr','Cv']
        if 'movingWindow' in entries:
            movingWindow = entries['movingWindow']
        else:
            movingWindow = 400
        if 'nEventsPerStat' in entries:
            nEventsPerStat = entries['nEventsPerStat']
        else:
            nEventsPerStat = 4000
        if 'lB' in entries:
            lB = entries['lB']
        else:
            lB = int(nEventsPerStat+movingWindow*3)
        seq_len = int((lB-nEventsPerStat)/movingWindow+1)
        if 'lbd' in entries:
            lbd = entries['lbd']
        else:
            lbd = (1-1/(nEventsPerStat*np.array([0.1, 0.5, 1, 5, 10, 50, 100]))).tolist()
        if 'assets' in entries:
            assets = entries['assets']
        else:
            assets = [1,2,3,4,7,8,10,11,12,13,14,16,17,19,27,28,29,30,31,32]
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
        if 'feature_keys' in entries:
            feature_keys = entries['feature_keys']
        else:
            feature_keys = [i for i in range(37)]
        if 'force_calculation_features' in entries:
            force_calculation_features = entries['force_calculation_features']
        else:
            force_calculation_features = [False for i in range(len(feature_keys))]
        if 'force_calulation_output' in entries:
            force_calulation_output = entries['force_calulation_output']
        else:
            force_calulation_output = False
        if 'noVarFeatsManual' in entries:
            noVarFeatsManual = entries['noVarFeatsManual']
        else:
            noVarFeatsManual = [8,9,12,17,18,21,23,24,25,26,27,28,29]#+[i for i in range(2000,2019)]
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
        if 'lookAheadVector' in entries:
            lookAheadVector = entries['lookAheadVector']
        else:
            lookAheadVector=[.1,.2,.5,1]
        if 'build_XY_mode' in entries:
            build_XY_mode = entries['build_XY_mode']
        else:
            build_XY_mode = 'K_fold'#'manual'
        if 'build_asset_relations' in entries:
            build_asset_relations = entries['build_asset_relations']
        else:
            build_asset_relations = ['direct', 'inverse']
        if 'phase_shift' in entries:
            phase_shift = entries['phase_shift']
        else:
            phase_shift = 2
        assert((movingWindow/phase_shift).is_integer())
        
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
            IDweights = 'W'+config_name[1:]
        if 'inverse_load' in entries:
            inverse_load = entries['inverse_load']
        else:
            inverse_load = True
        if 'feats_from_bids' in entries:
            feats_from_bids = entries['feats_from_bids']
        else:
            feats_from_bids = False
        if 'feats_from_all' in entries:
            feats_from_all = entries['feats_from_all']
        else:
            feats_from_all = False
        
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
            size_output_layer = 8
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
            loss_funcs = ['cross_entropy','cross_entropy','cross_entropy']
        if 'n_bits_outputs' in entries:
            n_bits_outputs = entries['n_bits_outputs']
        else:
            n_bits_outputs = [1,2,5]
        assert(sum(n_bits_outputs)==size_output_layer)
        # test-specific parameters
        if 'IDresults' in entries:
            IDresults = entries['IDresults']
        else:
            IDresults = 'R'+config_name[1:]#'100310INVO'
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
            save_stats = False
#        if 'load_features_from' in entries:
#            load_features_from = entries['load_features_from']
#        else:
#            load_features_from = 'manual' # {manual, tsfresh}
#        if 'build_partial_raw' in entries:
#            build_partial_raw = entries['build_partial_raw']
#        else:
#            build_partial_raw = False
        if 'build_test_db' in entries:
            build_test_db = entries['build_test_db']
        else:
            build_test_db = False
        assert(not (build_test_db and save_stats))
        if 'asset_relation' in entries:
            asset_relation = entries['asset_relation']
        else:
            asset_relation = 'direct' # {'direct','inverse'}
        
        # add parameters to config dictionary
        config = {'config_name':config_name,
                  
                  #'dateTest':dateTest,
                  'first_day':first_day,
                  'last_day':last_day,
                  'edge_dates':edge_dates,
                  'list_index2sets':list_index2sets,
                  'movingWindow':movingWindow,
                  'nEventsPerStat':nEventsPerStat,
                  'lB':lB,
                  'lbd':lbd,
                  'seq_len':seq_len,
                  'assets':assets,
                  'channels':channels,
                  'max_var':max_var,
                  'feature_keys_manual':feature_keys_manual,
                  'feature_keys':feature_keys,
                  'force_calculation_features':force_calculation_features,
                  'force_calulation_output':force_calulation_output,
                  'noVarFeatsManual':noVarFeatsManual,
                  'feature_keys_tsfresh':feature_keys_tsfresh,
                  'var_feat_keys':var_feat_keys,
                  'lookAheadIndex':lookAheadIndex,
                  'lookAheadVector':lookAheadVector,
                  'build_XY_mode':build_XY_mode,
                  'build_asset_relations':build_asset_relations,                  
                  'phase_shift':phase_shift,
                  
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
#                  'load_features_from':load_features_from,
#                  'build_partial_raw':build_partial_raw,
                  'feats_from_bids':feats_from_bids,
                  'feats_from_all':feats_from_all,
                  'build_test_db':build_test_db,
                  'asset_relation':asset_relation}
    
        # save config file for later use
    
        if not os.path.exists(local_vars.config_directory) and save:
            os.mkdir(local_vars.config_directory)
        print(config)
        if save:
            pickle.dump( config, open( config_filename, "wb" ))
            print("Config file "+config_filename+" saved")
        else:
            print("Config file NOT saved")
        
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

def save_config(config, from_server=False):
    """  """
    if not from_server:
        config_filename = local_vars.config_directory+config['config_name']+config_extension
    else:
        config_filename = local_vars.config_directory+config['config_name']+'remote'+config_extension
    pickle.dump( config, open( config_filename, "wb" ))
    if not os.path.exists(config_filename):
        
        print("Config file "+config_filename+" saved")
        return True
    else:
        print("WARNING! Config file "+config_filename+" overwritten")
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
    # 'T6N504060TESTv2': Agresive, i.e. 'R500AL','R500BS' open with pm_thr>=0.1 for sp<=1.3 or so
    # 'T6N504060TESTv1': first test version with three strategies per asset per direction
    config_name = 'T6N5060v1'#'T6N504060TESTv1'#'TESTPARAMUPDATE5'#'TTEST01010FS2NYREDOK2K52145314SR'#'TTESTv3'#
    config_filename = local_vars.config_directory+config_name+config_extension
    
    if not os.path.exists(config_filename):
        
        numberNetworks = 4
        IDresults = ['R500AL','R500BS']+['R600AL','R600BS']
        IDweights = [['W01010PS2NYk1K2A','W01010PS2NYk2K2A','WRNN01010k1K5A','WRNN01010k2K5A'],['W01010PS2NYk1K2A','W01010PS2NYk2K2A','WRNN01010k1K5A','WRNN01010k2K5A'],
                     ['WRNN01010k1K5A','WRNN01010k2K5A','W01060PS2Nk1K2A','W01060PS2Nk2K2A'],['WRNN01010k1K5A','WRNN01010k2K5A','W01060PS2Nk1K2A','W01060PS2Nk2K2A']]
        list_name = ['R500ALM2SP60','R500BSM2SP60']+['R0600ALM2SP60','R0600BSM2SP60']
        
        list_spread_ranges = [{'sp':[0]+[round(10*i)/10 for i in np.linspace(.5,5,num=46)],
                               'th':[(.5, .55)]+[(0.5, 0.58), (0.5, 0.58), (0.5, 0.58), (0.5, 0.58), (0.54, 0.58), (0.55, 0.58), (0.58, 0.58), (0.58, 0.58), (0.55, 0.59), (0.54, 0.6), (0.54, 0.6), 
                                     (0.54, 0.6), (0.55, 0.6), (0.58, 0.6), (0.55, 0.61), (0.58, 0.61), (0.58, 0.61), (0.65, 0.6), (0.65, 0.6), (0.65, 0.6), (0.65, 0.6), (0.65, 0.6), 
                                     (0.65, 0.6), (0.65, 0.6), (0.65, 0.6), (0.64, 0.61), (0.65, 0.61), (0.65, 0.61), (0.67, 0.61), (0.67, 0.61), (0.69, 0.61), (0.69, 0.61), (0.69, 0.61), 
                                     (0.71, 0.61), (0.71, 0.61), (0.71, 0.61), (0.65, 0.63), (0.73, 0.61), (0.75, 0.61), (0.75, 0.61), (0.75, 0.61), (0.75, 0.61), (0.75, 0.61), (0.75, 0.61), 
                                     (0.75, 0.61), (0.75, 0.61)],
                               'mar':[(0,0.0)]+[(0,0.02) for _ in range(46)],
                               'dir':'ASKS'},
                              {'sp':[0]+[round(10*i)/10 for i in np.linspace(.5,5,num=46)],
                               'th':[(0.51, 0.55)]+[(0.54, 0.57), (0.51, 0.58), (0.56, 0.57), (0.54, 0.58), (0.56, 0.58), (0.58, 0.58), (0.59, 0.58), (0.61, 0.58), (0.62, 0.58), (0.66, 0.57), (0.65, 0.58), 
                                     (0.66, 0.58), (0.66, 0.58), (0.67, 0.58), (0.67, 0.58), (0.67, 0.58), (0.68, 0.58), (0.68, 0.58), (0.71, 0.58), (0.71, 0.58), (0.71, 0.58), (0.71, 0.58), 
                                     (0.71, 0.58), (0.71, 0.58), (0.71, 0.58), (0.71, 0.58), (0.71, 0.58), (0.71, 0.58), (0.71, 0.58), (0.71, 0.58), (0.71, 0.58), (0.71, 0.58), (0.71, 0.58), 
                                     (0.72, 0.58), (0.72, 0.58), (0.72, 0.58), (0.73, 0.58), (0.74, 0.58), (0.74, 0.58), (0.74, 0.58), (0.74, 0.58), (0.74, 0.58), (0.74, 0.58), (0.74, 0.58), 
                                     (0.75, 0.58), (0.75, 0.58)],
                               'mar':[(0,0.0)]+[(0,0.02) for _ in range(46)],
                               'dir':'BIDS'}]+\
                            [{'sp':[0]+[round(10*i)/10 for i in np.linspace(.5,5,num=46)],
                              'th': [(0.5, 0.55)]+[(0.57, 0.58), (0.51, 0.59), (0.51, 0.6), (0.51, 0.6), (0.51, 0.6), (0.51, 0.6), (0.51, 0.6), (0.56, 0.6), (0.56, 0.6), (0.56, 0.6), (0.56, 0.6), (0.57, 0.6), (0.58, 0.6), 
                                       (0.6, 0.6), (0.6, 0.6), (0.62, 0.6), (0.59, 0.61), (0.6, 0.61), (0.62, 0.61), (0.62, 0.61), (0.62, 0.61), (0.64, 0.61), (0.67, 0.61), (0.67, 0.61), (0.67, 0.61), (0.67, 0.61), 
                                       (0.67, 0.61), (0.7, 0.61), (0.7, 0.61), (0.7, 0.61), (0.67, 0.62), (0.69, 0.62), (0.7, 0.62), (0.74, 0.61), (0.74, 0.61), (0.7, 0.63), (0.68, 0.64), (0.68, 0.64), (0.69, 0.64), 
                                       (0.69, 0.64), (0.7, 0.64), (0.7, 0.64), (0.7, 0.64), (0.7, 0.64), (0.77, 0.63), (0.77, 0.63)],
                               'mar':[(0,0.0)]+[(0,0.02) for _ in range(46)],
                               'dir':'ASKS'},
                              {'sp':[0]+[round(10*i)/10 for i in np.linspace(.5,5,num=46)],
                               'th': [(0.5, 0.55)]+[(0.52, 0.62), (0.53, 0.62), (0.64, 0.61), (0.64, 0.61), (0.65, 0.61), (0.66, 0.61), (0.67, 0.61), (0.68, 0.61), (0.63, 0.62), (0.63, 0.62), (0.68, 0.62), (0.68, 0.62), 
                                      (0.68, 0.62), (0.68, 0.62), (0.68, 0.62), (0.75, 0.61), (0.75, 0.61), (0.75, 0.61), (0.75, 0.61), (0.78, 0.59), (0.78, 0.59), (0.78, 0.59), (0.78, 0.59), (0.78, 0.59), (0.79, 0.59), 
                                      (0.79, 0.59), (0.79, 0.59), (0.79, 0.59), (0.79, 0.59), (0.79, 0.59), (0.79, 0.61), (0.79, 0.61), (0.79, 0.61), (0.79, 0.61), (0.81, 0.61), (0.81, 0.61), (0.81, 0.61), (0.81, 0.61), 
                                      (0.81, 0.61), (0.81, 0.61), (0.81, 0.61), (0.81, 0.61), (0.81, 0.61), (0.81, 0.61), (0.81, 0.61), (0.81, 0.61)],
                               'mar':[(0,0.0)]+[(0,0.02) for _ in range(46)],
                               'dir':'BIDS'}]
        
        
        mWs = [500, 500, 600, 600]
        nExSs = [5000, 5000, 6000, 6000]
        outputGains = [1, 1]+[1, 1]
        #lBs = [1300]
        list_feats_from_bids = [False, True]+[False, True]
        combine_ts = {'if_combine':False,'params_combine':[{'alg':'mean'}]}
        
        config_names = ['config_name'+str(i) for i in range(numberNetworks)]
        stacked = [4, 4]+[4, 4]
        max_opened_positions = 30
        
        entries_list = [[{'config_name':config_names[i],'IDweights':IDweights[i][st],
                         'results_from':list_spread_ranges[i]['dir'],
                         'feats_from_bids':list_feats_from_bids[i],
                       'size_output_layer':8,'n_bits_outputs':[1,2,5],'combine_ts':combine_ts,
                       'outputGain':outputGains[i],'movingWindow':mWs[i],
                       'nEventsPerStat':nExSs[i]}  for st in range(stacked[i])] for i in range(numberNetworks)]
        config_list = [[configuration(e, save=False) for e in entries] for entries in entries_list]
        IDepoch = [[5,2,14,14], [5,3,14,14], [14,14,19,19], [14,14,19,19]]
        netNames = ['500AL', '500BS','600AL', '600BS']
        list_t_indexs = [[0], [0], [0], [0]]
        list_inv_out = [True for i in range(numberNetworks)]
        #['B','B','B','A']# {B: from bid symbols, A: from ask symbols}
        list_entry_strategy = ['spread_ranges' for i in range(numberNetworks)] #'fixed_thr','gre' or 'spread_ranges', 'gre_v2'
        # {'S': short, 'L':long, 'C':combine} TODO: combine not supported yet
        #list_spread_ranges = [{'sp': [2], 'th': [(0.7, 0.7)],'dir':'C'}]
        list_priorities = [[0], [0], [0], [0]]#[[3],[2],[1],[0]]
        phase_shifts = [1 for i in range(numberNetworks)]
        
        
        list_lim_groi_ext = [-10 for i in range(numberNetworks)]
        list_thr_sl = [1000 for i in range(numberNetworks)]#50
        list_thr_tp = [1000 for i in range(numberNetworks)]
        list_max_lots_per_pos = [.15 for i in range(numberNetworks)]
#        list_invest_strategy = [{'name':'scale',
#                                 'steps':[.02, .04, .08, .16, .32, .5, .6, .7, .8, .9, 1.0],
#                                 'thrs':[2020, 2060, 2140, 4300, 8620, 11260, 12260, 14000, 16030, 18030, 20100]} for i in range(numberNetworks)]
        delays = [0 for i in range(numberNetworks)]
        list_w_str = ['55' for i in range(numberNetworks)]
        list_weights = [[.5,.5] for i in range(numberNetworks)]
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
        
        list_flexible_lot_ratio = [False for i in range(numberNetworks)]
        list_if_dir_change_close = [False for i in range(numberNetworks)]
        list_if_dir_change_extend = [False for i in range(numberNetworks)]
        
#        model_dict = {'size_hidden_layer':[100 for i in range(numberNetworks)],
#                      'L':[3 for i in range(numberNetworks)],
#                      'size_output_layer':[5 for i in range(numberNetworks)],
#                      'outputGain':[1 for i in range(numberNetworks)]}
        
        config = {'config_name':config_name,
                  'config_list':config_list,
                  'numberNetworks':numberNetworks,
                  'IDweights':IDweights,
                  'IDresults':IDresults,
                  #'lIDs':lIDs,
                  'list_name':list_name,
                  'IDepoch':IDepoch,
                  'netNames':netNames,
                  'list_t_indexs':list_t_indexs,
                  'list_inv_out':list_inv_out,
                  'list_entry_strategy':list_entry_strategy,
                  'list_spread_ranges':list_spread_ranges,
                  'list_priorities':list_priorities,
                  'max_opened_positions':max_opened_positions,
                  #'list_feats_from':list_feats_from,
                  'phase_shifts':phase_shifts,
                  'list_thr_sl':list_thr_sl,
                  'list_thr_tp':list_thr_tp,
                  'delays':delays,
                  'mWs':mWs,
                  'nExSs':nExSs,
                  #'lBs':lBs,
                  'list_lim_groi_ext':list_lim_groi_ext,
                  'list_w_str':list_w_str,
#                  'model_dict':model_dict,
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
#                  'list_invest_strategy':list_invest_strategy,
                  'list_flexible_lot_ratio':list_flexible_lot_ratio,
                  'list_if_dir_change_close':list_if_dir_change_close,
                  'list_if_dir_change_extend':list_if_dir_change_extend
                }
        if not os.path.exists(local_vars.config_directory):
            os.mkdir(local_vars.config_directory)
        pickle.dump( config, open( config_filename, "wb" ))
        #print(config)
        print("Config file "+config_filename+" saved")
        return None
    else:
        config = pickle.load( open( config_filename, "rb" ))
        if len(ins)>0:
            print("WARNING! Arguments not taken into consideration")
        print("Config file "+config_filename+" exists. Loaded from disk")
        return config

def change_spread_ranges(config, margin=(0.0,0.06)):
    for c in range(len(config['list_spread_ranges'])):
        for m in range(len(config['list_spread_ranges'][c]['mar'])):
            config['list_spread_ranges'][c]['mar'][m]=margin
    modify_config(config['config_name'],'list_spread_ranges',config['list_spread_ranges'])
        
if __name__=='__main__':
    #configuration()
    pass