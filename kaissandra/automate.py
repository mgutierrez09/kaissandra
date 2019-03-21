# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 17:25:12 2018

@author: mgutierrez
"""

#import time
import pandas as pd
import h5py
from multiprocessing import Process
import pickle

from kaissandra.trainRNN import train_RNN
from kaissandra.testRNN import test_RNN
from kaissandra.config import retrieve_config, configuration, write_log
from kaissandra.features import get_features
from kaissandra.preprocessing import K_fold
from kaissandra.local_config import local_vars
from kaissandra.models import RNN
from kaissandra.results2 import merge_results

def run_train_test(config, its, if_train, if_test, if_get_features, run_in_paralell):
    """
    
    """
    if if_get_features:
        get_features(config)
    # loop over iteratuibs
    for it in range(its):
        print("Iteration {0:d} of {1:d}".format(it,its-1))
        if if_train:
            print("IDweights: "+config['IDweights'])
        if if_test:
            print("IDresults: "+config['IDresults'])
        # launch train
        if if_train:
            # here we should check if the HDF5 file is used
            train_RNN(config)
        # launch test
        if if_test and not run_in_paralell:
            test_RNN(config)
        elif if_test and run_in_paralell:
            disp = Process(target=test_RNN, args=[config])
            disp.start()
            #time.sleep(1)

def automate(*ins):
    # init config
    if_get_features = False
    if_train = True
    if_test = True
    
    # retrieve list of config file names to run automatelly
    if len(ins)>0:
        configs = ins[0]
    else:
        configs = ['C0317INVO']
    if len(ins)>=2:
        its = ins[1]
    else:
        its = 100
    if len(ins)>=3:
        run_in_paralell = ins[2]
    else:
        run_in_paralell = False
    configs_list = []
     # load configuration files
    for config_name in configs:
        configs_list.append(retrieve_config(config_name))
        # run train/test
    for config in configs_list:
        # set automation-specific config fields
       config['num_epochs'] = 1
       config['startFrom'] = -1
       config['endAt'] = -1
       run_train_test(config, its, if_train, if_test, if_get_features, run_in_paralell)
            # parallelize
    #        disp = Process(target=run_train_test, args=[config, its, if_train, if_test, if_get_features])
    #        disp.start()
    #        time.sleep(1)

def automate_RNN(rootname_config, entries={}, K=5, tAt='TrTe', IDrs=[], build_IDrs=False,
                 its=15, sufix='', IDr_merged='',k_init=0, k_end=-1, log=''):
    """  """
    if 'feats_from_bids' in entries:
        feats_from_bids = entries['feats_from_bids']
    else:
        feats_from_bids = False
    if 'results_from' in entries:
        results_from = entries['results_from']
    else:
        results_from = 'COMB'
    if 'startFrom' in entries:
        startFrom = entries['startFrom']
    else:
        startFrom = -1
    if 'endAt' in entries:
        endAt = entries['endAt']
    else:
        endAt = -1
#        size_hidden_layer = 3
    if feats_from_bids==False:
        extW = 'A'
        extR = 'A'
    else:
        extW = 'B'
        extR = 'B'
    if results_from=='COMB':
        extR = extR+'C'
    elif results_from=='ASKS':
        extR = extR+'L'
    elif results_from=='BIDS':
        extR = extR+'S'
    else:
        raise ValueError('results_from not recognized')
            
    if IDr_merged=='':
        IDr_merged = rootname_config+'K'+str(K)+extR
    if k_end==-1:
        k_end=K
    for fold_idx in range(k_init,k_end):
        mess = "k "+str(fold_idx+1)+" of "+str(K)
        print(mess)
        if len(log)>0:
            write_log(mess)
        basename = rootname_config+'k'+str(fold_idx+1)+'K'+str(K)
        entries['config_name'] = 'C'+basename
        entries['IDweights'] = 'W'+basename+extW
        entries['IDresults'] = 'R'+basename+extR+sufix
        entries['IO_results_name'] = 'R'+basename+extR
        config = configuration(entries)
        IDresults = config['IDresults']
        if build_IDrs:
            IDrs.append(IDresults)
        
        
        dirfilename_tr, dirfilename_te, IO_results_name = K_fold(folds=K, \
                                                                 fold_idx=fold_idx, \
                                                                 config=config, 
                                                                 log=log)
        
        #dirfilename_tr = local_vars.IO_directory+tag+IDweights+ext
        f_IOtr = h5py.File(dirfilename_tr,'r')
        if 'Tr' in tAt:
            Ytr = f_IOtr['Y'][:]
            Xtr = f_IOtr['X'][:]
            #Rtr = f_IOtr['R'][:]
        #dirfilename_te = local_vars.IO_directory+tag+IDresults+ext
        f_IOte = h5py.File(dirfilename_te,'r')
        if 'Te' in tAt:
            Yte = f_IOte['Y'][:]
            Xte = f_IOte['X'][:]
        #Rte = f_IOte['R'][:]
        #IO_results_name = local_vars.IO_directory+'DTA_'+basename+'A.p'
        DTA = pickle.load( open( IO_results_name, "rb" ))
        
        
        epochs_per_it = 1
        for i in range(its):
            mess = "Iteration "+str(i)+" of "+str(its-1)
            print(mess)
            if len(log)>0:
                write_log(mess)
            if tAt=='TrTe':
                RNN(config).fit(Xtr, Ytr, num_epochs=epochs_per_it,log=log).\
                    cv(Xte, Yte, DTA, IDresults=IDresults, config=config, log=log)
            elif tAt=='Te':
                RNN(config).cv(Xte, Yte, DTA, IDresults=IDresults, \
                   startFrom=startFrom, endAt=endAt, config=config, log=log)
            elif tAt=='Tr':
                RNN(config).fit(Xtr,Ytr,num_epochs=epochs_per_it)
            else:
                raise ValueError('tAt value not known')
        
        f_IOtr.close()
        f_IOte.close()
    merge_results(IDrs, IDr_merged)
    
if __name__=='__main__':
    pass
    #automate(['C3012INVO'])