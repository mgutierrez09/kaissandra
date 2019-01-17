# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 17:25:12 2018

@author: mgutierrez
"""

import time
from multiprocessing import Process
from trainRNN import train_RNN
from testRNN import test_RNN
from config import retrieve_config
from features import get_features

def run_train_test(config, its, if_train, if_test, if_get_features):
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
        if if_test:
            test_RNN(config)

def automate(*ins):
    # init config
    if_get_features = False
    if_train = True
    if_test = True
    its = 20
    # retrieve list of config file names to run automatelly
    if len(ins)>0:
        configs = ins[0]
    else:
        configs = ['C0310INVO']
        
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
       run_train_test(config, its, if_train, if_test, if_get_features)
            # parallelize
    #        disp = Process(target=run_train_test, args=[config, its, if_train, if_test, if_get_features])
    #        disp.start()
    #        time.sleep(1)

if __name__=='__main__':
    automate(['C3012INVO'])