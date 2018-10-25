# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 17:25:12 2018

@author: mgutierrez
"""

from trainRNN import train_RNN
from testRNN import test_RNN
from config import configuration

def run_train_test(config):
    """
    
    """
#    config['num_epochs'] = 1
#    config['assets'] = [2]
#    config['startFrom'] = -1
#    config['endAt'] = -1
    its = 1
    # loop over iteratuibs
    for it in range(its):
        print("Iteration {0:d} of {1:d}".format(it,its-1))
        print("IDweights: "+config['IDweights'])
        print("IDresults: "+config['IDresults'])
        # launch train
        train_RNN(config)
        # launch test
        test_RNN(config)

if __name__=='__main__':
    # init config
    configs = ['C0000','C0001','C0002']
    configs_list = []
    # load configuration files
    for config_name in configs:
        configs_list.append(configuration(config_name))
    # run train/test
    for config in configs_list:
        run_train_test(config)