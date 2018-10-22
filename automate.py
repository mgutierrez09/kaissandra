# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 17:25:12 2018

@author: mgutierrez
"""

from trainRNN import train_RNN
from testRNN import test_RNN
from config import configuration

# init config
config = configuration()
# epochs per iteration
config['num_epochs'] = 1
its = 20
config['if_build_IO'] = True
# loop over iteratuibs
for it in range(its):
    print("Iteration {0:d}".fomat(it))
    print("IDweights: "+config['IDweights'])
    print("IDresults: "+config['IDresults'])
    # dont build IO if it's not the first iteration
    if it>0:
        config['if_build_IO'] = False
    # launch train
    train_RNN()
    # init test starting index
    config['startFrom'] = it*config['num_epochs']
    # launch test
    test_RNN()