# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 17:16:58 2019

@author: mgutierrez
"""

# edit config without saving for test purposes
from kaissandra.testRNN import test_RNN
from kaissandra.config import *
config=retrieve_config('C0350')
#config['config_name'] = 'C0520BSNFR'
#config['IDweights'] = '000541A'
config['IDresults'] = '100350TEST'
#config['results_from'] = 'ASKS'
config['commonY'] = 3
config['save_journal'] = False
config['startFrom'] = 0
config['endAt']= 0
resolution = 10
config['resolution'] = resolution
config['thresholds_mc'] = [.5+i/resolution for i in range(int(resolution/2))]
config['thresholds_md'] = [.5+i/resolution for i in range(int(resolution/2))]

#config['cost_name'] = '000318TI02'

test_RNN(config)