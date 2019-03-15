# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 14:20:05 2019

@author: mgutierrez
"""
import h5py
import pickle
import matplotlib.pyplot as plt
from kaissandra.local_config import local_vars
from kaissandra.preprocessing import build_IO_wrapper
from kaissandra.models import DNN, RNN, retrieve_costs
from kaissandra.config import retrieve_config, configuration

entries={'config_name':'CRNN00003','IDweights':'WRNN00003A', 'IDresults':'RRNN00003AC',
         'movingWindow':1000,'nEventsPerStat':10000,'feats_from_bids':False,
         'size_output_layer':3,'outputGain':1, 'results_from':'COMB'}
config = configuration(entries)
#config = retrieve_config('CRNN00001')
#config['config_name'] = 'CRNN00010RANDMB'
#config['IDweights'] = 'WRNN00001A'
#config['IDresults'] = 'RRNN00001A'
tag = 'IOE_'
ext = '.hdf5'
tAt = 'TrTe'# train and test

IDweights = config['IDweights']
IDresults = config['IDresults']

dirfilename_tr = build_IO_wrapper('tr',config)
dirfilename_te = build_IO_wrapper('te',config)

#dirfilename_tr = local_vars.IO_directory+tag+IDweights+ext
f_IOtr = h5py.File(dirfilename_tr,'r')
Ytr = f_IOtr['Y'][:]
Xtr = f_IOtr['X'][:]
#Rtr = f_IOtr['R'][:]

#dirfilename_te = local_vars.IO_directory+tag+IDresults+ext
f_IOte = h5py.File(dirfilename_te,'r')
Yte = f_IOte['Y'][:]
Xte = f_IOte['X'][:]
#Rte = f_IOte['R'][:]
IO_results_name = local_vars.IO_directory+'DTA_'+IDresults+'.p'
DTA = pickle.load( open( IO_results_name, "rb" ))

its = 100
epochs_per_it = 1
for i in range(its):
    print("Iteration "+str(i)+" of "+str(its-1))
    if tAt=='TrTe':
        RNN(config).fit(Xtr,Ytr,num_epochs=epochs_per_it).cv(Xte,Yte,DTA,IDresults=IDresults,config=config)
    elif tAt=='Te':
        RNN(config).cv(Xte,Yte,DTA,IDresults=IDresults,config=config)
    elif tAt=='Tr':
        RNN(config).fit(Xtr,Ytr,num_epochs=epochs_per_it)
    else:
        raise ValueError('tAt value not known')

f_IOtr.close()
f_IOte.close()