# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 08:12:49 2019

@author: mgutierrez
"""
import numpy as np
import h5py
import pickle
import matplotlib.pyplot as plt
from kaissandra.local_config import local_vars
from kaissandra.models import  DNN, RNN
from kaissandra.preprocessing import build_IO_wrapper
from kaissandra.config import retrieve_config

def sigmoid(x):
    """ """
    return (1/(1+np.exp(-x)))
tag = 'IOE_'
ext = '.hdf5'
tAt = 'TrTe'
evaluate_RNN = False
config = retrieve_config('CRNN00010')
config['IDweights'] = 'WRNN00010RANDMB'
config['IDresults'] = 'RRNN00010RANDMB'

IDweights = config['IDweights']
IDresults = config['IDresults']

dirfilename_tr = build_IO_wrapper('tr',config)
dirfilename_te = build_IO_wrapper('te',config)
f_IOtr = h5py.File(dirfilename_tr,'r')
Ytr = f_IOtr['Y']
Xtr = f_IOtr['X']
Rtr = f_IOtr['R']
f_IOte = h5py.File(dirfilename_te,'r')
Yte = f_IOte['Y']
Xte = f_IOte['X']
Rte = f_IOte['R']
IO_results_name = local_vars.IO_directory+'DTA_'+IDresults+'.p'
DTA = pickle.load( open( IO_results_name, "rb" ))
# evaluate RNN and save output
if evaluate_RNN:
    RNN(config).cv(Xtr, Ytr, config=config, save_output=True, save_cost=False,
                   if_get_results=False, startFrom=6, endAt=6, tag='MT_'+IDweights)\
               .cv(Xte, Yte, config=config, save_output=True, save_cost=False,
                   if_get_results=False, startFrom=6, endAt=6, tag='MT_'+IDresults)

# run meta model
if 'seq_len' in config:
    seq_len = config['seq_len']
else:
    seq_len = int((config['seq_len']-config['nEventsPerStat'])/config['movingWindow']+1)
size_output_layer = config['size_output_layer']

params = {'size_input_layer':int(seq_len*size_output_layer),'size_hidden_layers':[512, 64, 16],'IDweights':'WDNN00011',
        'IDresults':'RDNN00011','loss_func':'cross_entropy','out_act_func':'sigmoid',
        'act_funcs':['sigmoid','sigmoid','sigmoid'],'lR0':0.0001,'num_epochs':100,'epoch_predict':99,
        'dateTest':config['dateTest'],'save_journal':config['save_journal']}

#dirfilename_tr = local_vars.IO_directory+tag+IDweights+ext

RNN_Otr = h5py.File(local_vars.IO_directory+'MT_'+IDweights,'r')['output']

#dirfilename_te = local_vars.IO_directory+tag+IDresults+ext

RNN_Ote = h5py.File(local_vars.IO_directory+'MT_'+IDresults,'r')['output']

its = 100
epochs_per_it = 1
for i in range(its):
    print("Iteration "+str(i)+" of "+str(its-1))
    if tAt=='TrTe':
        DNN(params).fit(RNN_Otr[:].reshape([RNN_Otr.shape[0],-1]), sigmoid(Rtr[:]), num_epochs=epochs_per_it)\
           .cv(RNN_Ote[:].reshape([RNN_Ote.shape[0],-1]), sigmoid(Rte[:]), IDresults=params['IDresults'],DTA=DTA,config=params)
    elif tAt=='Te':
        DNN(params).fit(RNN_Otr[:].reshape([RNN_Otr.shape[0],-1]), sigmoid(Rtr[:]), num_epochs=epochs_per_it)
    elif tAt=='Tr':
        DNN(params).cv(RNN_Ote[:].reshape([RNN_Ote.shape[0],-1]), sigmoid(Rte[:]), IDresults=params['IDresults'],DTA=DTA,config=params)
    else:
        raise ValueError('tAt value not known')

f_IOtr.close()
f_IOte.close()
#meta = (self, X, Y, num_epochs=100, keep_prob_dropout=1.0)(Xte,Yte,DTA,IDresults=IDresults,config=config)
#DNN(params).fit(RNN_Otr[:].reshape([RNN_Otr.shape[0],-1]), sigmoid(Rtr[:]), num_epochs=100)\
#           .cv(RNN_Ote[:].reshape([RNN_Ote.shape[0],-1]), sigmoid(Rte[:]), IDresults=params['IDresults'],DTA=DTA,config=params)

#plt.plot(retrieve_costs(tOt='tr',params=params))
#plt.plot(retrieve_costs(tOt='te',params=params))
#meta.cv(RNN_O[:].reshape([RNN_O.shape[0],-1]), np.sort(np.tanh(R[:]/2)))


#input_train=np.random.randint(2,size=(1000,2))
#output_train=np.zeros((input_train.shape[0],1))
#output_train[:,0]=input_train[:,0]^input_train[:,1]
#input_cv=np.random.randint(2,size=(1000,2))
#output_cv=np.zeros((input_train.shape[0],1))
#output_cv[:,0]=input_cv[:,0]^input_cv[:,1]
#
#input_traint=-(-1)**input_train
#input_cvt=-(-1)**input_cv
#output_traint=np.zeros((input_train.shape[0],1))
#output_traint[:,0]=-(-1)**(input_train[:,0]^input_train[:,1])
#output_cvt=np.zeros((input_train.shape[0],1))
#output_cvt[:,0]=-(-1)**(input_cv[:,0]^input_cv[:,1])
#
#
#paramss={'size_input_layer':2,'size_hidden_layers':[10],'IDweights':'0TESTXORTANH',
#        'IDresults':'1TESTXORTANH','loss_func':'exponential','out_act_func':'sigmoid',
#        'act_funcs':['sigmoid'],'learning_rate':0.01,'num_epochs':500,'epoch_predict':499}
#xormodel=DNN(paramss)
#xormodel.fit(input_train,output_train,params=paramss).cv(input_cv,output_cvt,params=paramss).init_session(params=paramss).predict(np.array([[1,0],[0,0],[1,1],[0,1]]))
##xormodel.cv(input_cv,output_cvt,params=paramss)
##xormodel.init_session(params=paramss)
##xormodel.predict(np.array([[1,0],[0,0],[1,1],[0,1]]))
#
#paramst={'size_input_layer':2,'size_hidden_layers':[],'IDweights':'0TESTXORTANHNOHID',
#        'IDresults':'1TESTXORTANHNOHID','loss_func':'exponential','out_act_func':'tanh',
#        'act_funcs':[],'learning_rate':0.01,'num_epochs':500,'epoch_predict':499}
#xormodelt=DNN(paramst)
#xormodelt.fit(input_traint,output_traint,paramst).cv(input_cvt,output_cvt,params=paramst).init_session(params=paramst).predict(np.array([[1,-1],[-1,-1],[1,1],[-1,1]]))
##xormodelt.cv(input_cvt,output_cvt,params=paramst)
##xormodelt.init_session(params=paramst)
##xormodelt.predict(np.array([[1,-1],[-1,-1],[1,1],[-1,1]]))#
#
#plt.plot(retrieve_costs(tOt='tr',params=paramss))
#plt.plot(retrieve_costs(tOt='te',params=paramss))
#
#plt.plot(retrieve_costs(tOt='tr',params=paramst))
#plt.plot(retrieve_costs(tOt='te',params=paramst))
