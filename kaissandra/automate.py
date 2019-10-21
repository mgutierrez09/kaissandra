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
import numpy as np

from kaissandra.trainRNN import train_RNN
from kaissandra.testRNN import test_RNN
from kaissandra.config import retrieve_config, configuration, write_log
from kaissandra.features import get_features
from kaissandra.preprocessing import build_datasets, build_datasets_modular, build_datasets_modular_oneNet
from kaissandra.local_config import local_vars
from kaissandra.models import RNN#, LGBM
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

def wrapper_bild_datasets_Kfold(rootname_config, entries={}, K=5, build_IDrs=False,
                 sufix='',k_init=0, k_end=-1, log='', 
                 modular=False, sufix_io='', basename_IO='',sufix_re='', oneNet=False, extW=''):
    """  """
    if 'feats_from_bids' in entries:
        feats_from_bids = entries['feats_from_bids']
    else:
        feats_from_bids = False
    if 'results_from' in entries:
        results_from = entries['results_from']
    else:
        results_from = 'COMB'
    if 'build_asset_relations' in entries:
        build_asset_relations = entries['build_asset_relations']
    else:
        build_asset_relations = ['direct','inverse']
    
    if 'feats_from_all' in entries:
        feats_from_all = entries['feats_from_all']
    else:
        feats_from_all = False
    
    ext_rel_tr = ''
    if feats_from_all:
        if feats_from_bids==False:
            ext_rel_tr = ext_rel_tr+'B'
        else:
            ext_rel_tr = ext_rel_tr+'A'
    if 'direct' in build_asset_relations:
        ext_rel_tr = ext_rel_tr+'D'
    if 'inverse' in build_asset_relations:
        ext_rel_tr = ext_rel_tr+'I'
    if 'asset_relation' in entries:
        asset_relation = entries['asset_relation']
    else:
        asset_relation = 'direct'
    if asset_relation=='direct':
        ext_rel_cv = 'D'
    elif asset_relation=='inverse':
        ext_rel_cv = 'I'
    
#        size_hidden_layer = 3
    
    if feats_from_bids==False:
        extWtemp = 'A'
        extR = 'A'
        if feats_from_all:
            extWtemp+='B'
    else:
        extWtemp = 'B'
        extR = 'B'
        if feats_from_all:
            extWtemp+='A'
        #print("\n\n\n\nWARNING! extW = A not B\n\n\n\n")
    
    if len(extW)>0:
        print("\n\nWARNING! extW fixed manuely to "+str(extW)+"\n\n")
    else:
        extW = extWtemp
    if results_from=='COMB':
        extR = extR+'C'
    elif results_from=='ASKS':
        extR = extR+'L'
    elif results_from=='BIDS':
        extR = extR+'S'
    else:
        raise ValueError('results_from not recognized')
            
    if k_end==-1:
        k_end=K
    
    configs = []
    dirfilename_trs = []
    dirfilename_tes = []
    IO_results_names = []
    for fold_idx in range(k_init,k_end):
        mess = "k "+str(fold_idx+1)+" of "+str(K)
        print(mess)
        if len(log)>0:
            write_log(mess)
        basename = rootname_config+'k'+str(fold_idx+1)+'K'+str(K)
        entries['config_name'] = 'C'+basename
        entries['IDweights'] = 'W'+basename+extW+sufix
        entries['IDresults'] = 'R'+basename+extR+ext_rel_cv+sufix+sufix_re
        print('IDresults:')
        print(entries['IDresults'])
        if len(basename_IO)==0:
            entries['IO_results_name'] = 'R'+basename+extR+ext_rel_cv+sufix_io
        else:
            entries['IO_results_name'] = 'R'+basename_IO+extR+ext_rel_cv+sufix_io
        
        config = configuration(entries)
        if len(basename_IO)==0:
            config['IO_tr_name'] = basename+extR+ext_rel_tr+sufix_io
            config['IO_cv_name'] = basename+extR+ext_rel_cv+sufix_io
        else:
            config['IO_tr_name'] = basename_IO+extR+ext_rel_tr+sufix_io
            config['IO_cv_name'] = basename_IO+extR+ext_rel_cv+sufix_io
        if 'build_XY_mode' in entries:
            build_XY_mode = entries['build_XY_mode']
        else:
            build_XY_mode = 'K_fold'#'datebased'
        if build_XY_mode == 'datebased':
            assert(K==2 and k_init==0 and k_end==1)
        
            
        print("config['from_stats_file']")
        print(config['from_stats_file'])
        if not modular:
            dirfilename_tr, dirfilename_te, IO_results_name = build_datasets(folds=K, \
                                                                     fold_idx=fold_idx, \
                                                                     config=config, 
                                                                     log=log)
        elif not oneNet:
            dirfilename_tr, dirfilename_te, IO_results_name = build_datasets_modular(folds=K, \
                                                                     fold_idx=fold_idx, \
                                                                     config=config, 
                                                                     log=log)
        elif oneNet:
            dirfilename_tr, dirfilename_te, IO_results_name = build_datasets_modular_oneNet(folds=K, \
                                                                     fold_idx=fold_idx, \
                                                                     config=config, 
                                                                     log=log)
        configs.append(config)
        dirfilename_trs.append(dirfilename_tr)
        dirfilename_tes.append(dirfilename_te)
        IO_results_names.append(IO_results_name)
        
    return configs, dirfilename_trs, dirfilename_tes, IO_results_names

def automate_Kfold(rootname_config, entries={}, K=5, tAt='TrTe', IDrs=[], build_IDrs=False,
                 its=15, sufix='', IDr_merged='',k_init=0, k_end=-1, log='', just_build=False,
                 if_merge_results=False, modular=False, sufix_io='', basename_IO='', sufix_re='', 
                 oneNet=False, extW=''):
    """  """

    configs, dirfilename_trs, dirfilename_tes, IO_results_names = wrapper_bild_datasets_Kfold\
        (rootname_config, entries=entries, K=K, sufix=sufix, k_init=k_init, k_end=k_end, log=log,
         modular=modular, sufix_io=sufix_io, basename_IO=basename_IO, sufix_re=sufix_re, oneNet=oneNet, extW=extW)
    if 'startFrom' in entries:
        startFrom = entries['startFrom']
    else:
        startFrom = -1
    if 'endAt' in entries:
        endAt = entries['endAt']
    else:
        endAt = -1
    
    count = 0
    for fold_idx in range(k_init,k_end):
        config = configs[count]
        dirfilename_tr = dirfilename_trs[count]
        dirfilename_te = dirfilename_tes[count]
        IO_results_name = IO_results_names[count]
        IDresults = config['IDresults']
        if not just_build:
            f_IOtr = h5py.File(dirfilename_tr,'r')
            if 'Tr' in tAt:
                Ytr = f_IOtr['Y']
                Xtr = f_IOtr['X']
            f_IOte = h5py.File(dirfilename_te,'r')
            if 'Te' in tAt:
                Yte = f_IOte['Y']
                Xte = f_IOte['X']
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
                    RNN(config).fit(Xtr,Ytr,num_epochs=epochs_per_it, log=log)
                else:
                    raise ValueError('tAt value not known')
            
            f_IOtr.close()
            f_IOte.close()
        
        if build_IDrs:
            IDrs.append(IDresults)
        count += 1
    if not just_build and if_merge_results:
        merge_results(IDrs, IDr_merged)
        
def cross_entropy_loss(Y, Y_tilde):
    """  """
    return np.sum(Y*-np.log(Y_tilde)+(1-Y)*-np.log(1-Y_tilde))/Y.shape[0]

def extract_wrong_predictions(rootname_config, epoch, entries={}, K=5, 
                 sufix='' ,k_init=0, k_end=-1, log='', 
                 modular=False, sufix_io='', basename_IO=''):
    """  """
    import os
    configs, dirfilename_trs, dirfilename_tes, IO_results_names = wrapper_bild_datasets_Kfold\
        (rootname_config, entries=entries, K=K, sufix=sufix, k_init=k_init, k_end=k_end, log=log,
         modular=modular, sufix_io=sufix_io, basename_IO=basename_IO)
    
    count = 0
    for fold_idx in range(k_init,k_end):
        
        config = configs[count]
        t_indexes = config['t_indexes']
        IDresults = config['IDresults']
        IDweights = config['IDweights']
        
        dirfilename_tr = dirfilename_trs[count]
        dirfilename_te = dirfilename_tes[count]
        IO_results_name = IO_results_names[count]
        # Get IO sets
        f_IOtr = h5py.File(dirfilename_tr,'r')
        Ytr = f_IOtr['Y']
        Xtr = f_IOtr['X']
        f_IOte = h5py.File(dirfilename_te,'r')
        Yte = f_IOte['Y']
        Xte = f_IOte['X']
        DTA = pickle.load( open( IO_results_name, "rb" ))
        # get predictions from base model
#        print("Xtr.shape")
#        print(Xtr.shape)
#        print("Ytr.shape")
#        print(Ytr.shape)
        print("Getting Yte_tilde")
        Str_tilde = RNN(config).predict_batch(Xtr, epoch)
#        print("Getting Yte_tilde")
#        Ste_tilde = RNN(config).predict_batch(Xte, epoch)
        
        # extract wrong classified outputs
        short_bets_idx = 1*(np.mean(Str_tilde[:,:,1],1)<.5)
        short_out_idx = 1*(Ytr[:,0,1]<.5)
        errors_idx = np.abs(short_bets_idx-short_out_idx)>0
#        YerrTr = Ytr[errors_idx,:,:]
        # save error vector
        print(errors_idx)
        assert(max(abs(Yte[:,0,0]-Yte[:,-1,0]))==0)
        
        directory = local_vars.RNN_directory+'error/'
        if not os.path.exists(directory):
            os.makedirs(directory)
        pickle.dump(errors_idx, open(directory+ (dirfilename_tr.split('/')[-1]).split('.')[0]+'.p', "wb" ))
    print("DONE")
        
def boosting(rootname_config, epoch, entries={}, K=5, 
                 sufix='' ,k_init=0, k_end=-1, log='', 
                 modular=False, sufix_io='', basename_IO=''):
    """  """
    from kaissandra.results2 import init_results_dir, get_results
    from kaissandra.models import retrieve_costs
    
    configs, dirfilename_trs, dirfilename_tes, IO_results_names = wrapper_bild_datasets_Kfold\
        (rootname_config, entries=entries, K=K, sufix=sufix, k_init=k_init, k_end=k_end, log=log,
         modular=modular, sufix_io=sufix_io, basename_IO=basename_IO)
    
    
    # loop over k-foldings
    count = 0
    for fold_idx in range(k_init,k_end):
        
        config = configs[count]
        t_indexes = config['t_indexes']
        IDresults = config['IDresults']
        IDweights = config['IDweights']
        
        dirfilename_tr = dirfilename_trs[count]
        dirfilename_te = dirfilename_tes[count]
        IO_results_name = IO_results_names[count]
        # Get IO sets
        f_IOtr = h5py.File(dirfilename_tr,'r')
        Ytr = f_IOtr['Y']
        Xtr = f_IOtr['X']
        f_IOte = h5py.File(dirfilename_te,'r')
        Yte = f_IOte['Y']
        Xte = f_IOte['X']
        DTA = pickle.load( open( IO_results_name, "rb" ))
        # get predictions from base model
#        print("Xtr.shape")
#        print(Xtr.shape)
#        print("Ytr.shape")
#        print(Ytr.shape)
        print("Getting Ytr_tilde")
        Ytr_tilde = RNN(config).predict_batch(Xtr, epoch)
        print("Getting Yte_tilde")
        Yte_tilde = RNN(config).predict_batch(Xte, epoch)
#        print("Ytr[:10,:,:]")
#        print(Ytr[:10,:,:])
#        print("Y_tilde")
#        print(Ytr_tilde[:,0,1])
#        print(Ytr_tilde[:,0,2])
#        print(max(Ytr_tilde[:,0,2]-Ytr_tilde[:,0,1]))
        Ytr_tilde = Ytr_tilde[:,0,1]
#        Yte_tilde = Yte_tilde[:,0,1]
#        print("Y_tilde new")
#        print(Ytr_tilde.shape)
        
        costs_dict, costs = retrieve_costs(tOt='tr', IDweights=IDweights)
        Yte_tilde_2 = np.zeros(Yte_tilde.shape)
        Yte_tilde_2[:,:,:] = Yte_tilde[:,:,:]
        # LGB
        for t_index in t_indexes:
            # Get error set
            print("t_index = ",t_index)
#            print("Ytr[:,t_index,1]")
#            print(Ytr[:,t_index,1])
            err_tr = Ytr[:,t_index,1]-Ytr_tilde
#            print("err_tr")
#            print(err_tr)
            print("Fitting model")
            lgmb = LGBM(config).fit(Xtr[:,t_index,:], err_tr)
#            err_tr_tilde = lgmb.model.predict(Xtr[:,t_index,:])
#            print("rmsle: ")
#            print(lgmb.rmsle(err_tr, err_tr_tilde))
#            Ytr_tilde_2 = Ytr_tilde+err_tr_tilde
#            
            loss_tr = cross_entropy_loss(Ytr[:,t_index,1], Ytr_tilde)
#            loss_tr_2 = cross_entropy_loss(Ytr[:,t_index,1], Ytr_tilde_2)
            
            err_te = Yte[:,t_index,1]-Yte_tilde[:,t_index,1]
            print("Cross-validating model")
            err_te_tilde = lgmb.cv(Xte[:,t_index,:], err_te)
#            print("err_te_tilde")
#            print(err_te_tilde)
#            print(max(err_te_tilde))
#            print(min(err_te_tilde))
            Yte_tilde_2[:,t_index,1] = Yte_tilde[:,t_index,1]+err_te_tilde
#            print(Yte_tilde_2[:,t_index,1])
#            print(Yte_tilde[:,t_index,1])
            assert(max(Yte_tilde_2[:,t_index,1])<=1 and min(Yte_tilde_2[:,t_index,1])>=0)
            Yte_tilde_2[:,t_index,2] = 1-Yte_tilde_2[:,t_index,1]
            
            loss_te = cross_entropy_loss(Yte[:,t_index,1], Yte_tilde[:,t_index,1])
            loss_te_2 = cross_entropy_loss(Yte[:,t_index,1], Yte_tilde_2[:,t_index,1])
            
            print("loss_tr")
            print(loss_tr)
#            print("loss_tr_2")
#            print(loss_tr_2)
            print("loss_te")
            print(loss_te)
            print("loss_te_2")
            print(loss_te_2)
        
        results_filename, costs_filename = init_results_dir(local_vars.results_directory, 
                                                            IDresults)
        get_results(config, Yte[:], DTA, 
                    loss_te_2, Yte_tilde_2, costs_dict, epoch, -1, results_filename,
                    costs_filename, from_var=False)
        
        count += 1
    #return Ytr[:,t_index,1], Y_tilde, err_tr, err_tr_tilde

def automate_fixedEdges(rootname_config, entries={}, tAt='TrTe', IDrs=[], 
                        build_IDrs=False, its=15, sufix='', IDr_merged='', log='', 
                        just_build=False, if_merge_results=False,IDweights=''):
    """  """
    if 'build_XY_mode' in entries:
        build_XY_mode = entries['build_XY_mode']
    else:
        build_XY_mode = 'manual'
    assert(build_XY_mode=='manual')
    if 'edge_dates' in entries:
        edge_dates = entries['edge_dates']
    else:
        edge_dates = ['2017.09.27']
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
    tag = 'FE'# fixed edges
    entries['config_name'] = 'C'+rootname_config+tag
    if IDweights=='':
        entries['IDweights'] = 'W'+rootname_config+tag+extW
    else:
        entries['IDweights'] = IDweights
    entries['IDresults'] = 'R'+rootname_config+tag+extR+sufix
    entries['IO_results_name'] = 'R'+rootname_config+tag+extW
    entries['edge_dates'] = edge_dates
    assert(build_XY_mode=='manual')
    entries['build_XY_mode'] = build_XY_mode
    config = configuration(entries)
    IDresults = config['IDresults']
    
    dirfilename_tr, dirfilename_te, IO_results_name = build_datasets(config=config, 
                                                                     log=log)
    
    if not just_build:
        f_IOtr = h5py.File(dirfilename_tr,'r')
        if 'Tr' in tAt:
            Ytr = f_IOtr['Y']
            Xtr = f_IOtr['X']
        f_IOte = h5py.File(dirfilename_te,'r')
        if 'Te' in tAt:
            Yte = f_IOte['Y']
            Xte = f_IOte['X']
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
                RNN(config).fit(Xtr,Ytr,num_epochs=epochs_per_it, log=log)
            else:
                raise ValueError('tAt value not known')
            
        f_IOtr.close()
        f_IOte.close()
    if not just_build and if_merge_results:
        merge_results(IDrs, IDr_merged)
        
def combine_models(entries, model_names, epochs, rootname_config, sufix='', 
                   melting_func='mean', tag_from_till='',log='', modular=False, sufix_io='', 
                   get_results=True):
    """  """
    import numpy as np
    import os
    from kaissandra.results2 import init_results_dir, get_results
    
    results_directory = local_vars.results_directory
    if 'build_XY_mode' in entries:
        build_XY_mode = entries['build_XY_mode']
    else:
        build_XY_mode = 'manual'
    assert(build_XY_mode=='manual')
    if 'edge_dates' in entries:
        edge_dates = entries['edge_dates']
    else:
        edge_dates = ['2017.09.27']
    if 'feats_from_bids' in entries:
        feats_from_bids = entries['feats_from_bids']
    else:
        feats_from_bids = False
    if 'results_from' in entries:
        results_from = entries['results_from']
    else:
        results_from = 'COMB'
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
    if 'build_asset_relations' in entries:
        build_asset_relations = entries['build_asset_relations']
    else:
        build_asset_relations = ['direct']
    ext_rel_tr = ''
    if 'direct' in build_asset_relations:
        ext_rel_tr = ext_rel_tr+'D'
    if 'inverse' in build_asset_relations:
        ext_rel_tr = ext_rel_tr+'I'
    if 'asset_relation' in entries:
        asset_relation = entries['asset_relation']
    else:
        asset_relation = 'direct'
    if asset_relation=='direct':
        ext_rel_cv = 'D'
    elif asset_relation=='inverse':
        ext_rel_cv = 'I'
    tag = 'CM'+tag_from_till# combine models+from day till day
    
    entries['config_name'] = 'C'+rootname_config+tag
#    if IDweights=='':
#        entries['IDweights'] = 'W'+rootname_config+tag+extW
#    else:
    IDresults = 'R'+rootname_config+tag+extR+sufix
    entries['IDresults'] = IDresults
    entries['IO_results_name'] = 'R'+rootname_config+tag+extW+ext_rel_cv+sufix_io
    entries['edge_dates'] = edge_dates
    assert(build_XY_mode=='manual')
    entries['build_XY_mode'] = build_XY_mode
    
    config = configuration(entries)
    
    config['IO_tr_name'] = rootname_config+tag+extR+ext_rel_tr+sufix_io
    config['IO_cv_name'] = rootname_config+tag+extR+ext_rel_cv+sufix_io
        
    seq_len = config['seq_len']
    size_output_layer = config['size_output_layer']
#    dirfilename_tr, dirfilename_te, IO_results_name = build_datasets(config=config, 
#                                                                     log=log)
    if not modular:
        dirfilename_tr, dirfilename_te, IO_results_name = build_datasets(config=config, 
                                                                     log=log)
    else:
        dirfilename_tr, dirfilename_te, IO_results_name = build_datasets_modular(config=config, 
                                                                     log=log)
    f_IOte = h5py.File(dirfilename_te,'r')
    Yte = f_IOte['Y']
    Xte = f_IOte['X']
    DTA = pickle.load( open( IO_results_name, "rb" ))
    n_models = len(model_names)
    # save model names
    if not os.path.exists(local_vars.results_directory+IDresults):
        os.mkdir(local_vars.results_directory+IDresults)
    with open(local_vars.results_directory+IDresults+'/model_names.txt',"w") as file:    
        file.write('\n'.join(model_names))
        file.close()
    
    outputs_stacked = np.zeros((Yte.shape[0], n_models*seq_len, size_output_layer))
    #J_tests = 0
    for i, name in enumerate(model_names):
        config['IDweights'] = name
        startFrom = epochs[i]
        endAt = epochs[i]
        output, Js = RNN(config).cv(Xte, Yte, DTA, IDresults=IDresults, \
                       startFrom=startFrom, endAt=endAt, config=config, 
                       if_get_results=False, log=log, save_cost=False)
        #J_tests += Js
        outputs_stacked[:,i*seq_len:(i+1)*seq_len,:] = output
    #J_tests = J_tests/n_models
    # apply melting function
    if melting_func=='mean':
        outputs_melted = np.zeros((Yte.shape[0], 1, size_output_layer))
        outputs_melted[:,0,:] = np.mean(outputs_stacked, axis=1)
    else:
        raise ValueError("melting_func not supported")
    # get resutls
    if get_results:
        results_filename, costs_filename = init_results_dir(results_directory, IDresults)
        epoch = 0
        costs_dict = {str(epoch):0.0}
        get_results(config, Yte, DTA, Js, outputs_melted, costs_dict, epoch, -1, 
                    results_filename, costs_filename, from_var=False)
        return None
    else:
        return outputs_stacked, Yte
    
    
        
if __name__=='__main__':
    pass
    #automate(['C3012INVO'])