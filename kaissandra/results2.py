# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 18:01:01 2018

@author: mgutierrez

Module with functions related to the management of results.
"""

import pickle
import os
import numpy as np
import pandas as pd
import datetime as dt
import time
#import h5py
#import matplotlib.pyplot as plt
#import scipy.io as sio

from kaissandra.local_config import local_vars

def get_last_saved_epoch2(resultsDir, ID):
    """
    <DocString>
    """
    filename = resultsDir+ID+"/costs.csv"
    #print(filename)
    
    if os.path.exists(filename):
        TR = pd.read_csv(filename,sep='\t')
        try:
            last_saved_epoch = TR.epoch.iloc[-1]
        except IndexError:
            last_saved_epoch = -1
#        print(last_saved_epoch)
#        a=p
    else:
        last_saved_epoch = -1
    return last_saved_epoch

def get_mc_vectors(t_y, t_soft_tilde, thr_mc, ub_mc):
    """  """
#    y_mc = (t_y[:,0]>thr_mc) & (t_y[:,0]<=ub_mc) # non-zeros market change bits
#    y_md_down = y_mc & (t_y[:,1]>t_y[:,2]) # non-zeros market direction down
    
    # extract non-zeros (y_c0>0.5)
    y_mc = (t_y[:,0]>thr_mc) & (t_y[:,0]<=ub_mc)
    ys = {'y_mc':y_mc, # non-zeros market change bits
          'y_md_down':y_mc & (t_y[:,1]>t_y[:,2]), # non-zeros market direction down
          'y_md_up':y_mc & (t_y[:,1]<t_y[:,2]), # non-zeros market direction up
          'y_mc_tilde':(t_soft_tilde[:,0]>thr_mc) & (t_soft_tilde[:,0]<=ub_mc), # predicted non-zeros market change bits
          'probs_mc':t_soft_tilde[:,0]}
    return ys

def get_md_vectors(t_soft_tilde, t_y, ys_mc, n_classes, thr_md, ub_md):
    """  """
    y_dec_md_tilde = np.argmax(t_soft_tilde[:,1:3], 1)-1# predicted dec out
    y_md_down_tilde = ys_mc['y_mc_tilde'] & (t_soft_tilde[:,1]>thr_md) & (t_soft_tilde[:,1]<=ub_md)
    y_md_up_tilde = ys_mc['y_mc_tilde'] & (t_soft_tilde[:,2]>=thr_md) & (t_soft_tilde[:,2]<=ub_md)
    y_md_tilde = y_md_down_tilde | y_md_up_tilde
    y_dec_md_tilde = y_dec_md_tilde-(y_dec_md_tilde-1)*(-1)+2
    y_dec_md = np.argmax(t_y[:,3:], 1)-(n_classes-1)/2 # real output in decimal
    ys_md = {# non-zeros market direction down ([y_md1,y_md2]=10)
             'y_md_down_tilde':y_md_down_tilde,
             # non-zeros market direction down ([y_md1,y_md2]=10)
             'y_md_up_tilde':y_md_up_tilde,
             'y_md_tilde':y_md_tilde, #non-zeros market direction up ([y_c1,y_c2]=01)
             'nz_indexes':ys_mc['y_mc'] & y_md_tilde, # non-zero indexes index
             'y_md_down_intersect':ys_mc['y_md_down'] & y_md_down_tilde, # indicates up bits (y_c2) correctly predicted
             'y_md_up_intersect':ys_mc['y_md_up'] & y_md_up_tilde, # indicates up bits (y_c2) correctly predicted
             'y_dec_md':y_dec_md, # real output in decimal
             'y_dec_md_tilde':y_dec_md_tilde, 
             'y_dec_mg':np.argmax(t_y[y_md_tilde,3:], 1)-(n_classes-1)/2,
             'y_dec_mg_tilde':np.argmax(t_soft_tilde[y_md_tilde,3:], 1)-(n_classes-1)/2,
             # difference between y and y_tilde {0=no error, 1=error in mc. 2=error in md}
             'diff_y_y_tilde':np.abs(np.sign(y_dec_md_tilde[y_md_tilde])-np.sign(y_dec_md[y_md_tilde])),
             'probs_md':np.maximum(t_soft_tilde[y_md_tilde,2],t_soft_tilde[y_md_tilde,1])
            }
    
    return ys_md

def print_results(results, epoch, J_test, J_train, thr_md, thr_mc, t_str):
    """  """
    print("Epoch = "+str(epoch)+". Time index = "+t_str+
          ". Threshold MC = "+str(thr_mc)+". Threshold MD = "+str(thr_md))
    if thr_md==.5 and thr_mc==.5:
        print("J_test = "+str(J_test)+", J_train = "+
              str(J_train)+", Accuracy="+str(results["Acc"]))
    
    print("RD = {0:d} ".format(results["RD"])+
           "NZ = {0:d} ".format(results["NZ"])+
           "NZA = {0:d} ".format(results["NZA"])+
           "pNZ = {0:.3f}% ".format(results["pNZ"])+
           "pNZA = {0:.3f}% ".format(results["pNZA"])+
           "AD = {0:.2f}% ".format(results["AD"])+
           "ADA = {0:.2f}% ".format(results["ADA"])+
           "NO = {0:d} ".format(results["NO"])+
           "GSP = {0:.2f}% ".format(results["GSP"])+
           "NSP = {0:.2f}%".format(results["NSP"]))

    print("SI2 = {0:.2f} ".format(results["SI2"])+
          "SI = {0:.2f} ".format(results["SI"])+
          "eGROI = {0:.2f}% ".format(results["eGROI"])+
          "eROI = {0:.2f}% ".format(results["eROI"])+
          "eROI2 = {0:.2f}% ".format(results["eROI2"])+
          "eROI3 = {0:.2f}% ".format(results["eROI3"])+
          "mSpread = {0:.4f}%".format(results["mSpread"]))
    return None

def get_results_entries():
    """  """
    results_entries = ['epoch','t_index','thr_mc','thr_md','AD','ADA','GSP','NSP','NO',
                       'NZ','NZA','RD','NSP.5','NSP1','NSP1.5','NSP2','NSP3',
                       'NSP4','NSP5','SI.5','SI1','SI1.5','SI2','SI3','SI4','SI5','SI',
                       'eGROI','eROI.5','eROI1','eROI1.5','eROI2','eROI3','eROI4',
                       'eROI5','eROI','mSpread','pNZ','pNZA','tGROI','tROI','eRl1',
                       'eRl2','eGl1','eGl2','sharpe','NOl1','NOl2','eGROIL','eGROIS',
                       'NOL','NOS','GSPl','GSPs','99eGROI','99eROI','99eROI.5',
                       '99eROI1','99eROI1.5','99eROI2','99eROI3','99eROI4','99eROI5']
    # GRL: GROI for Long positions
    # GRS: GROI for short positions
    # NOL: Number of long openings
    # NOS: umber of short openings
    return results_entries

def get_costs_entries():
    """  """
    return ['epoch','J_train','J_test']

def get_kpi_names():
    """  """
    return ['ACC','PR','RC','F1']

def get_results_mg_entries(classes, t_indexes):
    """  """
    used = set()
    results_list = get_costs_entries()+\
            [kpi+'C'+str(c-int((classes-1)/2))+'T'+t_index if kpi!='ACC' \
             else kpi+'T'+t_index for kpi in get_kpi_names() \
             for t_index in t_indexes for c in range(classes)]
    unique = [x for x in results_list if x not in used and (used.add(x) or True)]
    return unique

def get_performance_entries():
    """  """
    entries = ['epoch','t_index','thr_mg',
               'pNZ','pNZA','AD','ADA',
               'NO','NZ','NZA','RD',
               'GSP','NSP','NSP.5','NSP1','NSP1.5','NSP2','NSP3','NSP4','NSP5',
               'eGROI','eROI','eROI.5','eROI1','eROI1.5','eROI2','eROI3','eROI4','eROI5',
               'SI.5','SI1','SI1.5','SI2','SI3','SI4','SI5','SI',
               '99eGROI','99eROI','99eROI.5','99eROI1','99eROI1.5','99eROI2','99eROI3','99eROI4','99eROI5',
               'mSpread','eRl1','eRl2','eGl1','eGl2','sharpe','NOl1','NOl2','eGROIL','eGROIS',
               'NOL','NOS','GSPl','GSPs']
    return entries

def kpi2func_merge():
    """  """
    mapper = {'epoch':'none',
               't_index':'none',
               'thr_mg':'none',
               'pNZ':'mean',
               'pNZA':'mean',
               'AD':'mean',
               'ADA':'mean',
               'GSP':'mean',
               'NSP':'mean',
               'NO':'sum',
               'NZ':'sum',
               'NZA':'sum',
               'RD':'sum',
               'NSP.5':'mean',
               'NSP1':'mean',
               'NSP1.5':'mean',
               'NSP2':'mean',
               'NSP3':'mean',
               'NSP4':'mean',
               'NSP5':'mean',
               'SI.5':'sum',
               'SI1':'sum',
               'SI1.5':'sum',
               'SI2':'sum',
               'SI3':'sum',
               'SI4':'sum',
               'SI5':'sum',
               'SI':'sum',
               'eGROI':'sum',
               'eROI.5':'sum',
               'eROI1':'sum',
               'eROI1.5':'sum',
               'eROI2':'sum',
               'eROI3':'sum',
               'eROI4':'sum',
               'eROI5':'sum',
               'eROI':'sum',
               '99eGROI':'sum',
               '99eROI':'sum',
               '99eROI.5':'sum',
               '99eROI1':'sum',
               '99eROI1.5':'sum',
               '99eROI2':'sum',
               '99eROI3':'sum',
               '99eROI4':'sum',
               '99eROI5':'sum',
               'mSpread':'mean',
               'eRl1':'sum',
               'eRl2':'sum',
               'eGl1':'sum',
               'eGl2':'sum',
               'sharpe':'mean',
               'NOl1':'sum',
               'NOl2':'sum',
               'eGROIL':'sum',
               'eGROIS':'sum',
               'NOL':'sum',
               'NOS':'sum',
               'GSPl':'mean',
               'GSPs':'mean'
              }
    return mapper
    
def init_results_dir(resultsDir, IDresults):
    """  """
    filedir = resultsDir+IDresults+'/'
    results_filename = resultsDir+IDresults+'/performance'
    costs_filename = resultsDir+IDresults+'/costs'
    if not os.path.exists(filedir):
        os.makedirs(filedir)
    if not os.path.exists(results_filename+'.csv'):
        pd.DataFrame(columns = get_results_entries()).to_csv(results_filename+'.csv', 
                            mode="w",index=False,sep='\t')
        pd.DataFrame(columns = get_results_entries()).to_csv(results_filename+'.txt', 
                            mode="w",index=False,sep='\t')
    else:
        # add new columns with NaNs if it's the case
        TR = pd.read_csv(results_filename+'.csv', sep='\t')
        for c in get_results_entries():
            if c not in TR.columns:
                TR[c] = np.nan
        success = 0
        while not success:
            try:
                TR.to_csv(results_filename+'.csv', mode='w', index=False, sep='\t')
                TR.to_csv(results_filename+'.txt', mode='w', index=False, sep='\t')
                success = 1
            except PermissionError:
                print("WARNING! PermissionError. Close programs using "+results_filename)
                time.sleep(1)
    if not os.path.exists(costs_filename+'.csv'):
        pd.DataFrame(columns = get_costs_entries()).to_csv(costs_filename+'.csv', 
                            mode="w",index=False,sep='\t')
        
    return results_filename, costs_filename

def init_results_mg_dir(resultsDir, IDresults, classes, t_indexes, get_performance=False):
    """  """
    filedir = resultsDir+IDresults+'/'
    results_filename = resultsDir+IDresults+'/results_mg'
    performance_filename = resultsDir+IDresults+'/performance'
    costs_filename = resultsDir+IDresults+'/costs'
    if not os.path.exists(filedir):
        os.makedirs(filedir)
    if not os.path.exists(results_filename+'.csv'):
        pd.DataFrame(columns = get_results_mg_entries(classes, t_indexes)).to_csv(results_filename+'.csv', 
                            mode="w",index=False,sep='\t')
        pd.DataFrame(columns = get_results_mg_entries(classes, t_indexes)).to_csv(results_filename+'.txt', 
                            mode="w",index=False,sep='\t')
        if get_performance:
            pd.DataFrame(columns = get_performance_entries()).to_csv(performance_filename+'.csv', 
                                mode="w",index=False,sep='\t')
            pd.DataFrame(columns = get_performance_entries()).to_csv(performance_filename+'.txt', 
                                mode="w",index=False,sep='\t')
    else:
        # add new columns with NaNs if it's the case
        TR = pd.read_csv(results_filename+'.csv', sep='\t')
        
        for c in get_results_mg_entries(classes, t_indexes):
            if c not in TR.columns:
                TR[c] = np.nan
        if get_performance:
            TP = pd.read_csv(performance_filename+'.csv', sep='\t')
            for c in get_performance_entries():
                if c not in TP.columns:
                    TP[c] = np.nan
        success = 0
        while not success:
            try:
                TR.to_csv(results_filename+'.csv', mode='w', index=False, sep='\t')
                TR.to_csv(results_filename+'.txt', mode='w', index=False, sep='\t')
                if get_performance:
                    TP.to_csv(performance_filename+'.csv', mode='w', index=False, sep='\t')
                    TP.to_csv(performance_filename+'.txt', mode='w', index=False, sep='\t')
                success = 1
            except PermissionError:
                print("WARNING! PermissionError. Close programs using "+results_filename+" or "+performance_filename)
                time.sleep(1)
    if not os.path.exists(costs_filename+'.csv'):
        pd.DataFrame(columns = get_costs_entries()).to_csv(costs_filename+'.csv', 
                            mode="w",index=False,sep='\t')
        
    return results_filename, costs_filename, performance_filename

def init_results_struct(epoch, thr_mc, thr_md, t_index):
    """  """
    results = {}
    results['epoch'] = epoch
    results['t_index'] = t_index
    results['thr_mc'] = thr_mc
    results['thr_md'] = thr_md
    return results

def get_basic_results_struct(ys_mc, ys_md, results, m):
    """  """
    # market change accuracy
    results['Acc'] = 1-np.sum(np.abs(ys_mc['y_mc']^ys_mc['y_mc_tilde']))/m
    results['NZA'] = np.sum(ys_md['y_md_tilde']) # number of non-zeros all
    results['NZ'] = np.sum(ys_md['nz_indexes']) # Number of non-zeros
    results['RD'] =  np.sum(ys_md['y_md_down_intersect'])+\
                     np.sum(ys_md['y_md_up_intersect']) # right direction
    #a=p
    if results['NZ']>0:
        results['AD'] = 100*results['RD']/results['NZ'] # accuracy direction
    else:
        results['AD'] = 0
    if results['NZA']>0:
        results['ADA'] = 100*results['RD']/results['NZA'] # accuracy direction all
    else:
        results['ADA'] = 0
    results['pNZ'] = 100*results['NZ']/m # percent of non-zeros
    results['pNZA'] = 100*results['NZA']/m # percent of non-zeros all
    
    return results

def save_journal_fn(journal, journal_dir, journal_id, ext):
    """ save journal in disk """
    journal.index.name = 'index'
    if not os.path.exists(journal_dir):
        os.makedirs(journal_dir)
    success = 0
    while not success:
        try:
            journal.to_csv(journal_dir+journal_id+ext,sep='\t')
            success = 1
        except PermissionError:
            print("WARNING! PermissionError. Close programs using "+journal_dir+journal_id+ext)
            time.sleep(1)
    
    return None

def save_results_fn(filename, results):
    """ save results in disc as pandas data frame """
    df = pd.DataFrame(results, index=[0])\
        [pd.DataFrame(columns = get_results_entries()).columns.tolist()]
    success = 0
    df.to_csv(filename+'.txt', mode='a', header=False, index=False, sep='\t')
    while not success:
        try:
            df.to_csv(filename+'.csv', mode='a', header=False, index=False, sep='\t')
            
            success = 1
        except PermissionError:
            print("WARNING! PermissionError. Close programs using "+filename)
            time.sleep(1)
    return None

def save_performance(filename, results):
    """ save results in disc as pandas data frame """
    df = pd.DataFrame(results, index=[0])\
        [pd.DataFrame(columns = get_performance_entries()).columns.tolist()]
    success = 0
    df.to_csv(filename+'.txt', mode='a', header=False, index=False, sep='\t')
    while not success:
        try:
            df.to_csv(filename+'.csv', mode='a', header=False, index=False, sep='\t')
            
            success = 1
        except PermissionError:
            print("WARNING! PermissionError. Close programs using "+filename)
            time.sleep(1)
    return None

def save_costs(costs_filename, entries):
    """  """
    costs_entries = get_costs_entries()
    dict_costs = {}
    for c in range(len(costs_entries)):
        dict_costs[costs_entries[c]] = entries[c]
    
    df = pd.DataFrame(dict_costs, index=[0])\
        [pd.DataFrame(columns = costs_entries).columns.tolist()]
    success = 0
    while not success:
        try:
            df.to_csv(costs_filename+'.csv', mode='a', header=False, index=False, sep='\t')
            success = 1
        except PermissionError:
            print("WARNING! PermissionError. Close programs using "+costs_filename)
            time.sleep(1)
    return None

def get_zero_result_dict(crt):
    """  """
    result_dict = {}
    for key in crt.keys():
        result_dict[key] = 0.0
    return result_dict

def get_single_result(CR_t, mc, md, thresholds_mc, thresholds_md):
    """  """
    sr = {}
    if mc<thresholds_mc[-1]:
        crt_mc = CR_t[mc+1][md]
    else:
        crt_mc = get_zero_result_dict(CR_t[mc][md])
    if md<thresholds_md[-1]:
        crt_md = CR_t[mc][md+1]
    else:
        crt_md = get_zero_result_dict(CR_t[mc][md])  
    if md==thresholds_md[-1] and mc==thresholds_mc[-1]:
        crt_mcd = get_zero_result_dict(CR_t[mc][md])  
    else:
        crt_mcd = CR_t[mc+1][md+1]
        
    for key in CR_t[mc][md].keys():
        sr[key] = CR_t[mc][md][key]-crt_mc[key]-crt_md[key]+crt_mcd[key]
    return sr

def get_best_results_list():
    """ get list containing the entries to get the best results from """
    return ['eGROI','eROI','eROI.5','eROI1','eROI1.5','eROI2','eROI3','eROI4',
           'eROI5','SI','SI.5','SI1','SI1.5','SI2','SI3','SI4','SI5',
           '99eGROI','99eROI','99eROI.5','99eROI1','99eROI1.5','99eROI2','99eROI3','99eROI4','99eROI5']
    
def get_best_results(TR, results_filename, resultsDir, IDresults, save=0, from_mg=False):
    """  """
    best_results_list = get_best_results_list()
    for c in best_results_list:
        if c not in TR.columns:
            TR[c] = -100000
    best_dir = resultsDir+IDresults+'/best/'
    if not os.path.exists(best_dir):
        os.mkdir(best_dir)
    if not save:
        file = open(best_dir+'best.txt',"w")
        file.close()
        file = open(best_dir+'best.txt',"a")
    for b in best_results_list:
        best_filename = best_dir+'best_'+b+'.csv'
        if not os.path.exists(best_filename):
            if not from_mg:
                pd.DataFrame(columns = get_results_entries()).to_csv(best_filename, 
                            mode="w",index=False,sep='\t')
            else:
                pd.DataFrame(columns = get_performance_entries()).to_csv(best_filename, 
                            mode="w",index=False,sep='\t')
        idx = TR[b].idxmax()
            
        # reset file
        
        if save:
            df = pd.DataFrame(TR.loc[idx:idx])
            success = 0
            while not success:
                try:
                    df.to_csv(best_filename, mode='a', header=False, index=False, sep='\t')
                    success = 1
                except PermissionError:
                    print("WARNING! PermissionError. Close programs using "+
                          best_filename)
                    time.sleep(1)
        if not from_mg:
            thr_text = " thr_mc "+str(TR['thr_mc'].loc[idx])+\
                  " thr_md "+str(TR['thr_md'].loc[idx])
        else:
            thr_text = " thr_mc "+str(TR['thr_mg'].loc[idx])
        
        out = "Best "+b+" = {0:.2f}".format(TR[b].loc[idx])+\
              " t_index "+str(TR['t_index'].loc[idx])+thr_text+\
              " epoch "+str(TR['epoch'].loc[idx])+" in "+str(TR['NO'].loc[idx])+\
              " GSP {0:.2f}".format(TR['GSP'].loc[idx])+\
              " ADA {0:.2f}".format(TR['ADA'].loc[idx])+\
              " AD {0:.2f}".format(TR['AD'].loc[idx])
        print(out)
        if not save:
            file.write(out+"\n")
    if not save:
        file.close()
    return None

def adc(AD, ADA):
    """ Accuracy directions combine """
    return max(1.7/3*(AD-.5)+1.3/3*(ADA-.33), 0)#2/3*AccDir+1/3*AccDirA#

def metric_combine(x):
    """ Combine metric for weights update """
    return max(x, 0)

def update_weights_combine(weights, t, w_idx, params, results):
    """ Update weights for combining based on algo """
    #map_idx2thr[w_idx]
    if params['alg'] == 'adc': # AD and ADA combine
        AD = results['AD']
        ADA = results['ADA']
        weights[t,w_idx,0] = adc(AD, ADA)
#        print("weights[t,w_idx,0]")
#        print(weights[t,w_idx,0])
    elif params['alg'] == 'roic': # ROI combine
        if 'spread' in params:
            spread = params['spread']
        else:
            spread = '2'
        weights[t,w_idx,0] = metric_combine(results['eROI'+spread])
    elif params['alg'] == 'mean': # mean
        weights[t,w_idx,0] = 1
    #thr_idx += 1
    return weights

def combine_ts_fn(seq_len, soft_tilde, weights, map_idx2thr, thresholds_mc, thresholds_md):
    """  """
    i_t_mc = 0
    i_thr = 0
    idx_thr = np.zeros((soft_tilde.shape[0], seq_len)).astype(int)
    
    for thr_mc in thresholds_mc:
        i_t_md = 0
        for thr_md in thresholds_md:
            for t_ in range(seq_len):
                
                idx_thr_mc = soft_tilde[:,t_,0]>thr_mc
                idx_thr_md = np.maximum(soft_tilde[:,t_,1],soft_tilde[:,t_,2])>thr_md
                idx_thr[idx_thr_mc & idx_thr_md,t_] = i_thr
            i_thr += 1
            i_t_md += 1
        i_t_mc += 1
                    
    sum_AD = np.zeros((idx_thr.shape[0],1))
    t_soft_tilde = np.zeros((soft_tilde.shape[0],soft_tilde.shape[2]))
    weights[np.isnan(weights)] = 0
    print("idx_thr")
    print(idx_thr)
    for t_ in range(seq_len):
#        print("idx_thr[:,t_]")
#        print(idx_thr[:,t_])
#        print("map_idx2thr[idx_thr[:,t_]]")
#        print(map_idx2thr[idx_thr[:,t_]])
#        print("weights[t_,map_idx2thr[idx_thr[:,t_]],:]")
#        print(weights[t_,map_idx2thr[idx_thr[:,t_]],:])
        sum_AD = sum_AD+weights[t_,map_idx2thr[idx_thr[:,t_]],:]
        t_soft_tilde = t_soft_tilde+weights[t_,map_idx2thr[idx_thr[:,t_]],:]*soft_tilde[:,t_,:]
    # normaluze t_soft_tilde
    t_soft_tilde = t_soft_tilde/sum_AD
    
    return t_soft_tilde

def save_results_mg(results, filename, classes, t_indexes):
    """ save results in disc as pandas data frame """
    df = pd.DataFrame(results, index=[0])\
        [pd.DataFrame(columns = get_results_mg_entries(classes, t_indexes)).columns.tolist()]
    success = 0
    df.to_csv(filename+'.txt', mode='a', header=False, index=False, sep='\t')
    while not success:
        try:
            df.to_csv(filename+'.csv', mode='a', header=False, index=False, sep='\t')
            
            success = 1
        except PermissionError:
            print("WARNING! PermissionError. Close programs using "+filename)
            time.sleep(1)
    
    return None

def update_TR_mg(TR, CN, acc, PR, RC, FS, t, t_idx, classes):
    """  """
    KPIs = get_kpi_names()
    for key in get_results_mg_entries(classes, [t_idx]):
        for kpi in KPIs:
            for c in range(classes):
                if 'PR' in key and key == kpi+'C'+str(c-int((classes-1)/2))+'T'+t_idx:
                    TR[key] = PR[c,t]
                elif 'RC' in key and key == kpi+'C'+str(c-int((classes-1)/2))+'T'+t_idx:
                    TR[key] = RC[c,t]
                elif 'F1' in key and key == kpi+'C'+str(c-int((classes-1)/2))+'T'+t_idx:
                    # TODO: make it compatible with more than one F-score
                    TR[key] = FS[c,t][0]
                elif 'ACC'in key and key == kpi+'T'+t_idx:
                    TR[key] = acc
    return TR

def get_results_meta(config, y, soft_tilde, costs, epoch, J_test, costs_filename,
                   results_filename, performance_filename, out_act_func, 
                   get_performance=False, DTA=None):
    """ Get results based on market gain """
    m = soft_tilde.shape[0]
    seq_len = 1
    n_classes = config['size_output_layer']
    if 'scores' in config:
        scores = config['scores']
    else:
        scores = [1]
    if 'cost_name' in config:
        cost_name = config['cost_name']
    else:
        cost_name = ''
    J_train = costs[cost_name+str(epoch)]
    
    accs = np.zeros((seq_len+1))
    PR = np.zeros((n_classes,seq_len+1)) # precision matrix
    RC = np.zeros((n_classes,seq_len+1)) # recall matrix
    FS = np.zeros((n_classes,seq_len+1,len(scores))) # F score matrix
    CM = np.zeros((n_classes, n_classes, seq_len+1))# confusion matrix
    # TODO: take +1 and 'mean' constants from config file
    t_indexes = [str(t) if t<seq_len else 'mean' for t in range(seq_len)]
    column_names = get_results_mg_entries(n_classes, t_indexes)
    # table results
    TR = {'epoch':epoch,
          'J_train':J_train,
          'J_test':J_test}
    for t in range(seq_len):
        if t<seq_len:
            y_tilde_dec = np.argmax(soft_tilde[:, t, :], axis=1)
            y_t = y[:,t,:].astype(int)
            t_idx = str(t)
        y_tilde = np.eye(n_classes)[y_tilde_dec].astype(int)
        y_dec = np.argmax(y_t, axis=1)
        #print(y_tilde_dec)
        #acc = 1-np.sum(np.abs(y_t^y_tilde))/m # xor
        acc = 1-np.sum(np.abs(np.sign(y_dec-y_tilde_dec)))/m # xor
        for i in range(n_classes):
            for j in range(n_classes):
                CM[i,j,t] = np.sum(y_tilde[:,i] & y_t[:,j]) # entry of conf. matrix
        for c in range(n_classes):
            if np.sum(CM[c,:,t])>0:
                PR[c,t] = CM[c,c,t]/np.sum(CM[c,:,t])
                RC[c,t] = CM[c,c,t]/np.sum(CM[:,c,t])
                FS[c,t,:] = [(1+score**2)*PR[c,t]*RC[c,t]/(score**2*PR[c,t]+RC[c,t]) for score in scores]
            else:
                PR[c,t] = 0
                RC[c,t] = 0
                FS[c,t,:] = [0 for score in scores]
        accs[t] = acc
        
        print_results_mg(acc, CM, PR, RC, FS, scores, t, t_idx)
        
        TR = update_TR_mg(TR, column_names, acc, PR, RC, FS, t, t_idx, n_classes)
        
        t_indexes.append(t_idx)
        #print(dicto)
    save_results_mg(TR, results_filename, n_classes, t_indexes)
    results = {'accs':accs,
               'CM':CM,
               'PR':PR,
               'RC':RC,
               'FS':FS}
    
    save_costs(costs_filename, [epoch, J_train, J_test])
    
    if get_performance:
        print("Getting performance")
        results_extended = get_performance_mg(config, y, soft_tilde, DTA, epoch, performance_filename)
        results.update(results_extended)
    return results

def get_results_mg(config, y, soft_tilde, costs, epoch, J_test, costs_filename,
                   results_filename, performance_filename, get_performance=False, DTA=None):
    """ Get results based on market gain """
    m = soft_tilde.shape[0]
    lB = config['lB']
    nEventsPerStat = config['nEventsPerStat']
    movingWindow = config['movingWindow']
    seq_len = int((lB-nEventsPerStat)/movingWindow+1)
    n_classes = config['size_output_layer']
    if 'scores' in config:
        scores = config['scores']
    else:
        scores = [1]
    if 'cost_name' in config:
        cost_name = config['cost_name']
    else:
        cost_name = ''
    J_train = costs[cost_name+str(epoch)]
    
    accs = np.zeros((seq_len+1))
    PR = np.zeros((n_classes,seq_len+1)) # precision matrix
    RC = np.zeros((n_classes,seq_len+1)) # recall matrix
    FS = np.zeros((n_classes,seq_len+1,len(scores))) # F score matrix
    CM = np.zeros((n_classes, n_classes, seq_len+1))# confusion matrix
    # TODO: take +1 and 'mean' constants from config file
    t_indexes = [str(t) if t<seq_len else 'mean' for t in range(seq_len+1)]
    column_names = get_results_mg_entries(n_classes, t_indexes)
    # table results
    TR = {'epoch':epoch,
          'J_train':J_train,
          'J_test':J_test}
    for t in range(seq_len+1):
        if t<seq_len:
            y_tilde_dec = np.argmax(soft_tilde[:, t, :], axis=1)
            y_t = y[:,t,:].astype(int)
            t_idx = str(t)
        else:
            y_tilde_dec = np.argmax(np.mean(soft_tilde, axis=1), axis=1)
            y_t = y[:,0,:].astype(int)
            t_idx = 'mean'
        y_tilde = np.eye(n_classes)[y_tilde_dec].astype(int)
        y_dec = np.argmax(y_t, axis=1)
        #print(y_tilde_dec)
        #acc = 1-np.sum(np.abs(y_t^y_tilde))/m # xor
        acc = 1-np.sum(np.abs(np.sign(y_dec-y_tilde_dec)))/m # xor
        for i in range(n_classes):
            for j in range(n_classes):
                CM[i,j,t] = np.sum(y_tilde[:,i] & y_t[:,j]) # entry of conf. matrix
        for c in range(n_classes):
            if np.sum(CM[c,:,t])>0:
                PR[c,t] = CM[c,c,t]/np.sum(CM[c,:,t])
                RC[c,t] = CM[c,c,t]/np.sum(CM[:,c,t])
                FS[c,t,:] = [(1+score**2)*PR[c,t]*RC[c,t]/(score**2*PR[c,t]+RC[c,t]) for score in scores]
            else:
                PR[c,t] = 0
                RC[c,t] = 0
                FS[c,t,:] = [0 for score in scores]
        accs[t] = acc
        
        print_results_mg(acc, CM, PR, RC, FS, scores, t, t_idx)
        
        TR = update_TR_mg(TR, column_names, acc, PR, RC, FS, t, t_idx, n_classes)
        
        t_indexes.append(t_idx)
        #print(dicto)
    save_results_mg(TR, results_filename, n_classes, t_indexes)
    results = {'accs':accs,
               'CM':CM,
               'PR':PR,
               'RC':RC,
               'FS':FS}
    
    save_costs(costs_filename, [epoch, J_train, J_test])
    
    if get_performance:
        print("Getting performance")
        results_extended = get_performance_mg(config, y, soft_tilde, DTA, epoch, performance_filename)
        results.update(results_extended)
    return results

def filter_journal(config, Journal):
    """  """
    if 'results_from' in config:
        results_from = config['results_from']
    else:
        results_from = 'BIDS'
    if results_from=='BIDS':
        # only get short bets (negative directions)
        Journal = Journal[Journal['Bet']<0]
    elif results_from=='ASKS':
        # only get long bets (positive directions)
        Journal = Journal[Journal['Bet']>0]
    elif results_from=='COMB':
        pass
    else:
        raise ValueError("Entry results_from not known. Options are SHORT/LONG/COMB")
    return Journal

def get_performance_meta(config, y, soft_tilde, DTA, epoch, 
                         performance_filename):
    """ Get ROI-based performance metrics """
    m = y.shape[0]
    seq_len = 1
    n_classes = 3
    n_days = len(config['dateTest'])
    if 'resolution' in config:
        resolution = config['resolution']
    else:
        resolution = 10
    IDresults = config['IDresults']
    save_journal = config['save_journal']
    if 'out_act_func' in config:
        out_act_func = config['out_act_func']
    else:
        out_act_func = 'sigmoid'
    if out_act_func=='sigmoid':
        thr = 0.5
    elif out_act_func=='tanh':
        thr = 0
    else:
        raise NotImplemented("out_act_func no implemented yet")
    if 'thresholds_mg' in config:
        thresholds_mg = config['thresholds_mg']
    else:
        thresholds_mg = thresholds_mg = [int(np.round((.5+i/resolution)*100))/100 for i in range(int(resolution/2))]
    if 'get_corr_signal' in config:
        get_corr_signal = config['get_corr_signal']
    else:
        get_corr_signal = False
    for t in range(seq_len):
        if t<seq_len:
            soft_tilde_t = soft_tilde
            y_t = y
            t_idx = str(t)
        else:
            y_t = y[:, 0, :]
            soft_tilde_t = np.mean(soft_tilde, axis=1)
            t_idx = 'mean'
        
        if out_act_func=='sigmoid':
            
            y_tilde_dec = (soft_tilde_t>thr).astype(int)-(soft_tilde_t<thr).astype(int)#np.argmax(soft_tilde_t, axis=1)-int((n_classes-1)/2)
            y_dec = (y_t>thr).astype(int)-(y_t<thr).astype(int)#np.argmax(y_t, axis=1)-int((n_classes-1)/2)
            p_bear = 1-soft_tilde_t#np.sum(soft_tilde_t[:, :int((n_classes-1)/2)],axis=1)
            p_bull = soft_tilde_t#np.sum(soft_tilde_t[:, int((n_classes-1)/2)+1:],axis=1)
        elif out_act_func=='tanh':
            
            y_tilde_dec = (soft_tilde_t>thr).astype(int)-(soft_tilde_t<thr).astype(int)#np.argmax(soft_tilde_t, axis=1)-int((n_classes-1)/2)
            y_dec = (y_t>thr).astype(int)-(y_t<thr).astype(int)#np.argmax(y_t, axis=1)-int((n_classes-1)/2)
            p_bear = (-1)*soft_tilde_t#np.sum(soft_tilde_t[:, :int((n_classes-1)/2)],axis=1)
            p_bull = soft_tilde_t#np.sum(soft_tilde_t[:, int((n_classes-1)/2)+1:],axis=1) 
        
        for mg, thr_mg in enumerate(thresholds_mg):
            indexes_thr_bear = p_bear>thr_mg
            indexes_thr_bull = p_bull>thr_mg
            indexes_mc = indexes_thr_bear | indexes_thr_bull
            diff_y_y_tilde = np.abs(np.sign(y_dec[indexes_mc])-np.sign(y_tilde_dec[indexes_mc]))
            DTAt = DTA.iloc[::seq_len,:]
            results = {}
            # market change accuracy
            results['NZA'] = np.sum(indexes_mc) # number of non-zeros all
            results['NZ'] = np.sum(indexes_mc & (y_dec!=0)) # Number of non-zeros
            results['RD'] =  np.sum(indexes_thr_bear & (y_dec<0))+\
                             np.sum(indexes_thr_bull & (y_dec>0)) # right direction
            #a=p
            if results['NZ']>0:
                results['AD'] = 100*results['RD']/results['NZ'] # accuracy direction
            else:
                results['AD'] = 0
            if results['NZA']>0:
                results['ADA'] = 100*results['RD']/results['NZA'] # accuracy direction all
            else:
                results['ADA'] = 0
            results['pNZ'] = 100*results['NZ']/m # percent of non-zeros
            results['pNZA'] = 100*results['NZA']/m # percent of non-zeros all
            # get journal
            Journal = get_journal(DTAt.iloc[indexes_mc[:,0]], 
                                  y_tilde_dec[indexes_mc], 
                                  y_dec[indexes_mc],
                                  diff_y_y_tilde,
                                  p_bear[indexes_mc], 
                                  p_bull[indexes_mc])
            
            Journal = filter_journal(config, Journal)
            if save_journal:
                journal_dir = local_vars.results_directory+IDresults+'/journal/'
                journal_id = 'J_E'+str(epoch)+'TI'+t_idx+'MG'+str(thr_mg)
                ext = '.csv'
                save_journal_fn(Journal, journal_dir, journal_id, ext)
                pos_dirname = local_vars.results_directory+IDresults+'/positions/'
                corr_dirname = local_vars.results_directory+IDresults+'/corr/'                    
                pos_filename = 'P_E'+str(epoch)+'TI'+t_idx+'MG'+str(thr_mg)+'.csv'
                corr_filename = 'C_E'+str(epoch)+'TI'+t_idx+'MG'+str(thr_mg)+'.p'
            else:
                corr_dirname = ''
                pos_dirname = ''
                pos_filename = ''
                corr_filename = ''
            results_extended, log = get_extended_results(Journal,
                                                n_classes,
                                                n_days,
                                                get_corr_signal=get_corr_signal,
                                                get_positions=save_journal,
                                                pos_dirname=pos_dirname,
                                                pos_filename=pos_filename,
                                                corr_filename=corr_filename,
                                                corr_dirname=corr_dirname)
            results_extended.update({'epoch':epoch,
                                     't_index':t_idx,
                                     'thr_mg':thr_mg})
            results.update(results_extended)
            print_performance(results, epoch, thr_mg, t_idx)
            save_performance(performance_filename, results)
    TP = pd.read_csv(performance_filename+'.csv', sep='\t')
    print('\n')
    get_best_results(TP[TP.epoch==epoch], performance_filename, 
                     local_vars.results_directory,
                     IDresults, save=1, from_mg=True)
    print("\nThe very best:")
    get_best_results(TP, performance_filename, local_vars.results_directory, IDresults, from_mg=True)
    return results_extended

def get_performance_mg(config, y, soft_tilde, DTA, epoch, performance_filename):
    """ Get ROI-based performance metrics """
    m = y.shape[0]
    lB = config['lB']
    nEventsPerStat = config['nEventsPerStat']
    movingWindow = config['movingWindow']
    seq_len = int((lB-nEventsPerStat)/movingWindow+1)
    n_classes = config['size_output_layer']
    n_days = len(config['dateTest'])
    resolution = config['resolution']
    IDresults = config['IDresults']
    save_journal = config['save_journal']
    if 'thresholds_mg' in config:
        thresholds_mg = config['thresholds_mg']
    else:
        thresholds_mg = [.5+i/resolution for i in range(int(resolution/2))]
    if 'get_corr_signal' in config:
        get_corr_signal = config['get_corr_signal']
    else:
        get_corr_signal = False
    for t in range(seq_len+1):
        if t<seq_len:
            soft_tilde_t = soft_tilde[:, t, :]
            y_t = y[:, t, :]
            t_idx = str(t)
        else:
            y_t = y[:, 0, :]
            soft_tilde_t = np.mean(soft_tilde, axis=1)
            t_idx = 'mean'
        y_tilde_dec = np.argmax(soft_tilde_t, axis=1)-int((n_classes-1)/2)
        y_dec = np.argmax(y_t, axis=1)-int((n_classes-1)/2)
        p_bear = np.sum(soft_tilde_t[:, :int((n_classes-1)/2)],axis=1)
        p_bull = np.sum(soft_tilde_t[:, int((n_classes-1)/2)+1:],axis=1)
        for mg, thr_mg in enumerate(thresholds_mg):
            indexes_thr_bear = p_bear>thr_mg
            indexes_thr_bull = p_bull>thr_mg
            indexes_mc = indexes_thr_bear | indexes_thr_bull
            diff_y_y_tilde = np.abs(np.sign(y_dec[indexes_mc])-\
                                            np.sign(y_tilde_dec[indexes_mc]))
            DTAt = DTA.iloc[::seq_len,:]
            results = {}
            # market change accuracy
            results['NZA'] = np.sum(indexes_mc) # number of non-zeros all
            results['NZ'] = np.sum(indexes_mc & (y_dec!=0)) # Number of non-zeros
            results['RD'] =  np.sum(indexes_thr_bear & (y_dec<0))+\
                             np.sum(indexes_thr_bull & (y_dec>0)) # right direction
            #a=p
            if results['NZ']>0:
                results['AD'] = 100*results['RD']/results['NZ'] # accuracy direction
            else:
                results['AD'] = 0
            if results['NZA']>0:
                results['ADA'] = 100*results['RD']/results['NZA'] # accuracy direction all
            else:
                results['ADA'] = 0
            results['pNZ'] = 100*results['NZ']/m # percent of non-zeros
            results['pNZA'] = 100*results['NZA']/m # percent of non-zeros all
            # get journal
            Journal = get_journal(DTAt.iloc[indexes_mc], 
                                  y_tilde_dec[indexes_mc], 
                                  y_dec[indexes_mc],
                                  diff_y_y_tilde,
                                  p_bear[indexes_mc], 
                                  p_bull[indexes_mc])
            
            Journal = filter_journal(config, Journal)
            if save_journal:
                journal_dir = local_vars.results_directory+IDresults+'/journal/'
                journal_id = 'J_E'+str(epoch)+'TI'+t_idx+'MG'+str(thr_mg)
                ext = '.csv'
                save_journal_fn(Journal, journal_dir, journal_id, ext)
                pos_dirname = local_vars.results_directory+IDresults+'/positions/'
                corr_dirname = local_vars.results_directory+IDresults+'/corr/'                    
                pos_filename = 'P_E'+str(epoch)+'TI'+t_idx+'MG'+str(thr_mg)+'.csv'
                corr_filename = 'C_E'+str(epoch)+'TI'+t_idx+'MG'+str(thr_mg)+'.p'
            else:
                corr_dirname = ''
                pos_dirname = ''
                pos_filename = ''
                corr_filename = ''
            results_extended, log = get_extended_results(Journal,
                                                n_classes,
                                                n_days,
                                                get_corr_signal=get_corr_signal,
                                                get_positions=save_journal,
                                                pos_dirname=pos_dirname,
                                                pos_filename=pos_filename,
                                                corr_filename=corr_filename,
                                                corr_dirname=corr_dirname)
            results_extended.update({'epoch':epoch,
                                     't_index':t_idx,
                                     'thr_mg':thr_mg})
            results.update(results_extended)
            print_performance(results, epoch, thr_mg, t_idx)
            save_performance(performance_filename, results)
    TP = pd.read_csv(performance_filename+'.csv', sep='\t')
    print('\n')
    get_best_results(TP[TP.epoch==epoch], performance_filename, 
                     local_vars.results_directory,
                     IDresults, save=1, from_mg=True)
    print("\nThe very best:")
    get_best_results(TP, performance_filename, local_vars.results_directory, IDresults, from_mg=True)
    return results_extended

def print_performance(results, epoch, thr_mg, t_idx):
    """  """
    print("Epoch = "+str(epoch)+". Time index = "+t_idx+
          ". Threshold MG = "+str(thr_mg))
    
    print("AD = {0:.2f}% ".format(results["AD"])+
          "ADA = {0:.2f}% ".format(results["ADA"])+
          "RD = {0:d} ".format(results["RD"])+
          "NZ = {0:d} ".format(results["NZ"])+
          "NZA = {0:d} ".format(results["NZA"])+
          "pNZ = {0:.4f}% ".format(results["pNZ"])+
          "pNZA = {0:.4f}% ".format(results["pNZA"]))
    
    print("NO = {0:d} ".format(results["NO"])+
          "NOL = {0:d} ".format(results["NOL"])+
          "NOS = {0:d} ".format(results["NOS"])+
          "GSP = {0:.2f}% ".format(results["GSP"])+
          "NSP = {0:.2f}% ".format(results["NSP"])+
          "NSP1 = {0:.2f}% ".format(results["NSP1"])+
          "NSP1.5 = {0:.2f}% ".format(results["NSP1.5"])+
          "NSP2 = {0:.2f}% ".format(results["NSP2"])+
          "NSP3 = {0:.2f}% ".format(results["NSP3"]))

    print("eGROI = {0:.2f}% ".format(results["eGROI"])+
          "eROI = {0:.2f}% ".format(results["eROI"])+
          "eROI1 = {0:.2f}% ".format(results["eROI1"])+
          "eROI1.5 = {0:.2f}% ".format(results["eROI1.5"])+
          "eROI2 = {0:.2f}% ".format(results["eROI2"])+
          "eROI3 = {0:.2f}% ".format(results["eROI3"])+
          "mSpread = {0:.4f}% ".format(results["mSpread"]))
    
    print("SI = {0:.2f} ".format(results["SI"])+
          "SI1 = {0:.2f} ".format(results["SI1"])+
          "SI1.5 = {0:.2f} ".format(results["SI1.5"])+
          "SI2 = {0:.2f} ".format(results["SI2"]))
    return None

def print_results_mg(acc, CM, PR, RC, FS, scores, t, t_idx):
    """  """
    # print results
#    print("J_test = "+str(J_test)+", J_train = "+
#          str(J_train)+", Accuracy="+str(results["Acc"]))
    print("t_index="+t_idx)
    print("Accuracy: {0:.2f}%".format(100*acc))
    #print("CM of t="+t_idx)
    #print(CM[:,:,t])
#    print("PR of t="+t_idx)
#    print(PR[:,t])
#    print("RC of t="+t_idx)
#    print(RC[:,t])
#    for s, score in enumerate(scores):
#        print("FS of t="+t_idx+" and score="+str(score))
#        print(FS[:,t,s])
    return None

def get_results(config, y, DTA, J_test, soft_tilde,
                 costs, epoch, lastTrained, results_filename,
                 costs_filename, from_var=False, keep_old_cost=False):
    """ Get results after one epoch.
    Args:
        - 
    Return:
        - """
#    print("y")
#    print(y)
#    print(y.shape)
#    print(np.sum(np.sum(y[:,0,1:3],axis=1)))
#    print(np.sum(1-y[:,0,5]))
#    print(np.sum(y[:,0,0]))
#    print("soft_tilde")
#    print(soft_tilde[:,-1,:])
#    mc5=soft_tilde[:,-1,0]>.5
#    print(soft_tilde[mc5,-1,:])
#    print(soft_tilde[mc5,-1,0])
#    a=p
    dateTest = config['dateTest']
    IDresults = config['IDresults']
    IDweights = config['IDweights']
    save_journal = config['save_journal']
    resolution = config['resolution']
    seq_len = config['seq_len']
    bits_mg = config['n_bits_outputs'][-1]
    resultsDir = local_vars.results_directory
    if 't_indexes' not in config:
        t_indexes = range(seq_len)
    else:
        t_indexes = config['t_indexes']
    if 'thresholds_mc' not in config:
        thresholds_mc = [.5+i/resolution for i in range(int(resolution/2))]
    else:
        thresholds_mc = config['thresholds_mc']
    if 'thresholds_md' not in config:
        thresholds_md = [.5+i/resolution for i in range(int(resolution/2))]
    else:
        thresholds_md = config['thresholds_md']
    if 'combine_ts' in config:
        combine_ts = config['combine_ts']
        
#        if_combine = config['combine_ts']['if_combine']
#        params_combine = config['combine_ts']['params_combine']
#        columns_AD = [str(tmc)+str(tmd) for tmc in thresholds_md for tmd in thresholds_md]#config['combine_ts']['columns_AD']
#        map_idx2thr = [columns_AD.index(str(int(tmc*10))+str(int(tmd*10))) \
#                       for tmc in thresholds_mc for tmd in thresholds_md]
#        extra_ts = len(params_combine)
#        weights_list = [np.zeros((model.seq_len+1,len(columns_AD),1)) for i in range(extra_ts)]
        
        if_combine = combine_ts['if_combine']
        if if_combine:
            t_indexes.append(seq_len+1)
        params_combine = combine_ts['params_combine']
        columns_AD = [str(int(tmc*10))+str(int(tmd*10)) for tmc in thresholds_mc for tmd in thresholds_md]#config['combine_ts']['columns_AD']

        map_idx2thr = np.array([columns_AD.index(str(int(tmc*10))+str(int(tmd*10))) \
                       for tmc in thresholds_mc for tmd in thresholds_md])

        extra_ts = len(params_combine)

        weights_list = [np.zeros((seq_len+1,len(columns_AD),1)) for i in range(extra_ts)]
        
        
        #map_idx2thr = np.zeros((len(thresholds_mc)*len(thresholds_md))).astype(int)
        
#        idx = 0
#        for tmc in thresholds_mc:
#            for tmd in thresholds_md:
#                col_name = str(int(tmc*10))+str(int(tmd*10))
#                map_idx2thr[idx] = columns_AD.index(col_name)
#                idx += 1
    else:
        if_combine = False
        extra_ts = 0
#        extra_ts = 0
        ### TEMP! ####
#        combine_ts = {'if_combine':True,
#                      'params_combine':[{'alg':'adc'}]}
#        if_combine = combine_ts['if_combine']
#        params_combine = combine_ts['params_combine']
#        columns_AD = [str(int(tmc*10))+str(int(tmd*10)) for tmc in thresholds_mc for tmd in thresholds_md]#config['combine_ts']['columns_AD']
#
#        map_idx2thr = np.array([columns_AD.index(str(int(tmc*10))+str(int(tmd*10))) \
#                       for tmc in thresholds_mc for tmd in thresholds_md])
#
#        extra_ts = len(params_combine)
#
#        weights_list = [np.zeros((model.seq_len+1,len(columns_AD),1)) for i in range(extra_ts)]
        #print("weights_list")
        #print(weights_list)
    m = y.shape[0]
    n_days = len(dateTest)
#    granularity = 1/resolution
    if 'cost_name' in config:
        cost_name = config['cost_name']
    else:
        cost_name = IDweights
    if keep_old_cost:
        J_train = costs[cost_name+str(epoch)]#
    else:
        J_train = costs[str(epoch)]#cost_name+
    # cum results per t_index and mc/md combination
    CR = [[[None for md in thresholds_md] for mc in thresholds_mc] for t in range(seq_len+1)]
    # single results (results per MCxMD)
    #SR = [[[None for md in thresholds_md] for mc in thresholds_mc] for t in range(model.seq_len+1)]
    # to acces CR: CR[t][mc][md]
    # save fuction cost
    save_costs(costs_filename, [epoch, J_train, J_test])
    print("Epoch "+str(epoch)+", J_train = "+str(J_train)+", J_test = "+str(J_test))
    # loop over t_indexes
    tic = time.time()
    
    for t_index in t_indexes:
        # init results dictionary
        thr_idx = 0
        if t_index>=seq_len:
            t_str = params_combine[0]['alg']
            # get MRC from all indexes
            weights_id = t_index-seq_len
#            print("weights_id")
#            print(weights_id)
            t_soft_tilde = combine_ts_fn(seq_len, soft_tilde, weights_list[weights_id], 
                                      map_idx2thr, thresholds_mc, thresholds_md)
        else:
            t_soft_tilde = soft_tilde[:,t_index,:]
            t_y = y[:,t_index,:]
            t_str = str(t_index)
            
        # loop over market change thresholds
        for mc in range(len(thresholds_mc)):
            thr_mc = thresholds_mc[mc]
            # upper bound
            ub_mc = 1#thr_mc+granularity
            ys_mc = get_mc_vectors(t_y, t_soft_tilde, thr_mc, ub_mc)
            # loop over market direction thresholds
            for md in range(len(thresholds_md)):
                thr_md = thresholds_md[md]
                results = init_results_struct(epoch, thr_mc, thr_md, t_str)
                # upper bound
                ub_md = 1#thr_md+granularity
                ys_md = get_md_vectors(t_soft_tilde, t_y, ys_mc, bits_mg, thr_md, ub_md)
                # extract DTA structure for t_index
                if from_var == True:
                    DTAt = DTA.iloc[:,:]
                else:
                    DTAt = DTA.iloc[::seq_len,:]
                # get journal
                Journal = get_journal(DTAt.iloc[ys_md['y_md_tilde']], 
                                      ys_md['y_dec_mg_tilde'], 
                                      ys_md['y_dec_mg'],
                                      ys_md['diff_y_y_tilde'], 
                                      ys_mc['probs_mc'][ys_md['y_md_tilde']], 
                                      ys_md['probs_md'])
                # Filter out positions with non-tackled direction
                Journal = filter_journal(config, Journal)
                ## calculate KPIs
                results = get_basic_results_struct(ys_mc, ys_md, results, m)
                results['tGROI'] = Journal['GROI'].sum()
                results['tROI'] = Journal['ROI'].sum()
                # init positions dir and filename
                if save_journal:
                    pos_dirname = resultsDir+IDresults+'/positions/'
                    pos_filename = 'P_E'+str(epoch)+'TI'+t_str+'MC'+str(thr_mc)+'MD'+str(thr_md)+'.csv'
                else:
                    pos_dirname = ''
                    pos_filename = ''
                # get results with extensions
                res_ext, log = get_extended_results(Journal,
                                                    bits_mg,
                                                    n_days, get_positions=save_journal,
                                                    pNZA=results['pNZA'],
                                                    pos_dirname=pos_dirname,
                                                    pos_filename=pos_filename)
                results.update(res_ext)
                # combine ts
                if if_combine:
                    
                    for i, params in enumerate(params_combine):
                        weights_list[i] = update_weights_combine(weights_list[i], 
                                    t_index, map_idx2thr[thr_idx], params, results)
                    thr_idx += 1
                # update cumm results list
                CR[t_index][mc][md] = results
                # print results
                print_results(results, epoch, J_test, J_train, thr_md, thr_mc, t_str)
                # save results
                save_results_fn(results_filename, results)
                # save journal
                if save_journal and (thr_mc>.5 or thr_md>.5):
                    journal_dir = resultsDir+IDresults+'/journal/'
                    journal_id = 'J_E'+str(epoch)+'TI'+t_str+'MC'+str(thr_mc)+'MD'+str(thr_md)
                    ext = '.csv'
                    save_journal_fn(Journal, journal_dir, journal_id, ext)
                
            # end of for thr_md in thresholds_md:
            print('')
        # end of for thr_mc in thresholds_mc:
    # end of for t_index in range(model.seq_len+1):
    # extract best result this epoch
    if resolution>0:
        TR = pd.read_csv(results_filename+'.csv', sep='\t')
        #TR = TR[TR.epoch==epoch]
        get_best_results(TR[TR.epoch==epoch], results_filename, 
                         resultsDir, IDresults, save=1)
        print("\nThe very best:")
        get_best_results(TR, results_filename, resultsDir, IDresults)
    # get results per MCxMD entry
#    for t_index in range(model.seq_len+1):
#        for mc in thresholds_mc:
#            for md in thresholds_md:
#                SR[t_index][mc][md] = get_single_result(CR[t_index], mc, md, 
#                                                        thresholds_mc, 
#                                                        thresholds_md)
    print("Time={0:.2f}".format(time.time()-tic)+" secs")
    return None

def get_journal(DTA, y_dec_tilde, y_dec, diff, probs_mc, probs_md):
    """
    Calculates trading journal given predictions.
    Args:
        - DTA: DataFrame containing Entry and Exit times, Bid and Ask.
        - y_dec_tilde: estimated output in decimal values.
        - y_dec: real output in decimal values.
        - diff: vector difference between real and estimated.
        - probs: probabilties vector of estimates.
    Returns:
        - Journal: DataFrame with info of each transaction.
    """
    #print("Getting journal...")
#    columns = ['DTi','DTo','Bi','Bo','Ai','Ao',
#                                    'Spread','GROI','ROI','Bet','Outcome',
#                                    'Diff','P_mc','P_md']
    Journal = pd.DataFrame()
    #Journal = DTA
    
    
    
    long_pos = y_dec_tilde>0
    short_pos = y_dec_tilde<0
    #grossROIs = np.sign(y_dec_tilde)*(DTA["B2"]-DTA["B1"])/DTA["A2"]
    grossROIs = np.zeros((y_dec_tilde.shape))
    LgrossROIs = np.zeros((y_dec_tilde.shape))
    SgrossROIs = np.zeros((y_dec_tilde.shape))
    tROIs = np.zeros((y_dec_tilde.shape))
    LtROIs = np.zeros((y_dec_tilde.shape))
    StROIs = np.zeros((y_dec_tilde.shape))
    LgrossROIs = (DTA["A2"]-DTA["A1"])/DTA["A1"]
    SgrossROIs = (DTA["B1"]-DTA["B2"])/DTA["A2"]
    grossROIs[long_pos] = LgrossROIs[long_pos]
    grossROIs[short_pos] = SgrossROIs[short_pos]
    LtROIs = (DTA["B2"]-DTA["A1"])/DTA["A1"]
    StROIs = (DTA["B1"]-DTA["A2"])/DTA["A2"]
    tROIs[long_pos] = LtROIs[long_pos]
    tROIs[short_pos] = StROIs[short_pos]
    espreads =  (DTA["A1"]-DTA["B1"])/DTA["A1"]
    spreads =  grossROIs-tROIs
    
    Journal['Asset'] = DTA['Asset'].iloc[:]
    Journal['DTi'] = DTA['DT1'].iloc[:]
    Journal['DTo'] = DTA['DT2'].iloc[:]
    Journal['GROI'] = 100*grossROIs
    Journal['ROI'] = 100*tROIs
    Journal['Spread'] = 100*spreads
    Journal['Espread'] = 100*espreads
    Journal['Bet'] = y_dec_tilde.astype(int)
    Journal['Outcome'] = y_dec.astype(int)
    Journal['Diff'] = diff.astype(int)
    Journal['Bi'] = DTA['B1'].iloc[:]
    Journal['Bo'] = DTA['B2'].iloc[:]
    Journal['Ai'] = DTA['A1'].iloc[:]
    Journal['Ao'] = DTA['A2'].iloc[:]
    Journal['P_mc'] = probs_mc
    Journal['P_md'] = probs_md
    
#    Journal['GROI'] = Journal['GROI'].astype(float)
#    Journal['Spread'] = Journal['Spread'].astype(float)
#    Journal['ROI'] = Journal['ROI'].astype(float)
    
    #Journal.index = range(Journal.shape[0])

    return Journal

def unroll_param(struct, param, name, extentions):
    """ unroll vectorian parameters and save in structure """
    for p in range(len(extentions)):
        struct[name+extentions[p]] = param[p]
    return struct

def unroll_dictionary(dictionary):
    """  """
    unrolled_dictionary = {}
    for k in dictionary.keys():
        if type(dictionary[k])==list:
            print("List length: "+str(len(dictionary[k]))+". Unrolling")
            unrolled_dictionary = unroll_param(unrolled_dictionary, dictionary[k], k, ['.'+str(i) for i in range(len(dictionary[k]))])
        else:
            print(k+" No list key. Add to unrolled dict. ")
            unrolled_dictionary[k] = dictionary[k]
    return unrolled_dictionary

def build_extended_res_struct(list_results):
#    (eGROI, eROI, eROIs, SI, SIs, sharpe, rROIxLevel, 
#                              rSampsXlevel, successes, mSpread)
    """  """
    res_w_ext = {}
    # list_results[i][0]: values, list_results[i][1]: name, list_results[i][2]:parameters
    for i in range(len(list_results)):
        if len(list_results[i])<3:
            res_w_ext.update({list_results[i][1]:list_results[i][0]})
        else:
            res_w_ext = unroll_param(res_w_ext, list_results[i][0], list_results[i][1], list_results[i][2])
    
    return res_w_ext

#def ma_pos2list():
#    """  """
#    this_list = [Journal['Asset'].iloc[e-1], Journal[DT1].iloc[eInit][:10],
#                 Journal[DT1].iloc[eInit][11:], Journal[DT2].iloc[e-1][:10],
#                 Journal[DT2].iloc[e-1][11:],100*GROI,100*ROI,
#                 100*thisSpread,this_pos_extended,direction,
#                 Bi,Bo,Ai,Ao]
#    return this_list
    
def remove_outliers(grois, spreads, thr=.99):
    """ Remove outliers from GROIs """
    n_positions = len(grois)
    high_thr_pos = int(np.ceil(n_positions*thr))
    low_thr_pos = int(np.ceil(n_positions*(1-thr)))
    if high_thr_pos-low_thr_pos>1:
        grois_no_outliers = np.sort(grois)[low_thr_pos:high_thr_pos+1]
        low_arg_goi = grois_no_outliers[0]
        high_arg_goi = grois_no_outliers[-1]
        idx_sorted = np.argsort(grois)
        spreads_sorted = spreads[idx_sorted]
        rois_no_outliers = grois_no_outliers-spreads_sorted[low_thr_pos:high_thr_pos+1]
    else:
        grois_no_outliers = []
        low_arg_goi = 0
        high_arg_goi = 0
        idx_sorted = []
        spreads_sorted = []
        rois_no_outliers = []
        
    return grois_no_outliers, rois_no_outliers, idx_sorted, low_arg_goi, high_arg_goi

def get_extended_results(Journal, n_classes, n_days, get_log=False, 
                         get_positions=False, pNZA=0,
                         pos_dirname='', pos_filename='', reference_date='2018.03.09',
                         end_date='2018.11.09 23:59:59', get_corr_signal=False,
                         corr_filename='', corr_dirname=''):
    """
    Function that calculates real ROI, GROI, spread...
    """
    print('Getting extended results...')
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    
    DT1 = 'DTi'
    DT2 = 'DTo'
    A2 = 'Ao'
    A1 = 'Ai'
    B1 = 'Bi'
    B2 = 'Bo'
    ref_date_dt = dt.datetime.strptime(reference_date,"%Y.%m.%d")
    log = pd.DataFrame(columns=['DateTime','Message'])
    # init positions
    if get_positions:
        columns_positions = ['Asset','Di','Ti','Do','To','GROI','ROI','spread',
                             'espread','ext','Dir','Bi','Bo','Ai','Ao']
        
        if not os.path.exists(pos_dirname):
            os.makedirs(pos_dirname)
        #if not os.path.exists(positions_dir+positions_filename):
        success = 0
        while not success:
            try:
                pd.DataFrame(columns=columns_positions).to_csv(pos_dirname+
                        pos_filename, mode="w",index=False,sep='\t')
                success = 1
            except PermissionError:
                print("WARNING! PermissionError. Close programs using "+pos_dirname+
                        pos_filename)
                time.sleep(1)

        
    # Add GROI and ROI with real spreads
    eGROI = 0.0
    eGROIL = 0.0
    eGROIS = 0.0
    eROI = 0.0
    mSpread = 0.0
    n_pos_opned = 1
    NOL = 0
    NOS = 0
    GSCl = 0 # gross succes counter long
    GSCs = 0
    n_pos_extended = 0
    gross_succ_counter = 0
    net_succ_counter = 0
    this_pos_extended = 0

    rROIxLevel = np.zeros((int(n_classes-1),3))
    rSampsXlevel = np.zeros((int(n_classes-1),2))
        
    fixed_spread_ratios = np.array([0.00005,0.0001,0.00015,0.0002,0.0003,0.0004,0.0005])
    fixed_extensions = ['.5','1','1.5','2','3','4','5']
    # fixed ratio success percent
    NSPs = np.zeros((fixed_spread_ratios.shape[0]))
    eROIs = np.zeros((fixed_spread_ratios.shape))
    ROI_vector = np.array([])
    GROI_vector = np.array([])
    spreads = np.array([])
    avGROI = 0.0 # average GROI for all trades happening concurrently and for the
               # same asset
#    if Journal.shape[0]>0:
#        log=log.append({'DateTime':Journal[DT1].iloc[0],
#                        'Message':Journal['Asset'].iloc[0]+" open" },
#                        ignore_index=True)
    eInit = 0
    if get_corr_signal and Journal.shape[0]>0:
        n_secs = int((dt.datetime.strptime(end_date,"%Y.%m.%d %H:%M:%S")-ref_date_dt).total_seconds())
        corr_signal = np.zeros((n_secs))+1j*np.zeros((n_secs))
        secInit = int((dt.datetime.strptime(Journal[DT1].iloc[eInit],"%Y.%m.%d %H:%M:%S")-ref_date_dt).total_seconds())
    e = 0
    if get_positions:
        list_pos = [[] for i in columns_positions]#[None for i in range(500)]
    # skip loop if both thresholds are .5
    if pNZA<10:
        end_of_loop = Journal.shape[0]
    else:
        end_of_loop = 0
    count_dif_dir = 0
    for e in tqdm(range(1,end_of_loop),mininterval=1):

        oldExitTime = dt.datetime.strptime(Journal[DT2].iloc[e-1],"%Y.%m.%d %H:%M:%S")
        newEntryTime = dt.datetime.strptime(Journal[DT1].iloc[e],"%Y.%m.%d %H:%M:%S")
        
        extendExitMarket = (newEntryTime-oldExitTime<=dt.timedelta(0))
        sameAss = Journal['Asset'].iloc[e] == Journal['Asset'].iloc[e-1] 
        if sameAss and extendExitMarket:# and sameDir:
            if get_log:
                log=log.append({'DateTime':Journal[DT1].iloc[e],
                                'Message':Journal['Asset'].iloc[e]+
                                " extended" },ignore_index=True)
            n_pos_extended += 1
            this_pos_extended += 1
            avGROI += Journal['GROI'].iloc[e]
            level = int(np.abs(Journal['Bet'].iloc[eInit])-1)
            rSampsXlevel[level,0] += 1
            
            
        else:
            direction = np.sign(Journal['Bet'].iloc[eInit])
            Ao = Journal[A2].iloc[e-1]
            Ai = Journal[A1].iloc[eInit]
            Bo = Journal[B2].iloc[e-1]
            Bi = Journal[B1].iloc[eInit]
            if get_corr_signal:
                # update correlation signal
                secEnd = int((dt.datetime.strptime(Journal[DT2].iloc[e-1],"%Y.%m.%d %H:%M:%S")-ref_date_dt).total_seconds())
            if direction>0:
                GROI = (Ao-Ai)/Ai#(Ao-Ai)/Ai
                ROI = (Bo-Ai)/Ai
                #GROI = (Bo-Bi)/Ai
                eGROIL += GROI
                NOL += 1
                if get_corr_signal:
                    corr_signal[secInit:secEnd] = corr_signal[secInit:secEnd]+1
            else:
                GROI = (Bi-Bo)/Ao
                ROI = (Bi-Ao)/Ao
                eGROIS += GROI
                NOS += 1
                if get_corr_signal:
                    corr_signal[secInit:secEnd] = corr_signal[secInit:secEnd]+1j
            thisSpread = GROI-ROI
            e_spread = (Journal[A1].iloc[eInit]-Journal[B1].iloc[eInit])/Journal[A1].iloc[eInit]
            if np.sign(Ao-Ai)!=np.sign(Bo-Bi):
                count_dif_dir += 1
            ### TEMP ###
#            GROI = -direction*(Bi-Bo)/Ao
#            thisSpread = (Ao-Bo)/Ao
#            ROI = GROI-thisSpread#-direction*(Bi-Ao)/Ao
#            if direction>0:
#                eGROIL += GROI
#                NOL += 1
#            else:
#                eGROIS += GROI
#                NOS += 1
            #thisSpread = (Journal[A2].iloc[e-1]-Journal[B2].iloc[e-1])/Journal[B1].iloc[e-1]  
            mSpread += thisSpread
            #GROI = np.sign(Journal['Bet'].iloc[eInit])*(Journal[B2].iloc[e-1]-Journal[B1].iloc[eInit])/Journal[B1].iloc[eInit]
            eGROI += GROI
            avGROI += Journal['GROI'].iloc[e]
            #ROI = GROI-thisSpread
            
            ROI_vector = np.append(ROI_vector,ROI)
            GROI_vector = np.append(GROI_vector,GROI)
            spreads = np.append(spreads,thisSpread)
            eROI += ROI
            eROIs = eROIs+GROI-fixed_spread_ratios
            level = int(np.abs(Journal['Bet'].iloc[eInit])-1)
            rROIxLevel[level,0] += 100*ROI
            rROIxLevel[level,1] += 100*GROI
            if this_pos_extended>0:
                rROIxLevel[level,2] += avGROI/this_pos_extended
            rSampsXlevel[level,0] += 1
            rSampsXlevel[level,1] += 1
            avGROI = 0.0
            
            
            if GROI>0:
                gross_succ_counter += 1
            if ROI>0:
                net_succ_counter += 1
            if GROI>0 and direction>0:
                GSCl += 1
            elif GROI>0 and direction<0:
                GSCs += 1
            NSPs = NSPs+((GROI-fixed_spread_ratios)>0)
            if get_positions:
                this_list = [Journal['Asset'].iloc[e-1],Journal[DT1].iloc[eInit][:10],
                                 Journal[DT1].iloc[eInit][11:],Journal[DT2].iloc[e-1][:10],
                                 Journal[DT2].iloc[e-1][11:],100*GROI,100*ROI,
                                 100*thisSpread,100*e_spread,this_pos_extended,direction,
                                 Bi,Bo,Ai,Ao]
                #print(this_list)
                for l in range(len(this_list)):
                    list_pos[l].append(this_list[l])
                
                #print(df.to_string())
            if get_log:
                log=log.append({'DateTime':Journal[DT2].iloc[e-1],'Message':" Close "+
                                Journal['Asset'].iloc[e-1]+
                                " entry bid {0:.4f}".format(Journal[B1].iloc[eInit])+
                                " exit bid {0:.4f}".format(Journal[B2].iloc[e-1])+
                                " GROI {0:.4f}% ".format(100*GROI)+
                                " ROI {0:.4f}% ".format(100*ROI)+
                                " tGROI {0:.4f}% ".format(100*eGROI)},
                                ignore_index=True)
                # WARNING!! Might be e-1. To be checked!!
                log=log.append({'DateTime':Journal[DT1].iloc[e],
                                'Message':Journal['Asset'].iloc[e]+
                                " open" },ignore_index=True)
            

            n_pos_opned += 1
            this_pos_extended = 0
            eInit = e
            if get_corr_signal:
                secInit = int((dt.datetime.strptime(Journal[DT1].iloc[eInit],"%Y.%m.%d %H:%M:%S")-ref_date_dt).total_seconds())
        # end of if (sameAss and extendExitMarket):
    # end of for e in range(1,Journal.shape[0]):
    
    if end_of_loop>0:
        direction = np.sign(Journal['Bet'].iloc[eInit])
        Ao = Journal[A2].iloc[-1]
        Ai = Journal[A1].iloc[eInit]
        Bo = Journal[B2].iloc[-1]
        Bi = Journal[B1].iloc[eInit]
        if get_corr_signal:
            secEnd = int((dt.datetime.strptime(Journal[DT2].iloc[-1],"%Y.%m.%d %H:%M:%S")-ref_date_dt).total_seconds())
        if direction>0:
            GROI = (Ao-Ai)/Ai#(Ao-Ai)/Ai
            ROI = (Bo-Ai)/Ai
            eGROIL += GROI
            NOL += 1
            if get_corr_signal:
                corr_signal[secInit:secEnd] = corr_signal[secInit:secEnd]+1
        else:
            GROI = (Bi-Bo)/Ao
            ROI = (Bi-Ao)/Ao
            eGROIS += GROI
            NOS += 1
            if get_corr_signal:
                corr_signal[secInit:secEnd] = corr_signal[secInit:secEnd]+1j
        if np.sign(Ao-Ai)!=np.sign(Bo-Bi):
            count_dif_dir += 1
        thisSpread = GROI-ROI
        e_spread = (Journal[A1].iloc[eInit]-Journal[B1].iloc[eInit])/Journal[A1].iloc[eInit]
        #thisSpread = (Journal[A2].iloc[-1]-Journal[B2].iloc[-1])/Journal[B1].iloc[-1]
        mSpread += thisSpread
        #GROI = np.sign(Journal['Bet'].iloc[eInit])*(Journal[B2].iloc[-1]-Journal[B1
        #            ].iloc[eInit])/Journal[B1].iloc[-1]
        eGROI += GROI
        avGROI += Journal['GROI'].iloc[-1]
        
        ROI = GROI-thisSpread
        ROI_vector = np.append(ROI_vector,ROI)
        GROI_vector = np.append(GROI_vector,GROI)
        spreads = np.append(spreads,thisSpread)
        eROI += ROI
        eROIs = eROIs+GROI-fixed_spread_ratios
        level = int(np.abs(Journal['Bet'].iloc[eInit])-1)
        rROIxLevel[level,0] += 100*ROI
        rROIxLevel[level,1] += 100*GROI
        if this_pos_extended>0:
            rROIxLevel[level,2] += avGROI/this_pos_extended
        rSampsXlevel[level,0] += 1
        rSampsXlevel[level,1] += 1
        
        if GROI>0:
            gross_succ_counter += 1
        if ROI>0:
            net_succ_counter += 1
        if get_log:
            log = log.append({'DateTime':Journal[DT2].iloc[eInit],
                              'Message':Journal['Asset'].iloc[eInit]+
                              " close GROI {0:.4f}% ".format(100*GROI)+
                              " ROI {0:.4f}% ".format(100*ROI)+
                              " TGROI {0:.4f}% ".format(100*eGROI) },
                              ignore_index=True)
        if get_positions:
                this_list = [Journal['Asset'].iloc[e-1], Journal[DT1].iloc[eInit][:10],
                                 Journal[DT1].iloc[eInit][11:], Journal[DT2].iloc[e-1][:10],
                                 Journal[DT2].iloc[e-1][11:],100*GROI,100*ROI,
                                 100*thisSpread,100*e_spread,this_pos_extended,direction,
                                 Bi,Bo,Ai,Ao]
                for l in range(len(this_list)):
                    list_pos[l].append(this_list[l])
    
    if get_positions:
        dict_pos = {columns_positions[i]:list_pos[i] for i in range(len(columns_positions))}
        df = pd.DataFrame(dict_pos)\
            [pd.DataFrame(columns = columns_positions).columns.tolist()]
        #df['DTi'] = df["Di"] +" "+ df["Ti"]
        if df.shape[0]>0:
            df['DTo'] = df["Do"] +" "+ df["To"]
            df = df.sort_values(by=['DTo']).drop('DTo',1)
            success = 0
            while not success:
                try:
                    df.to_csv(pos_dirname+pos_filename, mode='a', 
                      header=False, index=False, sep='\t')
                    success = 1
                except PermissionError:
                    print("WARNING! PermissionError. Close programs using "+
                          pos_dirname+pos_filename)
                    time.sleep(1)
#    print("count_dif_dir")
#    print(count_dif_dir)
#    print("percent_dif_dir")
#    print(100*count_dif_dir/n_pos_opned)
    gross_succ_per = gross_succ_counter/n_pos_opned
    net_succ_per = net_succ_counter/n_pos_opned
    NSPs = NSPs/n_pos_opned
    successes = [n_pos_opned, 100*gross_succ_per, 100*net_succ_per, 100*NSPs]
    mSpread = 100*mSpread/n_pos_opned
    try:
        GSPl = GSCl/NOL
    except ZeroDivisionError:
        GSPl = 0
    try:
        GSPs = GSCs/NOS
    except ZeroDivisionError:
        GSPs = 0
    n_bets = ROI_vector.shape[0]
    if np.var(ROI_vector)>0:
        sharpe = np.sqrt(n_bets)*np.mean(ROI_vector)/np.sqrt(np.var(ROI_vector))
    else:
        sharpe = 0.0
    # Success index per spread level
    SIs = n_pos_opned*(NSPs-.53)
    SI = n_pos_opned*(net_succ_per-.53)
    eGROI = 100*eGROI
    eROI = 100*eROI
    eROIs = 100*eROIs
    GROIS99, ROIS99, _, _, _ = remove_outliers(GROI_vector, spreads, thr=.99)
    GROI99 = sum(GROIS99)
#    print("GROI99")
#    print(GROI99)
    ROI99 = sum(ROIS99)
    ROIS99 = [100*sum(GROIS99-spread) for spread in fixed_spread_ratios]
#    print("ROI99")
#    print(ROI99)
#    print("ROIS99")
#    print(ROIS99)
    list_ext_results = [[eGROI,'eGROI'], [eROI,'eROI'], [successes[1],'GSP'], \
                        [successes[2],'NSP'], [successes[0],'NO'], [sharpe,'sharpe'], \
                        [SI,'SI'], [mSpread,'mSpread'], [rROIxLevel[:,0], 'eRl', ['1','2']], \
                        [rROIxLevel[:,1], 'eGl', ['1','2']], [rSampsXlevel[:,1], 'NOl', ['1','2']], \
                        [eROIs, 'eROI', fixed_extensions], \
                        [successes[3], 'NSP', fixed_extensions], \
                        [SIs, 'SI', fixed_extensions], [100*eGROIL, 'eGROIL'], \
                        [100*eGROIS, 'eGROIS'], [NOL, 'NOL'], [NOS, 'NOS'], \
                        [100*GSPl,'GSPl'],[100*GSPs,'GSPs'], \
                        [100*GROI99,'99eGROI'],[100*ROI99,'99eROI'],[ROIS99,'99eROI', fixed_extensions]]
#    print("list_ext_results")
#    print(list_ext_results)
    res_w_ext = build_extended_res_struct(list_ext_results)
#    plt.figure(np.random.randint(1000))
#    plt.plot(np.real(corr_signal))
#    plt.plot(np.imag(corr_signal))
#    plt.figure(np.random.randint(1000))
#    plt.plot(np.abs(corr_signal))
    if get_corr_signal and Journal.shape[0]>0:
        if not os.path.exists(corr_dirname):
            os.makedirs(corr_dirname)
        pickle.dump(corr_signal, open( corr_dirname+corr_filename, "wb" ))
    return res_w_ext, log

def print_real_ROI(Journal, n_days, fixed_spread=0, mc_thr=.5, md_thr=.5, spread_thr=1):
    """
    Function that calls get real ROI and prints out results
    """
    Journal = Journal.groupby(["Asset"]).apply(lambda x: x.sort_values(["Entry Time"], ascending = True)).reset_index(drop=True)
    # get positions with P>thrs
    if type(mc_thr)==list and len(mc_thr)==2:
        #print(J.P_md<=md_thr[1])
        Journal = Journal[(Journal.P_mc>mc_thr[0]) & (Journal.P_mc<=mc_thr[1]) & (Journal.P_md>md_thr[0]) & (Journal.P_md<=md_thr[1]) & (Journal.Spread<spread_thr)]
    else:
        Journal = Journal[(Journal.P_mc>mc_thr) & (Journal.P_md>md_thr) & (Journal.Spread<spread_thr)]
            
    rGROI, rROI, fROIs, sharpe_ratio, rROIxLevel, rSampsXlevel, log, varRet, successes = get_real_ROI_old(5, Journal, n_days, fixed_spread=fixed_spread)
    

    print("Sharpe = {0:.3f} rRl1 = {1:.4f}% rRl2 = {2:.4f}% rGROI = {3:.4f}% rROI = {4:.4f}%".format(
            sharpe_ratio,rROIxLevel[0,1],rROIxLevel[0,0],100*rGROI,100*rROI))
    
    return None

def get_summary_journal(mc_thr, md_thr, spread_thr, dir_file, print_table=False, fixed_spread=0):
    """
    
    """
    J=pd.read_csv(dir_file, sep='\t')
    
    if type(mc_thr)==list and len(mc_thr)==2:
        #print(J.P_md<=md_thr[1])
        nzs = J[(J.P_mc>mc_thr[0]) & (J.P_mc<=mc_thr[1]) & (J.P_md>md_thr[0]) & (J.P_md<=md_thr[1]) & (J.Spread<spread_thr)]
    else:
        nzs = J[(J.P_mc>mc_thr) & (J.P_md>md_thr) & (J.Spread<spread_thr)]
        
    if print_table:
        print(nzs.to_string())
    NZA=nzs.shape[0]
    RD=nzs[nzs.Diff==0].shape[0]
    NZ=nzs[nzs.Diff!=1].shape[0]
    tGROI=nzs.GROI.sum()
    if not fixed_spread:
        tROI=nzs.ROI.sum()
        RDTA = nzs[nzs.ROI>0].shape[0]
    else:
        tROI=tGROI-nzs.shape[0]*0.02
        RDTA = nzs[nzs.GROI-0.02>0].shape[0]
    RDT = nzs[nzs.GROI>0].shape[0]
    
    if NZ>0:
        print("RD={0:d}".format(RD)+" NZ={0:d}".format(NZ)+" NZA={0:d}".format(NZA)+" per RD={0:.2f}".format(100*RD/NZ)+"% per RDA={0:.2f}".format(100*RD/NZA)
        +"% per gross success={0:.2f}%".format(100*RDT/NZA)+" per nett success={0:.2f}%".format(100*RDTA/NZA)+" GROI={0:.2f}%".format(tGROI)+" ROI={0:.2f}%".format(tROI))
    else:
        print("RD="+str(RD)+" NZ="+str(NZ)+" NZA="+str(NZA)+" perNZ="+str(0)+"% perNZA="+str(0)+"% GROI="+str(tGROI)+" ROI="+str(tROI))
        
    print_real_ROI(J, 8, fixed_spread=fixed_spread, mc_thr=mc_thr, md_thr=md_thr,spread_thr=spread_thr)
    
    return None

def print_GRE(dir_origin, IDr, epoch):
    """ Print lower and upper bound GRE matrices """
    pip = 0.0001
    GRElb, GREub, NZs = load_GRE(dir_origin, IDr, epoch)
    seq_len = GRElb.shape[0]
    levels = GRElb.shape[3]
    for l in range(levels):
        for t in range(seq_len):
            print("t="+str(t)+" GRElb level "+str(l)+":")
            print(GRElb[t,:,:,l]/pip)
    for l in range(levels):
        for t in range(seq_len):
            print("t="+str(t)+" GREub level "+str(l)+":")
            print(GREub[t,:,:,l]/pip)
    print(NZs)
    return None

def merge_best_result_files(results_dir, IDr_m1, IDr_m2):
    """ Merge files for best results.
    Args:
        - results_dir (string): results directory. 
    Return:
        - merged_df (pandas DataFrame): table containing merged results. """
    best_cases = ['GROI','ROI','sharpe']
    
    # loop over best cases
    for case in best_cases:
        filename1 = IDr_m1+'_BR_'+case+'.txt'
        #filename2 = IDr_m2+'_BR_'+case+'.txt'
        
        df1 = pd.read_csv(results_dir+IDr_m1+'/'+filename1, sep='\t')
        #df2 = pd.read_csv(results_dir+IDr_m2+'/'+filename2,sep='\t')
        
#        print(df1)
#        print(df2)
        
        # find common epochs
        # TODO. Assume common epochs for now
        epochs = df1.epoch.iloc[:]
        # init results dict
        merged_df = pd.DataFrame()
        i = 0
        for epoch in epochs:
            merged_df = merged_df.append(pd.DataFrame(init_results_dict(), index=[i]))
            merged_df.epoch.iloc[-1] = epoch
            i += 1
        print(merged_df.to_string())
        #
        
    return merged_df

def merge_t_index_results(results_dir, IDr_m1, IDr_m2):
    """ Merge results of one t_index """
    t_index = 0
    t_index_dir1 = results_dir+IDr_m1+'/t'+str(t_index)+'/'
    t_index_dir2 = results_dir+IDr_m2+'/t'+str(t_index)+'/'
    while os.path.exists(t_index_dir1) and os.path.exists(t_index_dir2):
        #print(t_index_dir1)
        filename1 = t_index_dir1+IDr_m1+'t'+str(t_index)+'_results.csv'
        filename2 = t_index_dir2+IDr_m2+'t'+str(t_index)+'_results.csv'
        if os.path.exists(filename1) and os.path.exists(filename2):
#            print(filename1)
#            print(filename2)
            df1 = pd.read_csv(filename1, sep='\t')
            df2 = pd.read_csv(filename2,sep='\t')
#            print(df1)
#            print(df2)
            # find unique epoch values
            unique_epochs_df1 = df1.epoch.unique()
            unique_epochs_df2 = df2.epoch.unique()
            #print(unique_epochs_df1)
            #print(unique_epochs_df2)
            intersect = np.intersect1d(unique_epochs_df1, unique_epochs_df2)
            print(intersect)
            # find intersection between common values
            
            # combine columns for shared epochs
            
            
        t_index += 1
        t_index_dir1 = results_dir+IDr_m1+'/t'+str(t_index)+'/'
        t_index_dir2 = results_dir+IDr_m2+'/t'+str(t_index)+'/'
    if os.path.exists(t_index_dir1) or os.path.exists(t_index_dir2):
        raise ValueError("merge 1 and 2 must have same number of t_indexes")
    return None

def merge_results(IDrs, IDr_merged, from_mg=False):
    """ Merge results from two different test executions.
    Arg:
        - IDr_m1 (string): ID of first merging results
        - IDr_m2 (string): ID of second merging results
        - IDr_merged (string): ID of merged results
    Return:
        - None """
    from tqdm import tqdm
    results_dir = local_vars.results_directory
    
    columns = get_performance_entries()
    kpi2mergefunc = kpi2func_merge()
    resultsDir = local_vars.results_directory
#    TRT = pd.DataFrame(columns = columns)
        #[pd.DataFrame(columns = columns).columns.tolist()]
    for i, ID in enumerate(IDrs):
        print("Merging "+ID+"...")
        performance_filename = resultsDir+ID+'/performance'
        TP = pd.read_csv(performance_filename+'.csv', sep='\t')
        if i==0:
            TRT = TP
        # TODO: check if number of epochs is thhe same from all results
        epochs = TP['epoch'].unique()
        #t_indexes = TP['t_index'].unique()
        #thrs = TP['thr_mg'].unique()
        
#        for epoch in epochs:
#            for t_index in t_indexes:
#                for thr in thrs:
        #for index in tqdm(range(TRT.shape[0])):
        #print('index '+str(index)+" of "+str(TRT.shape[0]))
        for kpi, func in kpi2mergefunc.items():
            #print(kpi)
            #print(func)
            if kpi in TP:
                if func=='none':
                    TRT[kpi].iloc[:] = TP[kpi].iloc[:]
                elif func=='sum':
                   if i>0:
                       TRT[kpi].iloc[:] = TRT[kpi].iloc[:]+TP[kpi].iloc[:]
                elif func=='mean':
                   if i>0:
                       TRT[kpi].iloc[:] = TRT[kpi].iloc[:]+TP[kpi].iloc[:]
                       # last index: devide
                       if i==len(IDrs)-1:
                           TRT[kpi].iloc[:] = TRT[kpi].iloc[:]/len(IDrs)
#        for epoch in epochs:
#            for t in t_indexes:
    dir_merged = results_dir+IDr_merged+'/'
    if not os.path.exists(dir_merged):
        os.makedirs(dir_merged)
    #TRT = TRT.replace('nan', 0)
    TRT.to_csv(dir_merged+'performance.csv',sep='\t', index=False)
    for epoch in epochs:
        print('Epoch '+str(epoch))
        get_best_results(TRT[TRT.epoch==epoch], '', 
                         resultsDir,
                         IDr_merged, save=1, from_mg=from_mg)
        print('\n')
    print("\nThe very best:")
    get_best_results(TRT, '', resultsDir, IDr_merged, from_mg=from_mg)
    print('Results MERGED!')
    #merge_t_index_results(results_dir, IDr_m1, IDr_m2)
    
    return None

def merge_GREs(dir_origin, dir_destiny, gre_id, list_IDrs, epoch):
    """ Merge GREs to create a new one with statistics from all IDs.
    Arg: 
        - list_IDrs (list of strings): list with IDresult names of GREs to merge 
    Return:
        - GREs_merged (list of numpy matrix): merged GRE matrix """
    # list of lists containing GRElb and GRE lb and non-zeros counter
    list_GREs = [load_GRE(dir_origin, list_IDrs[i], epoch) 
                 for i in range(len(list_IDrs))]
    
    GRElbs = np.array([gre[0] for gre in list_GREs])
    GREubs = np.array([gre[1] for gre in list_GREs])
    NOs = np.array([gre[2] for gre in list_GREs])
    NO = sum(NOs)
    weights = np.nan_to_num(NOs/NO)
    
    GRElb = sum(weights*GRElbs)
    GREub = sum(weights*GREubs)
    
    pickle.dump([GRElb,GREub,NO], open( dir_destiny+gre_id+str(epoch)+".p", "wb" ))
    
    return list_GREs

def fillup_GRE(GRE):
    """ Fill up unknows in the GRE matrix """
    # TODO: implement linear regression to fill up the gaps
    thresholds_mc = [.5,.6,.7,.8,.9]
    thresholds_md = [.5,.6,.7,.8,.9]
    for t in range(GRE.shape[0]):
        for idx_mc in range(len(thresholds_mc)):
            min_md = GRE[t,idx_mc,:,:]
            for idx_md in range(len(thresholds_md)):
                min_mc = GRE[t,:,idx_md,:]    
                for l in range(int((5-1)/2)):
                    if GRE[t,idx_mc,idx_md,l]==0:
                        if idx_md==0:
                            if idx_mc==0:
                                if l==0:
                                    # all zeros, nothing to do
                                    pass
                                else:
                                    GRE[t,idx_mc,idx_md,l] = max(min_mc[idx_mc,:l])
                            else:
                                GRE[t,idx_mc,idx_md,l] = max(min_mc[:idx_mc,l])
                        else:
                            GRE[t,idx_mc,idx_md,l] = max(min_md[:idx_md,l])
    return GRE

def load_GRE(dir_origin, IDr, epoch):
    """ Load lower- and upper bound GREs from Disk. Return a list with
    both GRE matrices """
    # load GRE lower bound
    GRElb = pickle.load( open( dir_origin+IDr+
                                "/GRE_e"+str(epoch)+".p", "rb" ))
    # load GRE upper bound
    GREub = pickle.load( open( dir_origin+IDr+
                                "/GREex_e"+str(epoch)+".p", "rb" ))
    # load non-zero counter
    NZs = pickle.load( open( dir_origin+IDr+
                                "/NZpp_e"+str(epoch)+".p", "rb" ))
    return [GRElb, GREub, NZs]

def print_cost(IDw, epochs):
    """ Print cost value of weights IDw for some epochs.
    Args:
        - IDw (string): ID of weights.
        - epochs (list): list of ints representing the epochs. If list is empty, 
          the print all epochs """
    costs = pickle.load( open( "../RNN/weights/"+IDw+"/cost.p", "rb" ))
    
    print(costs)
    return None