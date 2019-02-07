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
import sqlite3
import time
#import h5py
#import matplotlib.pyplot as plt
import scipy.io as sio

from kaissandra.local_config import local_vars

def get_last_saved_epoch2(resultsDir, ID, t_index):
    """
    <DocString>
    """
    filename = resultsDir+ID+"/results.csv"
    #print(filename)
    
    if os.path.exists(filename):
        TR = pd.read_csv(filename,sep='\t')
        try:
            last_saved_epoch = TR.epoch.iloc[-1]
        except IndexError:
            last_saved_epoch = 0
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

def get_md_vectors(t_soft_tilde, t_y, ys_mc, size_output_layer, thr_md, ub_md):
    """  """
    y_dec_md_tilde = np.argmax(t_soft_tilde[:,1:3], 1)-1# predicted dec out
    y_md_down_tilde = ys_mc['y_mc_tilde'] & (t_soft_tilde[:,1]>thr_md) & (t_soft_tilde[:,1]<=ub_md)
    y_md_up_tilde = ys_mc['y_mc_tilde'] & (t_soft_tilde[:,2]>=thr_md) & (t_soft_tilde[:,2]<=ub_md)
    y_md_tilde = y_md_down_tilde | y_md_up_tilde
    y_dec_md_tilde = y_dec_md_tilde-(y_dec_md_tilde-1)*(-1)+2
    y_dec_md = np.argmax(t_y[:,3:], 1)-(size_output_layer-1)/2 # real output in decimal
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
             'y_dec_mg':np.argmax(t_y[y_md_tilde,3:], 1)-(size_output_layer-1)/2,
             'y_dec_mg_tilde':np.argmax(t_soft_tilde[y_md_tilde,3:], 1)-(size_output_layer-1)/2,
             # difference between y and y_tilde {0=no error, 1=error in mc. 2=error in md}
             'diff_y_y_tilde':np.abs(np.sign(y_dec_md_tilde[y_md_tilde])-np.sign(y_dec_md[y_md_tilde])),
             'probs_md':np.maximum(t_soft_tilde[y_md_tilde,2],t_soft_tilde[y_md_tilde,1])
            }
    
    return ys_md

def print_results(results, epoch, J_test, J_train, thr_md, thr_mc, t_index):
    """  """
    print("Epoch = "+str(epoch)+". Time index = "+str(t_index)+
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
                       'NZ','NZA','RD','NSP.5','NSP1','NSP2','NSP3',
                       'NSP4','NSP5','SI.5','SI1','SI2','SI3','SI4','SI5','SI',
                       'eGROI','eROI.5','eROI1','eROI2','eROI3','eROI4',
                       'eROI5','eROI','mSpread','pNZ','pNZA','tGROI','tROI','eRl1',
                       'eRl2','eGl1','eGl2','sharpe','NOl1','NOl2','eGROIL','eGROIS','NOL','NOS']
    # GRL: GROI for Long positions
    # GRS: GROI for short positions
    # NOL: Number of long openings
    # NOS: umber of short openings
    return results_entries

def get_costs_entries():
    """  """
    return ['epoch','J_train','J_test']
    
def init_results_dir(resultsDir, IDresults):
    """  """
    filedir = resultsDir+IDresults+'/'
    results_filename = resultsDir+IDresults+'/results'
    costs_filename = resultsDir+IDresults+'/costs'
    if not os.path.exists(filedir):
        os.makedirs(filedir)
    if not os.path.exists(results_filename+'.csv'):
        pd.DataFrame(columns = get_results_entries()).to_csv(results_filename+'.csv', 
                            mode="w",index=False,sep='\t')
        pd.DataFrame(columns = get_results_entries()).to_csv(results_filename+'.txt', 
                            mode="w",index=False,sep='\t')
    if not os.path.exists(costs_filename+'.csv'):
        pd.DataFrame(columns = get_costs_entries()).to_csv(costs_filename+'.csv', 
                            mode="w",index=False,sep='\t')
        
    return results_filename, costs_filename

def init_results_struct(epoch, thr_mc, thr_md, t_index):
    """  """
    results = {}
    results['epoch'] = epoch
    results['t_index'] = t_index
    results['thr_mc'] = thr_mc
    results['thr_md'] = thr_md
    return results

def get_basic_results_struct(ys_mc, ys_md, results, Journal, m):
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
    results['tGROI'] = Journal['GROI'].sum()
    results['tROI'] = Journal['ROI'].sum()
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
    df.to_csv(filename+'.txt', mode='a', header=False,float_format='%.2f', index=False, sep='\t')
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
    return ['eGROI','eROI','eROI.5','eROI1','eROI2','eROI3','eROI4',
           'eROI5','SI','SI.5','SI1','SI2','SI3','SI4','SI5']
    
def get_best_results(TR, results_filename, resultsDir, IDresults, save=0):
    """  """
    best_results_list = get_best_results_list()
    
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
            pd.DataFrame(columns = get_results_entries()).to_csv(best_filename, 
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
        
        
        out = "Best "+b+" = {0:.2f}".format(TR[b].loc[idx])+\
              " t_index "+str(TR['t_index'].loc[idx])+\
              " thr_mc "+str(TR['thr_mc'].loc[idx])+\
              " thr_md "+str(TR['thr_md'].loc[idx])+\
              " epoch "+str(TR['epoch'].loc[idx])
        print(out)
        if not save:
            file.write(out+"\n")
    if not save:
        file.close()
    return None

def get_results(config, model, y, DTA, J_test, soft_tilde,
                 costs, epoch, lastTrained, results_filename,
                 costs_filename, from_var=False):
    """ Get results after one epoch.
    Args:
        - 
    Return:
        - """
    
    dateTest = config['dateTest']
    IDresults = config['IDresults']
    IDweights = config['IDweights']
    save_journal = config['save_journal']
    resolution = config['resolution']
    resultsDir = local_vars.results_directory
    if 'thresholds_mc' not in config:
        thresholds_mc = [.5+i/resolution for i in range(int(resolution/2))]
    else:
        thresholds_mc = config['thresholds_mc']
    if 'thresholds_md' not in config:
        thresholds_md = [.5+i/resolution for i in range(int(resolution/2))]
    else:
        thresholds_md = config['thresholds_md']
    m = y.shape[0]
    n_days = len(dateTest)
#    granularity = 1/resolution
    if 'cost_name' in config:
        cost_name = config['cost_name']
    else:
        cost_name = IDweights
    J_train = costs[cost_name+str(epoch)]
    # cum results per t_index and mc/md combination
    CR = [[[None for md in thresholds_md] for mc in thresholds_mc] for t in range(model.seq_len+1)]
    # single results (results per MCxMD)
    #SR = [[[None for md in thresholds_md] for mc in thresholds_mc] for t in range(model.seq_len+1)]
    # to acces CR: CR[t][mc][md]
    # save fuction cost
    save_costs(costs_filename, [epoch, J_train, J_test])
    print("Epoch "+str(epoch)+", J_train = "+str(J_train)+", J_test = "+str(J_test))
    # loop over t_indexes
    tic = time.time()
    pos_dirname = ''
    pos_filename = ''
    for t_index in range(model.seq_len):
        # init results dictionary
        
        if t_index==model.seq_len:
            # get MRC from all indexes
            pass
        else:
            t_soft_tilde = soft_tilde[:,t_index,:]
            t_y = y[:,t_index,:]
        # loop over market change thresholds
        for mc in range(len(thresholds_mc)):
            thr_mc = thresholds_mc[mc]
            # upper bound
            ub_mc = 1#thr_mc+granularity
            ys_mc = get_mc_vectors(t_y, t_soft_tilde, thr_mc, ub_mc)
            # loop over market direction thresholds
            for md in range(len(thresholds_md)):
                thr_md = thresholds_md[md]
                results = init_results_struct(epoch, thr_mc, thr_md, t_index)
                # upper bound
                ub_md = 1#thr_md+granularity
                ys_md = get_md_vectors(t_soft_tilde, t_y, ys_mc, model.size_output_layer, thr_md, ub_md)
                # extract DTA structure for t_index
                if from_var == True:
                    DTAt = DTA.iloc[:,:]
                else:
                    DTAt = DTA.iloc[::model.seq_len,:]
                # get journal
                Journal = get_journal(DTAt.iloc[ys_md['y_md_tilde']], 
                                      ys_md['y_dec_mg_tilde'], 
                                      ys_md['y_dec_mg'],
                                      ys_md['diff_y_y_tilde'], 
                                      ys_mc['probs_mc'][ys_md['y_md_tilde']], 
                                      ys_md['probs_md'])
                ## calculate KPIs
                results = get_basic_results_struct(ys_mc, ys_md, results, Journal, m)
                # init positions dir and filename
                if save_journal:
                    pos_dirname = resultsDir+IDresults+'/positions/'
                    pos_filename = 'P_E'+str(epoch)+'TI'+str(t_index)+'MC'+str(thr_mc)+'MD'+str(thr_md)+'.csv'
                else:
                    pos_dirname = ''
                    pos_filename = ''
                # get results with extensions
                res_ext, log = get_extended_results(Journal,
                                                    model.size_output_layer,
                                                    n_days, get_positions=save_journal,
                                                    pNZA=results['pNZA'],
                                                    pos_dirname=pos_dirname,
                                                    pos_filename=pos_filename)
                results.update(res_ext)
                # update cumm results list
                CR[t_index][mc][md] = results
                # print results
                print_results(results, epoch, J_test, J_train, thr_md, thr_mc, t_index)
                # save results
                save_results_fn(results_filename, results)
                # save journal
                if save_journal and (thr_mc>.5 or thr_md>.5):
                    journal_dir = resultsDir+IDresults+'/journal/'
                    journal_id = 'J_E'+str(epoch)+'TI'+str(t_index)+'MC'+str(thr_mc)+'MD'+str(thr_md)
                    ext = '.csv'
                    save_journal_fn(Journal, journal_dir, journal_id, ext)
                
            # end of for thr_md in thresholds_md:
            print('')
        # end of for thr_mc in thresholds_mc:
    # end of for t_index in range(model.seq_len+1):
    # extract best result this epoch
    if resolution>0:
        TR = pd.read_csv(results_filename+'.csv', sep='\t')
        TR = TR[TR.epoch==epoch]
        get_best_results(TR, results_filename, resultsDir, IDresults, save=1)
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
    return results_filename

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
    Journal['Asset'] = DTA['Asset'].iloc[:]
    Journal['DTi'] = DTA['DT1'].iloc[:]
    Journal['DTo'] = DTA['DT2'].iloc[:]
    Journal['Bi'] = DTA['B1'].iloc[:]
    Journal['Bo'] = DTA['B2'].iloc[:]
    Journal['Ai'] = DTA['A1'].iloc[:]
    Journal['Ao'] = DTA['A2'].iloc[:]
    
    
    grossROIs = np.sign(y_dec_tilde)*(DTA["B2"]-DTA["B1"])/DTA["B1"]
    spreads =  (DTA["A2"]-DTA["B2"])/DTA["B1"]
    tROIs = grossROIs-spreads
    
    Journal['Spread'] = 100*spreads
    Journal['GROI'] = 100*grossROIs
    Journal['ROI'] = 100*tROIs
    Journal['Bet'] = y_dec_tilde.astype(int)
    Journal['Outcome'] = y_dec.astype(int)
    Journal['Diff'] = diff.astype(int)
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

def get_extended_results(Journal, size_output_layer, n_days, get_log=False, 
                         get_positions=False, pNZA=0,
                         pos_dirname='', pos_filename=''):
    """
    Function that calculates real ROI, GROI, spread...
    """
    
#    if 'DT1' in Journal.columns:
#        DT1 = 'DT1'
#        DT2 = 'DT2'
#        #A1 = 'A1'
#        A2 = 'A2'
#        B1 = 'B1'
#        B2 = 'B2'
#    else:
    from tqdm import tqdm
    
    DT1 = 'DTi'
    DT2 = 'DTo'
    A2 = 'Ao'
    A1 = 'Ai'
    B1 = 'Bi'
    B2 = 'Bo'
    
    log = pd.DataFrame(columns=['DateTime','Message'])
    # init positions
    if get_positions:
        columns_positions = ['Asset','Di','Ti','Do','To','GROI','ROI','spread','ext','Dir','Bi','Bo','Ai','Ao']
        
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
    n_pos_extended = 0
    gross_succ_counter = 0
    net_succ_counter = 0
    this_pos_extended = 0

    rROIxLevel = np.zeros((int(size_output_layer-1),3))
    rSampsXlevel = np.zeros((int(size_output_layer-1),2))
        
    fixed_spread_ratios = np.array([0.00005,0.0001,0.0002,0.0003,0.0004,0.0005])
    # fixed ratio success percent
    NSPs = np.zeros((fixed_spread_ratios.shape[0]))
    eROIs = np.zeros((fixed_spread_ratios.shape))
    ROI_vector = np.array([])
    GROI_vector = np.array([])
    avGROI = 0.0 # average GROI for all trades happening concurrently and for the
               # same asset
#    if Journal.shape[0]>0:
#        log=log.append({'DateTime':Journal[DT1].iloc[0],
#                        'Message':Journal['Asset'].iloc[0]+" open" },
#                        ignore_index=True)
    eInit = 0
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
            if direction>0:
                GROI = (Ao-Ai)/Ai#(Ao-Ai)/Ai
                ROI = (Bo-Ai)/Ai
                eGROIL += GROI
                NOL += 1
            else:
                GROI = (Bi-Bo)/Ao
                ROI = (Bi-Ao)/Ao
                eGROIS += GROI
                NOS += 1
            thisSpread = GROI-ROI
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
            
            NSPs = NSPs+((GROI-fixed_spread_ratios)>0)
            if get_positions:
                this_list = [Journal['Asset'].iloc[e-1],Journal[DT1].iloc[eInit][:10],
                                 Journal[DT1].iloc[eInit][11:],Journal[DT2].iloc[e-1][:10],
                                 Journal[DT2].iloc[e-1][11:],100*GROI,100*ROI,
                                 100*thisSpread,this_pos_extended,direction,
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
        # end of if (sameAss and extendExitMarket):
    # end of for e in range(1,Journal.shape[0]):
    
    if end_of_loop>0:
        direction = np.sign(Journal['Bet'].iloc[eInit])
        Ao = Journal[A2].iloc[-1]
        Ai = Journal[A1].iloc[eInit]
        Bo = Journal[B2].iloc[-1]
        Bi = Journal[B1].iloc[eInit]
        if direction>0:
            GROI = (Ao-Ai)/Ai#(Ao-Ai)/Ai
            ROI = (Bo-Ai)/Ai
            eGROIL += GROI
            NOL += 1
        else:
            GROI = (Bi-Bo)/Ao
            ROI = (Bi-Ao)/Ao
            eGROIS += GROI
            NOS += 1
        if np.sign(Ao-Ai)!=np.sign(Bo-Bi):
            count_dif_dir += 1
        thisSpread = GROI-ROI
        #thisSpread = (Journal[A2].iloc[-1]-Journal[B2].iloc[-1])/Journal[B1].iloc[-1]
        mSpread += thisSpread
        #GROI = np.sign(Journal['Bet'].iloc[eInit])*(Journal[B2].iloc[-1]-Journal[B1
        #            ].iloc[eInit])/Journal[B1].iloc[-1]
        eGROI += GROI
        avGROI += Journal['GROI'].iloc[e]
        
        ROI = GROI-thisSpread
        ROI_vector = np.append(ROI_vector,ROI)
        GROI_vector = np.append(GROI_vector,GROI)
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
                                 100*thisSpread,this_pos_extended,direction,
                                 Bi,Bo,Ai,Ao]
                for l in range(len(this_list)):
                    list_pos[l].append(this_list[l])
    
    if get_positions:
        dict_pos = {columns_positions[i]:list_pos[i] for i in range(len(columns_positions))}
#        dict_pos = {'Asset':list_pos[0],
#                    'Di':list_pos[1],
#                    'Ti':list_pos[2],
#                    'Do':list_pos[3],
#                    'To':list_pos[4],
#                    'GROI':list_pos[5],
#                    'ROI':list_pos[6],
#                    'spread':list_pos[7],
#                    'ext':list_pos[8],
#                    'Dir':list_pos[9],
#                    'Bi':list_pos[10],
#                    'Bo':list_pos[11],
#                    'Ai':list_pos[12],
#                    'Ao':list_pos[13]}
        df = pd.DataFrame(dict_pos)\
            [pd.DataFrame(columns = columns_positions).columns.tolist()]
        success = 0
        while not success:
            try:
                df.to_csv(pos_dirname+pos_filename, mode='a', 
                  header=False, index=False, sep='\t',float_format='%.4f')
                success = 1
            except PermissionError:
                print("WARNING! PermissionError. Close programs using "+
                      pos_dirname+pos_filename)
                time.sleep(1)
    print("count_dif_dir")
    print(count_dif_dir)
    print("percent_dif_dir")
    print(100*count_dif_dir/n_pos_opned)
    gross_succ_per = gross_succ_counter/n_pos_opned
    net_succ_per = net_succ_counter/n_pos_opned
    NSPs = NSPs/n_pos_opned
    successes = [n_pos_opned, 100*gross_succ_per, 100*net_succ_per, 100*NSPs]
    mSpread = 100*mSpread/n_pos_opned
    #varRet = [100000*np.var(ROI_vector), 100000*np.var(GROI_vector)]
    
    n_bets = ROI_vector.shape[0]
    if np.var(ROI_vector)>0:
        sharpe = np.sqrt(n_bets)*np.mean(ROI_vector)/np.sqrt(np.var(ROI_vector))
    else:
        sharpe = 0.0

    #rROI = rGROI-rSpread
    #results['NO']*(results['NSP2p']/100-.5)
    # Success index per spread level
    SIs = n_pos_opned*(NSPs-.55)
    SI = n_pos_opned*(net_succ_per-.55)
    eGROI = 100*eGROI
    eROI = 100*eROI
    eROIs = 100*eROIs
    
#    res_w_ext = {'eGROI':eGROI,
#                'eROI':eROI,
#                'GSP':successes[1],
#                'NSP':successes[2],
#                'NO':successes[0],
#                'sharpe':sharpe,
#                'SI':SI,
#                'mSpread':mSpread,
#                #'rROIxLevel':rROIxLevel,
#                #'rSampsXlevel':rSampsXlevel,
#                #'log':log,
#                #'varRet':varRet,
#                #'successes':successes
#                }
#    res_w_ext = unroll_param(res_w_ext, rROIxLevel[:,0], 'eRl', ['1','2'])
#    res_w_ext = unroll_param(res_w_ext, rROIxLevel[:,1], 'eGl', ['1','2'])
#    res_w_ext = unroll_param(res_w_ext, rSampsXlevel[:,1], 'NOl', ['1','2'])
#    res_w_ext = unroll_param(res_w_ext, eROIs, 'eROI', ['.5','1','2','3','4','5'])
#    res_w_ext = unroll_param(res_w_ext, successes[3], 'NSP', ['.5','1','2','3','4','5'])
#    res_w_ext = unroll_param(res_w_ext, SIs, 'SI', ['.5','1','2','3','4','5'])
    
    list_ext_results = [[eGROI,'eGROI'], [eROI,'eROI'], [successes[1],'GSP'], \
                        [successes[2],'NSP'], [successes[0],'NO'], [sharpe,'sharpe'], \
                        [SI,'SI'], [mSpread,'mSpread'], [rROIxLevel[:,0], 'eRl', ['1','2']], \
                        [rROIxLevel[:,1], 'eGl', ['1','2']], [rSampsXlevel[:,1], 'NOl', ['1','2']], \
                        [eROIs, 'eROI', ['.5','1','2','3','4','5']], \
                        [successes[3], 'NSP', ['.5','1','2','3','4','5']], \
                        [SIs, 'SI', ['.5','1','2','3','4','5']], [100*eGROIL, 'eGROIL'], \
                        [100*eGROIS, 'eGROIS'], [NOL, 'NOL'], [NOS, 'NOS']]
    
    res_w_ext = build_extended_res_struct(list_ext_results)
    
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

def merge_results(IDr_m1, IDr_m2, IDr_merged):
    """ Merge results from two different test executions.
    Arg:
        - IDr_m1 (string): ID of first merging results
        - IDr_m2 (string): ID of second merging results
        - IDr_merged (string): ID of merged results
    Return:
        - None """
    from local_config import local_vars
    results_dir = local_vars.results_directory
#    if os.path.exists(results_dir):
#        raise ValueError("Results directory already exists")
    # merge best results
    #merge_best_result_files(results_dir, IDr_m1, IDr_m2)
    merge_t_index_results(results_dir, IDr_m1, IDr_m2)
    
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