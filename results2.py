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

def plot_cost(IDweights,DBeval):
    
    conn = sqlite3.connect(DBeval)
    IDw = pd.read_sql_query("SELECT * FROM IDw", conn).set_index("index")
    #print(IDw)
    thisArg = IDw.loc[IDweights]
    print(thisArg)
    conn.close()
    
    allParameters = pickle.load( open( "../FCN/weights/"+IDweights+".p", "rb" ))
    N_epochs = allParameters["numEntries"]
    J = [allParameters["cost"+str(i)] for i in range(N_epochs)]
    plt.plot(J)
    sio.savemat("../MATLAB/"+"J"+IDweights+'.mat', {"J"+IDweights:J})
    #save_as_matfile("J"+IDweights,"J"+IDweights,J)
    return None

def init_TR(resultsDir,ID,t_index,thr_mc,save_results):
    
    #columns=['epoch','thr_mc','thr_md','J_test', 'J_train', 'AccMC','pNZ','pNZA','RD','NZ','NZA','AD','ADA','rROI','rGROI','tROI','tGROI']
        
    TRdf = pd.DataFrame()
    
    if save_results:
        if os.path.exists(resultsDir+ID+"/")==False:
            os.mkdir(resultsDir+ID+"/")
        if os.path.exists(resultsDir+ID+"/t"+str(t_index)+"/")==False:
            os.mkdir(resultsDir+ID+"/t"+str(t_index)+"/")
        
        filename = resultsDir+ID+"/t"+str(t_index)+"/"+ID+"t"+str(t_index)+"_results.csv"
        
        if not os.path.exists(filename):
            TRdf = pd.DataFrame()
            #TRdf = pd.DataFrame(columns=columns)
            TRdf.index.name = 'index'
        else:
            TRdf = pd.read_csv(filename,sep='\t',index_col='index')
            
    return TRdf

def save_results_v20(TRdf, t_index, thr_mc, epoch, lastTrained, save_results=0, resultsDir="",ID=""):
    
    if save_results:
        filename = resultsDir+ID+"/t"+str(t_index)+"/"+ID+"t"+str(t_index)+"_results.csv"
        # format table
        if epoch==lastTrained:#TRdf.shape[0]==1:
            TRdf["epoch"] = TRdf["epoch"].astype(int).fillna(0)
            TRdf["RD"] = TRdf["RD"].astype(int).fillna(0)
            TRdf["NZ"] = TRdf["NZ"].astype(int).fillna(0)
            TRdf["NZA"] = TRdf["NZA"].astype(int).fillna(0)
            TRdf["AD"] = TRdf["AD"].fillna(0)
            TRdf["ADA"] = TRdf["ADA"].fillna(0)
            TRdf["pNZ"] = TRdf["pNZ"].fillna(0)
            TRdf["pNZA"] = TRdf["pNZA"].fillna(0)
            TRdf["rGROI"] = TRdf["rGROI"].fillna(0)
            TRdf["rROI"] = TRdf["rROI"].fillna(0)
            TRdf["tGROI"] = TRdf["tGROI"].fillna(0)
            TRdf["tROI"] = TRdf["tROI"].fillna(0)
        
        #print("TR saved")
        #print(TRdf)
        #TRdf = TRdf.drop_duplicates(subset='tGROI')
        success = 0
        while not success:
            try:
                TRdf.to_csv(filename,sep='\t',float_format='%.3f',index_label='index')
                success = 1
            except PermissionError:
                print("WARNING! PermissionError. Close programs using "+filename)
                time.sleep(1)
    
    return None

def print_results_v20(TRdf, results, epoch, thr_md, thr_mc, t_index, save_results = 0):
    """  """
    print("Epoch = "+str(epoch)+". Time index = "+str(t_index)+
          ". Threshold MC = "+str(thr_mc)+". Threshold MD = "+str(thr_md))
    if thr_md==.5 and thr_mc==.5:
        print("J_test = "+str(results["J_test"])+", J_train = "+
              str(results["J_train"])+", Accuracy="+str(results["Acc"]))
    #print("".format())
    print(("RD = {0:d} NZ = {1:d} NZA = {2:d} pNZ = {5:.3f}% pNZA = {6:.3f}% "+
           "AD = {3:.2f}% ADA = {4:.2f}% NO = {9:d} GSP = {7:.2f}% NSP = {8:.2f}%")
          .format(results["RD"],results["NZ"],results["NZA"],results["AccDir"],
                  results["AccDirA"],results["perNZ"],results["perNZA"],
                  results["GSP"],results["NSP"],results["NO"]))
    #print(". AccDirA = {2:.2f}%".format(results["NZA"],results["accDirectionAll"]))
    print(("Sharpe = {1:.3f} rRl1 = {2:.4f}% rRl2 = {3:.4f}% rGROI = {4:.4f}%"+
           " rROI = {0:.4f}% tGROI = {5:.4f}% tROI = {6:.4f}%").format(
            results["rROI"],results["sharpe"],results["rROIxLevel"][0,0],
            results["rROIxLevel"][1,0],results["rGROI"],
            results["tGROI"],results["tROI"]))
    
    if save_results:
        #if epoch not in TRdf.index:
        #TRdf.loc[epoch] = np.zeros((TRdf.shape[1]))
        new_row = {'epoch':epoch,
                   'thr_mc':thr_mc,
                   'thr_md':thr_md,
                   'J_test':100*results["J_test"],
                   'J_train':100*results["J_train"],
                   'AccMC':results["Acc"],
                   'RD':results["RD"],
                   'NZ':results["NZ"],
                   'AD':results["AccDir"],
                   'NZA':results["NZA"],
                   'ADA':results["AccDirA"],
                   'pNZ':results["perNZ"],
                   'pNZA':results["perNZA"],
                   'rGROI':results["rGROI"],
                   'rROI':results["rROI"],
                   'tGROI':results["tGROI"],
                   'tROI':results["tROI"],
                   "varGROI":results["varGROI"],
                   "varROI":results["varROI"],
                   "GSP":results["GSP"],
                   "NSP":results["NSP"],
                   "NO":results["NO"]}
        # add ROIs with fixed spreads
        fixed_spread_ratios = ['.5','1','2','3','4','5']
        i = 0
        for froi in results["fROIs"]:
            new_row['fROI'+fixed_spread_ratios[i]] = results['fROIs'][i]
            i += 1
        i = 0
        for froi in results["fNSPs"]:
            new_row['fNSPs'+fixed_spread_ratios[i]] = results['fNSPs'][i]
            i += 1
        TRdf = TRdf.append(pd.DataFrame(data=new_row,index=[TRdf.shape[0]]))
    
    else:
        TRdf = pd.DataFrame()
        
    return TRdf



def get_last_saved_epoch(resultsDir, ID, t_index):
    """
    <DocString>
    """
    filename = resultsDir+ID+"/t"+str(t_index)+"/"+ID+"t"+str(t_index)+"_results.csv"
    if os.path.exists(filename):
        TR = pd.read_csv(filename,sep='\t',index_col="index")
        last_saved_epoch = TR.epoch.iloc[-1]
    else:
        last_saved_epoch = -1
    return last_saved_epoch

def get_last_saved_epoch2(resultsDir, ID, t_index):
    """
    <DocString>
    """
    filename = resultsDir+ID+"/results.csv"
    if os.path.exists(filename):
        TR = pd.read_csv(filename,sep='\t',index_col="index")
        last_saved_epoch = TR.epoch.iloc[-1]
    else:
        last_saved_epoch = -1
    return last_saved_epoch

def save_best_results(BR_GROI, BR_ROI, BR_sharpe, resultsDir, ID, save_results):
    
    if save_results:
        filename_GROIs = resultsDir+ID+"/"+ID+"_BR_GROI.csv"
        filename_ROIs = resultsDir+ID+"/"+ID+"_BR_ROI.csv"
        filename_sharpes = resultsDir+ID+"/"+ID+"_BR_sharpe.csv"
        
        
        if os.path.exists(filename_ROIs):
            BR_GROI = pd.read_csv(filename_GROIs,sep='\t',index_col="epoch").append(BR_GROI).sort_index()
            BR_GROI = BR_GROI.loc[BR_GROI.index.unique()]
            
            BR_ROI = pd.read_csv(filename_ROIs,sep='\t',index_col="epoch").append(BR_ROI).sort_index()
            BR_ROI = BR_ROI.loc[BR_ROI.index.unique()]
            
            BR_sharpe = pd.read_csv(filename_sharpes,sep='\t',index_col="epoch").append(BR_sharpe).sort_index()
            BR_sharpe = BR_sharpe.loc[BR_sharpe.index.unique()]
        
        BR_GROI = BR_GROI[~BR_GROI.index.duplicated()]
        BR_ROI = BR_ROI[~BR_ROI.index.duplicated()]
        BR_sharpe = BR_sharpe[~BR_sharpe.index.duplicated()]
        
        BR_GROI.to_csv(filename_GROIs,sep='\t',float_format='%.3f')
        BR_ROI.to_csv(filename_ROIs,sep='\t',float_format='%.3f')
        BR_sharpe.to_csv(filename_sharpes,sep='\t',float_format='%.3f')
    
    return None

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
             'y_dec_mg':np.argmax(t_y[:,3:], 1)-(size_output_layer-1)/2,
             'y_dec_mg_tilde':np.argmax(t_soft_tilde[:,3:], 1)-(size_output_layer-1)/2,
             # difference between y and y_tilde {0=no error, 1=error in mc. 2=error in md}
             'diff_y_y_tilde':np.abs(np.sign(y_dec_md_tilde)-np.sign(y_dec_md)),
             'probs_md':np.maximum(t_soft_tilde[:,2],t_soft_tilde[:,1])
            }
    
#    y_md_down_tilde = ys_mc['y_mc_tilde'] & (t_soft_tilde[:,1]>thr_md) & (t_soft_tilde[:,1]<=ub_md)
#    # non-zeros market direction up ([y_c1,y_c2]=01)
#    y_md_up_tilde = ys_mc['y_mc_tilde'] & (t_soft_tilde[:,2]>=thr_md) & (t_soft_tilde[:,2]<=ub_md)
#    # make it randon
#    #y_md_rand = np.random.rand(m)
#    #y_md_down_tilde = y_mc_tilde & (y_md_rand<(1-thr_md))
#    #y_md_up_tilde = y_mc_tilde & (y_md_rand>=thr_md)
#    # up and down indexes
#    y_md_tilde = y_md_down_tilde | y_md_up_tilde
#    nz_indexes = ys_mc['y_mc'] & y_md_tilde # non-zero indexes index
#    y_md_down_intersect = ys_mc['y_md_down'] & y_md_down_tilde # indicates down bits (y_c1) correctly predicted
#    y_md_up_intersect = ys_mc['y_md_up'] & y_md_up_tilde # indicates up bits (y_c2) correctly predicted
#    y_dec_md = np.argmax(t_y[:,3:], 1)-(size_output_layer-1)/2 # real output in decimal
#    y_dec_md_tilde = np.argmax(t_soft_tilde[:,1:3], 1)-1 # predicted dec out
#    y_dec_md_tilde = y_dec_md_tilde-(y_dec_md_tilde-1)*(-1)+2
#    # market gain
#    #y_mdg_down_tilde = y_md_down_tilde & y_down_tg
#    #y_mdg_up_tilde = y_md_up_tilde & y_up_tg
#    #y_mdg_tilde = y_mdg_down_tilde | y_mdg_up_tilde
#    #nz_mdg_indexes = y_mc & y_mdg_tilde
#    # y in decimal for MG
#    y_dec_mg = np.argmax(t_y[y_md_tilde,3:], 1)-(size_output_layer-1)/2
#    y_dec_mg_tilde = np.argmax(t_soft_tilde[y_md_tilde,3:], 1)-(size_output_layer-1)/2
#    
#    # difference between y and y_tilde {0=no error, 1=error in mc. 2=error in md}
#    diff_y_y_tilde = np.abs(np.sign(y_dec_md_tilde[y_md_tilde])-np.sign(y_dec_md[y_md_tilde]))
#    probs_md = np.maximum(t_soft_tilde[y_md_tilde,2],t_soft_tilde[y_md_tilde,1])
    
    return ys_md

def print_results(results, epoch, J_test, J_train, thr_md, thr_mc, t_index):
    """  """
    print("Epoch = "+str(epoch)+". Time index = "+str(t_index)+
          ". Threshold MC = "+str(thr_mc)+". Threshold MD = "+str(thr_md))
    if thr_md==.5 and thr_mc==.5:
        print("J_test = "+str(J_test)+", J_train = "+
              str(J_train)+", Accuracy="+str(results["Acc"]))
    #print("".format())
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

    #print(". AccDirA = {2:.2f}%".format(results["NZA"],results["accDirectionAll"]))
    print("SI2 = {0:.2f}% ".format(results["SI2"])+
          "SI = {0:.2f}% ".format(results["SI"])+
          "eGROI = {0:.2f}% ".format(results["eGROI"])+
          "eROI = {0:.2f}% ".format(results["eROI"])+
          "eROI1 = {0:.2f}% ".format(results["eROI2"])+
          "eROI2 = {0:.2f}%".format(results["eROI1"]))
    return None


def get_results_entries():
    """  """
    results_entries = ['epoch','t_index','thr_mc','thr_md','AD','ADA','GSP','NSP','NO',
                       'NZ','NZA','RD','NSP.5','NSP1','NSP2','NSP3',
                       'NSP4','NSP5','SI.5','SI1','SI2','SI3','SI4','SI5','SI',
                       'eGROI','eROI.5','eROI1','eROI2','eROI3','eROI4',
                       'eROI5','eROI','pNZ','pNZA','tGROI','tROI','eRl1',
                       'eRl2','eGl1','eGl2','sharpe','NOl1','NOl2']
    return results_entries

def get_costs_entries():
    """  """
    return ['epoch','J_train','J_test']
    
def init_results_dir(resultsDir, IDresults):
    """  """
    filedir = resultsDir+IDresults+'/'
    results_filename = resultsDir+IDresults+'/results.csv'
    costs_filename = resultsDir+IDresults+'/costs.csv'
    if not os.path.exists(filedir):
        os.makedirs(filedir)
    if not os.path.exists(results_filename):
        pd.DataFrame(columns = get_results_entries()).to_csv(results_filename, 
                            mode="w",index=False,sep='\t')
    if not os.path.exists(costs_filename):
        pd.DataFrame(columns = get_costs_entries()).to_csv(costs_filename, 
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

def save_journal_fn(journal, journal_dir, journal_id):
    """ save journal in disk """
    journal.index.name = 'index'
    if not os.path.exists(journal_dir):
        os.mkdirs(journal_dir)
    success = 0
    while not success:
        try:
            journal.to_csv(journal_dir+journal_id,sep='\t')
            success = 1
        except PermissionError:
            print("WARNING! PermissionError. Close programs using "+journal_dir+journal_id)
            time.sleep(1)
    
    return None

def save_results_fn(filename, results):
    """ save results in disc as pandas data frame """
    df = pd.DataFrame(results, index=[0])\
        [pd.DataFrame(columns = get_results_entries()).columns.tolist()]
    success = 0
    while not success:
        try:
            df.to_csv(filename, mode='a', header=False, index=False, sep='\t')
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
            df.to_csv(costs_filename, mode='a', header=False, index=False, sep='\t')
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

def get_results(data, model, y, DTA, IDresults, IDweights, J_test, soft_tilde, save_results,
                 costs, epoch, resultsDir, lastTrained, save_journal=False, resolution=10):
    """ Get results after for one epoch.
    Args:
        - 
    Return:
        - """
    results_filename, costs_filename = init_results_dir(resultsDir, IDresults)
    m = y.shape[0]
    n_days = len(data.dateTest)
    thresholds_mc = [.5+i/resolution for i in range(int(resolution/2))]
    thresholds_md = [.5+i/resolution for i in range(int(resolution/2))]
    granularity = 1/resolution
    J_train = costs[IDweights+str(epoch)]
    # cum results per t_index and mc/md combination
    CR = [[[None for md in thresholds_md] for mc in thresholds_mc] for t in range(model.seq_len+1)]
    # single results (results per MCxMD)
    SR = [[[None for md in thresholds_md] for mc in thresholds_mc] for t in range(model.seq_len+1)]
    # Journals
    J = [[[None for md in thresholds_md] for mc in thresholds_mc] for t in range(model.seq_len+1)]
    # to acces CR: CR[t][mc][md]
    # loop over t_indexes
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
            # extract non-zeros (y_c0>0.5)
#            y_mc = (t_y[:,0]>thr_mc) & (t_y[:,0]<=ub_mc) # non-zeros market change bits
#            y_md_down = y_mc & (t_y[:,1]>t_y[:,2]) # non-zeros market direction down
#            y_md_up = y_mc & (t_y[:,1]<t_y[:,2]) # non-zeros market direction up
#            y_mc_tilde = (t_soft_tilde[:,0]>thr_mc) & (t_soft_tilde[:,0]<=ub_mc) # predicted non-zeros market change bits
#            probs_mc = t_soft_tilde[:,0]
            # loop over market direction thresholds
            for md in range(len(thresholds_md)):
                thr_md = thresholds_mc[md]
                results = init_results_struct(epoch, thr_mc, thr_md, t_index)
                # upper bound
                ub_md = 1#thr_md+granularity
                ys_md = get_md_vectors(t_soft_tilde, t_y, ys_mc, model.size_output_layer, thr_md, ub_md)
#                # non-zeros market direction down ([y_md1,y_md2]=10)
#                y_md_down_tilde = ys_mc['y_mc_tilde'] & (t_soft_tilde[:,1]>thr_md) & (t_soft_tilde[:,1]<=ub_md)
#                # non-zeros market direction up ([y_c1,y_c2]=01)
#                y_md_up_tilde = ys_mc['y_mc_tilde'] & (t_soft_tilde[:,2]>=thr_md) & (t_soft_tilde[:,2]<=ub_md)
#                # make it randon
#                #y_md_rand = np.random.rand(m)
#                #y_md_down_tilde = y_mc_tilde & (y_md_rand<(1-thr_md))
#                #y_md_up_tilde = y_mc_tilde & (y_md_rand>=thr_md)
#                # up and down indexes
#                y_md_tilde = y_md_down_tilde | y_md_up_tilde
#                nz_indexes = ys_mc['y_mc'] & y_md_tilde # non-zero indexes index
#                y_md_down_intersect = ys_mc['y_md_down'] & y_md_down_tilde # indicates down bits (y_c1) correctly predicted
#                y_md_up_intersect = ys_mc['y_md_up'] & y_md_up_tilde # indicates up bits (y_c2) correctly predicted
#                y_dec_md = np.argmax(t_y[:,3:], 1)-(model.size_output_layer-1)/2 # real output in decimal
#                y_dec_md_tilde = np.argmax(t_soft_tilde[:,1:3], 1)-1 # predicted dec out
#                y_dec_md_tilde = y_dec_md_tilde-(y_dec_md_tilde-1)*(-1)+2
#                # market gain
#                #y_mdg_down_tilde = y_md_down_tilde & y_down_tg
#                #y_mdg_up_tilde = y_md_up_tilde & y_up_tg
#                #y_mdg_tilde = y_mdg_down_tilde | y_mdg_up_tilde
#                #nz_mdg_indexes = y_mc & y_mdg_tilde
#                # y in decimal for MG
#                y_dec_mg = np.argmax(t_y[y_md_tilde,3:], 1)-(model.size_output_layer-1)/2
#                y_dec_mg_tilde = np.argmax(t_soft_tilde[y_md_tilde,3:], 1)-(model.size_output_layer-1)/2
#                
#                # difference between y and y_tilde {0=no error, 1=error in mc. 2=error in md}
#                diff_y_y_tilde = np.abs(np.sign(y_dec_md_tilde[y_md_tilde])-np.sign(y_dec_md[y_md_tilde]))
#                probs_md = np.maximum(t_soft_tilde[y_md_tilde,2],t_soft_tilde[y_md_tilde,1])
                
                
                
                # extract DTA structure for t_index
                DTAt = DTA.iloc[::model.seq_len,:]
                # get journal
                Journal = get_journal(DTAt.iloc[ys_md['y_md_tilde']], 
                                      ys_md['y_dec_mg_tilde'][ys_md['y_md_tilde']], ys_md['y_dec_mg'][ys_md['y_md_tilde']],
                                      ys_md['diff_y_y_tilde'][ys_md['y_md_tilde']], 
                                      ys_mc['probs_mc'][ys_md['y_md_tilde']], 
                                      ys_md['probs_md'][ys_md['y_md_tilde']])
                ## calculate KPIs
                results = get_basic_results_struct(ys_mc, ys_md, results, Journal, m)
                # get results with extensions
                res_ext = get_extended_results(Journal, 
                                               model.size_output_layer,
                                               n_days)
                results.update(res_ext)
                # update cumm results list
                CR[t_index][mc][md] = results
                J[t_index][mc][md] = Journal
                # print results
                print_results(results, epoch, J_test, J_train, thr_md, thr_mc, t_index)
                # save results
                save_results_fn(results_filename, results)
                # save fuction cost
                save_costs(costs_filename, epoch, J_test, J_train)
            # end of for thr_md in thresholds_md:
            print('')
        # end of for thr_mc in thresholds_mc:
        if save_journal:
            journal_dir = resultsDir+IDresults+"/T"+str(t_index)+"/"
            journal_id = "J"+str(t_index)+".csv"
            save_journal_fn(J[t_index][0][0], journal_dir, journal_id, IDresults, t_index)
        
    # end of for t_index in range(model.seq_len+1):
    
    # get results per MCxMD entry
#    for t_index in range(model.seq_len+1):
#        for mc in thresholds_mc:
#            for md in thresholds_md:
#                SR[t_index][mc][md] = get_single_result(CR[t_index], mc, md, 
#                                                        thresholds_mc, 
#                                                        thresholds_md)
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

def build_extended_res_struct(eGROI, eROI, eROIs, SI, SIs, sharpe, rROIxLevel, 
                              rSampsXlevel, successes):
    """  """
    res_w_ext = {'eGROI':eGROI,
                'eROI':eROI,
                'GSP':successes[1],
                'NSP':successes[2],
                'NO':successes[0],
                'sharpe':sharpe,
                'SI':SI,
                #'rROIxLevel':rROIxLevel,
                #'rSampsXlevel':rSampsXlevel,
                #'log':log,
                #'varRet':varRet,
                #'successes':successes
                }
    res_w_ext = unroll_param(res_w_ext, rROIxLevel[:,0], 'eRl', ['1','2'])
    res_w_ext = unroll_param(res_w_ext, rROIxLevel[:,1], 'eGl', ['1','2'])
    res_w_ext = unroll_param(res_w_ext, rSampsXlevel[:,1], 'NOl', ['1','2'])
    res_w_ext = unroll_param(res_w_ext, eROIs, 'eROI', ['.5','1','2','3','4','5'])
    res_w_ext = unroll_param(res_w_ext, successes[3], 'NSP', ['.5','1','2','3','4','5'])
    res_w_ext = unroll_param(res_w_ext, SIs, 'SI', ['.5','1','2','3','4','5'])
    
    return res_w_ext

def get_extended_results(Journal, size_output_layer, n_days):
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
    DT1 = 'DTi'
    DT2 = 'DTo'
    A2 = 'Ao'
    B1 = 'Bi'
    B2 = 'Bo'
    
    log = pd.DataFrame(columns=['DateTime','Message'])
    
    # Add GROI and ROI with real spreads
    eGROI = 0.0
    eROI = 0.0
    eInit = 0#
    n_pos_opned = 1
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
    
    e = 0
    for e in range(1,Journal.shape[0]):

        oldExitTime = dt.datetime.strptime(Journal[DT2].iloc[e-1],"%Y.%m.%d %H:%M:%S")
        newEntryTime = dt.datetime.strptime(Journal[DT1].iloc[e],"%Y.%m.%d %H:%M:%S")

        extendExitMarket = (newEntryTime-oldExitTime<=dt.timedelta(0))
        sameAss = Journal['Asset'].iloc[e] == Journal['Asset'].iloc[e-1] 
        if sameAss and extendExitMarket:# and sameDir:
            #print("continue")
#            log=log.append({'DateTime':Journal[DT1].iloc[e],
#                            'Message':Journal['Asset'].iloc[e]+
#                            " extended" },ignore_index=True)
            n_pos_extended += 1
            this_pos_extended += 1
            avGROI += Journal['GROI'].iloc[e]
            level = int(np.abs(Journal['Bet'].iloc[eInit])-1)
            rSampsXlevel[level,0] += 1
        else:
            
            thisSpread = (Journal[A2].iloc[e-1]-Journal[B2].iloc[e-1])/Journal[B1].iloc[e-1]                
            GROI = np.sign(Journal['Bet'].iloc[eInit])*(Journal[B2].iloc[e-1]-Journal[B1].iloc[eInit])/Journal[B1].iloc[eInit]
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
            rROIxLevel[level,2] += avGROI/this_pos_extended
            rSampsXlevel[level,0] += 1
            rSampsXlevel[level,1] += 1
            avGROI = 0.0
            this_pos_extended = 0
            
            if GROI>0:
                gross_succ_counter += 1
            if ROI>0:
                net_succ_counter += 1
            
            NSPs = NSPs+((GROI-fixed_spread_ratios)>0)*1
                
#            log=log.append({'DateTime':Journal[DT2].iloc[e-1],'Message':" Close "+
#                            Journal['Asset'].iloc[e-1]+
#                            " entry bid {0:.4f}".format(Journal[B1].iloc[eInit])+
#                            " exit bid {0:.4f}".format(Journal[B2].iloc[e-1])+
#                            " GROI {0:.4f}% ".format(100*GROI)+
#                            " ROI {0:.4f}% ".format(100*ROI)+" tGROI {0:.4f}% ".format(100*rGROI)},
#                            ignore_index=True)
#            
#            log=log.append({'DateTime':Journal[DT1].iloc[e],
#                            'Message':Journal['Asset'].iloc[e]+
#                            " open" },ignore_index=True)
            n_pos_opned += 1

            eInit = e
        # end of if (sameAss and extendExitMarket):
    # end of for e in range(1,Journal.shape[0]):
    
    if Journal.shape[0]>0:
        thisSpread = (Journal[A2].iloc[-1]-Journal[B2].iloc[-1])/Journal[B1].iloc[-1]
        GROI = np.sign(Journal['Bet'].iloc[eInit])*(Journal[B2].iloc[-1]-Journal[B1
                    ].iloc[eInit])/Journal[B1].iloc[-1]
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
        rROIxLevel[level,2] += avGROI/this_pos_extended
        rSampsXlevel[level,0] += 1
        rSampsXlevel[level,1] += 1
        
        if GROI>0:
            gross_succ_counter += 1
        if ROI>0:
            net_succ_counter += 1
        
#        log = log.append({'DateTime':Journal[DT2].iloc[eInit],
#                          'Message':Journal['Asset'].iloc[eInit]+
#                          " close GROI {0:.4f}% ".format(100*GROI)+
#                          " ROI {0:.4f}% ".format(100*ROI)+
#                          " tGROI {0:.4f}% ".format(100*rGROI) },
#                          ignore_index=True)
    
    gross_succ_per = gross_succ_counter/n_pos_opned
    net_succ_per = net_succ_counter/n_pos_opned
    NSPs = NSPs/n_pos_opned
    successes = [n_pos_opned, 100*gross_succ_per, 100*net_succ_per, 100*NSPs]
    #varRet = [100000*np.var(ROI_vector), 100000*np.var(GROI_vector)]
    
    n_bets = ROI_vector.shape[0]
    if np.var(ROI_vector)>0:
        sharpe = np.sqrt(n_bets)*np.mean(ROI_vector)/np.sqrt(np.var(ROI_vector))
    else:
        sharpe = 0.0

    #rROI = rGROI-rSpread
    #results['NO']*(results['NSP2p']/100-.5)
    # Success index per spread level
    SIs = n_pos_opned*(NSPs-.5)
    SI = n_pos_opned*(net_succ_per-.5)
    eGROI = 100*eGROI
    eROI = 100*eROI
    eROIs = 100*eROIs
    res_w_ext = build_extended_res_struct(eGROI, eROI, eROIs, SI, SIs, sharpe, 
                                          rROIxLevel, rSampsXlevel, successes)
    
    return res_w_ext

def evaluate_RNN(data, model, y, DTA, IDresults, IDweights, J_test, soft_tilde, save_results,
                 costs, epoch, resultsDir, lastTrained, save_journal=False):
    """
    Evaluate RNN results for one epoch.
    """
    
    m = y.shape[0]
    n_days = len(data.dateTest)
    
    # extract y_c vector, if necessary
    if model.commonY == 3:
        
        # init thresholds
        
        thresholds_mc = [.5,.6,.7,.8,.9]
        thresholds_md = [.5,.6,.7,.8,.9]
        
        # init structures for best result tracking
        best_ROI = 0.0
        best_ROI_profile = {"epoch":epoch,
                            "t_index":-1,
                            "J_test":0.0,
                            "J_train":0.0,
                            "AccMC":0.0,
                            "RD":0,
                            "NZ":0,
                            "NZA":0,
                            "AD":0.0,
                            "ADA":0.0,
                            "pNZ":0.0,
                            "pNZA":0.0,
                            "rGROI":0.0,
                            "rROI":0.0,
                            "sharpe":0.0,
                            "varGROI":0.0,
                            "varROI":0.0,
                            "thr_mc":0.0,
                            "thr_md":0.0,
                            "GSP":0.0,
                            "NSP":0.0,
                            "NO":0}
        best_GROI = 0.0
        best_GROI_profile = {"epoch":epoch,
                            "t_index":-1,
                            "J_test":0.0,
                            "J_train":0.0,
                            "AccMC":0.0,
                            "RD":0,
                            "NZ":0,
                            "NZA":0,
                            "AD":0.0,
                            "ADA":0.0,
                            "pNZ":0.0,
                            "pNZA":0.0,
                            "rGROI":0.0,
                            "rROI":0.0,
                            "sharpe":0.0,
                            "varGROI":0.0,
                            "varROI":0.0,
                            "thr_mc":0.0,
                            "thr_md":0.0,
                            "GSP":0.0,
                            "NSP":0.0,
                            "NO":0}
        best_sharpe = 0.0
        best_sharpe_profile = {"epoch":epoch,
                            "t_index":-1,
                            "J_test":0.0,
                            "J_train":0.0,
                            "AccMC":0.0,
                            "RD":0,
                            "NZ":0,
                            "NZA":0,
                            "AD":0.0,
                            "ADA":0.0,
                            "pNZ":0.0,
                            "pNZA":0.0,
                            "rGROI":0.0,
                            "rROI":0.0,
                            "sharpe":0.0,
                            "varGROI":0.0,
                            "varROI":0.0,
                            "thr_mc":0.0,
                            "thr_md":0.0,
                            "GSP":0.0,
                            "NSP":0.0,
                            "NO":0}
        
        # init alphas (weight vector for consensus)
        #alphas = np.zeros((model.seq_len, len(thresholds_mc),len(thresholds_md)))
        # init columns for accuracy direction resumé
        columns_AD = ['55','56','57','58','59','65','66','67','68','69',
                      '75','76','77','78','79','85','86','87','88','89',
                      '95','96','97','98','99']
        # init expected ROI and GROI per profile
        # shape = seq_lens x thr_mc x thr_md x levels x 2 (ROI/GROI)
        eROIpp = np.zeros((model.seq_len+1, len(thresholds_mc), len(thresholds_md), int((model.size_output_layer-1)/2),2))
        NZpp = np.zeros((model.seq_len+1, len(thresholds_mc), len(thresholds_md), int((model.size_output_layer-1)/2),2)).astype(int)
        GRE = np.zeros((model.seq_len+1, len(thresholds_mc), len(thresholds_md), int((model.size_output_layer-1)/2)))
        GREav = np.zeros((model.seq_len+1, len(thresholds_mc), len(thresholds_md), int((model.size_output_layer-1)/2)))
        GREex = np.zeros((model.seq_len+1, len(thresholds_mc), len(thresholds_md), int((model.size_output_layer-1)/2)))
        # generate map of idx to threholds
        map_idx2thr = np.zeros((len(thresholds_mc)*len(thresholds_md))).astype(int)
        idx = 0
        for tmc in thresholds_mc:
            for tmd in thresholds_md:
                col_name = str(int(tmc*10))+str(int(tmd*10))
                map_idx2thr[idx] = columns_AD.index(col_name)
                idx += 1
        
        AD_resumes = np.zeros((model.seq_len+1,len(columns_AD),1))
        
        for t in range(model.seq_len+1): 
            
            t_out = model.seq_len-t
            t_index = model.seq_len-t_out
            
            
            # init AD resume
            if save_journal:
                AD_filename = resultsDir+IDresults+"/t"+str(t_index)+"/"+IDresults+"t"+str(t_index)+"AD_resume.txt"
                if os.path.exists(AD_filename):
                    AD_pd = pd.read_csv(AD_filename,sep='\t',index_col="epoch")
                else:
                    AD_pd = pd.DataFrame(columns=columns_AD)
                    AD_pd.index.name = 'epoch'
            
            # init vector of ADs
            #AD_resume = np.zeros((1,len(columns_AD)))
            
            if t==model.seq_len:
                # build diversity combining output
                
                i_t_mc = 0
                i_thr = 0
                idx_thr = np.zeros((soft_tilde.shape[0],model.seq_len)).astype(int)
                for thr_mc in thresholds_mc:
                    i_t_md = 0
                    for thr_md in thresholds_md:
                        for t_ in range(model.seq_len):
                            
                            idx_thr_mc = soft_tilde[:,t_,0]>thr_mc
                            idx_thr_md = np.maximum(soft_tilde[:,t_,1],soft_tilde[:,t_,2])>thr_md
                            idx_thr[idx_thr_mc & idx_thr_md,t_] = i_thr

                        i_thr += 1
                        i_t_md += 1
                    i_t_mc += 1
                    
                sum_AD = np.zeros((idx_thr.shape[0],1))
                t_soft_tilde = np.zeros((soft_tilde.shape[0],soft_tilde.shape[2]))

                AD_resumes[np.isnan(AD_resumes)] = 0
                #MC
                #t_soft_tilde = np.max(soft_tilde,1)
                #MRC
                for t_ in range(model.seq_len):
                    
                    sum_AD = sum_AD+AD_resumes[t_,map_idx2thr[idx_thr[:,t_]],:]
                    t_soft_tilde = t_soft_tilde+AD_resumes[t_,map_idx2thr[idx_thr[:,t_]],:]*soft_tilde[:,t_,:]
                # normaluze t_soft_tilde
#                print(soft_tilde[0,:,:])
#                print("sum_AD[0,0] "+str(sum_AD[0,0]))
                t_soft_tilde = t_soft_tilde/sum_AD
                
                #t_soft_tilde = np.max(soft_tilde,axis=1)
                
            else:
                t_soft_tilde = soft_tilde[:,-t_out,:]
                t_y = y[:,-t_out,:]
                #t_out = 1
#            print(t_soft_tilde)
            thr_idx = 0
            ## New approach ##
            if np.min(t_soft_tilde[:,1]+t_soft_tilde[:,2])<0.99:
                print("Warning! np.min(t_soft_tilde[:,1]+t_soft_tilde[:,2])="+str(np.min(t_soft_tilde[:,1]+t_soft_tilde[:,2])))
            # loop over market change probability thresholds
            i_t_mc = 0
            for thr_mc in thresholds_mc:
                
                TRdf = init_TR(resultsDir,IDresults,t_index,thr_mc,save_results)
                
                if save_journal:
                    # loop over upper bound
                    if thr_mc==.9:
                        upper_bound_mc = [1,round((thr_mc+.1)*10)/10+0.0000000001]
                    else:
                        upper_bound_mc = [1,round((thr_mc+.1)*10)/10]
                    
                else:
                    upper_bound_mc = [1]
                    
                for ub_mc in upper_bound_mc:
                    # extract non-zeros (y_c0>0.5)
                    y_mc = (t_y[:,0]>thr_mc) & (t_y[:,0]<=ub_mc) # non-zeros market change bits
                    y_md_down = y_mc & (t_y[:,1]>t_y[:,2]) # non-zeros market direction down
                    y_md_up = y_mc & (t_y[:,1]<t_y[:,2]) # non-zeros market direction up
                    y_mc_tilde = (t_soft_tilde[:,0]>thr_mc) & (t_soft_tilde[:,0]<=ub_mc)# predicted non-zeros market change bits
                    probs_mc = t_soft_tilde[:,0]
                    # random y_mc
                    #y_mc_tilde = np.random.rand(m)>=thr_mc
                    i_t_md = 0
                    # loop over market direction probability thresholds
                    for thr_md in thresholds_md:
                        
                        if save_journal:
                            # loop over upper bound
                            
                            if thr_md==.9:
                                upper_bound_md = [1,round((thr_md+.1)*10)/10+0.0000000001]
                            else:
                                upper_bound_md = [1,round((thr_md+.1)*10)/10]
                        else:
                            upper_bound_md = [1]
                            
                        for ub_md in upper_bound_md:
                        
                            # non-zeros market direction down ([y_md1,y_md2]=10)
                            y_md_down_tilde = y_mc_tilde & (t_soft_tilde[:,1]>thr_md) & (t_soft_tilde[:,1]<=ub_md)
                            # non-zeros market direction up ([y_c1,y_c2]=01)
                            y_md_up_tilde = y_mc_tilde & (t_soft_tilde[:,2]>=thr_md) & (t_soft_tilde[:,2]<=ub_md)
                            # make it randon
                            #y_md_rand = np.random.rand(m)
                            #y_md_down_tilde = y_mc_tilde & (y_md_rand<(1-thr_md))
                            #y_md_up_tilde = y_mc_tilde & (y_md_rand>=thr_md)
                            # up and down indexes
                            y_md_tilde = y_md_down_tilde | y_md_up_tilde
                            nz_indexes = y_mc & y_md_tilde # non-zero indexes index
                            y_md_down_intersect = y_md_down & y_md_down_tilde # indicates down bits (y_c1) correctly predicted
                            y_md_up_intersect = y_md_up & y_md_up_tilde # indicates up bits (y_c2) correctly predicted
                            y_dec_md = np.argmax(t_y[:,3:], 1)-(model.size_output_layer-1)/2 # real output in decimal
                            y_dec_md_tilde = np.argmax(t_soft_tilde[:,1:3], 1)-1 # predicted dec out
                            y_dec_md_tilde = y_dec_md_tilde-(y_dec_md_tilde-1)*(-1)+2
                            # market gain
                            #y_mdg_down_tilde = y_md_down_tilde & y_down_tg
                            #y_mdg_up_tilde = y_md_up_tilde & y_up_tg
                            #y_mdg_tilde = y_mdg_down_tilde | y_mdg_up_tilde
                            #nz_mdg_indexes = y_mc & y_mdg_tilde
                            # y in decimal for MG
                            y_dec_mg = np.argmax(t_y[:,3:], 1)-(model.size_output_layer-1)/2
                            y_dec_mg_tilde = np.argmax(t_soft_tilde[:,3:], 1)-(model.size_output_layer-1)/2
                            
                            # difference between y and y_tilde {0=no error, 1=error in mc. 2=error in md}
                            diff_y_y_tilde = np.abs(np.sign(y_dec_md_tilde[y_md_tilde])-np.sign(y_dec_md[y_md_tilde]))
                            probs_md = np.maximum(t_soft_tilde[:,2],t_soft_tilde[:,1])
                            # calculate KPIs
                            Acc = 1-np.sum(np.abs(y_mc^y_mc_tilde))/m # market change accuracy
                            NZA = np.sum(y_md_tilde) # number of non-zeros all
                            NZ = np.sum(nz_indexes) # Number of non-zeros
                            RD =  np.sum(y_md_down_intersect)+np.sum(y_md_up_intersect) # right direction
                            AccDir = RD/NZ # accuracy direction
                            AccDirA = RD/NZA # accuracy direction all
                            perNZ = NZ/m # percent of non-zeros
                            perNZA = NZA/m # percent of non-zeros all
                            # get this DateTime/Asset
                            if t<model.seq_len:
                                DTAt = DTA.iloc[t_index::model.seq_len,:]
                            else:
                                # for MRC, DTA indexes correspond of those for t_index=0
                                DTAt = DTA.iloc[::model.seq_len,:]
                            
                            
                            # get and save results if upper bound is 1
                            if ((ub_mc==1 and ub_md==1) or (ub_mc!=1 and ub_md!=1)):
                                
                                
                                # get Journal
                                if (thr_mc==.5 and thr_md==.5):# or (thr_mc==.6 and thr_md==.5) or (thr_mc==.5 and thr_md==.6)
                                    get_real = False
                                else:
                                    get_real = True
                                
                                Journal,rROIxLevel, rSampsXlevel, summary, log_journal = getJournal_v20(DTAt.iloc[y_md_tilde], y_dec_mg_tilde[y_md_tilde], y_dec_mg[y_md_tilde],
                                             diff_y_y_tilde, probs_mc[y_md_tilde], probs_md[y_md_tilde] ,model.size_output_layer, n_days, fixed_spread=1, get_real=get_real, save_journal=save_journal)
                                
                                if save_journal and ub_mc==1 and ub_md==1:
                                    
                                    #print("Saving journal...")
                                    
                                    Journal.index.name = 'index'
                                    journal_dir = resultsDir+IDresults+"/t"+str(t_index)+"/"+IDresults+"t"+str(t_index)+"mc"+str(thr_mc)+"/"
                                    
                                    if os.path.exists(journal_dir)==False:
                                        os.mkdir(journal_dir)
                                    
                                    journal_dir = journal_dir+IDresults+"t"+str(t_index)+"mc"+str(thr_mc)+"md"+str(thr_md)+"/"
                                    
                                    if os.path.exists(journal_dir)==False:
                                        os.mkdir(journal_dir)
                                    
                                    journal_id = IDresults+"t"+str(t_index)+"mc"+str(thr_mc)+"md"+str(thr_md)+"e"+str(epoch)+".csv"
                                    Journal.to_csv(journal_dir+'J'+journal_id,sep='\t')#float_format='%.3f'
                                    log_journal.to_csv(journal_dir+'LG'+journal_id,sep='\t',float_format='%.3f')
                                    log_journal.sort_values(by=['DateTime']).to_csv(journal_dir+'LGS'+journal_id,float_format='%.3f')
                                    
                                    
                                #print(rROIxLevel)
                                # save results in structure
                                results = {}
                                results["J_test"] = J_test
                                results["J_train"] = costs[IDweights+str(epoch)]
                                results["Acc"] = 100*Acc # tbd
                                results["RD"] = RD
                                results["NZ"] = NZ
                                results["NZA"] = NZA
                                results["AccDir"] = 100*AccDir
                                results["AccDirA"] = 100*AccDirA
                                results["perNZ"] = 100*perNZ
                                results["perNZA"] = 100*perNZA
                                results["rROI"] = summary['rROI']
                                results["rGROI"] = summary['rGROI']
                                results["fROIs"] = summary['fROIs']
                                results["tROI"] = summary['tROI']
                                results["tGROI"] = summary['tGROI']
                                results["sharpe"] = summary['Sharpe']
                                results["varROI"] = summary['varRet'][0]
                                results["varGROI"] = summary['varRet'][1]
                                results['NO'] = summary['successes'][0]# number opened positions
                                results['GSP'] = summary['successes'][1] # gross success percentage
                                results['NSP'] = summary['successes'][2] # net success percentage
                                results['fNSPs'] = summary['successes'][3]
                                results['NSP2p'] = 100*results['fNSPs'][2]
                                results['SI2p'] = results['NO']*(results['NSP2p']/100-.5)# success indicator spread=2p
                                results['rSI'] = results['NO']*(results['NSP']/100-.5)# success indicator
#                                print(results['NSP2p'])
#                                print(results['SI2p'])
#                                print(results['NSP'])
#                                print(results['rSI'])
                                results["rROIxLevel"] = rROIxLevel
                                results["rSampsXlevel"] = rSampsXlevel
                                
                                # update alphas
                                if t<model.seq_len and ub_mc==1 and ub_md==1:
                                    # update structures for diversity combining
            
                                    #alphas[t,i_t_mc,i_t_md] = AccDirA
                                    idxs = np.zeros((y_md_tilde.shape))<1 # init an all trues array
                                    idxs[:model.seq_len-t] = False
                                    idxs[m-t+1:] = False
                                    
                                    #t_soft_mdc[y_md_tilde[model.seq_len-t-1:m-t],0,:] = t_soft_mdc[y_md_tilde[model.seq_len-t-1:m-t],0,:]+AccDirA*t_soft_tilde[y_md_tilde & idxs,:]
                                    
                                #print(alphas)
                                # update best ROI
                                if results["rROI"]>best_ROI and ub_mc==1 and ub_md==1:
                                    best_ROI = results["rROI"]
                                    best_ROI_profile = {"epoch":epoch,
                                                        "t_index":t_index,
                                                        "J_test":results["J_test"],
                                                        "J_train":results["J_train"],
                                                        "AccMC":results["Acc"],
                                                        "RD":results["RD"],
                                                        "NZ":results["NZ"],
                                                        "NZA":results["NZA"],
                                                        "AD":results["AccDir"],
                                                        "ADA":results["AccDirA"],
                                                        "pNZ":results["perNZ"],
                                                        "pNZA":results["perNZA"],
                                                        "rGROI":results["rGROI"],
                                                        "rROI":best_ROI,
                                                        "sharpe":results["sharpe"],
                                                        "varGROI":results["varGROI"],
                                                        "GSP":results["GSP"],
                                                        "NSP":results["NSP"],
                                                        "NSP2p":results['NSP2p'],
                                                        "SI2p":results['SI2p'],
                                                        "RSI":results['rSI'],
                                                        "NO":results['NO'],
                                                        "thr_mc":thr_mc,
                                                        "thr_md":thr_md}
                                
                                # update best GROI
                                if results["rGROI"]>best_GROI and ub_mc==1 and ub_md==1:
                                    best_GROI = results["rGROI"]
                                    best_GROI_profile = {"epoch":epoch,
                                                        "t_index":t_index,
                                                        "J_test":results["J_test"],
                                                        "J_train":results["J_train"],
                                                        "AccMC":results["Acc"],
                                                        "RD":results["RD"],
                                                        "NZ":results["NZ"],
                                                        "NZA":results["NZA"],
                                                        "AD":results["AccDir"],
                                                        "ADA":results["AccDirA"],
                                                        "pNZ":results["perNZ"],
                                                        "pNZA":results["perNZA"],
                                                        "rGROI":results["rGROI"],
                                                        "rROI":results["rROI"],
                                                        "sharpe":results["sharpe"],
                                                        "varGROI":results["varGROI"],
                                                        "varROI":results["varROI"],
                                                        "GSP":results["GSP"],
                                                        "NSP":results["NSP"],
                                                        "NO":results['NO'],
                                                        "NSP2p":results['NSP2p'],
                                                        "SI2p":results['SI2p'],
                                                        "RSI":results['rSI'],
                                                        "thr_mc":thr_mc,
                                                        "thr_md":thr_md}
                                # update best sharpe ratio
                                if (results["sharpe"]>best_sharpe and 
                                    results["sharpe"] != float('Inf') and 
                                    results["NZA"]>=20 and ub_mc==1 and ub_md==1):
                                    best_sharpe = results["sharpe"]
                                    best_sharpe_profile = {"epoch":epoch,
                                                        "t_index":t_index,
                                                        "J_test":results["J_test"],
                                                        "J_train":results["J_train"],
                                                        "AccMC":results["Acc"],
                                                        "RD":results["RD"],
                                                        "NZ":results["NZ"],
                                                        "NZA":results["NZA"],
                                                        "AD":results["AccDir"],
                                                        "ADA":results["AccDirA"],
                                                        "pNZ":results["perNZ"],
                                                        "pNZA":results["perNZA"],
                                                        "rGROI":results["rGROI"],
                                                        "rROI":results["rROI"],
                                                        "sharpe":results["sharpe"],
                                                        "varGROI":results["varGROI"],
                                                        "varROI":results["varROI"],
                                                        "GSP":results["GSP"],
                                                        "NSP":results["NSP"],
                                                        "NO":results['NO'],
                                                        "NSP2p":results['NSP2p'],
                                                        "SI2p":results['SI2p'],
                                                        "RSI":results['rSI'],
                                                        "thr_mc":thr_mc,
                                                        "thr_md":thr_md}
                                    
                                if ub_mc==1 and ub_md==1:
                                    TRdf = print_results_v20(TRdf, results, epoch, thr_md, thr_mc, t_index, save_results=save_results)
                                    # get AD KPI
                                    #print(2/3*AccDir+1/3*AccDirA)
        #                            AD_resume[0,thr_idx] = 2/3*AccDir+1/3*AccDirA
                                    #print(thr_idx)
                                    #if t<model.seq_len:
                                    AD_resumes[t,map_idx2thr[thr_idx],0] = max(1.7/3*(AccDir-.5)+1.3/3*(AccDirA-.33), 0)#2/3*AccDir+1/3*AccDirA#
                                    thr_idx += 1
                                else:
                                    print("eROIpp for mc between "+
                                          str(round(thr_mc*10)/10)+
                                          "-"+str(round(ub_mc*10)/10)+
                                          " and md "+str(round(thr_md*10)/10)+"-"
                                          +str(round(ub_md*10)/10))
#                                    print(np.sum(np.abs(y_dec_mg_tilde[y_md_tilde])==0))
                                    for b in range(int((model.size_output_layer-1)/2)):
#                                        print(np.sum(np.abs(y_dec_mg_tilde[y_md_tilde])==(b+1)))
                                        NZpp[t,i_t_mc,i_t_md, b, 0] = int(rSampsXlevel[b,0])#int(np.sum(np.abs(y_dec_mg_tilde[y_md_tilde]).astype(int)==(b+1)))
                                        NZpp[t,i_t_mc,i_t_md, b, 1] = int(rSampsXlevel[b,1])
                                        eROIpp[t,i_t_mc,i_t_md, b, 0] = rROIxLevel[b,0]/100
                                        eROIpp[t,i_t_mc,i_t_md, b, 1] = rROIxLevel[b,1]/100
                                        if NZpp[t,i_t_mc,i_t_md, b, 0]>0:
                                            GRE[t,i_t_mc,i_t_md, b] = eROIpp[t,i_t_mc,i_t_md, b, 1]/NZpp[t,i_t_mc,i_t_md, b, 0]
                                        print("GRE level "+str(b)+": "+str(GRE[t,i_t_mc,i_t_md, b]/0.0001)+" pips")
                                        print("Nonzero entries = "+str(NZpp[t,i_t_mc,i_t_md, b, 0]))
                                        # GRE new
                                        if rSampsXlevel[b,1]>0:
                                            GREav[t,i_t_mc,i_t_md, b] = rROIxLevel[b,0]/(100*rSampsXlevel[b,1])
                                        print("GRE av level "+str(b)+": "+str(GREav[t,i_t_mc,i_t_md, b]/0.0001)+" pips")
                                        print("Nonzero entries = "+str(int(rSampsXlevel[b,1])))
                                        if rSampsXlevel[b,1]>0:
                                            GREex[t,i_t_mc,i_t_md, b] = eROIpp[t,i_t_mc,i_t_md, b, 1]/rSampsXlevel[b,1]
                                        print("GRE ex level "+str(b)+": "+str(GREex[t,i_t_mc,i_t_md, b]/0.0001)+" pips")
                                        print("Nonzero entries = "+str(int(rSampsXlevel[b,1])))
                                        # weighted GRE
#                                        rROIxLevelxExt, rSampsXlevelXExt
#                                        for ex in range(10):
#                                            p = rSampsXlevelXExt[ex]/NZpp[t,i_t_mc,i_t_md, b]
#                                            GREw[t,i_t_mc,i_t_md, b] += p*rROIxLevelxExt[b,1]
                                        
                        # end of for ub_md in upper_bound_md:
                        
                        i_t_md += 1
                        save_results_v20(TRdf, t_index, thr_mc, epoch, lastTrained, save_results=save_results, resultsDir=resultsDir,ID=IDresults)
                    #end of for thr_md in thresholds_md:
                    
                # end of for ub_mc in upper_bound_mc:
                i_t_mc += 1
                print("\n")
                
                
            # end of for thr_mc in thresholds_mc:
            if save_journal:
                # get eROI per bet 
                # add AD_resume to dataframe
                AD_pd = AD_pd.append(pd.DataFrame(data=AD_resumes[t:t+1,:,0],columns=columns_AD,index=[epoch]))
                AD_pd.index.name = 'epoch'
                #print(AD_pd)
                AD_pd.to_csv(AD_filename,sep='\t')
        # end of for t in range(model.seq_len): 
        
        if save_journal:
#            GRE = fill_up_GRE(GRE, model, thresholds_mc, thresholds_md)
#            GRE_new = fill_up_GRE(GRE_new, model, thresholds_mc, thresholds_md)
            # Fillthe gaps of GRE
#            for t in range(model.seq_len+1):
#                for idx_mc in range(len(thresholds_mc)):
#                    min_md = GRE[t,idx_mc,:,:]
#                    for idx_md in range(len(thresholds_md)):
#                        min_mc = GRE[t,:,idx_md,:]    
#                        for l in range(int((model.size_output_layer-1)/2)):
#                            if GRE[t,idx_mc,idx_md,l]==0:
#                                if idx_md==0:
#                                    if idx_mc==0:
#                                        if l==0:
#                                            # all zeros, nothing to do
#                                            pass
#                                        else:
#                                            GRE[t,idx_mc,idx_md,l] = max(min_mc[idx_mc,:l])
#                                    else:
#                                        GRE[t,idx_mc,idx_md,l] = max(min_mc[:idx_mc,l])
#                                else:
#                                    GRE[t,idx_mc,idx_md,l] = max(min_md[:idx_md,l])
                                    
                                    
                                
                    
                    
            pickle.dump( AD_resumes, open( resultsDir+IDresults+"/AD_e"+str(epoch)+".p", "wb" ))
            pickle.dump( NZpp, open( resultsDir+IDresults+"/NZpp_e"+str(epoch)+".p", "wb" ))
            pickle.dump( eROIpp, open( resultsDir+IDresults+"/eROIpp_e"+str(epoch)+".p", "wb" ))
            pickle.dump( GRE, open( resultsDir+IDresults+"/GRE_e"+str(epoch)+".p", "wb" ))
            pickle.dump( GREav, open( resultsDir+IDresults+"/GREav_e"+str(epoch)+".p", "wb" ))
            pickle.dump( GREex, open( resultsDir+IDresults+"/GREex_e"+str(epoch)+".p", "wb" ))
            print("GRE:")
            for b in range(int((model.size_output_layer-1)/2)):
                print("Level "+str(b))
                print(GRE[:,:,:,b]/0.0001)
            print("GRE av:")
            for b in range(int((model.size_output_layer-1)/2)):
                print("Level "+str(b))
                print(GREav[:,:,:,b]/0.0001)
            print("GRE ex:")
            for b in range(int((model.size_output_layer-1)/2)):
                print("Level "+str(b))
                print(GREex[:,:,:,b]/0.0001)
        # add best ROI to best results table
        BR_GROI = pd.DataFrame(best_GROI_profile,index=[0]).set_index("epoch")
        print(BR_GROI.to_string())
        BR_ROI = pd.DataFrame(best_ROI_profile,index=[0]).set_index("epoch")
        print(BR_ROI.to_string())
        BR_sharpe = pd.DataFrame(best_sharpe_profile,index=[0]).set_index("epoch")
        print(BR_sharpe.to_string())
        
        # save best results in files
        
    # 4 commonY means that market gain tagets have no zero but only up and down bits
    elif model.commonY == 4:
        pass
    
    return best_ROI, BR_GROI, BR_ROI, BR_sharpe

def fill_up_GRE(GRE, model, thresholds_mc, thresholds_md):
    """
    
    """
    for t in range(model.seq_len+1):
        for idx_mc in range(len(thresholds_mc)):
            min_md = GRE[t,idx_mc,:,:]
            for idx_md in range(len(thresholds_md)):
                min_mc = GRE[t,:,idx_md,:]    
                for l in range(int((model.size_output_layer-1)/2)):
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

def get_real_ROI_old(size_output_layer, Journal, n_days, fixed_spread=0):
    """
    Function that calculates real ROI, GROI, spread...
    """
    
    if 'DT1' in Journal.columns:
        DT1 = 'DT1'
        DT2 = 'DT2'
        #A1 = 'A1'
        A2 = 'A2'
        B1 = 'B1'
        B2 = 'B2'
    else:
        DT1 = 'Entry Time'
        DT2 = 'Exit Time'
        A2 = 'Ao'
        B1 = 'Bi'
        B2 = 'Bo'
    
    log = pd.DataFrame(columns=['DateTime','Message'])
    
    # Add GROI and ROI with real spreads
    rGROI = 0.0
    rROI = 0.0
    eInit = 0#
    n_pos_opned = 1
    n_pos_extended = 0
    gross_succ_counter = 0
    net_succ_counter = 0
    this_pos_extended = 0

    rROIxLevel = np.zeros((int(size_output_layer-1),3))
    rSampsXlevel = np.zeros((int(size_output_layer-1),2))
        
    fixed_spread_ratios = np.array([0.00005,0.0001,0.0002,0.0003,0.0004,0.0005])
    # fixed ratio success percent
    fNSP = np.zeros((fixed_spread_ratios.shape[0]))
    fROIs = np.zeros((fixed_spread_ratios.shape))
    ROI_vector = np.array([])
    GROI_vector = np.array([])
    avGROI = 0.0 # average GROI for all trades happening concurrently and for the
               # same asset
#    if Journal.shape[0]>0:
#        log=log.append({'DateTime':Journal[DT1].iloc[0],
#                        'Message':Journal['Asset'].iloc[0]+" open" },
#                        ignore_index=True)
    
    e = 0
    for e in range(1,Journal.shape[0]):

        oldExitTime = dt.datetime.strptime(Journal[DT2].iloc[e-1],"%Y.%m.%d %H:%M:%S")
        newEntryTime = dt.datetime.strptime(Journal[DT1].iloc[e],"%Y.%m.%d %H:%M:%S")

        extendExitMarket = (newEntryTime-oldExitTime<=dt.timedelta(0))
        sameAss = Journal['Asset'].iloc[e] == Journal['Asset'].iloc[e-1] 
        if sameAss and extendExitMarket:# and sameDir:
            #print("continue")
#            log=log.append({'DateTime':Journal[DT1].iloc[e],
#                            'Message':Journal['Asset'].iloc[e]+
#                            " extended" },ignore_index=True)
            n_pos_extended += 1
            this_pos_extended += 1
            avGROI += Journal['GROI'].iloc[e]
            level = int(np.abs(Journal['Bet'].iloc[eInit])-1)
            rSampsXlevel[level,0] += 1
            continue
        else:
            
            thisSpread = (Journal[A2].iloc[e-1]-Journal[B2].iloc[e-1])/Journal[B1].iloc[e-1]                
            GROI = np.sign(Journal['Bet'].iloc[eInit])*(Journal[B2].iloc[e-1]-Journal[B1].iloc[eInit])/Journal[B1].iloc[eInit]
            rGROI += GROI
            avGROI += Journal['GROI'].iloc[e]
            ROI = GROI-thisSpread
            
            ROI_vector = np.append(ROI_vector,ROI)
            GROI_vector = np.append(GROI_vector,GROI)
            rROI += ROI
            fROIs = fROIs+GROI-fixed_spread_ratios
            level = int(np.abs(Journal['Bet'].iloc[eInit])-1)
            rROIxLevel[level,0] += 100*ROI
            rROIxLevel[level,1] += 100*GROI
            rROIxLevel[level,2] += avGROI/this_pos_extended
            rSampsXlevel[level,0] += 1
            rSampsXlevel[level,1] += 1
            avGROI = 0.0
            this_pos_extended = 0
            
            if GROI>0:
                gross_succ_counter += 1
            if ROI>0:
                net_succ_counter += 1
            
            fNSP = fNSP+((GROI-fixed_spread_ratios)>0)*1
                
#            log=log.append({'DateTime':Journal[DT2].iloc[e-1],'Message':" Close "+
#                            Journal['Asset'].iloc[e-1]+
#                            " entry bid {0:.4f}".format(Journal[B1].iloc[eInit])+
#                            " exit bid {0:.4f}".format(Journal[B2].iloc[e-1])+
#                            " GROI {0:.4f}% ".format(100*GROI)+
#                            " ROI {0:.4f}% ".format(100*ROI)+" tGROI {0:.4f}% ".format(100*rGROI)},
#                            ignore_index=True)
#            
#            log=log.append({'DateTime':Journal[DT1].iloc[e],
#                            'Message':Journal['Asset'].iloc[e]+
#                            " open" },ignore_index=True)
            n_pos_opned += 1

            eInit = e
        # end of if (sameAss and extendExitMarket):
    # end of for e in range(1,Journal.shape[0]):
    
    if Journal.shape[0]>0:
        thisSpread = (Journal[A2].iloc[-1]-Journal[B2].iloc[-1])/Journal[B1].iloc[-1]
        GROI = np.sign(Journal['Bet'].iloc[eInit])*(Journal[B2].iloc[-1]-Journal[B1
                    ].iloc[eInit])/Journal[B1].iloc[-1]
        rGROI += GROI
        avGROI += Journal['GROI'].iloc[e]
        
        ROI = GROI-thisSpread
        ROI_vector = np.append(ROI_vector,ROI)
        GROI_vector = np.append(GROI_vector,GROI)
        rROI += ROI
        fROIs = fROIs+GROI-fixed_spread_ratios
        level = int(np.abs(Journal['Bet'].iloc[eInit])-1)
        rROIxLevel[level,0] += 100*ROI
        rROIxLevel[level,1] += 100*GROI
        rROIxLevel[level,2] += avGROI/this_pos_extended
        rSampsXlevel[level,0] += 1
        rSampsXlevel[level,1] += 1
        
        if GROI>0:
            gross_succ_counter += 1
        if ROI>0:
            net_succ_counter += 1
        
#        log = log.append({'DateTime':Journal[DT2].iloc[eInit],
#                          'Message':Journal['Asset'].iloc[eInit]+
#                          " close GROI {0:.4f}% ".format(100*GROI)+
#                          " ROI {0:.4f}% ".format(100*ROI)+
#                          " tGROI {0:.4f}% ".format(100*rGROI) },
#                          ignore_index=True)
    
    gross_succ_per = gross_succ_counter/n_pos_opned
    net_succ_per = net_succ_counter/n_pos_opned
    fNSP = fNSP/n_pos_opned
    successes = [n_pos_opned, 100*gross_succ_per, 100*net_succ_per, fNSP]
    varRet = [100000*np.var(ROI_vector), 100000*np.var(GROI_vector)]
    
    n_bets = ROI_vector.shape[0]
    sharpe_ratio = np.sqrt(n_bets)*np.mean(ROI_vector)/(np.sqrt(np.var(ROI_vector))*n_days)

    #rROI = rGROI-rSpread
    
    return rGROI, rROI, fROIs, sharpe_ratio, rROIxLevel, rSampsXlevel, log, varRet, successes

def getJournal_v20(DTA, y_dec_tilde, y_dec, diff, probs_mc, probs_md, 
                   size_output_layer, n_days, fixed_spread=0, 
                   get_real=1, save_journal=False):
    """
    Version 2.0 of get journal. Calculates trading journal given predictions.
    Args:
        - DTA: DataFrame containing Entry and Exit times, Bid and Ask.
        - y_dec_tilde: estimated output in decimal values.
        - y_dec: real output in decimal values.
        - diff: vector difference between real and estimated.
        - probs: probabilties vector of estimates.
        - size_output_layer: size of output layer.
        - fixed_spread: boolean indicating type of spread calulation. Fixed or  real.
    Returns:
        - Journal: DataFrame with info of each transaction.
        - rGOIxLevel: real GROI per level.
        - summury: summmary info
    """
    #print("Getting journal...")
    Journal = DTA
    #print(infoBets)
    #Journal.columns = ['Entry Time         ','Exit Time          ','AssgetJournal_v20et']
    #print(y_dec_tilde)
    #print(DTA)
    grossROIs = np.sign(y_dec_tilde)*(DTA["B2"]-DTA["B1"])/DTA["B1"]
    spreads =  (DTA["A2"]-DTA["B2"])/DTA["B1"]
    #print(stoploss)
    #grossROIs = np.maximum(grossROIs, -stoploss)
    tROIs = grossROIs-spreads
    #print(grossROIs)
    
    #Journal.loc[:,'Bid'] = infoBets["B1"] 
    Journal.loc[:,'GROI'] = 100*grossROIs
    #Journal['GROI'] = Journal['GROI'].astype(float)
    Journal.loc[:,'Spread'] = 100*spreads
    Journal.loc[:,'ROI'] = 100*tROIs
    Journal.loc[:,'Bet'] = y_dec_tilde.astype(int)
    Journal.loc[:,'Outcome'] = y_dec.astype(int)
    Journal.loc[:,'Diff'] = diff.astype(int)
    Journal.loc[:,"P_mc"] = probs_mc
    Journal.loc[:,"P_md"] = probs_md
    
    Journal.index = range(Journal.shape[0])
    
    if get_real:
        [rGROI, rROI, fROIs, sharpe_ratio, rROIxLevel, rSampsXlevel, log, 
         varRet, successes] = get_real_ROI_old(size_output_layer, 
                                           Journal, n_days, 
                                           fixed_spread=fixed_spread)
        
        summary = {'tGROI':Journal['GROI'].sum(),
               'tROI':Journal['ROI'].sum(),
               'tSpread':Journal['Spread'].sum(),
               'rGROI':100*rGROI,
               'rROI':100*rROI,
               'fROIs':100*fROIs,
               'Sharpe':sharpe_ratio,
               'varRet':varRet,
               'successes':successes}
    else:
        rROIxLevel = np.zeros((int(size_output_layer-1),3))
        
        rSampsXlevel = np.zeros((int(size_output_layer-1),2))
        
#        rROIxLevelxExt = np.zeros((int(size_output_layer-1),10,2))
#        rSampsXlevelXExt = np.zeros((int(size_output_layer-1),10))
        
        log = pd.DataFrame(columns=['DateTime','Message'])
        
        fixed_spread_ratios = np.array([0.00005,0.0001,0.0002,0.0003,0.0004,0.0005])
        fROIs = np.zeros((fixed_spread_ratios.shape))
        
        summary = {'tGROI':Journal['GROI'].sum(),
               'tROI':Journal['ROI'].sum(),
               'tSpread':Journal['Spread'].sum(),
               'rGROI':0,
               'rROI':0,
               'fROIs':fROIs,
               'Sharpe':0,
               'varRet':[0,0],
               'successes':[0,0,0,fROIs]}
    
#    summary = {'tGROI':Journal['GROI'].sum(),
#               'tROI':Journal['ROI'].sum(),
#               'tSpread':Journal['Spread'].sum(),
#               'rGROI':100*rGROI,
#               'rROI':100*rROI,
#               'fROIs':100*fROIs,
#               'Sharpe':sharpe_ratio,
#               'varRet':varRet,
#               'successes':successes}
    
    Journal['GROI'] = Journal['GROI'].astype(float)
    Journal['Spread'] = Journal['Spread'].astype(float)
    Journal['ROI'] = Journal['ROI'].astype(float)
    #print(Journal)
    return Journal,rROIxLevel,rSampsXlevel,summary,log

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

def init_results_dict(origin='merge'):
    """  """
    if origin == 'merge':
        results_dict = {"epoch":0,
                   "t_index":0,
                   "RD":0,
                   "NZ":0,
                   "NZA":0,
                   "AD":0.0,
                   "ADA":0.0,
                   "pNZ":0.0,
                   "pNZA":0.0,
                   "rGROI":0.0,
                   "rROI":0.0,
                   "sharpe":0.0,
                   "varGROI":0.0,
                   "varROI":0.0,
                   "GSP":0.0,
                   "NSP":0.0,
                   "NO":0,
                   "NSP2p":0.0,
                   "SI2p":0.0,
                   "RSI":0.0,
                   "thr_mc":0.0,
                   "thr_md":0.0}
    return results_dict

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
    results_dir = '../RNN/results/'
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