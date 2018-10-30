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
import h5py
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

def init_TR_v11(resultsDir, ID, t_index, thr_mc, save_results, epoch):
    """
    Version 1.1 of initializing table results. This one without loading previously results yet.
    """
    # init columns and dtypes
#    columns=['J_test', 'J_train', 'AccMC',
#                 '0.5RD','0.5NZ','0.5NZA','0.5AD','0.5ADA','0.5pNZ','0.5pNZA','0.5rROI','0.5GROI',
#                 '0.6RD','0.6NZ','0.6NZA','0.6AD','0.6ADA','0.6pNZ','0.6pNZA','0.6rROI','0.6GROI',
#                 '0.7RD','0.7NZ','0.7NZA','0.7AD','0.7ADA','0.7pNZ','0.7pNZA','0.7rROI','0.7GROI',
#                 '0.8RD','0.8NZ','0.8NZA','0.8AD','0.8ADA','0.8pNZ','0.8pNZA','0.8rROI','0.8GROI',
#                 '0.9RD','0.9NZ','0.9NZA','0.9AD','0.9ADA','0.9pNZ','0.9pNZA','0.9rROI','0.9GROI']
    
    input_dict={'J_test':0.0,'J_train':0.0,'AccMC':0.0,
            '0.5RD':0,'0.5NZ':0,'0.5NZA':int,'0.5AD':0.0,'0.5ADA':0.0,'0.5pNZ':0.0,'0.5pNZA':0.0,'0.5rROI':0.0,'0.5GROI':0.0,
            '0.6RD':0,'0.6NZ':0,'0.6NZA':int,'0.6AD':0.0,'0.6ADA':0.0,'0.6pNZ':0.0,'0.6pNZA':0.0,'0.6rROI':0.0,'0.6GROI':0.0,
            '0.7RD':0,'0.7NZ':0,'0.7NZA':int,'0.7AD':0.0,'0.7ADA':0.0,'0.7pNZ':0.0,'0.7pNZA':0.0,'0.7rROI':0.0,'0.7GROI':0.0,
            '0.8RD':0,'0.8NZ':0,'0.8NZA':int,'0.8AD':0.0,'0.8ADA':0.0,'0.8pNZ':0.0,'0.8pNZA':0.0,'0.8rROI':0.0,'0.8GROI':0.0,
            '0.9RD':0,'0.9NZ':0,'0.9NZA':int,'0.9AD':0.0,'0.9ADA':0.0,'0.9pNZ':0.0,'0.9pNZA':0.0,'0.9rROI':0.0,'0.9GROI':0.0}
        
#    dtypes={'J_test':float,'J_train':float,'AccMC':float,
#            '0.5RD':int,'0.5NZ':int,'0.5NZA':int,'0.5AD':float,'0.5ADA':float,'0.5pNZ':float,'0.5pNZA':float,'0.5rROI':float,'0.5GROI':float,
#            '0.6RD':int,'0.6NZ':int,'0.6NZA':int,'0.6AD':float,'0.6ADA':float,'0.6pNZ':float,'0.6pNZA':float,'0.6rROI':float,'0.6GROI':float,
#            '0.7RD':int,'0.7NZ':int,'0.7NZA':int,'0.7AD':float,'0.7ADA':float,'0.7pNZ':float,'0.7pNZA':float,'0.7rROI':float,'0.7GROI':float,
#            '0.8RD':int,'0.8NZ':int,'0.8NZA':int,'0.8AD':float,'0.8ADA':float,'0.8pNZ':float,'0.8pNZA':float,'0.8rROI':float,'0.8GROI':float,
#            '0.9RD':int,'0.9NZ':int,'0.9NZA':int,'0.9AD':float,'0.9ADA':float,'0.9pNZ':float,'0.9pNZA':float,'0.9rROI':float,'0.9GROI':float}
    
    TRdf = pd.DataFrame(input_dict, index=[epoch])
    TRdf.index.name = 'Epoch'
    filename = ""
    
    if save_results:
        if os.path.exists(resultsDir+ID+"/")==False:
            os.mkdir(resultsDir+ID+"/")
        if os.path.exists(resultsDir+ID+"/t"+str(t_index)+"/")==False:
            os.mkdir(resultsDir+ID+"/t"+str(t_index)+"/")
        
        filename = resultsDir+ID+"/t"+str(t_index)+"/"+ID+"t"+str(t_index)+"thrmc"+str(thr_mc)+".txt"
        
#        if not os.path.exists(filename):
#            TRdf = pd.DataFrame(columns=columns)
#            TRdf.index.name = 'Epoch'
#        else:
#            TRdf = pd.read_csv(filename,sep='\t',index_col="Epoch")
            
    return TRdf, filename

def init_TR_v20(resultsDir,ID,t_index,thr_mc,save_results):
    
    #columns=['epoch','thr_mc','thr_md','J_test', 'J_train', 'AccMC','pNZ','pNZA','RD','NZ','NZA','AD','ADA','rROI','rGROI','tROI','tGROI']
        
    TRdf = pd.DataFrame()
    
    if save_results:
        if os.path.exists(resultsDir+ID+"/")==False:
            os.mkdir(resultsDir+ID+"/")
        if os.path.exists(resultsDir+ID+"/t"+str(t_index)+"/")==False:
            os.mkdir(resultsDir+ID+"/t"+str(t_index)+"/")
        
        filename = resultsDir+ID+"/t"+str(t_index)+"/"+ID+"t"+str(t_index)+"_results.txt"
        
        if not os.path.exists(filename):
            TRdf = pd.DataFrame()
            #TRdf = pd.DataFrame(columns=columns)
            TRdf.index.name = 'index'
        else:
            TRdf = pd.read_csv(filename,sep='\t',index_col='index')
            
    return TRdf

def init_TR_v10(resultsDir,ID,t_index,thr_mc,save_results):
    
    columns=['J_test', 'J_train', 'AccMC',
                 '0.5RD','0.5NZ','0.5NZA','0.5AD','0.5ADA','0.5pNZ','0.5pNZA','0.5rROI','0.5GROI',
                 '0.6RD','0.6NZ','0.6NZA','0.6AD','0.6ADA','0.6pNZ','0.6pNZA','0.6rROI','0.6GROI',
                 '0.7RD','0.7NZ','0.7NZA','0.7AD','0.7ADA','0.7pNZ','0.7pNZA','0.7rROI','0.7GROI',
                 '0.8RD','0.8NZ','0.8NZA','0.8AD','0.8ADA','0.8pNZ','0.8pNZA','0.8rROI','0.8GROI',
                 '0.9RD','0.9NZ','0.9NZA','0.9AD','0.9ADA','0.9pNZ','0.9pNZA','0.9rROI','0.9GROI']
        
#        dtypes={'J_test':float,'J_train':float,'AccMC':float,
#                '0.5RD':int,'0.5NZ':int,'0.5NZA':int,'0.5AD':float,'0.5ADA':float,'0.5pNZ':float,'0.5pNZA':float,'0.5rROI':float,'0.5GROI':float,
#                '0.6RD':int,'0.6NZ':int,'0.6NZA':int,'0.6AD':float,'0.6ADA':float,'0.6pNZ':float,'0.6pNZA':float,'0.6rROI':float,'0.6GROI':float,
#                '0.7RD':int,'0.7NZ':int,'0.7NZA':int,'0.7AD':float,'0.7ADA':float,'0.7pNZ':float,'0.7pNZA':float,'0.7rROI':float,'0.7GROI':float,
#                '0.8RD':int,'0.8NZ':int,'0.8NZA':int,'0.8AD':float,'0.8ADA':float,'0.8pNZ':float,'0.8pNZA':float,'0.8rROI':float,'0.8GROI':float,
#                '0.9RD':int,'0.9NZ':int,'0.9NZA':int,'0.9AD':float,'0.9ADA':float,'0.9pNZ':float,'0.9pNZA':float,'0.9rROI':float,'0.9GROI':float}
    
    TRdf = pd.DataFrame()
    
    if save_results:
        if os.path.exists(resultsDir+ID+"/")==False:
            os.mkdir(resultsDir+ID+"/")
        if os.path.exists(resultsDir+ID+"/t"+str(t_index)+"/")==False:
            os.mkdir(resultsDir+ID+"/t"+str(t_index)+"/")
        
        filename = resultsDir+ID+"/t"+str(t_index)+"/"+ID+"t"+str(t_index)+"thrmc"+str(thr_mc)+".txt"
        
        if not os.path.exists(filename):
            TRdf = pd.DataFrame(columns=columns)
            TRdf.index.name = 'Epoch'
        else:
            TRdf = pd.read_csv(filename,sep='\t',index_col="Epoch")
            
    return TRdf

def save_results_v20(TRdf, t_index, thr_mc, epoch, lastTrained, save_results=0, resultsDir="",ID=""):
    
    if save_results:
        filename = resultsDir+ID+"/t"+str(t_index)+"/"+ID+"t"+str(t_index)+"_results.txt"
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
        TRdf.to_csv(filename,sep='\t',float_format='%.3f',index_label='index')
    
    return None

def print_results_v20(TRdf, results, epoch, thr_md, thr_mc, t_index, save_results = 0):
    
    print("Epoch = "+str(epoch)+". Time index = "+str(t_index)+". Threshold MC = "+str(thr_mc)+". Threshold MD = "+str(thr_md))
    if thr_md==.5 and thr_mc==.5:
        print("J_test = "+str(results["J_test"])+", J_train = "+str(results["J_train"])+", Accuracy="+str(results["Acc"]))
    #print("".format())
    print("RD = {0:d} NZ = {1:d} NZA = {2:d} pNZ = {5:.3f}% pNZA = {6:.3f}% AD = {3:.2f}% ADA = {4:.2f}% NO = {9:d} GSP = {7:.2f}% NSP = {8:.2f}%"
          .format(results["RD"],results["NZ"],results["NZA"],results["AccDir"],results["AccDirA"],
            results["perNZ"],results["perNZA"],results["GSP"],results["NSP"],results["NO"]))
    #print(". AccDirA = {2:.2f}%".format(results["NZA"],results["accDirectionAll"]))
    print("Sharpe = {1:.3f} rRl1 = {2:.4f}% rRl2 = {3:.4f}% rGROI = {4:.4f}% rROI = {0:.4f}% tGROI = {5:.4f}% tROI = {6:.4f}%".format(
            results["rROI"],results["sharpe"],results["rROIxLevel"][0,0],results["rROIxLevel"][1,0],
            results["rGROI"],results["tGROI"],results["tROI"]))
    
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
        TRdf = TRdf.append(pd.DataFrame(data=new_row,index=[TRdf.shape[0]]))
        
#        TRdf.loc[epoch]["thr_mc"] = thr_mc
#        TRdf.loc[epoch]["thr_md"] = thr_md
#        TRdf.loc[epoch]["J_test"] = 100*results["J_test"]
#        TRdf.loc[epoch]["J_train"] = 100*results["J_train"]
#        TRdf.loc[epoch]["AccMC"] = results["Acc"]
#        TRdf["RD"].loc[epoch] = results["RD"]
#        TRdf["NZ"].loc[epoch] = results["NZ"]
#        TRdf["AD"].loc[epoch] = results["AccDir"]
#        TRdf["NZA"].loc[epoch] = results["NZA"]
#        TRdf["ADA"].loc[epoch] = results["AccDirA"]
#        TRdf["pNZ"].loc[epoch] = results["perNZ"]
#        TRdf["pNZA"].loc[epoch] = results["perNZA"]
#        TRdf["rGROI"].loc[epoch] = results["rGROI"]
#        TRdf["rROI"].loc[epoch] = results["rROI"]
#        TRdf["tROI"].loc[epoch] = results["tGROI"]
#        TRdf["tROI"].loc[epoch] = results["tROI"]
    
    else:
        TRdf = pd.DataFrame()
        
    return TRdf



def get_last_saved_epoch(resultsDir, ID, t_index):
    """
    <DocString>
    """
    filename = resultsDir+ID+"/t"+str(t_index)+"/"+ID+"t"+str(t_index)+"_results.txt"
    if os.path.exists(filename):
        TR = pd.read_csv(filename,sep='\t',index_col="index")
        last_saved_epoch = TR.epoch.iloc[-1]
    else:
        last_saved_epoch = -1
    return last_saved_epoch

def save_best_results(BR_GROI, BR_ROI, BR_sharpe, resultsDir, ID, save_results):
    
    if save_results:
        filename_GROIs = resultsDir+ID+"/"+ID+"_BR_GROI.txt"
        filename_ROIs = resultsDir+ID+"/"+ID+"_BR_ROI.txt"
        filename_sharpes = resultsDir+ID+"/"+ID+"_BR_sharpe.txt"
        
        
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

def evaluate_RNN(data, model, y, DTA, IDresults, IDweights, J_test, soft_tilde, save_results,
                 costs, epoch, resultsDir, lastTrained, save_journal=False):
    """
    Evaluate RNN results for one epoch.
    """
    
    m = y.shape[0]
    n_days = len(data.dateTest)
    
    # extract y_c vector, if necessary
    if model.commonY == 0:
        #pass
        for t in range(model.seq_len-1,model.seq_len):
            t_out = model.seq_len-t
            t_index = model.seq_len-t_out
            #print(y[:,-t_out,:])
            #print(soft_tilde[:,-t_out,:])
            Y_real = np.argmax(y[:,-t_out,:], 1)-(model.size_output_layer-1)/2
            Y_tilde = np.argmax(soft_tilde[:,-t_out,:], 1)-(model.size_output_layer-1)/2
            
            diff = Y_tilde-Y_real
            
            zerosYt = Y_tilde==0
            nonZerosYt = Y_tilde!=0
            nonZerosYr = Y_real!=0
            nonZerosYtYr = nonZerosYr*nonZerosYt
            accuracy = np.sum(diff==0)/Y_tilde.shape[0]
            accDirectionInd = np.sign(Y_tilde[nonZerosYtYr])-np.sign(Y_real[nonZerosYtYr])
            accDirectionAllInd = np.abs(np.sign(Y_tilde[nonZerosYt])-np.sign(Y_real[nonZerosYt]))
            #print(accDirectionAllInd)
            #print(np.sign(Y_real[nonZerosYt]))
            Z = np.sum(zerosYt)
            RD = np.sum(accDirectionInd==0) # right direction
            NZ = np.sum(nonZerosYtYr) # non zeros
            accDirection = RD/NZ
            RDA = np.sum(accDirectionAllInd==0) #right direction All
            NZA = np.sum(nonZerosYt) #Total all
            accDirectionAll = RDA/NZA
            percentNonZeros = np.sum(nonZerosYtYr)/m
            percentNonZerosAll = np.sum(nonZerosYt)/m
            
            print("J_test = "+str(J_test)+", J_train = "+str(costs[IDweights+str(epoch)])+", Accuracy="+str(100*accuracy))
            print("Epoch "+str(epoch)+" "+"t_index="+str(t_index)+":")
            print("accDirection "+str(100*accDirection))
            print("accDirectionAll "+str(100*accDirectionAll))
            print("percentNonZeros "+str(100*percentNonZeros))
            print("percentNonZerosAll "+str(100*percentNonZerosAll))
            print("RD "+str(RD))
            print("NZ "+str(NZ))
            #print("RDA "+str(RDA))
            print("NZA "+str(NZA))
            print("Z "+str(Z))
            print("m "+str(Y_tilde.shape[0]))
        
    elif model.commonY == 1:
        # extract non-zeros (y_c0>0.5)
        for t in range(model.seq_len): 
            t_out = model.seq_len-t
            t_index = model.seq_len-t_out
            
            y_mc = y[:,-t_out,0]>=.5
            y_mc_tilde = soft_tilde[:,-t_out,0]>=.5
            #yc_0_tilde = soft_tilde[:,-t_out,0]>=np.sort(soft_tilde[:,-t_out,0])
            print(np.max(soft_tilde[:,-t_out,0]))
            print(np.min(soft_tilde[:,-t_out,0]))
            
            #print(y[:,-t_out,:])
            #print(np.min(soft_tilde[:,-t_out,0]))
            #print(np.max(yc_0_tilde))
            
            Acc = 1-np.sum(np.abs(y_mc-y_mc_tilde))/m
            print("J_test = "+str(J_test)+", J_train = "+str(costs[IDweights+str(epoch)])+", Accuracy="+str(100*Acc))
            
            

    elif model.commonY == 2:
        # extract down outputs ([y_c1,y_c2]=10)
        pass
    elif model.commonY == 3:
        
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
        NZpp = np.zeros((model.seq_len+1, len(thresholds_mc), len(thresholds_md), int((model.size_output_layer-1)/2))).astype(int)
        GRE = np.zeros((model.seq_len+1, len(thresholds_mc), len(thresholds_md), int((model.size_output_layer-1)/2)))
        GREw = np.zeros((model.seq_len+1, len(thresholds_mc), len(thresholds_md), int((model.size_output_layer-1)/2)))
        
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
                
                TRdf = init_TR_v20(resultsDir,IDresults,t_index,thr_mc,save_results)
                
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
                                DTAt = DTA.iloc[t-1::model.seq_len,:]
                            
                            
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
                                    
                                    journal_id = IDresults+"t"+str(t_index)+"mc"+str(thr_mc)+"md"+str(thr_md)+"e"+str(epoch)+".txt"
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
                                                        "thr_mc":thr_mc,
                                                        "thr_md":thr_md}
                                # update best sharpe ratio
                                if results["sharpe"]>best_sharpe and results["sharpe"] != float('Inf') and results["NZA"]>=20 and ub_mc==1 and ub_md==1:
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
                                                        "thr_mc":thr_mc,
                                                        "thr_md":thr_md}
                                    
                                if ub_mc==1 and ub_md==1:
                                    TRdf = print_results_v20(TRdf, results, epoch, thr_md, thr_mc, t_index, save_results=save_results)
                                    # get AD KPI
                                    #print(2/3*AccDir+1/3*AccDirA)
        #                            AD_resume[0,thr_idx] = 2/3*AccDir+1/3*AccDirA
                                    #print(thr_idx)
                                    #if t<model.seq_len:
                                    AD_resumes[t,map_idx2thr[thr_idx],0] = 2/3*AccDir+1/3*AccDirA
                                    thr_idx += 1
                                else:
                                    print("eROIpp for ub_mc="+str(round(ub_mc*10)/10)+" and ub_md="+str(ub_md))
#                                    print(np.sum(np.abs(y_dec_mg_tilde[y_md_tilde])==0))
                                    for b in range(int((model.size_output_layer-1)/2)):
#                                        print(np.sum(np.abs(y_dec_mg_tilde[y_md_tilde])==(b+1)))
                                        NZpp[t,i_t_mc,i_t_md, b] = int(rSampsXlevel[b])#int(np.sum(np.abs(y_dec_mg_tilde[y_md_tilde]).astype(int)==(b+1)))
                                        eROIpp[t,i_t_mc,i_t_md, b, 0] = rROIxLevel[b,0]/100
                                        eROIpp[t,i_t_mc,i_t_md, b, 1] = rROIxLevel[b,1]/100
                                        if NZpp[t,i_t_mc,i_t_md, b]>0:
                                            GRE[t,i_t_mc,i_t_md, b] = eROIpp[t,i_t_mc,i_t_md, b, 1]/NZpp[t,i_t_mc,i_t_md, b]
                                        print("GRE level "+str(b)+": "+str(GRE[t,i_t_mc,i_t_md, b]/0.0001)+" pips")
                                        print("Nonzero entries = "+str(NZpp[t,i_t_mc,i_t_md, b]))
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
            # Fillthe gaps of GRE
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
                                    
                                    
                                
                    
                    
            pickle.dump( AD_resumes, open( resultsDir+IDresults+"/AD_e"+str(epoch)+".p", "wb" ))
            pickle.dump( NZpp, open( resultsDir+IDresults+"/NZpp_e"+str(epoch)+".p", "wb" ))
            pickle.dump( eROIpp, open( resultsDir+IDresults+"/eROIpp_e"+str(epoch)+".p", "wb" ))
            pickle.dump( GRE, open( resultsDir+IDresults+"/GRE_e"+str(epoch)+".p", "wb" ))
            print("GRE")
            print(GRE)
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
            
    rGROI, rROI, fROIs, sharpe_ratio, rROIxLevel, rSampsXlevel, log, varRet, successes = get_real_ROI(5, Journal, n_days, fixed_spread=fixed_spread)
    

    print("Sharpe = {0:.3f} rRl1 = {1:.4f}% rRl2 = {2:.4f}% rGROI = {3:.4f}% rROI = {4:.4f}%".format(
            sharpe_ratio,rROIxLevel[0,1],rROIxLevel[0,0],100*rGROI,100*rROI))
    
    return None

def get_real_ROI(size_output_layer, Journal, n_days, fixed_spread=0):
    """
    Function that calculates real ROI, GROI, spread...
    """
    
    if 'DT1' in Journal.columns:
        DT1 = 'DT1'
        DT2 = 'DT2'
        A1 = 'A1'
        A2 = 'A2'
        B1 = 'B1'
        B2 = 'B2'
    else:
        DT1 = 'Entry Time'
        DT2 = 'Exit Time'
        A1 = 'Ai'
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

    rROIxLevel = np.zeros((int(size_output_layer-1),2))
    rSampsXlevel = np.zeros((int(size_output_layer-1)))
    
    maxExtentions = 10
    
#    rROIxLevelxExt = np.zeros((int(size_output_layer-1),maxExtentions,2))
#    rSampsXlevelXExt = np.zeros((int(size_output_layer-1),maxExtentions))
    pip = 0.0001
    
    fixed_spread_ratios = np.array([0.00005,0.0001,0.0002,0.0003,0.0004,0.0005])
    fROIs = np.zeros((fixed_spread_ratios.shape))
    ROI_vector = np.array([])
    GROI_vector = np.array([])
    if Journal.shape[0]>0:
        log=log.append({'DateTime':Journal[DT1].iloc[0],'Message':Journal['Asset'].iloc[0]+" open" },ignore_index=True)
        
    #print(Journal)
    for e in range(1,Journal.shape[0]):

        oldExitTime = dt.datetime.strptime(Journal[DT2].iloc[e-1],"%Y.%m.%d %H:%M:%S")
        newEntryTime = dt.datetime.strptime(Journal[DT1].iloc[e],"%Y.%m.%d %H:%M:%S")

        extendExitMarket = (newEntryTime-oldExitTime<=dt.timedelta(0))
        sameAss = Journal['Asset'].iloc[e] == Journal['Asset'].iloc[e-1] 
        sameDir = Journal['Bet'].iloc[e-1]*Journal['Bet'].iloc[e]>0

        if sameAss and extendExitMarket:# and sameDir:
            #print("continue")
            log=log.append({'DateTime':Journal[DT1].iloc[e],'Message':Journal['Asset'].iloc[e]+" extended" },ignore_index=True)
            n_pos_extended += 1
            this_pos_extended += 1
            rSampsXlevel[int(np.abs(Journal['Bet'].iloc[eInit])-1)] += 1
            continue
        else:
            
            thisSpread = (Journal[A2].iloc[e-1]-Journal[B2].iloc[e-1])/Journal[B1].iloc[e-1]
                
            GROI = np.sign(Journal['Bet'].iloc[eInit])*(Journal[B2].iloc[e-1]-Journal[B1].iloc[eInit])/Journal[B1].iloc[eInit]

            rGROI += GROI
            
            ROI = GROI-thisSpread
            ROI_vector = np.append(ROI_vector,ROI)
            GROI_vector = np.append(GROI_vector,GROI)
            rROI += ROI
            fROIs = fROIs+GROI-fixed_spread_ratios
            rROIxLevel[int(np.abs(Journal['Bet'].iloc[eInit])-1),0] += 100*ROI
            rROIxLevel[int(np.abs(Journal['Bet'].iloc[eInit])-1),1] += 100*GROI
            rSampsXlevel[int(np.abs(Journal['Bet'].iloc[eInit])-1)] += 1

#            rROIxLevelxExt[int(np.abs(Journal['Bet'].iloc[eInit])-1),np.min(maxExtentions,this_pos_extended),:] += [ROI/pip,GROI/pip]
#            rSampsXlevelXExt[int(np.abs(Journal['Bet'].iloc[eInit])-1),np.min(maxExtentions,this_pos_extended)] += 1
            
            this_pos_extended = 0
            
            if GROI>0:
                gross_succ_counter += 1
            if ROI>0:
                net_succ_counter += 1
                
            log=log.append({'DateTime':Journal[DT2].iloc[e-1],'Message':" Close "+Journal['Asset'].iloc[e-1]+
                            " entry bid {0:.4f}".format(Journal[B1].iloc[eInit])+" exit bid {0:.4f}".format(Journal[B2].iloc[e-1])+
                            " GROI {0:.4f}% ".format(100*GROI)+" ROI {0:.4f}% ".format(100*ROI)+" tGROI {0:.4f}% ".format(100*rGROI) },ignore_index=True)
            #if e<Journal.shape[0]-1:
            log=log.append({'DateTime':Journal[DT1].iloc[e],'Message':Journal['Asset'].iloc[e]+" open" },ignore_index=True)
            n_pos_opned += 1

            eInit = e
        # end of if (sameAss and extendExitMarket):
    # end of for e in range(1,Journal.shape[0]):
    
    if Journal.shape[0]>0:     
        #print("last")
        #print((infoBets["A2"].iloc[e]-infoBets["B2"].iloc[e])/infoBets["B1"].iloc[e])
        thisSpread = (Journal[A2].iloc[-1]-Journal[B2].iloc[-1])/Journal[B1].iloc[-1]

        #print((infoBets["B2"].iloc[e]-infoBets["B1"].iloc[eInit])/infoBets["B1"].iloc[e])
        GROI = np.sign(Journal['Bet'].iloc[eInit])*(Journal[B2].iloc[-1]-Journal[B1
                    ].iloc[eInit])/Journal[B1].iloc[-1]

        rGROI += GROI

        ROI = GROI-thisSpread
        ROI_vector = np.append(ROI_vector,ROI)
        GROI_vector = np.append(GROI_vector,GROI)
        rROI += ROI
        fROIs = fROIs+GROI-fixed_spread_ratios
        rROIxLevel[int(np.abs(Journal['Bet'].iloc[eInit])-1),0] += 100*ROI
        rROIxLevel[int(np.abs(Journal['Bet'].iloc[eInit])-1),1] += 100*GROI
        
        #            rROIxLevelxExt[int(np.abs(Journal['Bet'].iloc[eInit])-1),np.min(maxExtentions,this_pos_extended),:] += [ROI/pip,GROI/pip]
#            rSampsXlevelXExt[int(np.abs(Journal['Bet'].iloc[eInit])-1),np.min(maxExtentions,this_pos_extended)] += 1
        
        rSampsXlevel[int(np.abs(Journal['Bet'].iloc[eInit])-1)] += 1
        
        if GROI>0:
            gross_succ_counter += 1
        if ROI>0:
            net_succ_counter += 1
        
        log = log.append({'DateTime':Journal[DT2].iloc[eInit],'Message':Journal['Asset'].iloc[eInit]+" close GROI {0:.4f}% ".format(100*GROI)+" ROI {0:.4f}% ".format(100*ROI)+
                        " tGROI {0:.4f}% ".format(100*rGROI) },ignore_index=True)
            
        #thisProbIndex = np.argmax((np.floor(10*prob[eInit])/10==percents))-1
        #rROIxPercent[thisProbIndex] += 100*(thisGROI-thisSpread)
#    print("n_pos_opned="+str(n_pos_opned))
#    print("n_pos_extended="+str(n_pos_extended))
#    print("journal_entries="+str(Journal.shape[0]))
#    print(log.to_string())
#    print(Journal.shape[0])
    
    gross_succ_per = gross_succ_counter/n_pos_opned
    net_succ_per = net_succ_counter/n_pos_opned
    successes = [n_pos_opned, 100*gross_succ_per, 100*net_succ_per]
    varRet = [100000*np.var(ROI_vector), 100000*np.var(GROI_vector)]
    
    n_bets = ROI_vector.shape[0]
    sharpe_ratio = np.sqrt(n_bets)*np.mean(ROI_vector)/(np.sqrt(np.var(ROI_vector))*n_days)

    #rROI = rGROI-rSpread
    
    return rGROI, rROI, fROIs, sharpe_ratio, rROIxLevel, rSampsXlevel, log, varRet, successes

def getJournal_v20(DTA, y_dec_tilde, y_dec, diff, probs_mc, probs_md, size_output_layer, n_days, fixed_spread=0, get_real=1, save_journal=False):
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
        rGROI, rROI, fROIs, sharpe_ratio, rROIxLevel, rSampsXlevel, log, varRet, successes = get_real_ROI(size_output_layer, Journal, n_days, fixed_spread=fixed_spread)
        
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
        rROIxLevel = np.zeros((int(size_output_layer-1),2))
        
        rSampsXlevel = np.zeros((int(size_output_layer-1)))
        
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
               'successes':[0,0,0]}
    
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

#[[  1.89644665e-01   4.49722797e-01   5.50277174e-01   1.61192399e-02
#    4.24829900e-01   8.16543881e-12   5.45770705e-01   1.32801458e-02]
# [  2.44395345e-01   4.44633484e-01   5.55366457e-01   2.40209624e-02
#    4.14886951e-01   3.51778828e-10   5.42823434e-01   1.82686001e-02]
# [  2.50142217e-01   4.36904818e-01   5.63095212e-01   1.55553315e-02
#    4.06303853e-01   5.46768131e-10   5.65299928e-01   1.28408233e-02]]
#sum_AD[0,0] 1.60078084114
#[[  2.28120242e-01   4.43772570e-01   5.56227410e-01 ...,   3.02071873e-10
#    5.51217537e-01   1.48221295e-02]
# [  2.30687945e-01   4.68803048e-01   5.31197011e-01 ...,   4.92423057e-10
#    5.22672293e-01   1.61536893e-02]
# [  1.89086640e-01   4.28530739e-01   5.71469301e-01 ...,   7.28038209e-10
#    5.70109309e-01   9.16342052e-03]
# ..., 
# [  4.41239828e-01   4.63852478e-01   5.36147502e-01 ...,   3.39533312e-10
#    4.90207105e-01   4.52598002e-02]
# [  4.80076799e-01   5.01303982e-01   4.98696028e-01 ...,   7.48612705e-10
#    4.31808985e-01   6.18215955e-02]
# [  4.29431010e-01   4.80957717e-01   5.19042254e-01 ...,   2.71653928e-10
#    4.78077795e-01   4.12066225e-02]]