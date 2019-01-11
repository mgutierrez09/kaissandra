# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 12:00:20 2018

@author: mgutierrez
Script to save features, results and sats into hdf5 files
"""

import numpy as np
import time
import pandas as pd
import h5py
import datetime as dt
import pickle
import scipy.io as sio
import os

class Data:
    
    AllAssets = {"0":"[USDX]",
             "1":'AUDCAD',
             "2":'EURAUD',
             "3":'EURCAD',
             "4":'EURCHF',
             "5":'EURCZK',
             "6":'EURDKK',
             "7":'EURGBP',
             "8":'EURNZD',
             "9":'EURPLN',
             "10":'EURUSD',
             "11":'GBPAUD',
             "12":'GBPCAD',
             "13":'GBPCHF',
             "14":'GBPUSD',
             "15":'GOLD',
             "16":'USDCAD',
             "17":'USDCHF',
             "18":'USDHKD',
             "19":'USDJPY',
             "20":'USDMXN',
             "21":'USDNOK',
             "22":'USDPLN',
             "23":'USDRUB',
             "24":'USDSGD',
             "25":'XAGUSD',
             "26":'XAUUSD',
             "27":"CADJPY",
             "28":"EURJPY",
             "29":"AUDJPY",
             "30":"CHFJPY",
             "31":"GBPJPY",
             "32":"NZDUSD"}
    
    cwt_coeff = [3, 9,2,3,11,12,5,4,8,14,7,6,13,10,13,10,5,11, 5,13,14, 3,1, 4,
                 12, 6,10,4, 7,2, 9, 8,9,1,12, 5, 2, 1, 8,11, 4, 7, 6, 2,14, 3]
    cwt_weights = [5,10,2,2,10,10,5,5,5,10,5,5,10,10, 5, 5,2, 5,10,20,20,10,5,10,
                   5,10,20,2,10,5,20,10,5,2,20,20,10,10,20,20,20,20,20,20, 5,20]
    
    AllFeatures = {# Manueal features
                "0":"bids",
                "1":"EMA01",
                "2":"EMA05",
                "3":"EMA1",
                "4":"EMA5",
                "5":"EMA10",
                "6":"EMA50",
                "7":"EMA100",
                "8":"variance",
                "9":"timeInterval",
                "10":"parSARhigh",
                "11":"parSARlow",
                "12": "time",
                "13":"parSARhigh2",
                "14":"parSARlow2",
                "15":"difVariance",
                "16":"difTimeInterval",
                "17":"maxValue",
                "18":"minValue",
                "19":"difMaxValue",
                "20":"difMinValue",
                "21":"minOmax",
                "22":"difMinOmax",
                "23":"bidOema01",
                "24":"bidOema05",
                "25":"bidOema1",
                "26":"bidOema5",
                "27":"bidOema10",
                "28":"bidOema50",
                "29":"bidOema100",
                "30":"difBidOema01",
                "31":"difBidOema05",
                "32":"difBidOema1",
                "33":"difBidOema5",
                "34":"difBidOema10",
                "35":"difBidOema50",
                "36":"difBidOema100",
                # TSFRESH features
                "37":["quantile",0.3,1],
                "38":["quantile",0.4,1],
                "39":["quantile",0.2,1],
                "40":["fft_coefficient",0,"real",1],
                "41":["fft_coefficient",0,"abs",1],
                "42":["sum_values",1],
                "43":["median",1],
                "44":["c3",3,1],
                "45":["c3",2,1],
                "46":["c3",1,1],
                "47":["mean",1],
                "48":["minimum",1],
                "49":["median",1],
                "50":["cwt_coefficients",[2, 5, 10, 20],cwt_coeff,cwt_weights,len(cwt_weights)],
                "51":["maximum",1],
                "52":["quantile",0.1,1],
                "53":["quantile",0.8,1],
                "54":["quantile",0.9,1],
                "55":["quantile",0.6,1],
                "56":["quantile",0.7,1],
                "57":["abs_energy",1],
                "58":["linear_trend","intercept",1],
                "59":["agg_linear_trend","min",50,"intercept",1],
                "60":["agg_linear_trend","min",5,"intercept",1],
                "61":["agg_linear_trend","max",10,"intercept",1],
                "62":["agg_linear_trend","mean",50,"intercept",1],
                "63":["agg_linear_trend","max",50,"intercept",1],
                "64":["agg_linear_trend","mean",5,"intercept",1],
                "65":["agg_linear_trend","min",10,"intercept",1],
                "66":["agg_linear_trend","mean",10,"intercept",1],
                "67":["agg_linear_trend","max",5,"intercept",1],
                # Variations-based features
                "68":"vars",
                "69":"EMAvar01",
                "70":"EMAvar05",
                "71":"EMAvar1",
                "72":"EMAvar5",
                "73":"EMAvar10",
                "74":"EMAvar50",
                "75":"EMAvar100",
                "76":"varVar",
                "77":"timeIntervalVar",
                "78":"parSARhigh20Var",
                "79":"parSARlow20Var",
                "80":"parSARhigh2Var",
                "81":"parSARlow2Var",
                "82": "timeVar",
                "83":"maxValueVar",
                "84":"minValueVar",
                "85":"minOmaxVar",
                "86":"varOema01",
                "87":"varOema05",
                "88":"varOema1",
                "89":"varOema5",
                "90":"varOema10",
                "91":"varOema50",
                "92":"varOema100"}
    
    
    average_over = np.array([0.1, 0.5, 1, 5, 10, 50, 100])
    std_var = 0.1
    std_time = 0.1
    lookAheadVector=[.1,.2,.5,1]
    parsars = [20,2]
    
    def __init__(self, movingWindow=100,nEventsPerStat=1000,lB=1200,
                 channels=[0],divideLookAhead=1,lookAheadIndex=3,
                 feature_keys_manual=[i for i in range(37)],
                 dateTest=['2017.09.15','2017.11.06','2017.11.07'],
                 assets=[1,2,3,4,7,8,10,11,12,13,14,15,16,17,19,27,28,29,30,31,32],
                 max_var=10, feature_keys_tsfresh=[],#[i for i in range(37,68)]
                 noVarFeatsManual=[8,9,12,17,18,21,23,24,25,26,27,28,29],
                 trsfresh_from_variations=True,
                 var_feat_keys=[]):
        
        self.movingWindow = movingWindow
        self.nEventsPerStat = nEventsPerStat
        
        self.feature_keys_manual = feature_keys_manual
        self.var_feat_keys = var_feat_keys
        self.n_feats_manual = len(self.feature_keys_manual)
        self.feature_keys_tsfresh = feature_keys_tsfresh
        self.lB = lB
        self.divideLookAhead = divideLookAhead
        
        assert(max(channels)<nEventsPerStat/movingWindow)
        self.channels = channels
        
        self.lbd=1-1/(self.nEventsPerStat*self.average_over)
        self.feature_keys = feature_keys_manual+feature_keys_tsfresh
        self.nFeatures = len(self.feature_keys)
        self.dateTest = dateTest
        self.assets = assets
        if not trsfresh_from_variations:
            self.noVarFeats = noVarFeatsManual+feature_keys_tsfresh
        else:
            self.noVarFeats = noVarFeatsManual
        self.lookAheadIndex = lookAheadIndex
        self.max_var = max_var
        self.n_feats_tsfresh = self._get_n_feats_tsfresh()#76
     
    def _get_n_feats_tsfresh(self):
        """ private function to get number of tsfresh features"""
        n_feats_tsfresh = 0
        # loop over key features vector
        for key in self.feature_keys_tsfresh:
            # get parameters from dictionary
            params = self.AllFeatures[str(key)]
            # add last parameter entry (number of features)
            n_feats_tsfresh += params[-1]
        return n_feats_tsfresh
    
def initFeaturesLive(data,tradeInfoLive):
    """
    <DocString>
    """
    class parSarInit:
        # old parsar=> 20
        #periodSAR = data.nEventsPerStat
        HP20 = 0
        HP2 = 0
        LP20 = 100000
        LP2 = 100000
        stepAF = 0.02
        AFH20 = stepAF
        AFH2 = stepAF
        AFL20 = stepAF
        AFL2 = stepAF
        maxAF20 = 20*stepAF    
        maxAF2 = 2*stepAF
    
    parSarStruct = parSarInit
    nF = len(data.feature_keys_manual)
    featuresLive = np.zeros((nF,1))
    
    initRange = int(data.nEventsPerStat/data.movingWindow)
    em = np.zeros((data.lbd.shape))+tradeInfoLive.SymbolBid.loc[tradeInfoLive.SymbolBid.index[0]]
    for i in range(initRange*data.movingWindow-data.movingWindow):
        em = data.lbd*em+(1-data.lbd)*tradeInfoLive.SymbolBid.loc[i+tradeInfoLive.SymbolBid.index[0]]
    
    
    if 1 in data.feature_keys:
        featuresLive[1:1+data.lbd.shape[0],0] = tradeInfoLive.SymbolBid.iloc[0]
        for i in range(int(data.nEventsPerStat*(1-data.movingWindow/data.nEventsPerStat))):
            featuresLive[1:1+data.lbd.shape[0],0] = data.lbd*featuresLive[1:1+data.lbd.shape[0],0]+(
                    1-data.lbd)*tradeInfoLive.SymbolBid.iloc[i]
    #print(em-featuresLive[1:1+data.lbd.shape[0],0])
    if 10 in data.feature_keys:
        featuresLive[10,0] = tradeInfoLive.SymbolBid.iloc[0]
        featuresLive[11,0] = tradeInfoLive.SymbolBid.iloc[0]
    
    if 13 in data.feature_keys:
        featuresLive[13,0] = tradeInfoLive.SymbolBid.iloc[0]
        featuresLive[14,0] = tradeInfoLive.SymbolBid.iloc[0]
    
    return featuresLive,parSarStruct,em

def extractFeaturesLive(tradeInfoLive, data, featuresLive,parSarStruct,em):
    """
    <DocString>
    """
    nEvents = data.nEventsPerStat
    
    
    initRange = int(data.nEventsPerStat/data.movingWindow)
    endIndex = initRange*data.movingWindow+tradeInfoLive.SymbolBid.index[0]
    newBidsIndex = range(endIndex-data.movingWindow,endIndex)
    for i in newBidsIndex:
        #print(tradeInfoLive.SymbolBid.loc[i])
        em = data.lbd*em+(1-data.lbd)*tradeInfoLive.SymbolBid.loc[i]
    
    
    if 1 in data.feature_keys:
        newEventsRange = range(int(nEvents*(1-data.movingWindow/data.nEventsPerStat)),nEvents)
        #print(newEventsRange)
        eml = featuresLive[1:1+data.lbd.shape[0],0]
        for il in newEventsRange:
            #print(tradeInfoLive.SymbolBid.iloc[il])
            eml = data.lbd*eml+(
                    1-data.lbd)*tradeInfoLive.SymbolBid.iloc[il]
        
        featuresLive[1:1+data.lbd.shape[0],0] = eml
        
    if 10 in data.feature_keys:
        parSarStruct.HP20 = np.max([np.max(tradeInfoLive.SymbolBid.iloc[:]),parSarStruct.HP20])
        parSarStruct.LP20 = np.min([np.min(tradeInfoLive.SymbolBid.iloc[:]),parSarStruct.LP20])
        featuresLive[10,0] = featuresLive[10,0]+parSarStruct.AFH20*(parSarStruct.HP20-featuresLive[10,0]) #parSar high
        featuresLive[11,0] = featuresLive[11,0]-parSarStruct.AFL20*(featuresLive[11,0]-parSarStruct.LP20) # parSar low
        if featuresLive[10,0]<parSarStruct.HP20:
            parSarStruct.AFH20 = np.min([parSarStruct.AFH20+parSarStruct.stepAF,parSarStruct.maxAF20])
            parSarStruct.LP20 = np.min(tradeInfoLive.SymbolBid.iloc[:])
        if featuresLive[11,0]>parSarStruct.LP20:
            parSarStruct.AFL20 = np.min([parSarStruct.AFH20+parSarStruct.stepAF,parSarStruct.maxAF20])
            parSarStruct.HP20 = np.max(tradeInfoLive.SymbolBid.iloc[:])
        
    if 13 in data.feature_keys:
        parSarStruct.HP2 = np.max([np.max(tradeInfoLive.SymbolBid.iloc[:]),parSarStruct.HP2])
        parSarStruct.LP2 = np.min([np.min(tradeInfoLive.SymbolBid.iloc[:]),parSarStruct.LP2])
        featuresLive[13,0] = featuresLive[13,0]+parSarStruct.AFH2*(parSarStruct.HP2-featuresLive[13,0]) #parSar high
        featuresLive[14,0] = featuresLive[14,0]-parSarStruct.AFL2*(featuresLive[14,0]-parSarStruct.LP2) # parSar low
        if featuresLive[13,0]<parSarStruct.HP2:
            parSarStruct.AFH2 = np.min([parSarStruct.AFH2+parSarStruct.stepAF,parSarStruct.maxAF2])
            parSarStruct.LP2 = np.min(tradeInfoLive.SymbolBid.iloc[:])
        if featuresLive[14,0]>parSarStruct.LP2:
            parSarStruct.AFL2 = np.min([parSarStruct.AFH2+parSarStruct.stepAF,parSarStruct.maxAF2])
            parSarStruct.HP2 = np.max(tradeInfoLive.SymbolBid.iloc[:])

    if 0 in data.feature_keys:
        featuresLive[0,0] = tradeInfoLive.SymbolBid.iloc[-1]
    
    
    if 8 in data.feature_keys:
            featuresLive[8,0] = 10*np.log10(np.var(tradeInfoLive.SymbolBid.iloc[:])/data.std_var+1e-10)
        
    if 9 in data.feature_keys:
        te = pd.to_datetime(tradeInfoLive.iloc[-1].DateTime)
        t0 = pd.to_datetime(tradeInfoLive.iloc[0].DateTime)
        timeInterval = (te-t0).seconds/nEvents
        featuresLive[9,0] = 10*np.log10(timeInterval/data.std_time+0.01)

    if 12 in data.feature_keys:
        secsInDay = 86400.0
        if 9 not in data.feature_keys:
            # calculate te if not yet calculated
            te = pd.to_datetime(tradeInfoLive.iloc[-1].DateTime)
        timeSec = (te.hour*60*60+te.minute*60+te.second)/secsInDay
        featuresLive[12,0] = timeSec
    
    # Repeat non-variation features to inclue variation betwen first and second input
    if 15 in data.feature_keys:
        featuresLive[15,0] = featuresLive[8,0]
    
    if 16 in data.feature_keys:
        featuresLive[16,0] = featuresLive[9,0]
        
    if 17 in data.feature_keys:
        featuresLive[17,0] = np.max(tradeInfoLive.SymbolBid.iloc[:])-featuresLive[0,0]
    
    if 18 in data.feature_keys:
        featuresLive[18,0] = featuresLive[0,0]-np.min(tradeInfoLive.SymbolBid.iloc[:])
    
    if 19 in data.feature_keys:
        featuresLive[19,0] = np.max(tradeInfoLive.SymbolBid.iloc[:])-featuresLive[0,0]
        
    if 20 in data.feature_keys:
        featuresLive[20,0] = featuresLive[0,0]-np.min(tradeInfoLive.SymbolBid.iloc[:])
    
    if 21 in data.feature_keys:
        featuresLive[21,0] = np.min(tradeInfoLive.SymbolBid.iloc[:])/np.max(tradeInfoLive.SymbolBid.iloc[:])
    
    if 22 in data.feature_keys:
        featuresLive[22,0] = np.min(tradeInfoLive.SymbolBid.iloc[:])/np.max(tradeInfoLive.SymbolBid.iloc[:])
    
    for i in range(data.lbd.shape[0]):
        #nF = nF+1
        if 23+i in data.feature_keys:                 
            featuresLive[23+i,0] = featuresLive[0,0]/eml[i]
    
    for i in range(data.lbd.shape[0]):
        #nF = nF+1
        if 30+i in data.feature_keys:                 
            featuresLive[30+i,0] = featuresLive[0,0]/eml[i]
    
    return featuresLive,parSarStruct,em

def extractSeparators(tradeInfo,minThresDay,minThresNight,bidThresDay,bidThresNight,dateTest, tOt="tr"):
    """
    <DocString>
    """
    belong2tOt = pd.Series(tradeInfo.DateTime.str.contains('%&§&&§')) # initialize to all false
    for d in dateTest:
        belong2tOt = belong2tOt | tradeInfo.DateTime.str.contains('^'+d)
    
    if tOt=="te":
        belong2tOt = ~belong2tOt
    
    separators = pd.DataFrame(data=[],columns=['DateTime','SymbolBid','SymbolAsk'])
    #separators.index.name = "real_index"
    # append first and last entry already if not all zeros
    if belong2tOt.min()==0:
        separators = separators.append(tradeInfo.loc[(~belong2tOt).argmax()]).append(tradeInfo.loc[(~belong2tOt)[::-1].argmax()])
    
    dTs = pd.to_datetime(tradeInfo["DateTime"],format='%Y.%m.%d %H:%M:%S').to_frame()
    dTs1 = dTs[:-1]
    dTs2 = dTs[1:].set_index(dTs1.index)
    dayIndex = (dTs1.DateTime.dt.hour<22) & (dTs1.DateTime.dt.hour>5)

    tID1 = belong2tOt[:-1]
    tID2 = belong2tOt[1:]
    tID2.index = tID1.index
    #print(belong2tOt)
    tOtTransition = np.logical_xor(belong2tOt.iloc[:-1],belong2tOt.iloc[1:])
    #print(tOtTransition.index[tOtTransition])
    
    indexesTrans = np.union1d(tOtTransition.index[tOtTransition],tOtTransition.index[tOtTransition]+1)
    #print(indexesTrans)
    tOtTransition.loc[indexesTrans] = True
    tOtTransition = tOtTransition & (~belong2tOt).iloc[:-1]
    #print(tradeInfo.loc[tID1[tOtTransition].index])
    separators = separators.append(tradeInfo.loc[tID1[tOtTransition].index])
    separators = separators[~separators.index.duplicated(keep='first')]

    bids1 = tradeInfo.SymbolBid[:-1]
    bids2 = tradeInfo.SymbolBid[1:]
    bidsIndex = bids1.index
    bids2.index = bidsIndex

    deltaBids = ((bids1-bids2).abs())*np.mean(bids1)
    warningsBid = (((deltaBids>=bidThresDay) & (~tID1)) & dayIndex) | (((deltaBids>=bidThresNight) & (~tID1)) & ~dayIndex)
    deltasInsSecs = (dTs2-dTs1).astype('timedelta64[s]')
    # Find days transition indexes
    dayChange = (dTs1.DateTime.dt.day!=dTs2.DateTime.dt.day)
    deltasInsSecs.DateTime = deltasInsSecs.DateTime-60*6*dayChange

    # Find weekend transition indexes
    weekChange = (dTs1.DateTime.dt.dayofweek==4) & (dTs2.DateTime.dt.dayofweek==0)

    # Update seconds difference
    deltasInsSecs.DateTime = deltasInsSecs.DateTime-(2*24*60*60)*weekChange

    # Find deltas of more than 5 minutes
    warningTime = ((deltasInsSecs.DateTime>60*minThresDay) & (~tID1) & dayIndex) | ((deltasInsSecs.DateTime>60*minThresNight) & (~tID1) & ~dayIndex)
    allWarnings = warningTime & warningsBid & (~tOtTransition)
    endChunckIndex = deltaBids[allWarnings].index
    beginningChunckIndex = []
    for i in range(endChunckIndex.shape[0]):
        beginningChunckIndex.append(tradeInfo.index.get_loc(endChunckIndex[i])+1)

    separators = separators.append(tradeInfo.loc[endChunckIndex]).append(tradeInfo.iloc[beginningChunckIndex]).sort_index()

    #delete all those that have same index as beginning and end, i.g., chunk length is zero
    separators = separators[~separators.index.duplicated(False)]

    return separators

def save_as_matfile(filename,varname,var):
    
    sio.savemat("../MATLAB/"+filename+'.mat', {varname:var})
    print('MAT file saved')
    
    return None

def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y

def get_outputGain(stdO, bid, spread):
    """
    Function that obtains variable output gain for one asset
    """
    outputGain = np.min([1,stdO/(4*spread*bid)])
    return outputGain

def _build_bin_output(model, Output, batch_size):
    """
    Function that builds output binary Y based on real-valued returns Output vector.
    Args:
        - model: object containing model parameters
        - Output: real-valued returns vector
        - batch_size: scalar representing the batch size 
    Returns:
        - output binary matrix y_bin
    """
    # init y
    y = np.zeros((Output.shape))
    #print(y.shape)
    # quantize output to get y
    out_quantized = np.minimum(np.maximum(np.sign(Output)*np.round(abs(Output)*model.outputGain),-
        (model.size_output_layer-1)/2),(model.size_output_layer-1)/2)
    
    y = out_quantized+int((model.size_output_layer-1)/2)
    # conver y as integer
    y_dec=y.astype(int)
    # one hot output
    y_one_hot = convert_to_one_hot(y_dec, model.size_output_layer).T.reshape(
        batch_size,model.seq_len,model.size_output_layer)
    #print("y_one_hot.shape")
    #print(y_one_hot.shape)

    # add y_c bits if proceed
    y_c = np.zeros((y_one_hot.shape[0],y_one_hot.shape[1],0))
    
    if model.commonY == 1 or model.commonY == 3:
        y_c0 = np.zeros((y_one_hot.shape[0],y_one_hot.shape[1],1))
        # find up/down outputs (=> those with a zero in the middle bit of one-hot vector)
        nonZeroYs = y_one_hot[:,:,int((model.size_output_layer-1)/2)]!=1
        # set 1s in y_c0 vector at non-zero entries
        y_c0[nonZeroYs,0] = 1
        y_c = np.append(y_c,y_c0,axis=2)
    if model.commonY == 2 or model.commonY == 3:
        # build y_c1 and y_c2 vectors. y_c1 indicates all down outputs. y_c2
        # indicates all up outputs.
#        print("y_one_hot.shape")
#        print(y_one_hot.shape)
        y_c1 = np.zeros((y_one_hot.shape[0],y_one_hot.shape[1],1))
        y_c2 = np.zeros((y_one_hot.shape[0],y_one_hot.shape[1],1))
        # find negative (positive) returns
        negativeYs = out_quantized<0
        positiveYs = out_quantized>0

        # set to 1 the corresponding entries
        y_c1[np.squeeze(negativeYs,axis=2),0] = 1
        y_c2[np.squeeze(positiveYs,axis=2),0] = 1

        y_c = np.append(y_c,y_c1,axis=2)
        y_c = np.append(y_c,y_c2,axis=2)

    # build output vector

    y_bin = np.append(y_c,y_one_hot,axis=2)

    return y_bin, y_dec

def get_features_from_raw_par(data, features, DateTime, SymbolBid):
    """
    Function that calculates features from raw data in per batches
    Args:
        - data
        - features
        - DateTime
        - SymbolBid
    Returns:
        - features
    """
    
    tic = time.time()
    # init scalars
    nExS = data.nEventsPerStat
    mW = data.movingWindow
    nE = DateTime.shape[0]
    m = int(np.floor((nE/nExS-1)*nExS/mW)+1)
    secsInDay = 86400.0
    nEMAs = data.lbd.shape[0]
    
    # init exponetial means
    em = np.zeros((data.lbd.shape))+SymbolBid[0]
    for i in range(nExS-mW):
        em = data.lbd*em+(1-data.lbd)*SymbolBid[i]
    #/(1-np.maximum(data.lbd**i,1e-3))
    
    
    oldSARh20 = SymbolBid[0]
    oldSARh2 = SymbolBid[0]
    oldSARl20 = SymbolBid[0]
    oldSARl2 = SymbolBid[0]
    HP20 = 0
    HP2 = 0
    LP20 = 100000
    LP2 = 100000
    stepAF = 0.02
    AFH20 = stepAF
    AFH2 = stepAF
    AFL20 = stepAF
    AFL2 = stepAF
    maxAF20 = 20*stepAF    
    maxAF2 = 2*stepAF
    
    batch_size = 10000000
    par_batches = int(np.ceil(m/batch_size))
    l_index = 0
    # loop over batched
    for b in range(par_batches):
        # get m
        m_i = np.min([batch_size, m-b*batch_size])
        
        # init structures
        EMA = np.zeros((m_i,nEMAs))
        bids = np.zeros((m_i))
        variance = np.zeros((m_i))
        maxValue = np.zeros((m_i))
        minValue = np.zeros((m_i))
        timeInterval = np.zeros((m_i))
        timeSecs = np.zeros((m_i))
        parSARhigh20 = np.zeros((m_i))
        parSARhigh2 = np.zeros((m_i))
        parSARlow20 = np.zeros((m_i))
        parSARlow2 = np.zeros((m_i))
        
        
        for mm in range(m_i):
            
            startIndex = l_index+mm*mW
            endIndex = startIndex+nExS
            thisPeriod = range(startIndex,endIndex)
            thisPeriodBids = SymbolBid[thisPeriod]
            
            newBidsIndex = range(endIndex-mW,endIndex)
            for i in newBidsIndex:
                #a=data.lbd*em/(1-data.lbd**i)+(1-data.lbd)*tradeInfo.SymbolBid.loc[i]
                em = data.lbd*em+(1-data.lbd)*SymbolBid[i]
                
            t0 = dt.datetime.strptime(DateTime[thisPeriod[0]].decode("utf-8"),'%Y.%m.%d %H:%M:%S')
            te = dt.datetime.strptime(DateTime[thisPeriod[-1]].decode("utf-8"),'%Y.%m.%d %H:%M:%S')

            bids[mm] = SymbolBid[thisPeriod[-1]]
            EMA[mm,:] = em
            variance[mm] = np.var(thisPeriodBids)
            timeInterval[mm] = (te-t0).seconds/data.nEventsPerStat
            maxValue[mm] = np.max(thisPeriodBids)
            minValue[mm] = np.min(thisPeriodBids)
            timeSecs[mm] = (te.hour*60*60+te.minute*60+te.second)/secsInDay
            
            HP20 = np.max([maxValue[mm],HP20])
            LP20 = np.min([minValue[mm],LP20])
            parSARhigh20[mm] = oldSARh20+AFH20*(HP20-oldSARh20)
            parSARlow20[mm] = oldSARl20-AFL20*(oldSARl20-LP20)
            if parSARhigh20[mm]<HP20:
                AFH20 = np.min([AFH20+stepAF,maxAF20])
                LP20 = np.min(thisPeriodBids)
            if parSARlow20[mm]>LP20:
                AFL20 = np.min([AFH20+stepAF,maxAF20])
                HP20 = np.max(thisPeriodBids)
            oldSARh20 = parSARhigh20[mm]
            oldSARl20 = parSARlow20[mm]
            
            HP2 = np.max([maxValue[mm],HP2])
            LP2 = np.min([minValue[mm],LP2])
            parSARhigh2[mm] = oldSARh2+AFH2*(HP2-oldSARh2)
            parSARlow2[mm] = oldSARl2-AFL2*(oldSARl2-LP2)
            if parSARhigh2[mm]<HP2:
                AFH2 = np.min([AFH2+stepAF,maxAF2])
                LP2 = np.min(thisPeriodBids)
            if parSARlow2[mm]>LP2:
                AFL2 = np.min([AFH2+stepAF,maxAF2])
                HP2 = np.max(thisPeriodBids)
            oldSARh2 = parSARhigh2[mm]
            oldSARl2 = parSARlow2[mm]
        # end of for mm in range(m_i):
        l_index = startIndex+mW
        #print(l_index)
        toc = time.time()
        print("\t\tmm="+str(b*batch_size+mm+1)+" of "+str(m)+". Total time: "+str(np.floor(toc-tic))+"s")
        # update features vector
        init_idx = b*batch_size
        end_idx = b*batch_size+m_i

        nF = 0
        features[init_idx:end_idx,nF] = bids

        nF += 1
        features[init_idx:end_idx,nF:nF+data.lbd.shape[0]] = EMA

        nF += data.lbd.shape[0]
        logVar = 10*np.log10(variance/data.std_var+1e-10)
        features[init_idx:end_idx,nF] = logVar

        nF += 1
        logInt = 10*np.log10(timeInterval/data.std_time+0.01)
        features[init_idx:end_idx,nF] = logInt
        
        nF += 1
        features[init_idx:end_idx,nF] = parSARhigh20
        features[init_idx:end_idx,nF+1] = parSARlow20
        
        nF += 2
        features[init_idx:end_idx,nF] = timeSecs
        
        nF += 1
        features[init_idx:end_idx,nF] = parSARhigh2
        features[init_idx:end_idx,nF+1] = parSARlow2
        
        # repeat
        nF += 2
        features[init_idx:end_idx,nF] = logVar
        
        nF += 1
        features[init_idx:end_idx,nF] = logInt
        
        nF += 1
        features[init_idx:end_idx,nF] = maxValue-bids
        
        nF += 1
        features[init_idx:end_idx,nF] = bids-minValue
        
        nF += 1
        features[init_idx:end_idx,nF] = maxValue-bids
        
        nF += 1
        features[init_idx:end_idx,nF] = bids-minValue
        
        nF += 1
        features[init_idx:end_idx,nF] = minValue/maxValue
        
        nF += 1
        features[init_idx:end_idx,nF] = minValue/maxValue
        
        for i in range(data.lbd.shape[0]):          
            nF += 1        
            features[init_idx:end_idx,nF] = bids/EMA[:,i]
        
        for i in range(data.lbd.shape[0]):
            nF += 1        
            features[init_idx:end_idx,nF] = bids/EMA[:,i]
            
    return features

def get_features_from_tsfresh(data, DateTime, SymbolBid):
    """
    Funtion that extracts features with the TSFRESH tool
    """
    
    return None

def get_returns_from_raw(data, returns, ret_idx, DT, B, A, 
                         idx_init, DateTime, SymbolBid, SymbolAsk):
    """
    Function that obtains the outputs from raw data.
    Args:
        - data:
        - DateTime:
        - SymbolBid:
        - SymbolAsk:
    Returns:
        - outputs:
        - ret_idx:
    """
    nExS = data.nEventsPerStat
    mW = data.movingWindow
    nE = DateTime.shape[0]
    m = int(np.floor((nE/nExS-1)*nExS/mW)+1)
    initRange = int(nExS/mW)
    
    np_00 = initRange*data.movingWindow-1
    np_e0 = m*data.movingWindow-1
    
    indexOrigins = [i for i in range(np_00,np_e0,data.movingWindow)]
    ret_idx[:,0] = indexOrigins+idx_init
    DT[:,0] = DateTime[indexOrigins]
    B[:,0] = SymbolBid[indexOrigins]
    A[:,0] = SymbolAsk[indexOrigins]
    #origins = np.array(tradeInfo.SymbolBid.iloc[np_00:np_e0:data.movingWindow])
    
    for nr in range(len(data.lookAheadVector)):
        #print("nr")
        #print(nr)
        unp_0e = int(initRange*data.movingWindow+np.floor(data.nEventsPerStat*data.lookAheadVector[nr])-2)
        unp_ee = int(np.min([m*data.movingWindow+np.floor(
                data.nEventsPerStat*data.lookAheadVector[nr])-2,nE]))

        indexEnds = [i for i in range(unp_0e,unp_ee,data.movingWindow)]
        #fill ends wth last value
        for i in range(len(indexEnds),len(indexOrigins)):
            indexEnds.append(nE-1)
        returns[:,nr] = SymbolBid[indexEnds]-SymbolBid[indexOrigins]
        ret_idx[:,nr+1] = indexEnds+idx_init
        DT[:,nr+1] = DateTime[indexEnds]
        B[:,nr+1] = SymbolBid[indexEnds]
        A[:,nr+1] = SymbolAsk[indexEnds]
    
    return returns, ret_idx, DT, B, A

def get_normalization_stats(data, features, returns, hdf5_directory):
    """
    Function that calcultes mean and var of a group.
    Args:
        - data:
        - features:
        - returns:
        - nChannels:
    Returns:
        - means_in:
        - stds_in:
        - means_out:
        - stds_out:
    """
    tic = time.time()
    nChannels = int(data.nEventsPerStat/data.movingWindow)
    nF = len(data.feature_keys_manual)
    # open temporal file for variations
    try:
        # create file
        ft = h5py.File(hdf5_directory+'temp.hdf5','w')
        # create group
        group_temp = ft.create_group('temp')
        # reserve memory space for variations and normalized variations
        variations = group_temp.create_dataset("variations", (features.shape[0],nF,nChannels), dtype=float)
        # init variations and normalized variations to 999 (impossible value)
        
        #variations[:] = variations[:]+999
        print("\t getting variations")
        # loop over channels
        for r in range(nChannels):
            variations[r+1:,:,r] = features[r+1:,:]-features[:-(r+1),:]
            variations[r+1:,data.noVarFeats,r] = features[:-(r+1),data.noVarFeats]
        print("\t time for variations: "+str(time.time()-tic))
        # init stats    
        means_in = np.zeros((nChannels,nF))
        stds_in = np.zeros((nChannels,nF))
        print("\t getting means and stds")
        # loop over channels
        for r in range(nChannels):
            #nonZeros = variations[:,0,r]!=999
            #print(np.sum(nonZeros))
            means_in[r,:] = np.mean(variations[nChannels:,:,r],axis=0,keepdims=1)
            stds_in[r,:] = np.std(variations[nChannels:,:,r],axis=0,keepdims=1)  
    #       
            # get output stats
            stds_out = np.std(returns,axis=0)
            means_out = np.mean(returns,axis=0)
        print("\t Total time for stats: "+str(time.time()-tic))
    except KeyboardInterrupt:
        ft.close()
        print("ERROR! Closing file and exiting.")
        raise KeyboardInterrupt
    ft.close()
    return [means_in, stds_in, means_out, stds_out]

def load_returns(data, hdf5_directory, thisAsset, separators, s):
    """
    Function that extracts features, results and normalization stats from already saved
    structures.
    Args:
        - data:
        - thisAsset
        - separators
        - f_prep_IO
        - group
        - hdf5_directory
        - s
    Returns:
        - features 
        - returns
        - ret_idx
    """
    filename_prep_IO = (hdf5_directory+'IO_mW'+str(data.movingWindow)+'_nE'+
                        str(data.nEventsPerStat)+'_nF'+str(37)+'.hdf5')
    f_prep_IO = h5py.File(filename_prep_IO,'r')
    # init structures
    returns = []
    ret_idx = []
    # number of events
    nE = separators.index[s+1]-separators.index[s]+1
    # check if number of events is not enough to build two features and one return
    if nE>=2*data.nEventsPerStat:
#        print("\tSeparator batch {0:d} out of {1:d}".format(int(s/2),int(len(separators)/2-1)))
#        print("\t"+separators.DateTime.iloc[s]+" to "+separators.DateTime.iloc[s+1])
        # get init and end dates of these separators
        init_date = dt.datetime.strftime(dt.datetime.strptime(
                separators.DateTime.iloc[s],'%Y.%m.%d %H:%M:%S'),'%y%m%d%H%M%S')
        end_date = dt.datetime.strftime(dt.datetime.strptime(
                separators.DateTime.iloc[s+1],'%Y.%m.%d %H:%M:%S'),'%y%m%d%H%M%S')
        
        # group name of separator s
        group_name = thisAsset+'/'+init_date+end_date
        
        # create new gruop if not yet in file
        if group_name not in f_prep_IO:
            # create group, its attributes and its datasets
            print("Group should already exist. Run get_features_results_stats_from_raw first to create it.")
            raise ValueError
        else:
            # get group from file
            group = f_prep_IO[group_name]
        # get features, returns and stats if don't exist
        if group.attrs.get("means_in") is None:
            print("Group should already exist. Run get_features_results_stats_from_raw first to create it.")
            raise ValueError
        else:
            # get data sets
            returns = group['returns']
            ret_idx = group['ret_idx']
            if 'DT' in group:
                DT = group['DT']
                B = group['B']
                A = group['A']
    # end of if separators.index[s+1]-separators.index[s]>=2*data.nEventsPerStat:
    # save results in a dictionary
    returns_struct = {}
    returns_struct['returns'] = returns
    returns_struct['ret_idx'] = ret_idx
    if 'DT' in group:
        returns_struct['DT'] = DT
        returns_struct['B'] = B
        returns_struct['A'] = A
    return returns_struct

def load_manual_features(data, thisAsset, separators, f_prep_IO, s):
    """
    Function that extracts features from previously saved structures.
    Args:
        - data:
        - thisAsset
        - separators
        - f_prep_IO
        - group
        - hdf5_directory
        - s
    Returns:
        - features
    """

    # init structures
    features = []
    # number of events
    nE = separators.index[s+1]-separators.index[s]+1
    # check if number of events is not enough to build two features and one return
    if nE>=2*data.nEventsPerStat:
#        print("\tSeparator batch {0:d} out of {1:d}".format(int(s/2),int(len(separators)/2-1)))
#        print("\t"+separators.DateTime.iloc[s]+" to "+separators.DateTime.iloc[s+1])
        # get init and end dates of these separators
        init_date = dt.datetime.strftime(dt.datetime.strptime(
                separators.DateTime.iloc[s],'%Y.%m.%d %H:%M:%S'),'%y%m%d%H%M%S')
        end_date = dt.datetime.strftime(dt.datetime.strptime(
                separators.DateTime.iloc[s+1],'%Y.%m.%d %H:%M:%S'),'%y%m%d%H%M%S')
        
        # group name of separator s
        group_name = thisAsset+'/'+init_date+end_date
        
        # create new gruop if not yet in file
        if group_name not in f_prep_IO:
            # create group, its attributes and its datasets
            raise ValueError("Group should already exist. Run get_features_results_stats_from_raw first to create it.")
        else:
            # get group from file
            group = f_prep_IO[group_name]
        # get features, returns and stats if don't exist
        if group.attrs.get("means_in") is None:
            raise ValueError("Group should already exist. Run get_features_results_stats_from_raw first to create it.")
        else:
            # get data sets
            features = group['features']
            
    # end of if separators.index[s+1]-separators.index[s]>=2*data.nEventsPerStat:
    # save results in a dictionary
#    features_struct = {}
#    features_struct['features'] = features
    
    return features

def load_tsf_features(data, thisAsset, separators, f_feats_tsf, s):
    """ Load features from TSFRESH tool """
    nE = separators.index[s+1]-separators.index[s]+1
    # check if number of events is not enough to build two features and one return
    if nE>=2*data.nEventsPerStat:
        # get init and end dates of these separators
        init_date = dt.datetime.strftime(dt.datetime.strptime(
                separators.DateTime.iloc[s],'%Y.%m.%d %H:%M:%S'),'%y%m%d%H%M%S')
        end_date = dt.datetime.strftime(dt.datetime.strptime(
                separators.DateTime.iloc[s+1],'%Y.%m.%d %H:%M:%S'),'%y%m%d%H%M%S')
        # group name of separator s
        group_name_chunck = thisAsset+'/'+init_date+end_date
        c = 0
        # save features in HDF5 file
        for f in data.feature_keys_tsfresh:
            group_name_feat = group_name_chunck+'/'+str(f)
            #print(group_name_feat)
            params = data.AllFeatures[str(f)]
            n_new_feats = params[-1]
            #if group_name_feat in file_features_tsf:
            #    del file_features_tsf[group_name_feat]
            if group_name_feat not in f_feats_tsf:
                print("Group should already exist. Run get_features_results_"+
                      "stats_from_raw first to create it.")
                raise ValueError
            else: # load features from HDF5 file if they are saved already
                params = data.AllFeatures[str(f)]
                n_new_feats = params[-1]
                success = False
                tries = 0
                while not success and tries<10:
                    try:
                        group_chunck = f_feats_tsf[group_name_feat]
                        success = True
                    except KeyError:
                        print("KeyError reading "+group_name_feat)
                        tries += 1
                if not success:
                    raise KeyError(group_name_feat+" File corrupted. Delete and retrieve again")
                if c==0:
                    features = np.zeros((group_chunck['feature'].shape[0], data.n_feats_tsfresh))
                features[:,c:c+n_new_feats] = group_chunck['feature']
            c += n_new_feats
    else:
        features = np.array([])
    return features

def get_features_results_stats_from_raw(data, thisAsset, separators, f_prep_IO, group_raw,
                               stats, hdf5_directory, s, save_stats):
    """
    Function that extracts features, results and normalization stats from raw data.
    Args:
        - data:
        - thisAsset
        - separators
        - f_prep_IO
        - group
        - stats
        - hdf5_directory
        - s
    Returns:
        - features 
        - returns
        - ret_idx
        - stats
    """
    
    # get trade info datasets
    DateTime = group_raw["DateTime"]
    SymbolBid = group_raw["SymbolBid"]
    SymbolAsk = group_raw["SymbolAsk"]
    
    # init structures
    features = []
    returns = []
    ret_idx = []
    # number of events
    nE = separators.index[s+1]-separators.index[s]+1
    # check if number of events is not enough to build two features and one return
    if nE>=2*data.nEventsPerStat:
#        print("\tSeparator batch {0:d} out of {1:d}".format(int(s/2),int(len(separators)/2-1)))
#        print("\t"+separators.DateTime.iloc[s]+" to "+separators.DateTime.iloc[s+1])
        # get init and end dates of these separators
        init_date = dt.datetime.strftime(dt.datetime.strptime(
                separators.DateTime.iloc[s],'%Y.%m.%d %H:%M:%S'),'%y%m%d%H%M%S')
        end_date = dt.datetime.strftime(dt.datetime.strptime(
                separators.DateTime.iloc[s+1],'%Y.%m.%d %H:%M:%S'),'%y%m%d%H%M%S')
#        print(DateTime.shape)
#        if thisAsset=='AUDJPY':
#            print(DateTime.shape)
#            print(separators.index)
#            print(separators.index[s])
#            print(separators.index[s+1])
#            print(DateTime[separators.index[s]])
#            print(DateTime[separators.index[s+1]])
        # group name of separator s
        group_name = thisAsset+'/'+init_date+end_date
        
        # create new gruop if not yet in file
        if group_name not in f_prep_IO:
            # create group, its attributes and its datasets
            group = f_prep_IO.create_group(group_name)
        else:
            # get group from file
            group = f_prep_IO[group_name]
        # get features, returns and stats if don't exist
        if group.attrs.get("means_in") is None:
            # number of samples
            nExS = data.nEventsPerStat
            mW = data.movingWindow
            nF = len(data.feature_keys_manual)
            # number of features and number of returns
            m_in = int(np.floor((nE/nExS-1)*nExS/mW)+1)
            m_out = int(m_in-nExS/mW)#len(range(int(nExS/mW)*mW-1,m_in*mW-1,mW))
            # save as attributes
            group.attrs.create("m_in", m_in, dtype=int)
            group.attrs.create("m_out", m_out, dtype=int)
            # create datasets
            try:
                features = group.create_dataset("features", (m_in,nF),dtype=float)
                returns = group.create_dataset("returns", (m_out,len(data.lookAheadVector)),dtype=float)
                ret_idx = group.create_dataset("ret_idx", (m_out,len(data.lookAheadVector)+1),dtype=int)
                DT = group.create_dataset("DT", (m_out,len(data.lookAheadVector)+1),dtype='S19')
                B = group.create_dataset("B", (m_out,len(data.lookAheadVector)+1),dtype=float)
                A = group.create_dataset("A", (m_out,len(data.lookAheadVector)+1),dtype=float)
                
            except (ValueError,RuntimeError):
                print("WARNING: RunTimeError. Trying to recover from it.")
                features = group['features']
                returns = group['returns']
                ret_idx = group['ret_idx']
                DT = group['DT']
                B = group['B']
                A = group['A']
                
            print("\tgetting features from raw data...")
            # get structures and save them in a hdf5 file
            features = get_features_from_raw_par(data,features,
                                             DateTime[separators.index[s]:separators.index[s+1]+1], 
                                             SymbolBid[separators.index[s]:separators.index[s+1]+1])

            returns, ret_idx, DT, B, A = get_returns_from_raw(data,returns,ret_idx,DT,B,A,separators.index[s],
                                             DateTime[separators.index[s]:separators.index[s+1]+1], 
                                             SymbolBid[separators.index[s]:separators.index[s+1]+1],
                                             SymbolAsk[separators.index[s]:separators.index[s+1]+1])

#            print(DT[0,0])
#            print(DT[-1,0])
            assert(ret_idx[-1,0]<=separators.index[-1])
            # get stats
            means_in, stds_in, means_out, stds_out = get_normalization_stats(data, features, returns, hdf5_directory)
            # save means and variances as atributes
            group.attrs.create("means_in", means_in, dtype=float)
            group.attrs.create("stds_in", stds_in, dtype=float)
            group.attrs.create("means_out", means_out, dtype=float)
            group.attrs.create("stds_out", stds_out, dtype=float)
            
        else:
            # get data sets
            features = group['features']
            returns = group['returns']
            ret_idx = group['ret_idx']
            # get attributes
            m_in = group.attrs.get("m_in")
            m_out = group.attrs.get("m_out")
            means_in = group.attrs.get("means_in")
            stds_in = group.attrs.get("stds_in")
            means_out = group.attrs.get("means_out")
            stds_out = group.attrs.get("stds_out")
            #print("\tIO loaded from HDF5 file.")
        
        if save_stats:
        # update combined stats of all data sets
            stats["means_t_in"] += m_in*means_in
            stats["stds_t_in"] += m_in*stds_in
            stats["means_t_out"] += m_out*means_out
            stats["stds_t_out"] += m_out*stds_out
            stats["m_t_in"] += m_in
            stats["m_t_out"] += m_out
    # end of if separators.index[s+1]-separators.index[s]>=2*data.nEventsPerStat:
#    else:
#        print("\tSeparator batch {0:d} out of {1:d} skkiped. Not enough entries".format(int(s/2),int(len(separators)/2-1)))
    # save results in a dictionary
    IO_prep = {}
    IO_prep['features'] = features
    IO_prep['returns'] = returns
    IO_prep['ret_idx'] = ret_idx
    
    return IO_prep, stats

def build_IO(file_temp, data, model, features_manual,features_tsf,returns_struct,
             stats_manual,stats_tsf,stats_output,IO,totalSampsPerLevel, 
             s, nE, thisAsset):
    """
    Function that builds X and Y from data contained in a HDF5 file.
    """
    # total number of possible channels
    nChannels = int(data.nEventsPerStat/data.movingWindow)
    # extract means and stats
    if len(stats_manual)>0:
        means_in_manual = stats_manual['means_t_in']
        stds_in_manual = stats_manual['stds_t_in']
    else:
        means_in_manual = np.zeros((nChannels,0))
        stds_in_manual = np.zeros((nChannels,0))
        
    if len(stats_tsf)>0:
        means_in_tsf = stats_tsf['means_t_in']
        stds_in_tsf = stats_tsf['stds_t_in']
    else:
        means_in_tsf = np.zeros((nChannels,0))
        stds_in_tsf = np.zeros((nChannels,0))
    
    # concatenate features and stats
    features = np.append(features_manual, features_tsf, 1)
    stds_in = np.append(stds_in_manual, stds_in_tsf, 1)
    means_in = np.append(means_in_manual, means_in_tsf, 1)
    
    stds_out = stats_output['stds_t_out']
    # extract features and returns
    returns = returns_struct['returns']
    ret_idx = returns_struct['ret_idx']
    # add dateTimes, bids and asks if are included in file
    all_info = 0
    if 'D' in IO:
        all_info = 1
        dts = returns_struct['DT']
        bids = returns_struct['B']
        asks = returns_struct['A']
        
        D = IO['D']
        B = IO['B']
        A = IO['A']

    # extract IO structures
    X = IO['X']
    Y = IO['Y']
    I = IO['I']
    pointer = IO['pointer']
    # number of channels
    nC = len(data.channels)
    # sequence length
    seq_len = model.seq_len#int((data.lB-data.nEventsPerStat)/data.movingWindow)
    # samples allocation per batch
    aloc = 2**20
    # number of features
    nF = data.nFeatures
    # create group
    group_temp = file_temp.create_group('temp')
    # reserve memory space for variations and normalized variations
    variations = group_temp.create_dataset('variations', (features.shape[0],nF,nC), dtype=float)
    variations_normed = group_temp.create_dataset('variations_normed', (features.shape[0],nF,nC), dtype=float)
    # init variations and normalized variations to 999 (impossible value)
    variations[:] = variations[:]+999
    variations_normed[:] = variations[:]
    nonVarFeats = np.intersect1d(data.noVarFeats,data.feature_keys)
    #non-variation idx for variations normed
    nonVarIdx = np.zeros((len(nonVarFeats))).astype(int)
    nv = 0
    for allnv in range(nF):
        if data.feature_keys[allnv] in nonVarFeats:
            nonVarIdx[nv] = int(allnv)
            nv += 1
    # loop over channels
    for r in range(nC):
        variations[data.channels[r]+1:,:,r] = (features[data.channels[r]+1:,
                                               data.feature_keys]-features[
                                                :-(data.channels[r]+1),
                                                data.feature_keys])
        if nonVarFeats.shape[0]>0:
            variations[data.channels[r]+1:,nonVarIdx,r] = features[:-(data.channels[r]+1),nonVarFeats]
            
        variations_normed[data.channels[r]+1:,:,r] = np.minimum(np.maximum((variations[data.channels[r]+1:,
                          :,r]-means_in[r,data.feature_keys])/stds_in[r,data.feature_keys],-data.max_var),data.max_var)
    # remove the unaltered entries
    nonremoveEntries = range(nChannels,variations_normed.shape[0])#variations_normed[:,0,-1]!=999
    # create new variations 
    variations_normed_new = group_temp.create_dataset('variations_normed_new', variations_normed[nChannels:,:,:].shape, dtype=float)
    variations_normed_new[:] = variations_normed[nonremoveEntries,:,:]
    del group_temp['variations_normed']
    del group_temp['variations']
    # get some scalars
    nSamps = variations_normed_new.shape[0]
    samp_remaining = nSamps-nChannels-seq_len-1
    chunks = int(np.ceil(samp_remaining/aloc))
    # init counter of samps processed
    offset = 0
    # loop over chunks
    for i in range(chunks):
        # this batch length
        batch = np.min([samp_remaining,aloc])
        init_idx_rets = nChannels+offset+seq_len-1
        end_idx_rets = nChannels+offset+batch+2*seq_len-1
        # create support numpy vectors to speed up iterations
        v_support = variations_normed_new[offset:offset+batch+seq_len, :, :]
        r_support = returns[init_idx_rets:end_idx_rets, data.lookAheadIndex]
#        if len(stats_manual)>0:
#            tag = '_m_'
#        else:
#            tag = '_a_'
#        save_as_matfile(thisAsset+tag+str(int(s/2)),thisAsset+tag+str(int(s/2)),v_support)
#        raise KeyboardInterrupt
        # we only take here the entry time index, and later at DTA building the 
        # exit time index is derived from the entry time and the number of events to
        # look ahead
        i_support = ret_idx[init_idx_rets:end_idx_rets, [0,data.lookAheadIndex+1]]
        if all_info:
            dt_support = dts[init_idx_rets:end_idx_rets, [0,data.lookAheadIndex+1]]
            b_support = bids[init_idx_rets:end_idx_rets, [0,data.lookAheadIndex+1]]
            a_support = asks[init_idx_rets:end_idx_rets, [0,data.lookAheadIndex+1]]
        # update remaining samps to proceed
        samp_remaining = samp_remaining-batch
        # init formatted input and output
        X_i = np.zeros((batch, seq_len, model.nFeatures))
        # real-valued output
        O_i = np.zeros((batch, seq_len, 1))
        # output index vector
        I_i = np.zeros((batch, seq_len, 2))
        
        D_i = np.chararray((batch, seq_len, 2),itemsize=19)
        B_i = np.zeros((batch, seq_len, 2))
        A_i = np.zeros((batch, seq_len, 2))
        
        for nI in range(batch):
            # init channels counter
            cc = 0
            for r in range(nC):
                # get input
                v_s_s = v_support[nI:nI+seq_len, :, r]
                X_i[nI,:,cc*nF:(cc+1)*nF] = v_s_s[::-1,:]#v_support[nI+seq_len-1:nI-1:-1, :, r]#[nI:nI+seq_len, :, r]
                cc += 1
            # due to substraction of features for variation, output gets the 
            # feature one entry later
            O_i[nI,:,0] = r_support[nI]
            I_i[nI,:,:] = i_support[nI,:]
            if all_info:
                D_i[nI,:,:] = dt_support[nI,:]
                B_i[nI,:,:] = b_support[nI,:]
                A_i[nI,:,:] = a_support[nI,:]
        
        
        # normalize output
        O_i = O_i/stds_out[0,data.lookAheadIndex]#stdO#
        # update counters
        offset = offset+batch
        # get decimal and binary outputs
        Y_i, y_dec = _build_bin_output(model, O_i, batch)
        # get samples per level
        for l in range(model.size_output_layer):
            totalSampsPerLevel[l] = totalSampsPerLevel[l]+np.sum(y_dec[:,-1,0]==l)
        # resize IO structures
        X.resize((pointer+batch, seq_len, model.nFeatures))
        Y.resize((pointer+batch, seq_len,model.commonY+model.size_output_layer))
        I.resize((pointer+batch, seq_len, 2))
        # update IO structures
        X[pointer:pointer+batch,:,:] = X_i
        Y[pointer:pointer+batch,:,:] = Y_i
        I[pointer:pointer+batch,:,:] = I_i
        if all_info:
            # resize
            D.resize((pointer+batch, seq_len, 2))
            B.resize((pointer+batch, seq_len, 2))
            A.resize((pointer+batch, seq_len, 2))
            # update
            D[pointer:pointer+batch,:,:] = D_i
            B[pointer:pointer+batch,:,:] = B_i
            A[pointer:pointer+batch,:,:] = A_i
#        save_as_matfile('X_h_n_'+str(int(s/2)),'X_h_n'+str(int(s/2)),X_i)
#        save_as_matfile('O_h_n_'+str(int(s/2)),'O_h_n'+str(int(s/2)),O_i)
        
        # uodate pointer
        pointer += batch
        #print(pointer)
    # end of for i in range(chunks):
    # update dictionary
    IO['X'] = X
    IO['Y'] = Y
    IO['I'] = I
    IO['pointer'] = pointer
    if all_info:
        IO['D'] = D
        IO['B'] = B
        IO['A'] = A
    
    return IO, totalSampsPerLevel

# Function Build IO from var
def build_IO_from_var(data, model, stats, IO, totalSampsPerLevel, features, returns, calculate_roi):
    # total number of possible channels
    nExS = data.nEventsPerStat
    mW = data.movingWindow
    nChannels = int(nExS/mW)
    # sequence length
    seq_len = model.seq_len#int((data.lB-data.nEventsPerStat)/data.movingWindow)
    # samples allocation per batch
    aloc = 2**20
    # extract means and stats
    means_in = stats['means_in']
    stds_in = stats['stds_in']
    m_in = stats['m_in']
    stds_out = stats['stds_out']
    m_out = stats['m_out']
    print("m_in")
    print(m_in)
    print("m_out")
    print(m_out)
    # add dateTimes, bids and asks if are included in file
    all_info = 0
    if calculate_roi:
        raise NotImplemented
        all_info = 1
        dts = symbols['DT']
        bids = symbols['B']
        asks = symbols['A']

        D = IO['D']
        B = IO['B']
        A = IO['A']

    # extract IO structures
    X = IO['X']
    Y = IO['Y']
    #I = IO['I']
    pointer = IO['pointer']

    feats_var_normed = np.minimum(np.maximum((features-means_in)/\
                         stds_in,-data.max_var),data.max_var)
    # get some scalars
    #nSamps = feats_var_normed.shape[0]
    samp_remaining = m_out-nChannels-seq_len-1
    print("samp_remaining")
    print(samp_remaining)
    chunks = int(np.ceil(samp_remaining/aloc))
    # init counter of samps processed
    offset = 0
    # loop over chunks
    for i in range(chunks):
        # this batch length
        batch = np.min([samp_remaining, aloc])
        print("batch")
        print(batch)
        # create support numpy vectors to speed up iterations
        v_support = feats_var_normed[offset:offset+batch+seq_len, :]
        # get init and end indexes for returns
        init_idx_rets = nChannels+offset+seq_len-1
        end_idx_rets = nChannels+offset+batch+2*seq_len-1
        r_support = returns[init_idx_rets:end_idx_rets, data.lookAheadIndex]
        # we only take here the entry time index, and later at DTA building the 
        # exit time index is derived from the entry time and the number of events to
        # look ahead
        if calculate_roi:
            dt_support = dts[init_idx_rets:end_idx_rets, [0,data.lookAheadIndex+1]]
            b_support = bids[init_idx_rets:end_idx_rets, [0,data.lookAheadIndex+1]]
            a_support = asks[init_idx_rets:end_idx_rets, [0,data.lookAheadIndex+1]]
        # update remaining samps to proceed
        samp_remaining = samp_remaining-batch
        # init formatted input and output
        X_i = np.zeros((batch, seq_len, features.shape[1]))
        # real-valued output
        O_i = np.zeros((batch, seq_len, 1))    
        if calculate_roi:
            # last dimension is to incorporate in and out symbols
            D_i = np.chararray((batch, 2),itemsize=19)
            B_i = np.zeros((batch, 2))
            A_i = np.zeros((batch, 2))

        for nI in range(batch):
            # get input
            v_s_s = v_support[nI:nI+seq_len, :]
            X_i[nI,:,:] = v_s_s[::-1,:]#v_support[nI:nI+seq_len, :]            
            # due to substraction of features for variation, output gets the 
            # feature one entry later
            O_i[nI,:,0] = r_support[nI]
            if calculate_roi:
                D_i[nI,:] = dt_support[nI,:]
                B_i[nI,:] = b_support[nI,:]
                A_i[nI,:] = a_support[nI,:]

        # normalize output
        #a=1
        O_i = O_i/stds_out[data.lookAheadIndex]
        # update counters
        offset = offset+batch
        # get decimal and binary outputs
        Y_i, y_dec = _build_bin_output(model, O_i, batch)
        # get samples per level
        for l in range(model.size_output_layer):
            totalSampsPerLevel[l] = totalSampsPerLevel[l]+np.sum(y_dec[:,-1,0]==l)
        # resize IO structures
        X.resize((pointer+batch, seq_len,features.shape[1]))
        Y.resize((pointer+batch, seq_len,model.commonY+model.size_output_layer))
        # update IO structures
        X[pointer:pointer+batch,:,:] = X_i
        Y[pointer:pointer+batch,:,:] = Y_i
        if calculate_roi:
            # resize
            D.resize((pointer+batch, 2))
            B.resize((pointer+batch, 2))
            A.resize((pointer+batch, 2))
            # update
            D[pointer:pointer+batch,:] = D_i
            B[pointer:pointer+batch,:] = B_i
            A[pointer:pointer+batch,:] = A_i
    #        save_as_matfile('X_h_n_'+str(int(s/2)),'X_h_n'+str(int(s/2)),X_i)
    #        save_as_matfile('O_h_n_'+str(int(s/2)),'O_h_n'+str(int(s/2)),O_i)

        # uodate pointer
        pointer += batch
    # end of for i in range(chunks):
    # update dictionary
    IO['X'] = X
    IO['Y'] = Y
    IO['pointer'] = pointer
    if calculate_roi:
        IO['D'] = D
        IO['B'] = B
        IO['A'] = A
    
    return IO, totalSampsPerLevel

def build_IO_from_variations(file_temp, data, model, feats_var ,features_tsf, 
                             returns, symbols, stats_manual, stats_in, stats_out,
                             IO, totalSampsPerLevel, s, nE, thisAsset):
    """
    Function that builds RNN inputs/outputs pairs from data contained in a HDF5 file.
    """
    nExS = data.nEventsPerStat
    mW = data.movingWindow
    # total number of possible channels
    nChannels = int(nExS/mW)
    # sequence length
    seq_len = model.seq_len#int((data.lB-data.nEventsPerStat)/data.movingWindow)
    # samples allocation per batch
    aloc = 2**20
    # extract means and stats
    means_in = stats_in['means']
    stds_in = stats_in['stds']
    stds_out = stats_out['stds']
    # add dateTimes, bids and asks if are included in file
    all_info = 0
    if 'D' in IO:
        all_info = 1
        dts = symbols['DT']
        bids = symbols['B']
        asks = symbols['A']
        
        D = IO['D']
        B = IO['B']
        A = IO['A']

    # extract IO structures
    X = IO['X']
    Y = IO['Y']
    I = IO['I']
    pointer = IO['pointer']
    
    feats_var_normed = np.minimum(np.maximum((feats_var-means_in)/\
                     stds_in,-data.max_var),data.max_var)
    # get some scalars
    nSamps = feats_var_normed.shape[0]
    samp_remaining = nSamps-nChannels-seq_len-1
    chunks = int(np.ceil(samp_remaining/aloc))
    # init counter of samps processed
    offset = 0
    # loop over chunks
    for i in range(chunks):
        # this batch length
        batch = np.min([samp_remaining,aloc])
        # create support numpy vectors to speed up iterations
        v_support = feats_var_normed[offset:offset+batch+seq_len, :]
        r_support = returns[nChannels+offset+2:nChannels+offset+batch+seq_len+2, data.lookAheadIndex]
#        if len(stats_manual)>0:
#            tag = '_m_'
#        else:
#            tag = '_a_'
#        save_as_matfile(thisAsset+tag+str(int(s/2)),thisAsset+tag+str(int(s/2)),v_support)
#        raise KeyboardInterrupt
        # we only take here the entry time index, and later at DTA building the 
        # exit time index is derived from the entry time and the number of events to
        # look ahead
        if all_info:
            dt_support = dts[nChannels+offset+2:nChannels+offset+batch+seq_len+2, [0,data.lookAheadIndex+1]]
            b_support = bids[nChannels+offset+2:nChannels+offset+batch+seq_len+2, [0,data.lookAheadIndex+1]]
            a_support = asks[nChannels+offset+2:nChannels+offset+batch+seq_len+2, [0,data.lookAheadIndex+1]]
        # update remaining samps to proceed
        samp_remaining = samp_remaining-batch
        # init formatted input and output
        X_i = np.zeros((batch, seq_len, model.nFeatures))
        # real-valued output
        O_i = np.zeros((batch, seq_len, 1))
        
        D_i = np.chararray((batch, seq_len, 2),itemsize=19)
        B_i = np.zeros((batch, seq_len, 2))
        A_i = np.zeros((batch, seq_len, 2))
        
        for nI in range(batch):
            # get input
            X_i[nI,:,:] = v_support[nI:nI+seq_len, :]            
            # due to substraction of features for variation, output gets the 
            # feature one entry later
            O_i[nI,:,0] = r_support[nI:nI+seq_len]
            if all_info:
                D_i[nI,:,:] = dt_support[nI:nI+seq_len,:]
                B_i[nI,:,:] = b_support[nI:nI+seq_len,:]
                A_i[nI,:,:] = a_support[nI:nI+seq_len,:]
        
        # normalize output
        O_i = O_i/stds_out[0,data.lookAheadIndex]#stdO#
        # update counters
        offset = offset+batch
        # get decimal and binary outputs
        Y_i, y_dec = _build_bin_output(model, O_i, batch)
        # get samples per level
        for l in range(model.size_output_layer):
            totalSampsPerLevel[l] = totalSampsPerLevel[l]+np.sum(y_dec[:,-1,0]==l)
        # resize IO structures
        X.resize((pointer+batch, seq_len, model.nFeatures))
        Y.resize((pointer+batch, seq_len,model.commonY+model.size_output_layer))
        I.resize((pointer+batch, seq_len, 2))
        # update IO structures
        X[pointer:pointer+batch,:,:] = X_i
        Y[pointer:pointer+batch,:,:] = Y_i
        if all_info:
            # resize
            D.resize((pointer+batch, seq_len, 2))
            B.resize((pointer+batch, seq_len, 2))
            A.resize((pointer+batch, seq_len, 2))
            # update
            D[pointer:pointer+batch,:,:] = D_i
            B[pointer:pointer+batch,:,:] = B_i
            A[pointer:pointer+batch,:,:] = A_i
#        save_as_matfile('X_h_n_'+str(int(s/2)),'X_h_n'+str(int(s/2)),X_i)
#        save_as_matfile('O_h_n_'+str(int(s/2)),'O_h_n'+str(int(s/2)),O_i)
        
        # uodate pointer
        pointer += batch
        #print(pointer)
    # end of for i in range(chunks):
    # update dictionary
    IO['X'] = X
    IO['Y'] = Y
    IO['pointer'] = pointer
    if all_info:
        IO['D'] = D
        IO['B'] = B
        IO['A'] = A
    
    return IO, totalSampsPerLevel

def load_separators(data, thisAsset, separators_directory, tOt='tr', from_txt=1):
    """
    Function that loads and segments separators according to beginning and end dates.
    
    """
    if from_txt:
        # separators file name
        separators_filename = thisAsset+'_separators.txt'
        # load separators
        if os.path.exists(separators_directory+separators_filename):
            separators = pd.read_csv(separators_directory+separators_filename, index_col='Pointer')
        else:
            separators = []
    else:
        print("Depricated load separators from DB. Use text instead")
        raise ValueError
    return separators

def build_DTA(data, D, B, A, ass_IO_ass):
    """
    Function that builds structure based on IO to later get Journal and ROIs.
    Args:
        - data
        - I: structure containing indexes
        - ass_IO_ass: asset to IO assignment
    """
    # init columns
    columns = ["DT1","DT2","B1","B2","A1","A2","Asset"]
    # init DTA
    DTA = pd.DataFrame()
    # init hdf5 file with raw data
    

    ass_index = 0
    last_ass_IO_ass = 0
    # loop over assets
    for ass in data.assets:
        # get this asset's name
        thisAsset = data.AllAssets[str(ass)]
        print(thisAsset)
        # init DTA for this asset
        DTA_i = pd.DataFrame(columns = columns)
#        entry_idx = I[last_ass_IO_ass:ass_IO_ass[ass_index],:,0].reshape((-1))
#        exit_idx = I[last_ass_IO_ass:ass_IO_ass[ass_index],:,1].reshape((-1))
        # fill DTA_i up
        DTA_i['DT1'] = D[last_ass_IO_ass:ass_IO_ass[ass_index],:,0].reshape((-1))
        if DTA_i.shape[0]>0:
            DTA_i['DT1'] = DTA_i['DT1'].str.decode('utf-8')
            print(DTA_i['DT1'].iloc[0])
            print(DTA_i['DT1'].iloc[-1])
            if DTA_i['DT1'].iloc[0][:10] not in data.dateTest:
                print("WARNING!!! DTA_i['DT1'].iloc[0][:10] not in data.dateTest")
            #assert(DTA_i['DT1'].iloc[0][:10] in data.dateTest)
#            assert(DTA_i['DT1'].iloc[-1][:10] in data.dateTest)
            if DTA_i['DT1'].iloc[-1][:10] not in data.dateTest:
                print("WARNING!!! DTA_i['DT1'].iloc[-1][:10] not in data.dateTest")
            #DTA_i['DT1'] = DTA_i['DT1'].str.decode('utf-8')
            DTA_i['B1'] = B[last_ass_IO_ass:ass_IO_ass[ass_index],:,0].reshape((-1))
            DTA_i['A1'] = A[last_ass_IO_ass:ass_IO_ass[ass_index],:,0].reshape((-1))
            
            DTA_i['DT2'] = D[last_ass_IO_ass:ass_IO_ass[ass_index],:,1].reshape((-1))
            DTA_i['DT2'] = DTA_i['DT2'].str.decode('utf-8')
            DTA_i['B2'] = B[last_ass_IO_ass:ass_IO_ass[ass_index],:,1].reshape((-1))
            DTA_i['A2'] = A[last_ass_IO_ass:ass_IO_ass[ass_index],:,1].reshape((-1))
            DTA_i['Asset'] = thisAsset
    #        print(DTA_i['DT1'].iloc[0])
    #        print(DTA_i['DT1'].iloc[-1])
            # append DTA this asset to all DTAs
            DTA = DTA.append(DTA_i,ignore_index=True)
        last_ass_IO_ass = ass_IO_ass[ass_index]
        ass_index += 1
    # end of for ass in data.assets:
    return DTA

def load_stats_output(data, hdf5_directory, thisAsset):
    """
    Load output stats
    """
    # TODO: pass output stats to their own container and load them from there
    stats = pickle.load( open( hdf5_directory+'stats/'+thisAsset+'_stats_mW'+
                                      str(data.movingWindow)+
                                     '_nE'+str(data.nEventsPerStat)+
                                     '_nF37'+'.p', 'rb' ))
    stats_output = {'m_t_out':stats['m_t_out'],
                    'stds_t_out':stats['stds_t_out']}
    return stats_output
    
def load_stats(data, thisAsset, ass_group, save_stats, from_stats_file=False, 
               hdf5_directory='', save_pickle=False):
    """
    Function that loads stats
    """
    nChannels = int(data.nEventsPerStat/data.movingWindow)
    nF = len(data.feature_keys_manual)
    # init or load total stats
    stats = {}
    if save_stats:
        
        stats["means_t_in"] = np.zeros((nChannels,nF))
        stats["stds_t_in"] = np.zeros((nChannels,nF))
        stats["means_t_out"] = np.zeros((1,len(data.lookAheadVector)))
        stats["stds_t_out"] = np.zeros((1,len(data.lookAheadVector)))
        stats["m_t_in"] = 0
        stats["m_t_out"]  = 0
    elif not from_stats_file:
        stats["means_t_in"] = ass_group.attrs.get("means_t_in")
        stats["stds_t_in"] = ass_group.attrs.get("stds_t_in")
        stats["means_t_out"] = ass_group.attrs.get("means_t_out")
        stats["stds_t_out"] = ass_group.attrs.get("stds_t_out")
        stats["m_t_in"] = ass_group.attrs.get("m_t_in")
        stats["m_t_out"] = ass_group.attrs.get("m_t_out")
    
    elif from_stats_file:
        try:
            stats = pickle.load( open( hdf5_directory+thisAsset+'_stats_mW'+
                                      str(data.movingWindow)+
                                     '_nE'+str(data.nEventsPerStat)+
                                     '_nF'+str(nF)+".p", "rb" ))
        except FileNotFoundError:
            print("WARNING FileNotFoundError: [Errno 2] No such file or directory."+
                  " Getting stats from HDF5 file")
            stats["means_t_in"] = ass_group.attrs.get("means_t_in")
            stats["stds_t_in"] = ass_group.attrs.get("stds_t_in")
            stats["means_t_out"] = ass_group.attrs.get("means_t_out")
            stats["stds_t_out"] = ass_group.attrs.get("stds_t_out")
            stats["m_t_in"] = ass_group.attrs.get("m_t_in")
            stats["m_t_out"] = ass_group.attrs.get("m_t_out")
    else:
        #print("EROR: Not a possible combination of input parameters")
        raise ValueError("Not a possible combination of input parameters")
        
    # save stats individually
    if save_pickle:
        pickle.dump( stats, open( hdf5_directory+thisAsset+'_stats_mW'+
                                 str(data.movingWindow)+
                                 '_nE'+str(data.nEventsPerStat)+
                                 '_nF'+str(nF)+".p", "wb" ))
    return stats

def load_stats_manual(data, thisAsset, ass_group, from_stats_file=False, 
               hdf5_directory='', save_pickle=False):
    """
    Function that loads stats
    """
    nF = len(data.feature_keys_manual)
    # init or load total stats
    stats = {}

    if not from_stats_file:
        stats["means_t_in"] = ass_group.attrs.get("means_t_in")
        stats["stds_t_in"] = ass_group.attrs.get("stds_t_in")
        stats["m_t_in"] = ass_group.attrs.get("m_t_in")
    
    elif from_stats_file:
        try:
            stats = pickle.load( open( hdf5_directory+thisAsset+'_stats_mW'+
                                      str(data.movingWindow)+
                                     '_nE'+str(data.nEventsPerStat)+
                                     '_nF'+str(nF)+".p", "rb" ))
        except FileNotFoundError:
            print("WARNING FileNotFoundError: [Errno 2] No such file or directory."+
                  " Getting stats from HDF5 file")
            stats["means_t_in"] = ass_group.attrs.get("means_t_in")
            stats["stds_t_in"] = ass_group.attrs.get("stds_t_in")
            stats["m_t_in"] = ass_group.attrs.get("m_t_in")
    else:
        print("EROR: Not a possible combination of input parameters")
        raise ValueError
        
    # save stats individually
    if save_pickle:
        pickle.dump( stats, open( hdf5_directory+thisAsset+'_stats_mW'+
                                 str(data.movingWindow)+
                                 '_nE'+str(data.nEventsPerStat)+
                                 '_nF'+str(nF)+".p", "wb" ))
    return stats

def load_stats_tsf(data, thisAsset, root_directory, features_file, load_from_stats_file=True):
    """ Load stats (mean and std) from HDF5 file for TSFRESH features """
    # TODO: to be tested
    # init means and stds
    window_size = data.nEventsPerStat
    sprite_length = data.movingWindow
    
    nMaxChannels = int(window_size/sprite_length)
    
    means = np.zeros((nMaxChannels,data.n_feats_tsfresh))
    stds = np.zeros((nMaxChannels,data.n_feats_tsfresh))
    
    stats_directory = root_directory+'stats/'
    filename = ('mW'+str(sprite_length)+
                '_nE'+str(window_size)+'_2.hdf5')
    # init hdf5 container
    stats_file = h5py.File(stats_directory+'stats_'+filename,'r')
    #features_file = h5py.File(root_directory+'feats_'+filename,'r')
    # loop over features
    c = 0
    for f in data.feature_keys_tsfresh:
        means_name = 'means_in'+str(f)
        stds_name = 'stds_in'+str(f)
        n_new_feats = data.AllFeatures[str(f)][-1]
        # check first in stats file
        if load_from_stats_file and thisAsset in stats_file and means_name in list(stats_file[thisAsset].attrs.keys()):
            means[:,c:c+n_new_feats] = stats_file[thisAsset].attrs.get(means_name)
            stds[:,c:c+n_new_feats] = stats_file[thisAsset].attrs.get(stds_name)
        # if not, check in features file
        elif thisAsset in features_file and means_name in list(features_file[thisAsset].attrs.keys()):
            means[:,c:c+n_new_feats] = features_file[thisAsset].attrs.get(means_name)
            stds[:,c:c+n_new_feats] = features_file[thisAsset].attrs.get(stds_name)
            print("Here!")
            # add stats to stats file
#            stats_file[thisAsset].attrs.create(means_name, means[:,c:c+n_new_feats], dtype=float)
#            stats_file[thisAsset].attrs.create(stds_name, stds[:,c:c+n_new_feats], dtype=float)
        else: # error
            raise ValueError("Stats not found")
        c += n_new_feats
    # add loaded stats to dictionary
    stats_tsf = {'means_t_in':means,
                 'stds_t_in':stds}
    stats_file.close()
    #features_file.close()
    return stats_tsf