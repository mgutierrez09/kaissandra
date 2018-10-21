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
import sqlite3
import scipy.io as sio
import pickle
import re

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

    AllFeatures = {"0":"bids",
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
                "36":"difBidOema100"}
    
    
    average_over = np.array([0.1, 0.5, 1, 5, 10, 50, 100])
    #average_over = np.array([0.11,0.2,0.3,.4,.5,.6,.7,.8,.9,1,1.5,2,2.5,3,4,5,6,7,8,9,10,20,30,40,50,60,70,80,90,100])
    
    train = "Train"
    test = "Test"
    
    def __init__(self, movingWindow=40,nEventsPerStat=40,lB=400,std_var=0.1,nFeaturesAuto=0,channels=[0],
                 lookAhead=1,comments='',save_data_every=1,save_IO=1,divideLookAhead=1,lookAheadIndex=3,
                 features=[i for i in range(37)],BDeval='../DB/EVAL.sqlite',lookAheadVector=[.1,.2,.5,1,2,5,10],
                 dateStart='2017.08.14',dateEnd='2017.09.19',dateTest=['2017.09.15','2017.11.06','2017.11.07'],
                 assets=[1,2,3,4,7,8,10,11,12,13,14,15,16,17,19,27,28,29,30,31,32], IDweights=None,FCNID=None):
        
        if IDweights==None:
            
            
            self.movingWindow = movingWindow
            self.nEventsPerStat = nEventsPerStat
            self.nFeatures = len(features)
            
            self.lB = lB
            self.lookBack = int(lB/movingWindow) # lookback 500 events
            
            self.lookAhead = lookAhead
            self.maxTimeDif = dt.timedelta(minutes=7) # maximum time allowed between files to be concatenated
            self.divideLookAhead = divideLookAhead
            
            assert(max(channels)<nEventsPerStat/movingWindow)
            self.channels = channels
            
            
        else:
            IDstr = "{0:06d}".format(int(IDweights))
            conn = sqlite3.connect(BDeval)
            parameters = pd.read_sql_query("SELECT * FROM IDw WHERE `index`=="+"'"+IDstr+"'", conn)
            #self.dateEnd = parameters["dateEnd"].iloc[0]
            #self.dateStart = parameters["dateStart"].iloc[0]
            self.lookAhead = parameters["lookAhead"].iloc[0]
            self.movingWindow = parameters["movingWindow"].iloc[0]
            if "lookBack" in parameters.columns:
                self.lookBack = parameters["lookBack"].iloc[0]
                self.lB = self.lookBack*self.movingWindow
            else:
                self.lB = parameters["lB"].iloc[0]
                self.lookBack = int(self.lB/self.movingWindow)
                
            self.nEventsPerStat = parameters["nEventsPerStat"].iloc[0]
            self.nFeatures = parameters["nFeatures"].iloc[0]
            self.divideLookAhead = parameters["divideLookAhead"].iloc[0]
            
            p = re.compile("1")
            '''
            self.assets = []
            for m in p.finditer(parameters["assets"].iloc[0]):
                #print(m.start())
                self.assets.append(m.start())
            '''
            testDaysIndex =[]
            testDates = []
            dateDT = dt.datetime.strptime(dateStart,"%Y.%m.%d")
            for m in p.finditer(parameters["dateTest"].iloc[0]):
                #print(m.start())
                testDaysIndex.append(m.start())
                testDate = dt.datetime.strftime(dateDT+dt.timedelta(days=m.start()),"%Y.%m.%d")
                if testDate not in dateTest:
                    print("Warning! Test Dates don't match!")
                testDates.append(testDate)
            #print(testDates)
            conn.close()
        
        self.lbd=1-1/(self.nEventsPerStat*self.average_over)
        self.features=features
        self.std_var = std_var
        self.std_time = std_var
        self.comments = comments
        self.save_data_every = save_data_every
        self.save_IO = save_IO
        self.dateTest = dateTest
        self.dateStart = dateStart
        self.dateEnd = dateEnd
        self.assets = assets
        self.noVarFeats = [8,9,12,17,18,21,23,24,25,26,27,28,29]
        self.lookAheadVector = lookAheadVector
        self.lookAheadIndex = lookAheadIndex
        self.FCNID = FCNID
        self.nFeaturesAuto = nFeaturesAuto

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
    
    featuresLive = np.zeros((data.nFeatures,1))
    
    initRange = int(data.nEventsPerStat/data.movingWindow)
    em = np.zeros((data.lbd.shape))+tradeInfoLive.SymbolBid.loc[0+tradeInfoLive.SymbolBid.index[0]]
    for i in range(initRange*data.movingWindow-data.movingWindow):
        em = data.lbd*em+(1-data.lbd)*tradeInfoLive.SymbolBid.loc[i+tradeInfoLive.SymbolBid.index[0]]
    
    
    if 1 in data.features:
        featuresLive[1:1+data.lbd.shape[0],0] = tradeInfoLive.SymbolBid.iloc[0]
        for i in range(int(data.nEventsPerStat*(1-data.movingWindow/data.nEventsPerStat))):
            featuresLive[1:1+data.lbd.shape[0],0] = data.lbd*featuresLive[1:1+data.lbd.shape[0],0]+(
                    1-data.lbd)*tradeInfoLive.SymbolBid.iloc[i]
    #print(em-featuresLive[1:1+data.lbd.shape[0],0])
    if 10 in data.features:
        featuresLive[10,0] = tradeInfoLive.SymbolBid.iloc[0]
        featuresLive[11,0] = tradeInfoLive.SymbolBid.iloc[0]
    
    if 13 in data.features:
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
    
    
    if 1 in data.features:
        newEventsRange = range(int(nEvents*(1-data.movingWindow/data.nEventsPerStat)),nEvents)
        #print(newEventsRange)
        eml = featuresLive[1:1+data.lbd.shape[0],0]
        for il in newEventsRange:
            #print(tradeInfoLive.SymbolBid.iloc[il])
            eml = data.lbd*eml+(
                    1-data.lbd)*tradeInfoLive.SymbolBid.iloc[il]
        
        featuresLive[1:1+data.lbd.shape[0],0] = eml
        
    if 10 in data.features:
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
        
    if 13 in data.features:
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

    if 0 in data.features:
        featuresLive[0,0] = tradeInfoLive.SymbolBid.iloc[-1]
    
    
    if 8 in data.features:
            featuresLive[8,0] = 10*np.log10(np.var(tradeInfoLive.SymbolBid.iloc[:])/data.std_var+1e-10)
        
    if 9 in data.features:
        te = pd.to_datetime(tradeInfoLive.iloc[-1].DateTime)
        t0 = pd.to_datetime(tradeInfoLive.iloc[0].DateTime)
        timeInterval = (te-t0).seconds/nEvents
        featuresLive[9,0] = 10*np.log10(timeInterval/data.std_time+0.01)

    if 12 in data.features:
        secsInDay = 86400.0
        if 9 not in data.features:
            # calculate te if not yet calculated
            te = pd.to_datetime(tradeInfoLive.iloc[-1].DateTime)
        timeSec = (te.hour*60*60+te.minute*60+te.second)/secsInDay
        featuresLive[12,0] = timeSec
    
    # Repeat non-variation features to inclue variation betwen first and second input
    if 15 in data.features:
        featuresLive[15,0] = featuresLive[8,0]
    
    if 16 in data.features:
        featuresLive[16,0] = featuresLive[9,0]
        
    if 17 in data.features:
        featuresLive[17,0] = np.max(tradeInfoLive.SymbolBid.iloc[:])-featuresLive[0,0]
    
    if 18 in data.features:
        featuresLive[18,0] = featuresLive[0,0]-np.min(tradeInfoLive.SymbolBid.iloc[:])
    
    if 19 in data.features:
        featuresLive[19,0] = np.max(tradeInfoLive.SymbolBid.iloc[:])-featuresLive[0,0]
        
    if 20 in data.features:
        featuresLive[20,0] = featuresLive[0,0]-np.min(tradeInfoLive.SymbolBid.iloc[:])
    
    if 21 in data.features:
        featuresLive[21,0] = np.min(tradeInfoLive.SymbolBid.iloc[:])/np.max(tradeInfoLive.SymbolBid.iloc[:])
    
    if 22 in data.features:
        featuresLive[22,0] = np.min(tradeInfoLive.SymbolBid.iloc[:])/np.max(tradeInfoLive.SymbolBid.iloc[:])
    
    for i in range(data.lbd.shape[0]):
        #nF = nF+1
        if 23+i in data.features:                 
            featuresLive[23+i,0] = featuresLive[0,0]/eml[i]
    
    for i in range(data.lbd.shape[0]):
        #nF = nF+1
        if 30+i in data.features:                 
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

def build_output(model, Output, batch_size):
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
#        print("negativeYs.shape")
#        print(negativeYs.shape)
        # set to 1 the corresponding entries
        y_c1[np.squeeze(negativeYs),0] = 1
        y_c2[np.squeeze(positiveYs),0] = 1
#        print("y_c1")
#        print(y_c1)
        # append to y_c
#        print("y_c.shape")
#        print(y_c.shape)
#        print("y_c1.shape")
#        print(y_c1.shape)
        y_c = np.append(y_c,y_c1,axis=2)
        y_c = np.append(y_c,y_c2,axis=2)
#        print(y_c)
#        print("y_c.shape")
#        print(y_c.shape)
    # build output vector
    #print(y_c)

    y_bin = np.append(y_c,y_one_hot,axis=2)
#    print("y_bin.shape")
#    print(y_bin.shape)

    return y_bin, y_dec

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

def build_bin_output(model, Output, batch_size):
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
        l_index = startIndex
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
        #print(nF)
    # end of for b in range(par_batches):
#    save_as_matfile('features','features',features[:])
#    a=p
    return features

def get_returns_from_raw(data, returns, ret_idx, DT, B, A, idx_init, DateTime, SymbolBid, SymbolAsk):
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
    # open temporal file for variations
    try:
        # create file
        ft = h5py.File(hdf5_directory+'temp.hdf5','w')
        # create group
        group_temp = ft.create_group('temp')
        # reserve memory space for variations and normalized variations
        variations = group_temp.create_dataset("variations", (features.shape[0],data.nFeatures,nChannels), dtype=float)
        # init variations and normalized variations to 999 (impossible value)
        
        #variations[:] = variations[:]+999
        print("\t getting variations")
        # loop over channels
        for r in range(nChannels):
            variations[r+1:,:,r] = features[r+1:,:]-features[:-(r+1),:]
            variations[r+1:,data.noVarFeats,r] = features[:-(r+1),data.noVarFeats]
        print("\t time for variations: "+str(time.time()-tic))
        # init stats    
        means_in = np.zeros((nChannels,data.nFeatures))
        stds_in = np.zeros((nChannels,data.nFeatures))
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
        error()
    ft.close()
    return [means_in, stds_in, means_out, stds_out]

def load_features_results(data, thisAsset, separators, f_prep_IO, s):
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
        
        # group name of separator s
        group_name = thisAsset+'/'+init_date+end_date
        
        # create new gruop if not yet in file
        if group_name not in f_prep_IO:
            # create group, its attributes and its datasets
            print("Group should already exist. Run get_features_results_stats_from_raw first to create it.")
            error()
        else:
            # get group from file
            group = f_prep_IO[group_name]
        # get features, returns and stats if don't exist
        if group.attrs.get("means_in") is None:
            print("Group should already exist. Run get_features_results_stats_from_raw first to create it.")
            error()
        else:
            # get data sets
            features = group['features']
            returns = group['returns']
            ret_idx = group['ret_idx']
            if 'DT' in group:
                DT = group['DT']
                B = group['B']
                A = group['A']
    # end of if separators.index[s+1]-separators.index[s]>=2*data.nEventsPerStat:
    # save results in a dictionary
    IO_prep = {}
    IO_prep['features'] = features
    IO_prep['returns'] = returns
    IO_prep['ret_idx'] = ret_idx
    if 'DT' in group:
        IO_prep['DT'] = DT
        #print(DT.shape)
        #print(DT[0,0])
        #print(DT[-1,-1])
        IO_prep['B'] = B
        IO_prep['A'] = A
    
    return IO_prep

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
            
            # number of features and number of returns
            m_in = int(np.floor((nE/nExS-1)*nExS/mW)+1)
            m_out = int(m_in-nExS/mW)#len(range(int(nExS/mW)*mW-1,m_in*mW-1,mW))
            # save as attributes
            group.attrs.create("m_in", m_in, dtype=int)
            group.attrs.create("m_out", m_out, dtype=int)
            # create datasets
            try:
                features = group.create_dataset("features", (m_in,data.nFeatures),dtype=float)
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
                
            print("\tgetting IO from raw data...")
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

def build_IO(file_temp, data, model, IO_prep, stats, IO, nSampsPerLevel, s, nE, thisAsset):
    """
    Function that builds X and Y from data contained in a HDF5 file.
    """
    # extract means and stats
    means_in = stats['means_t_in']
    stds_in = stats['stds_t_in']
    stds_out = stats['stds_t_out']
    # extract features and returns
    features = IO_prep['features']
    returns = IO_prep['returns']
    ret_idx = IO_prep['ret_idx']
    # add dateTimes, bids and asks if are included in file
    all_info = 0
    if 'D' in IO:
        all_info = 1
        dts = IO_prep['DT']
        bids = IO_prep['B']
        asks = IO_prep['A']
        
        D = IO['D']
        B = IO['B']
        A = IO['A']

    # extract IO structures
    X = IO['X']
    Y = IO['Y']
    I = IO['I']
    pointer = IO['pointer']
    # total number of possible channels
    nChannels = int(data.nEventsPerStat/data.movingWindow)
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
    nonVarFeats = np.intersect1d(data.noVarFeats,data.features)
    #non-variation idx for variations normed
    nonVarIdx = np.zeros((len(nonVarFeats))).astype(int)
    nv = 0
    for allnv in range(nF):
        if data.features[allnv] in nonVarFeats:
            nonVarIdx[nv] = int(allnv)
            nv += 1
    # loop over channels
    for r in range(nC):
        variations[data.channels[r]+1:,:,r] = features[data.channels[r]+1:,data.features]-features[:-(data.channels[r]+1),data.features]
        if nonVarFeats.shape[0]>0:
            variations[data.channels[r]+1:,nonVarIdx,data.channels[r]] = features[:-(data.channels[r]+1),nonVarFeats]
        variations_normed[data.channels[r]+1:,:,data.channels[r]] = np.minimum(np.maximum((variations[data.channels[r]+1:,
                          :,data.channels[r]]-means_in[r,data.features])/stds_in[r,data.features],-10),10)
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
        # create support numpy vectors to speed up iterations
        v_support = variations_normed_new[offset:offset+batch+seq_len, :, :]
        r_support = returns[nChannels+offset+2:nChannels+offset+batch+seq_len+2, data.lookAheadIndex]
        # we only take here the entry time index, and later at DTA building the 
        # exit time index is derived from the entry time and the number of events to
        # look ahead
        i_support = ret_idx[nChannels+offset+2:nChannels+offset+batch+seq_len+2, [0,data.lookAheadIndex+1]]
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
                X_i[nI,:,cc*nF:(cc+1)*nF] = v_support[nI:nI+seq_len, :, r]
                cc += 1
            # due to substraction of features for variation, output gets the 
            # feature one entry later
            O_i[nI,:,0] = r_support[nI:nI+seq_len]
            I_i[nI,:,:] = i_support[nI:nI+seq_len,:]
            if all_info:
                D_i[nI,:,:] = dt_support[nI:nI+seq_len,:]
                B_i[nI,:,:] = b_support[nI:nI+seq_len,:]
                A_i[nI,:,:] = a_support[nI:nI+seq_len,:]
        
        
        # normalize output
        O_i = O_i/stds_out[0,data.lookAheadIndex]#stdO#
        # update counters
        offset = offset+batch
        # get decimal and binary outputs
        Y_i, y_dec = build_bin_output(model, O_i, batch)
        # get samples per level
        for l in range(model.size_output_layer):
            nSampsPerLevel[l] = nSampsPerLevel[l]+np.sum(y_dec[:,-1,0]==l)
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
    
    return IO, nSampsPerLevel

def load_separators(data, thisAsset, separators_directory, tOt='tr', from_txt=1):
    """
    Function that loads and segments separators according to beginning and end dates.
    
    """
    if from_txt:
        # separators file name
        separators_filename = thisAsset+'_separators.txt'
        # load separators
        separators = pd.read_csv(separators_directory+separators_filename, index_col='Pointer')
    else:
        print("Depricated load separators from DB. Use text instead")
        error()
    return separators

def build_DTA(data, I, ass_IO_ass, hdf5_directory):
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
    
    filename_raw = hdf5_directory+'tradeinfo.hdf5'
    f_raw = h5py.File(filename_raw,'r')
    ass_index = 0
    last_ass_IO_ass = 0
    # loop over assets
    for ass in data.assets:
        # get this asset's name
        thisAsset = data.AllAssets[str(ass)]
        print(thisAsset)
        # init DTA for this asset
        DTA_i = pd.DataFrame(columns = columns)
        entry_idx = I[last_ass_IO_ass:ass_IO_ass[ass_index],:,0].reshape((-1))
        exit_idx = I[last_ass_IO_ass:ass_IO_ass[ass_index],:,1].reshape((-1))
        last_ass_IO_ass = ass_IO_ass[ass_index]
        ass_index += 1
        # get raw data for this asset
        group_raw = f_raw[thisAsset]
        # fill DTA_i up
        DTA_i['DT1'] = group_raw['DateTime'][:][entry_idx]
        #print(DTA_i['DT1'])
        DTA_i['DT1'] = DTA_i['DT1'].str.decode('utf-8')
        DTA_i['B1'] = group_raw['SymbolBid'][:][entry_idx]
        DTA_i['A1'] = group_raw['SymbolAsk'][:][entry_idx]
        
        DTA_i['DT2'] = group_raw['DateTime'][:][exit_idx]
        DTA_i['DT2'] = DTA_i['DT2'].str.decode('utf-8')
        DTA_i['B2'] = group_raw['SymbolBid'][:][exit_idx]
        DTA_i['A2'] = group_raw['SymbolAsk'][:][exit_idx]
        DTA_i['Asset'] = thisAsset
        print(DTA_i['DT1'].iloc[0])
        print(DTA_i['DT1'].iloc[-1])
        # append DTA this asset to all DTAs
        DTA = DTA.append(DTA_i,ignore_index=True)
    # end of for ass in data.assets:
    # close raw file
    f_raw.close()
    return DTA

def build_DTA_v10(data, D, B, A, ass_IO_ass):
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
        DTA_i['DT1'] = DTA_i['DT1'].str.decode('utf-8')
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

def load_stats(data, thisAsset, ass_group, save_stats, from_stats_file=False, hdf5_directory='', save_pickle=False):
    """
    Function that loads stats
    """
    nChannels = int(data.nEventsPerStat/data.movingWindow)
    # init or load total stats
    stats = {}
    if save_stats:
        
        stats["means_t_in"] = np.zeros((nChannels,data.nFeatures))
        stats["stds_t_in"] = np.zeros((nChannels,data.nFeatures))
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
        
        # save stats individually
        if save_pickle:
            pickle.dump( stats, open( hdf5_directory+thisAsset+'_stats_mW'+str(data.movingWindow)+
                                     '_nE'+str(data.nEventsPerStat)+'_nF'+str(data.nFeatures)+".p", "wb" ))
    
    elif from_stats_file:
        try:
            stats = pickle.load( open( hdf5_directory+thisAsset+'_stats_mW'+str(data.movingWindow)+
                                     '_nE'+str(data.nEventsPerStat)+'_nF'+str(data.nFeatures)+".p", "rb" ))
        except FileNotFoundError:
            print("WARNING FileNotFoundError: [Errno 2] No such file or directory. Getting stats from HDF5 file")
            stats["means_t_in"] = ass_group.attrs.get("means_t_in")
            stats["stds_t_in"] = ass_group.attrs.get("stds_t_in")
            stats["means_t_out"] = ass_group.attrs.get("means_t_out")
            stats["stds_t_out"] = ass_group.attrs.get("stds_t_out")
            stats["m_t_in"] = ass_group.attrs.get("m_t_in")
            stats["m_t_out"] = ass_group.attrs.get("m_t_out")
    else:
        print("EROR: Not a possible combination of input parameters")
        error()
        
    
    return stats