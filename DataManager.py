# -*- coding: utf-8 -*-
"""
Created on Thu Oct 5 21:39:48 2017

@author: mgutierrez

Module containing all relevant functions related to data manipulation, storage
and formatting
"""

#import h5py
import numpy as np
import pandas as pd
import re
import os
import datetime as dt
import time
import pickle
import scipy.io as sio
import sqlite3
#from DNN import getOutputsLayersFCN


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

def extractSeparators_v21(tradeInfo,minThresDay,minThresNight,bidThresDay,bidThresNight,dateTest, tOt="tr"):
    
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

def mergeSeparators_v21(oldSeparators,newSeparators):
    
    newSepIndex = newSeparators.index
    #print(newSepIndex[:])
    separators = oldSeparators.append(newSeparators).sort_index()
    #print(separators)
    
    allIndexes = separators.index
    for i in allIndexes[1:-1]:
        ilocIndex = allIndexes.get_loc(i)
        locIndexPrev = allIndexes[ilocIndex-1]
        locIndexNext = allIndexes[ilocIndex+1]
        
        # find transition indexes
        if i in newSepIndex and locIndexPrev not in newSepIndex:
            #print("transition to new sep")
            #print("locIndexPrev "+str(locIndexPrev))
            
            if allIndexes[ilocIndex]-1==allIndexes[ilocIndex-1]:
                #print("remove seps")
                ilocIsep = separators.index.get_loc(i)
                separators = separators.drop(separators.index[ilocIsep]).drop(separators.index[ilocIsep-1])
        elif i in newSepIndex and locIndexNext not in newSepIndex:
            if allIndexes[ilocIndex]+1==allIndexes[ilocIndex+1]:
                ilocIsep = separators.index.get_loc(i)
                separators = separators.drop(separators.index[ilocIsep]).drop(separators.index[ilocIsep+1])

    return separators

def extractFromSeparators_v20(separators, dateStart, dateEnd, asset, DBforex):
    
    
    initChunkSep = separators["DateTime"][::2]
    endChunkSep = separators["DateTime"][1::2]
    
    replaceInitStartIndex = ((pd.to_datetime(initChunkSep)-dt.datetime.strptime(dateStart,'%Y.%m.%d'))<=dt.timedelta(0))#[::-1].argmax() #.str.find('2017.09.11')>=0
    replaceEndStartIndex = ((pd.to_datetime(endChunkSep)-dt.datetime.strptime(dateStart,'%Y.%m.%d'))<=dt.timedelta(0))
    
    replaceEndEndtIndex = ((pd.to_datetime(endChunkSep)-(dt.datetime.strptime(dateEnd,'%Y.%m.%d')+dt.timedelta(days=1)))>=dt.timedelta(0))
    replaceInitEndtIndex = ((pd.to_datetime(initChunkSep)-(dt.datetime.strptime(dateEnd,'%Y.%m.%d')+dt.timedelta(days=1)))>=dt.timedelta(0))
    
    #print(np.logical_xor(replaceInitStartIndex,replaceEndStartIndex))
    
    maxLogical = np.logical_xor(replaceInitStartIndex,replaceEndStartIndex).max()
    #print(np.isnan(maxLogical))
    if maxLogical and ~np.isnan(maxLogical):
        conn = sqlite3.connect(DBforex)
        conn.create_function("REGEXP", 2, regexp)
        dropStartIdxs = separators.loc[initChunkSep[replaceInitStartIndex].index[0]:initChunkSep[replaceInitStartIndex].index[-1]].index
        firstSeparator = pd.read_sql_query("SELECT * FROM "+asset+" WHERE DateTime REGEXP ('"+
                                  dateStart+"') LIMIT 1", conn).set_index("real_index")
        if firstSeparator.shape[0] == 0:
            print("Error! endSeparator cannot be empty!")
            error()
    elif replaceInitStartIndex.max()==False or np.isnan(maxLogical):
        dropStartIdxs = []
        firstSeparator = pd.DataFrame(data=[],columns=['DateTime','SymbolBid','SymbolAsk'])
    else:
        dropStartIdxs = separators.loc[initChunkSep[replaceInitStartIndex].index[0]:initChunkSep[replaceInitStartIndex].index[-1]].index
        dropStartIdxs = dropStartIdxs.append(separators.loc[endChunkSep[replaceEndStartIndex].index[0]:endChunkSep[replaceEndStartIndex].index[-1]].index)
        firstSeparator = pd.DataFrame(data=[],columns=['DateTime','SymbolBid','SymbolAsk'])
    
    maxLogical = np.logical_xor(replaceEndEndtIndex,replaceInitEndtIndex).max()
    if maxLogical and ~np.isnan(maxLogical):
        conn = sqlite3.connect(DBforex)
        conn.create_function("REGEXP", 2, regexp)
        dropEndtIdxs = separators.loc[endChunkSep[replaceEndEndtIndex].index[0]:endChunkSep[replaceEndEndtIndex].index[-1]].index
        endSeparator = pd.read_sql_query("SELECT * FROM "+asset+
                                " DESC WHERE DateTime REGEXP ('"+dateEnd+
                                "')  ORDER BY real_index DESC LIMIT 1", conn).set_index("real_index")
        if endSeparator.shape[0] == 0:
            print("Error! endSeparator cannot be empty!")
            error()
            
    elif replaceEndEndtIndex.max()==False or np.isnan(maxLogical):
        dropEndtIdxs = []
        endSeparator = pd.DataFrame(data=[],columns=['DateTime','SymbolBid','SymbolAsk'])
    else:
        dropEndtIdxs = separators.loc[endChunkSep[replaceEndEndtIndex].index[0]:endChunkSep[replaceEndEndtIndex].index[-1]].index
        dropEndtIdxs = dropEndtIdxs.append(separators.loc[initChunkSep[replaceInitEndtIndex].index[0]:initChunkSep[replaceInitEndtIndex].index[-1]].index)
        endSeparator = pd.DataFrame(data=[],columns=['DateTime','SymbolBid','SymbolAsk'])
        
    separatorsExt = pd.DataFrame(data=[],columns=['DateTime','SymbolBid','SymbolAsk'])
    separatorsExt = separatorsExt.append(firstSeparator).append(separators.drop(dropStartIdxs)).drop(dropEndtIdxs).append(endSeparator)
    
    return separatorsExt

def regexp(expr, item):
    reg = re.compile(expr)
    return reg.search(item) is not None

def extractFeaturesFromFCN_v00(model, data, SymbolBid, weights, separators,
                               mean, std, s, aloc = 100000):
    
    nE = data.nEventsPerStat
    mW = data.movingWindow
    
    bids = SymbolBid.loc[separators.index[s]:separators.index[s+1]+1].as_matrix()

    nEvents = bids.shape[0]
    nEintoMovW = nEvents/data.movingWindow
    allSamps = np.floor(nEintoMovW)-np.ceil(data.nEventsPerStat/data.movingWindow)
    allSamps = allSamps.astype(int)
    initRange = int(data.nEventsPerStat/data.movingWindow)
    nSamps = allSamps-initRange
            
    #m = int(np.floor(L/nE-1)*nE/mW+1)-int(nE/mW) # substract the lost sample due to difference calculations
    m = nSamps
    #print("m "+str(m))
    chunks = int(np.ceil(m/aloc))
    Z = np.zeros((model.size_hidden_layer,0,model.L-2))
    offset = 0
    for e in range(chunks):
        
        batch = np.min([m,aloc])
        X_i = np.zeros((batch, nE))
        for i in range(nE):
            diffBids = bids[mW+offset+i:mW+offset+mW*m+i:mW,0]-bids[offset+i:offset+mW*m+i:mW,0]
            X_i[:,i] = diffBids
        
        X_i = (X_i-mean)/std
        # extract features
        Z_e = getOutputsLayersFCN(model, X_i.T, weights)
        #print(Z_e.shape)
        Z = np.append(Z,Z_e,axis=1)
        offset = offset+batch
        
    # serialize all layers
    print("Features obtained from FCN")
    n_layers = int(data.nFeaturesAuto/model.size_hidden_layer)
    Z_l = Z[:,:,-n_layers:].reshape(data.nFeaturesAuto,Z.shape[1])
    #Z_l = Z.reshape((Z.shape[0]*Z.shape[2],Z.shape[1]))
    # check percent of saturated feature points
    for l in range(Z.shape[2]):
        print("Percent of 1s in layer "+str(l)+" = "+str(np.sum(np.abs(Z[:,:,l])==1)/Z[:,:,l].size))
    #a=p
    return Z_l

def buildIORNN_v12(model, data, asset, trainOrTest, DBforex, DBDD, DBfeatures, DBresults, DBstats,
                           X, Y, DTA, filesCounter, aloc = 100000,logFile="",ID="",
                           saveDTA=False,autoFeatures=0,modelFCN=None):
    """
    Version 1.2 of build-IO-for-RNN functions. Despite its name, it is an older
    version of the 1.1, since this one builds one specific case of the more genenal IO
    construction of version 1.1. More specifically, this version builds an input 
    matrix X with only one channel, corresponding to the furthest one to the current
    entry. 
    """
    print("\nLoading "+ asset)
    if logFile!="":
        file = open(logFile,"a")
        file.write("\nLoading "+ asset+"\n")
        file.close()
    
    if trainOrTest==data.train:
        tOt = "tr"
    else:
        tOt = "te"
        
    #print(separators)
    
    means = np.zeros((1,data.nFeatures))
    stds = np.zeros((1,data.nFeatures))
    

    stats = loadStatsFromDB_v33(DBstats, data, asset, trainOrTest)
        
    means[0,:] = stats.as_matrix().T[0,0:data.nFeatures]
    stds[0,:] = stats.as_matrix().T[1,0:data.nFeatures]
    stdO = float(stats.as_matrix().T[1,-1])
    #print("stdO")
    #print(stdO)
    connDD = sqlite3.connect(DBDD)
    separators = pd.read_sql_query("SELECT * FROM "+asset+tOt+"DISP",connDD).set_index("real_index")
    connDD.close()
    #print(separators)
    separators = extractFromSeparators_v20(separators, data.dateStart, data.dateEnd, asset, DBforex)
    
    seq_len = model.seq_len
    
    totalSamps = 0
    nSampsPerLevel = np.zeros((model.size_output_layer))
    nF = model.nFeatures
    
    resolution = int(data.nEventsPerStat/data.movingWindow)
    
    if autoFeatures:

        conn = sqlite3.connect(DBforex)
        SymbolBid = pd.read_sql_query("SELECT SymbolBid FROM "+asset,conn)
        conn.close()
        allParameters = pickle.load( open( "../FCN/weights/"+data.FCNID[:6]+".p", "rb" ))
        if len(data.FCNID)==6:
            epoch = allParameters["numEntries"]-1
        else:
            #print(data.FCNID[6:])
            #print(int(data.FCNID[6:]))
            epoch = int(data.FCNID[6:])
        
        #print("Epoch "+str(epoch))
        thisEpoch = allParameters["epochIndexes"][epoch]
        print("Weights of epoch "+str(thisEpoch))
        weights = allParameters["epoch"+str(thisEpoch)]
        #nF = modelFCN.size_hidden_layer

    #tic = time.time()
    for s in range(0,len(separators)-1,2):#range(len(separators)-1)
        
        if separators.index[s+1]-separators.index[s]>=2*data.nEventsPerStat:
            print("Symbols batch {0:d} out of {1:d}".format(int(s/2),int(len(separators)/2-1)))
            print(separators.DateTime.iloc[s]+" to "+separators.DateTime.iloc[s+1])
            # get init and end dates of these separators
            sepInitDate = dt.datetime.strftime(dt.datetime.strptime(
                    separators.DateTime.iloc[s],'%Y.%m.%d %H:%M:%S'),'%y%m%d%H%M%S')
            sepEndDate = dt.datetime.strftime(dt.datetime.strptime(
                    separators.DateTime.iloc[s+1],'%Y.%m.%d %H:%M:%S'),'%y%m%d%H%M%S')
    
            # Load features from DB
            features, tableRets, DTs = extractFeaturesFromDB_v31(asset, sepInitDate, sepEndDate, data, DBfeatures)
            tableRets,DTs = extractResultsFromDB_v31(asset, sepInitDate, sepEndDate, data, DBresults)
            Returns = tableRets[str(data.lookAheadIndex+1)].as_matrix()

            variation = features[resolution:,:]-features[0:-resolution,:]
            nVF = np.intersect1d(data.noVarFeats,data.features)
            if nVF.shape[0]>0:
                variation[:,nVF] = features[:-resolution,nVF]
                varNormed = np.minimum(np.maximum((variation-means)/stds,-10),10)
            else:
                variation = np.zeros([features.shape[0]-resolution,0])
                varNormed = np.zeros([features.shape[0]-resolution,0])
                #print(varNormedFixed.shape)
            
            if autoFeatures:
                meanAuto = float(stats.as_matrix().T[0,0])
                stdAuto = float(stats.as_matrix().T[1,0])
                varNormedAuto = extractFeaturesFromFCN_v00(modelFCN, data, SymbolBid, weights, separators,
                                           meanAuto, stdAuto, s, aloc = aloc).T
            else:
                varNormedAuto = np.zeros([features.shape[0],0])
                
            nSamps = varNormed.shape[0]
            m = nSamps-seq_len
            epochs = int(np.ceil(m/aloc))
            #print("epochs="+str(epochs))
            if logFile!="":
                file = open(logFile,"a")
                file.write("epochs="+str(epochs)+"\n")
                file.close()
                
            offset = 0

            for e in range(epochs):
                
                batch = np.min([m,aloc])
                #print("batch="+str(batch))
                if logFile!="":
                    file = open(logFile,"a")
                    file.write("batch="+str(batch)+"\n")
                    file.close()
            
                X_i = np.zeros((batch, seq_len, nF))
                Output_i = np.zeros((batch, seq_len, 1))
                
                columns = ["DTe","DT"+str(data.lookAheadIndex),"Be","B"+str(data.lookAheadIndex),
                           "Ae","A"+str(data.lookAheadIndex)]
                
                DTA_i = pd.DataFrame()
                
                if saveDTA:
                    appendIndexes = np.array([]).astype(int)
                    for nI in range(batch):
                        X_i[nI,:,:] = varNormed[offset+nI:offset+nI+seq_len, :]
                        # due to substraction of features for variation, output gets the 
                        # feature one entry later
                        Output_i[nI,:,0] = Returns[resolution+offset+nI:resolution+offset+nI+seq_len]
                        appendIndexes = np.append(appendIndexes,[i for i in range(resolution+offset+nI,resolution+offset+nI+seq_len)])
                        #DTA_i = DTA_i.append(DTs[columns].iloc[1+offset+nI:1+offset+nI+seq_len])
                    
                    DTA_i = DTs[columns].iloc[appendIndexes]
                    #print(DTA_i)
                    #print(DTA2_i)
                    DTA_i.loc[:,"Asset"] = asset
                    DTA_i.columns = ["DT1","DT2","B1","B2","A1","A2","Asset"]
                else:
                    for nI in range(batch):
                        X_i[nI,:,:] = varNormed[offset+nI:offset+nI+seq_len, :]
                        Output_i[nI,:,0] = Returns[resolution+offset+nI:resolution+offset+nI+seq_len]

                Output_i = Output_i/stdO
                
                offset = offset+batch
                totalSamps = totalSamps+batch
                
                y = np.zeros((Output_i.shape))
                y = np.minimum(np.maximum(np.sign(Output_i)*np.round(abs(Output_i)*model.outputGain),-
                 (model.size_output_layer-1)/2),(model.size_output_layer-1)/2)+int((model.size_output_layer-1)/2)
                    
                Y_dec=y.astype(int)
                y_bin = convert_to_one_hot(Y_dec, model.size_output_layer).T.reshape(
                        batch,seq_len,model.size_output_layer)
                
                for l in range(model.size_output_layer):
                    nSampsPerLevel[l] = nSampsPerLevel[l]+np.sum(Y_dec[:,-1,0]==l)
                    
                if logFile!="":
                    file = open(logFile,"a")
                    file.write("Stats applyed to X_"+str(e)+"\n")
                    file.close()
                
                #print("X_i.T.shape")
                #print(X_i.T.shape)
                X = np.append(X,X_i,axis=0)
                Y = np.append(Y,y_bin,axis=0)
                DTA = DTA.append(DTA_i,ignore_index=True)
                if X.shape[0]>aloc:
                    tempX = X
                    tempY = Y
                    tempDT = DTA
                    X = X[:aloc,:,:]
                    Y = Y[:aloc,:,:]
                    DTA = DTA.iloc[:aloc]
                    pickle.dump( X, open( "../RNN/IO/X"+str(filesCounter)+"_"+ID+tOt+".p", "wb" ))
                    pickle.dump( Y, open( "../RNN/IO/Y"+str(filesCounter)+"_"+ID+tOt+".p", "wb" ))
                    pickle.dump( DTA, open( "../RNN/IO/DTA"+str(filesCounter)+"_"+ID+tOt+".p", "wb" ))
                    X = tempX[aloc:,:,:]
                    Y = tempY[aloc:,:,:]
                    DTA = tempDT.iloc[aloc:]
                    print("IO "+str(filesCounter)+" saved")
                    #print(X.shape)
                    filesCounter = filesCounter+1
                    
                m = m-batch
        else:
            print("Skyping symbols batch {0:d}. Not enough entries.".format(int(s/2)))
    
    return X,Y,DTA,filesCounter,totalSamps,nSampsPerLevel

def save_for_assertion(var, var_name):
    """
    Function that saves in disk a variable neccesary to assert some function
    Args:
        - var: variable to save
        - var_name: string containing variable name
    """
    directory = "../Assertion/"
    # create directory if does not exist
    if not os.path.exists(directory):
        os.mkdir(directory)
        
    pickle.dump( var, open(directory+var_name+".p", "wb" ))
    
    print("var "+var_name+" saved for assertion")
    
    return None

def build_output_v11(model, Output, batch_size):
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

def buildIORNN_v11(model, data, asset, trainOrTest, DBforex, DBDD, DBfeatures, DBresults, DBstats,
                           X, Y, DTA, filesCounter, aloc = 100000,logFile="",ID="",
                           saveDTA=False,autoFeatures=0,modelFCN=None):
    """
    Version 11 of buildIORNN. It builds the Input and output to the RNN, taking 
    as input the features extracted from extractFeaturesFromDB_v31, the outputs 
    extracted from extractResultsFromDB_v31, and the normalization stats from 
    loadStatsFromDB_v32. This function is the first one compatible with different
    number of channels in the inputs.
    """
    print("\nLoading "+ asset)
    if logFile!="":
        file = open(logFile,"a")
        file.write("\nLoading "+ asset+"\n")
        file.close()
    
    if trainOrTest==data.train:
        tOt = "tr"
    else:
        tOt = "te"
    
    maxNchannels = int(data.nEventsPerStat/data.movingWindow)
    means = np.zeros((maxNchannels,data.nFeatures))
    stds = np.zeros((maxNchannels,data.nFeatures))

    stats = loadStatsFromDB_v32(DBstats, data, asset)
    #print(stats)
    
    for r in range(maxNchannels):
        means[r,:] = stats.as_matrix().T[0,r*data.nFeatures:(r+1)*data.nFeatures]
        stds[r,:] = stats.as_matrix().T[1,r*data.nFeatures:(r+1)*data.nFeatures]
    #print(means)
    stdO = float(stats.as_matrix().T[1,-1])
    #print(stdO-stdO_v31)
    #print("stdO")
    #print(stdO)
    connDD = sqlite3.connect(DBDD)
    separators = pd.read_sql_query("SELECT * FROM "+asset+tOt+"DISP",connDD).set_index("real_index")
    connDD.close()
    separators = extractFromSeparators_v20(separators, data.dateStart, data.dateEnd, asset, DBforex)
    
    seq_len = model.seq_len
    
    totalSamps = 0
    nSampsPerLevel = np.zeros((model.size_output_layer))
    nF = model.nFeatures
    
    if autoFeatures:

        conn = sqlite3.connect(DBforex)
        SymbolBid = pd.read_sql_query("SELECT SymbolBid FROM "+asset,conn)
        conn.close()
        allParameters = pickle.load( open( "../FCN/weights/"+data.FCNID[:6]+".p", "rb" ))
        if len(data.FCNID)==6:
            epoch = allParameters["numEntries"]-1
        else:
            epoch = int(data.FCNID[6:])
        
        thisEpoch = allParameters["epochIndexes"][epoch]
        print("Weights of epoch "+str(thisEpoch))
        weights = allParameters["epoch"+str(thisEpoch)]

    for s in range(0,len(separators)-1,2):#range(len(separators)-1)
        
        if separators.index[s+1]-separators.index[s]>=2*data.nEventsPerStat:
            print("Symbols batch {0:d} out of {1:d}".format(int(s/2),int(len(separators)/2-1)))
            print(separators.DateTime.iloc[s]+" to "+separators.DateTime.iloc[s+1])
            # get init and end dates of these separators
            sepInitDate = dt.datetime.strftime(dt.datetime.strptime(
                    separators.DateTime.iloc[s],'%Y.%m.%d %H:%M:%S'),'%y%m%d%H%M%S')
            sepEndDate = dt.datetime.strftime(dt.datetime.strptime(
                    separators.DateTime.iloc[s+1],'%Y.%m.%d %H:%M:%S'),'%y%m%d%H%M%S')
    
            # Load features from DB
            features, _, _ = extractFeaturesFromDB_v31(asset, sepInitDate, sepEndDate, data, DBfeatures)
            #print("features shape "+str(features.shape))
            
            tableRets,DTs = extractResultsFromDB_v31(asset, sepInitDate, sepEndDate, data, DBresults)
            Returns = tableRets[str(data.lookAheadIndex+1)].as_matrix()
            
            variation_v32 = np.zeros((features.shape[0],data.nFeatures,maxNchannels))
            varNormed = 999+np.zeros((features.shape[0],data.nFeatures,maxNchannels))
            nVF = np.intersect1d(data.noVarFeats,data.features)
            for r in range(maxNchannels):
                variation_v32[r+1:,:,r] = features[r+1:,:]-features[:-(r+1),:]
                #if r!=0:
                variation_v32[r+1:,data.noVarFeats,r] = features[:-(r+1),data.noVarFeats]
                varNormed[r+1:,:,r] = np.minimum(np.maximum((variation_v32[r+1:,:,r]-means[r,:])/stds[r,:],-10),10)
#            print(means[0,:])
#            print(stds[0,:])
#            save_as_matfile('variation','variation',variation_v32)
#            save_as_matfile('means','means',means)
#            save_as_matfile('stds','stds',stds)
            nonremoveEntries = varNormed[:,0,-1]!=999
            varNormed = varNormed[nonremoveEntries,:,:]
            variation = features[maxNchannels:,:]-features[0:-maxNchannels,:]

            
            if nVF.shape[0]>0:
                variation[:,nVF] = features[:-maxNchannels,nVF]
                varNormedFixed = np.minimum(np.maximum((variation-means[0,:])/stds[0,:],-10),10)
            else:
                variation = np.zeros([features.shape[0]-1,0])
                varNormedFixed = np.zeros([features.shape[0]-1,0])

            if autoFeatures:
                meanAuto = float(stats.as_matrix().T[0,0])
                stdAuto = float(stats.as_matrix().T[1,0])
                varNormedAuto = extractFeaturesFromFCN_v00(modelFCN, data, SymbolBid, weights, separators,
                                           meanAuto, stdAuto, s, aloc = aloc).T
            else:
                varNormedAuto = np.zeros([features.shape[0],0])
            
            #varNormed = np.append(varNormedFixed,varNormedAuto[:-resolution,:],axis=1)

            nSamps = varNormed.shape[0]
            m = nSamps-seq_len
            chuncks = int(np.ceil(m/aloc))
            #print("epochs="+str(epochs))
            if logFile!="":
                file = open(logFile,"a")
                file.write("epochs="+str(chuncks)+"\n")
                file.close()
                
            offset = 0

            for e in range(chuncks):
                
                batch = np.min([m,aloc])
                print(batch)
                #print("batch="+str(batch))
                if logFile!="":
                    file = open(logFile,"a")
                    file.write("batch="+str(batch)+"\n")
                    file.close()
            
                X_i = np.zeros((batch, seq_len, nF))
                #print(X_i.shape)
                Output_i = np.zeros((batch, seq_len, 1))
                
                columns = ["DTe","DT"+str(data.lookAheadIndex),"Be","B"+str(data.lookAheadIndex),
                           "Ae","A"+str(data.lookAheadIndex)]
                
                DTA_i = pd.DataFrame()
                #rs = [maxNchannels-1]#range(resolution)
                
                if saveDTA:
                    appendIndexes = np.array([]).astype(int)
                    for nI in range(batch):
                        countRs = 0
                        for r in data.channels:#range(resolution)
                            X_i[nI,:,countRs*data.nFeatures:(countRs+1)*data.nFeatures] = varNormed[offset+nI:offset+nI+seq_len, :,r]#
                            countRs += 1
                        # due to substraction of features for variation, output gets the 
                        # feature one entry later
                        Output_i[nI,:,0] = Returns[maxNchannels+offset+nI:maxNchannels+offset+nI+seq_len]
                        appendIndexes = np.append(appendIndexes,[i for i in range(maxNchannels+offset+nI,maxNchannels+offset+nI+seq_len)])
                        #DTA_i = DTA_i.append(DTs[columns].iloc[1+offset+nI:1+offset+nI+seq_len])
                    
                    DTA_i = DTs[columns].iloc[appendIndexes]
                    DTA_i.loc[:,"Asset"] = asset
                    DTA_i.columns = ["DT1","DT2","B1","B2","A1","A2","Asset"]
                else:
                    for nI in range(batch):
                        countRs = 0
                        for r in data.channels:
                            X_i[nI,:,countRs*data.nFeatures:(countRs+1)*data.nFeatures] = varNormed[offset+nI:offset+nI+seq_len, :,r]
                            countRs += 1
                            
                        Output_i[nI,:,0] = Returns[maxNchannels+offset+nI:maxNchannels+offset+nI+seq_len]
#                save_as_matfile("X","X",X_i)
#                save_as_matfile("varNormed","varNormed",varNormed)
#                save_as_matfile("Output_un","Output_un",Output_i)
#                save_as_matfile("features"+str(s),"features"+str(s),features)
                Output_i = Output_i/stdO
                save_as_matfile('X'+str(int(s/2)),'X'+str(int(s/2)),X_i)
                save_as_matfile('O'+str(int(s/2)),'O'+str(int(s/2)),Output_i)
#                print(stdO)
#                save_as_matfile("Output","Output",Output_i)
#                if s>=0:
#                    a=p
#                save_for_assertion(Output_i, "Output")
#                a=p
                
                offset = offset+batch
                totalSamps = totalSamps+batch
                
                
                # get decimal and binary outputs
                y_bin, y_dec = build_output_v11(model,Output_i,batch)
                
                for l in range(model.size_output_layer):
                    nSampsPerLevel[l] = nSampsPerLevel[l]+np.sum(y_dec[:,-1,0]==l)
                    
                if logFile!="":
                    file = open(logFile,"a")
                    file.write("Stats applyed to X_"+str(e)+"\n")
                    file.close()
                
                X = np.append(X,X_i,axis=0)
                #save_as_matfile("X","X",X)
                #save_as_matfile("O","O",Output_i)
                Y = np.append(Y,y_bin,axis=0)
                DTA = DTA.append(DTA_i,ignore_index=True)
                if X.shape[0]>aloc:
                    tempX = X
                    tempY = Y
                    tempDT = DTA
                    X = X[:aloc,:,:]
                    Y = Y[:aloc,:,:]
                    DTA = DTA.iloc[:aloc]
                    pickle.dump( X, open( "../RNN/IO/X"+str(filesCounter)+"_"+ID+tOt+".p", "wb" ))
                    pickle.dump( Y, open( "../RNN/IO/Y"+str(filesCounter)+"_"+ID+tOt+".p", "wb" ))
                    pickle.dump( DTA, open( "../RNN/IO/DTA"+str(filesCounter)+"_"+ID+tOt+".p", "wb" ))
                    X = tempX[aloc:,:,:]
                    Y = tempY[aloc:,:,:]
                    DTA = tempDT.iloc[aloc:]
                    print("IO "+str(filesCounter)+" saved")
                    #print(X.shape)
                    filesCounter = filesCounter+1
                    
                m = m-batch
        else:
            print("Skyping symbols batch {0:d}. Not enough entries.".format(int(s/2)))
    
    return X,Y,DTA,filesCounter,totalSamps,nSampsPerLevel

def buildIORNN_v10(model, data, asset, trainOrTest, DBforex, DBDD, DBfeatures, DBresults,
                           X, Y, DTA, filesCounter, aloc = 100000,logFile="",ID="",
                           saveDTA=False,newReturns=0,autoFeatures=0,modelFCN=None):
    
    print("\nLoading "+ asset)
    if logFile!="":
        file = open(logFile,"a")
        file.write("\nLoading "+ asset+"\n")
        file.close()
    
    if trainOrTest==data.train:
        tOt = "tr"
    else:
        tOt = "te"
        
    #print(separators)
    
    means = np.zeros((1,data.nFeatures))
    stds = np.zeros((1,data.nFeatures))
    
    if newReturns==0:
        stats = loadStatsFromDB_v27(DBfeatures, data, asset, trainOrTest)
    else:
        stats = loadStatsFromDB_v31(DBfeatures, data, asset, trainOrTest)
        
    means[0,:] = stats.as_matrix().T[0,0:data.nFeatures]
    stds[0,:] = stats.as_matrix().T[1,0:data.nFeatures]
    stdO = float(stats.as_matrix().T[1,-1])
    #print("stdO")
    #print(stdO)
    connDD = sqlite3.connect(DBDD)
    separators = pd.read_sql_query("SELECT * FROM "+asset+tOt+"DISP",connDD).set_index("real_index")
    connDD.close()
    #print(separators)
    separators = extractFromSeparators_v20(separators, data.dateStart, data.dateEnd, asset, DBforex)
    
    seq_len = model.seq_len
    
    totalSamps = 0
    nSampsPerLevel = np.zeros((model.size_output_layer))
    nF = model.nFeatures
    
    if autoFeatures:

        conn = sqlite3.connect(DBforex)
        SymbolBid = pd.read_sql_query("SELECT SymbolBid FROM "+asset,conn)
        conn.close()
        allParameters = pickle.load( open( "../FCN/weights/"+data.FCNID[:6]+".p", "rb" ))
        if len(data.FCNID)==6:
            epoch = allParameters["numEntries"]-1
        else:
            #print(data.FCNID[6:])
            #print(int(data.FCNID[6:]))
            epoch = int(data.FCNID[6:])
        
        #print("Epoch "+str(epoch))
        thisEpoch = allParameters["epochIndexes"][epoch]
        print("Weights of epoch "+str(thisEpoch))
        weights = allParameters["epoch"+str(thisEpoch)]
        #nF = modelFCN.size_hidden_layer

    #tic = time.time()
    for s in range(0,len(separators)-1,2):#range(len(separators)-1)
        
        if separators.index[s+1]-separators.index[s]>=2*data.nEventsPerStat:
            print("Symbols batch {0:d} out of {1:d}".format(int(s/2),int(len(separators)/2-1)))
            print(separators.DateTime.iloc[s]+" to "+separators.DateTime.iloc[s+1])
            # get init and end dates of these separators
            sepInitDate = dt.datetime.strftime(dt.datetime.strptime(
                    separators.DateTime.iloc[s],'%Y.%m.%d %H:%M:%S'),'%y%m%d%H%M%S')
            sepEndDate = dt.datetime.strftime(dt.datetime.strptime(
                    separators.DateTime.iloc[s+1],'%Y.%m.%d %H:%M:%S'),'%y%m%d%H%M%S')
    
            # Load features from DB
            features, tableRets, DTs = extractFeaturesFromDB_v31(asset, sepInitDate, sepEndDate, data, DBfeatures)
            #print("features shape "+str(features.shape))
            
            if newReturns:
                tableRets,DTs = extractResultsFromDB_v31(asset, sepInitDate, sepEndDate, data, DBresults)
                #print(tableRets)
                #print(DTs)
                Returns = tableRets[str(data.lookAheadIndex+1)].as_matrix()
            else:
                Returns = tableRets[str(data.divideLookAhead)].as_matrix()
            #save_as_matfile("features"+str(int(s/2)),"features"+str(int(s/2)),features)
            #a=p
            #print(features.shape)
            variation = features[1:features.shape[0],:]-features[0:features.shape[0]-1,:]
            nVF = np.intersect1d(data.noVarFeats,data.features)
            if nVF.shape[0]>0:
                variation[:,nVF] = features[:-1,nVF]
                varNormedFixed = np.minimum(np.maximum((variation-means)/stds,-10),10)
            else:
                variation = np.zeros([features.shape[0]-1,0])
                varNormedFixed = np.zeros([features.shape[0]-1,0])
                #print(varNormedFixed.shape)
            
            if autoFeatures:
                meanAuto = float(stats.as_matrix().T[0,0])
                stdAuto = float(stats.as_matrix().T[1,0])
                varNormedAuto = extractFeaturesFromFCN_v00(modelFCN, data, SymbolBid, weights, separators,
                                           meanAuto, stdAuto, s, aloc = aloc).T
            else:
                varNormedAuto = np.zeros([features.shape[0],0])
                #print("varNormed shape"+str(varNormed.shape))
                #print("varNormedAuto shape"+str(varNormedAuto[:-1,:].shape))
            #save_as_matfile("varNormed"+str(int(s/2)),"varNormed"+str(int(s/2)),varNormed)
            varNormed = np.append(varNormedFixed,varNormedAuto[:-1,:],axis=1)
            #print(varNormed.shape)
            nSamps = varNormed.shape[0]
            #print("Returns shape "+str(Returns.shape))
            #print("Variation shape"+str(variation.shape))
            #m_i = np.floor(nSamps-shiftSampsInOut)
            #m_i = m_i.astype(int)
            
            #m = m_i
            m = nSamps-seq_len
            epochs = int(np.ceil(m/aloc))
            #print("epochs="+str(epochs))
            if logFile!="":
                file = open(logFile,"a")
                file.write("epochs="+str(epochs)+"\n")
                file.close()
                
            offset = 0

            for e in range(epochs):
                
                batch = np.min([m,aloc])
                #print("batch="+str(batch))
                if logFile!="":
                    file = open(logFile,"a")
                    file.write("batch="+str(batch)+"\n")
                    file.close()
            
                X_i = np.zeros((batch, seq_len, nF))
                Output_i = np.zeros((batch, seq_len, 1))
                
                if newReturns:
                    columns = ["DTe","DT"+str(data.lookAheadIndex),"Be","B"+str(data.lookAheadIndex),
                           "Ae","A"+str(data.lookAheadIndex)]
                else:
                    columns = ["DT1","DT"+str(data.divideLookAhead+1),"B1","B"+str(data.divideLookAhead+1),
                           "A1","A"+str(data.divideLookAhead+1)]
                DTA_i = pd.DataFrame()
                
                #print(Returns.shape)
                #a=p
                
                if saveDTA:
                    appendIndexes = np.array([]).astype(int)
                    for nI in range(batch):
                        X_i[nI,:,:] = varNormed[offset+nI:offset+nI+seq_len, :]
                        # due to substraction of features for variation, output gets the 
                        # feature one entry later
                        Output_i[nI,:,0] = Returns[1+offset+nI:1+offset+nI+seq_len]
                        appendIndexes = np.append(appendIndexes,[i for i in range(1+offset+nI,1+offset+nI+seq_len)])
                        #DTA_i = DTA_i.append(DTs[columns].iloc[1+offset+nI:1+offset+nI+seq_len])
                    
                    DTA_i = DTs[columns].iloc[appendIndexes]
                    #print(DTA_i)
                    #print(DTA2_i)
                    DTA_i.loc[:,"Asset"] = asset
                    DTA_i.columns = ["DT1","DT2","B1","B2","A1","A2","Asset"]
                else:
                    for nI in range(batch):
                        X_i[nI,:,:] = varNormed[offset+nI:offset+nI+seq_len, :]
                        Output_i[nI,:,0] = Returns[1+offset+nI:1+offset+nI+seq_len]
                
                #print("last index varNormed in X_i: "+str(offset+nI+seq_len-1))
                #save_as_matfile(asset+"varNormed"+str(s)+"_"+str(newReturns),"varNormed"+str(s)+"_"+str(newReturns),varNormed)
                
                Output_i = Output_i/stdO
                #save_as_matfile(asset+"X_"+str(s)+"_"+str(newReturns),
                #                "X_"+str(s)+"_"+str(newReturns),X_i)
                #save_as_matfile(asset+"Output_"+str(s)+"_"+str(newReturns),
                #                "Output_"+str(s)+"_"+str(newReturns),Output_i)
                
#                a=p
                #Output_i = variation[data.lookBack+shift:data.lookBack+shift+batch,data.nFeatures]
                
                #print(DTs)
                #DTA_i = DTs[columns].iloc[offset:offset+batch]#+data.lookAhead
                #DTA_i.columns=["DT1","DT2","B1","B2","A1","A2"]
                
                
                #print(DTA_i)
                #print(Output_i.shape)
                #print(DTA_i.shape)
                
                offset = offset+batch
                totalSamps = totalSamps+batch
                
                y = np.zeros((Output_i.shape))
                y = np.minimum(np.maximum(np.sign(Output_i)*np.round(abs(Output_i)*model.outputGain),-
                 (model.size_output_layer-1)/2),(model.size_output_layer-1)/2)+int((model.size_output_layer-1)/2)
                    
                Y_dec=y.astype(int)
                y_bin = convert_to_one_hot(Y_dec, model.size_output_layer).T.reshape(
                        batch,seq_len,model.size_output_layer)
                
                for l in range(model.size_output_layer):
                    nSampsPerLevel[l] = nSampsPerLevel[l]+np.sum(Y_dec[:,-1,0]==l)
                    
                if logFile!="":
                    file = open(logFile,"a")
                    file.write("Stats applyed to X_"+str(e)+"\n")
                    file.close()
                
                #print("X_i.T.shape")
                #print(X_i.T.shape)
                X = np.append(X,X_i,axis=0)
                Y = np.append(Y,y_bin,axis=0)
                DTA = DTA.append(DTA_i,ignore_index=True)
                if X.shape[0]>aloc:
                    tempX = X
                    tempY = Y
                    tempDT = DTA
                    X = X[:aloc,:,:]
                    Y = Y[:aloc,:,:]
                    DTA = DTA.iloc[:aloc]
                    pickle.dump( X, open( "../RNN/IO/X"+str(filesCounter)+"_"+ID+tOt+".p", "wb" ))
                    pickle.dump( Y, open( "../RNN/IO/Y"+str(filesCounter)+"_"+ID+tOt+".p", "wb" ))
                    pickle.dump( DTA, open( "../RNN/IO/DTA"+str(filesCounter)+"_"+ID+tOt+".p", "wb" ))
                    X = tempX[aloc:,:,:]
                    Y = tempY[aloc:,:,:]
                    DTA = tempDT.iloc[aloc:]
                    print("IO "+str(filesCounter)+" saved")
                    #print(X.shape)
                    filesCounter = filesCounter+1
                    
                m = m-batch
        else:
            print("Skyping symbols batch {0:d}. Not enough entries.".format(int(s/2)))
    
    return X,Y,DTA,filesCounter,totalSamps,nSampsPerLevel

def buildIOFCN_v00(model, data, asset, trainOrTest, DBforex, DBDD, DBfeatures,
                           X, Y, filesCounter, IDweights, aloc = 100000,ID=""):
    
    print("\nLoading "+ asset)
    
    if trainOrTest==data.train:
        tOt = "tr"
    else:
        tOt = "te"

    conn = sqlite3.connect(DBforex)
    SymbolBid = pd.read_sql_query("SELECT SymbolBid FROM "+asset,conn)
    conn.close()
    
    stats = loadStatsFromDB_v31(DBfeatures, data, asset, trainOrTest)
    
    stdO = float(stats.as_matrix().T[1,-1])
    mean = stats.as_matrix().T[0,0]
    std = stats.as_matrix().T[1,0]

    connDD = sqlite3.connect(DBDD)
    separators = pd.read_sql_query("SELECT * FROM "+asset+tOt+"DISP",connDD).set_index("real_index")
    connDD.close()
    #print(separators)
    separators = extractFromSeparators_v20(separators, data.dateStart, data.dateEnd, asset, DBforex)
    
    totalSamps = 0
    nSampsPerLevel = np.zeros((model.size_output_layer))
    nE = data.nEventsPerStat
    mW = data.movingWindow
    #tic = time.time()
    for s in range(0,len(separators)-1,2):#range(len(separators)-1)
        
        if separators.index[s+1]-separators.index[s]>=2*data.nEventsPerStat:
            print("Symbols batch {0:d} out of {1:d}".format(int(s/2),int(len(separators)/2-1)))
            print(separators.DateTime.iloc[s]+" to "+separators.DateTime.iloc[s+1])

            bids = SymbolBid.loc[separators.index[s]:separators.index[s+1]+1].as_matrix()
            nEvents = bids.shape[0]
            nEintoMovW = nEvents/data.movingWindow
            allSamps = np.floor(nEintoMovW)-np.ceil(data.nEventsPerStat/data.movingWindow)
            allSamps = allSamps.astype(int)
            initRange = int(data.nEventsPerStat/data.movingWindow)
            nSamps = allSamps-initRange
            #print("nSamps "+str(nSamps))
            L = bids.shape[0]
            m = nSamps
            #m = int(np.floor(L/nE-1)*nE/mW+1)-int(mW/mW) # substract the lost sample due to difference calculations
            #print("m "+str(m))
            chunks = int(np.ceil(m/aloc))
            offset = 0

            for e in range(chunks):
                
                batch = np.min([m,aloc])
                X_i = np.zeros((batch, model.size_init_layer))

                for i in range(model.size_init_layer):
                    diffBids = bids[mW+offset+i:mW+offset+mW*m+i:mW,0]-bids[offset+i:offset+mW*m+i:mW,0]
                    X_i[:,i] = diffBids

                Output_i = bids[mW+offset+2*nE-2:mW+offset+nE-2+mW*m:mW,0]-bids[mW+offset+nE-1:mW+offset+mW*m-1:mW,0]

                X_i = (X_i[:-int(nE/mW),:]-mean)/std # normalize and remove last entry to much with ouptup
                Output_i = Output_i/stdO
                
                offset = offset+batch
                totalSamps = totalSamps+batch
                
                #y = np.zeros((Output_i.shape))
                y = np.minimum(np.maximum(np.sign(Output_i)*np.round(abs(Output_i)*model.outputGain),-
                 (model.size_output_layer-1)/2),(model.size_output_layer-1)/2)+int((model.size_output_layer-1)/2)
                    
                Y_dec=y.astype(int)
                y_bin = convert_to_one_hot(Y_dec, model.size_output_layer)
                
                # extract features
                
#                allParameters = pickle.load( open( "../FCN/weights/"+IDweights+".p", "rb" ))
#                epoch = allParameters["numEntries"]-1
#                #epoch = 0
#                print("Epoch "+str(epoch))
#                thisEpoch = allParameters["epochIndexes"][epoch]
#                weights = allParameters["epoch"+str(thisEpoch)]
#                Z = getOutputsLayersFCN(model, X_i.T, weights, epoch)
#                save_as_matfile("X","X",X_i)
#                save_as_matfile("Z","Z",Z)
#                a=p
                #print("Y_dec shape "+str(Y_dec.shape))
                for l in range(model.size_output_layer):
                    nSampsPerLevel[l] = nSampsPerLevel[l]+np.sum(Y_dec==l)

                X = np.append(X,X_i.T,axis=1)
                Y = np.append(Y,y_bin,axis=1)
                if X.shape[1]>aloc:
                    tempX = X
                    tempY = Y
                    X = X[:,:aloc]
                    Y = Y[:,:aloc]
                    #DTA = DTA.iloc[:aloc]
                    pickle.dump( X, open( "../FCN/IO/X"+str(filesCounter)+"_"+ID+tOt+".p", "wb" ))
                    pickle.dump( Y, open( "../FCN/IO/Y"+str(filesCounter)+"_"+ID+tOt+".p", "wb" ))
                    X = tempX[:,aloc:]
                    Y = tempY[:,aloc:]
                    print("IOs "+str(filesCounter)+"_"+ID+tOt+".p"+" saved")
                    filesCounter = filesCounter+1
                    
                m = m-batch
        else:
            print("Skyping symbols batch {0:d}. Not enough entries.".format(int(s/2)))
    
    return X,Y,filesCounter,totalSamps,nSampsPerLevel

def initFeaturesLive_11(data,tradeInfoLive):
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

def extractFeaturesLive_11(tradeInfoLive, data, featuresLive,parSarStruct,em):
    
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

def initFeaturesLive_00(data,tradeInfoLive):
    class parSarInit:
    
        periodSAR = data.nEventsPerStat
        HP = 0
        LP = 100000
        stepAF = 0.02
        AFH = stepAF
        AFL = stepAF
        maxAF = 20*stepAF   
    
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
    
    return featuresLive,parSarStruct,em

def extractFeaturesLive_00(tradeInfoLive, data, featuresLive,parSarStruct,em):
    
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
        parSarStruct.HP = np.max([np.max(tradeInfoLive.SymbolBid.iloc[:]),parSarStruct.HP])
        parSarStruct.LP = np.min([np.min(tradeInfoLive.SymbolBid.iloc[:]),parSarStruct.LP])
        featuresLive[10,0] = featuresLive[10,0]+parSarStruct.AFH*(parSarStruct.HP-featuresLive[10,0]) #parSar high
        featuresLive[11,0] = featuresLive[11,0]-parSarStruct.AFL*(featuresLive[11,0]-parSarStruct.LP) # parSar low
        if featuresLive[10,0]<parSarStruct.HP:
            parSarStruct.AFH = np.min([parSarStruct.AFH+parSarStruct.stepAF,parSarStruct.maxAF])
            parSarStruct.LP = np.min(tradeInfoLive.SymbolBid.iloc[:])
        if featuresLive[11,0]>parSarStruct.LP:
            parSarStruct.AFL = np.min([parSarStruct.AFH+parSarStruct.stepAF,parSarStruct.maxAF])
            parSarStruct.HP = np.max(tradeInfoLive.SymbolBid.iloc[:])
    
    
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
    
    
    return featuresLive,parSarStruct,em

def extractFeatures_v40(tradeInfo, data):
    
    tic = time.time()
    
    #average_over = np.array([0.1,0.2,0.3,.4,.5,.6,.7,.8,.9,1,1.5,2,2.5,3,4,5,6,7,8,9,10,20,30,40,50,60,70,80,90,100])
    #lbd=1-1/(data.nEventsPerStat*average_over)
    nEMAs = data.lbd.shape[0]
    
    nExS = data.nEventsPerStat
    mW = data.movingWindow
    nE = tradeInfo.shape[0]
    m = int(np.floor(nE/nExS-1)*nExS/mW+1)
    
    EMA = np.zeros((m,nEMAs))
    bids = np.zeros((m))
    variance = np.zeros((m))
    maxValue = np.zeros((m))
    minValue = np.zeros((m))
    timeInterval = np.zeros((m))
    timeSecs = np.zeros((m))
    #Return_old = np.zeros((nSamps))
    features = np.zeros((m,data.nFeatures))
    allBids = tradeInfo.SymbolBid
    
    secsInDay = 86400.0
    
    # init exponetial means
    em = np.zeros((data.lbd.shape))+allBids.iloc[0]
    for i in range(nExS-mW):
        em = data.lbd*em+(1-data.lbd)*allBids.iloc[i]
    #/(1-np.maximum(data.lbd**i,1e-3))
    
    parSARhigh20 = np.zeros((m))
    parSARhigh2 = np.zeros((m))
    oldSARh20 = allBids.iloc[0]
    oldSARh2 = allBids.iloc[0]
    parSARlow20 = np.zeros((m))
    parSARlow2 = np.zeros((m))
    oldSARl20 = allBids.iloc[0]
    oldSARl2 = allBids.iloc[0]
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

    for mm in range(m):
        
        endIndex = mm*mW+nExS
        startIndex = mm*mW
        thisPeriod = range(startIndex,endIndex)
        thisPeriodBids = allBids.iloc[thisPeriod]
        
        newBidsIndex = range(endIndex-mW,endIndex)
        for i in newBidsIndex:
            #a=data.lbd*em/(1-data.lbd**i)+(1-data.lbd)*tradeInfo.SymbolBid.loc[i]
            em = data.lbd*em+(1-data.lbd)*allBids.iloc[i]

        if mm%5000==0:
            toc = time.time()
            print("mm="+str(mm)+" of "+str(m)+". Total time: "+str(np.floor(toc-tic))+"s")
            
        t0 = dt.datetime.strptime(tradeInfo.DateTime.iloc[thisPeriod[0]],'%Y.%m.%d %H:%M:%S')
        te = dt.datetime.strptime(tradeInfo.DateTime.iloc[thisPeriod[-1]],'%Y.%m.%d %H:%M:%S')
        
        maxValue[mm] = np.max(thisPeriodBids)
        minValue[mm] = np.min(thisPeriodBids)
        
        if 10 in data.features:
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
            
        if 13 in data.features:
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
        
        bids[mm] = tradeInfo.SymbolBid.loc[thisPeriod[len(thisPeriod)-1]]
        EMA[mm,:] = em
        variance[mm] = np.var(thisPeriodBids)
        timeInterval[mm] = (te-t0).seconds/data.nEventsPerStat
        timeSecs[mm] = (te.hour*60*60+te.minute*60+te.second)/secsInDay
        

    #print(dateTime)
    nF = 0
    if 0 in data.features:
        features[:,nF] = bids
    
    for i in range(data.lbd.shape[0]):
        #nF = nF+1
        if nF+1 in data.features:            
            nF += 1        
            features[:,nF] = EMA[:,i]
        
    if 8 in data.features:
        nF+=1
        logVar = 10*np.log10(variance/data.std_var+1e-10)
        features[:,nF] = logVar
    
    if 9 in data.features:
        nF = nF+1
        logTimeInt = 10*np.log10(timeInterval/data.std_time+0.01)
        features[:,nF] = logTimeInt
    
    if 10 in data.features:
        nF = nF+1  
        features[:,nF] = parSARhigh20
    
    if 11 in data.features:
        nF = nF+1
        features[:,nF] = parSARlow20
    
    if 12 in data.features:
        nF = nF+1
        features[:,nF] = timeSecs
    
    if 13 in data.features:
        nF = nF+1  
        features[:,nF] = parSARhigh2
    
    if 14 in data.features:
        nF = nF+1
        features[:,nF] = parSARlow2
    
    # Repeat non-variation features to inclue variation betwen first and second input
    if 15 in data.features:
        nF+=1
        logVar = 10*np.log10(variance/data.std_var+1e-10)
        features[:,nF] = logVar
    
    if 16 in data.features:
        nF = nF+1
        logTimeInt = 10*np.log10(timeInterval/data.std_time+0.01)
        features[:,nF] = logTimeInt
        
    if 17 in data.features:
        nF = nF+1
        features[:,nF] = maxValue-bids
    
    if 18 in data.features:
        nF = nF+1
        features[:,nF] = bids-minValue
    
    if 19 in data.features:
        nF = nF+1
        features[:,nF] = maxValue-bids
        
    if 20 in data.features:
        nF = nF+1
        features[:,nF] = bids-minValue
    
    if 21 in data.features:
        nF = nF+1
        features[:,nF] = minValue/maxValue
    
    if 22 in data.features:
        nF = nF+1
        features[:,nF] = minValue/maxValue
    
    for i in range(data.lbd.shape[0]):
        #nF = nF+1
        if nF+1 in data.features:            
            nF += 1        
            features[:,nF] = bids/EMA[:,i]
    
    for i in range(data.lbd.shape[0]):
        #nF = nF+1
        if nF+1 in data.features:
            nF += 1        
            features[:,nF] = bids/EMA[:,i]
    
    return features

def extractReturns_v40(tradeInfo, data):

    
    
#    nEvents = tradeInfo.shape[0]
#    nEintoMovW = nEvents/data.movingWindow
#    allSamps = np.floor(nEintoMovW)-np.ceil(data.nEventsPerStat/data.movingWindow)
#    allSamps = allSamps.astype(int)
#    initRange = int(data.nEventsPerStat/data.movingWindow)
#    nSamps = allSamps-initRange
    
    nExS = data.nEventsPerStat
    mW = data.movingWindow
    nE = tradeInfo.shape[0]
    m = int(np.floor(nE/nExS-1)*nExS/mW+1)
    initRange = int(nExS/mW)
    
    np_00 = initRange*data.movingWindow-1
    np_0e = initRange*data.movingWindow+data.nEventsPerStat-2
    np_ee = m*data.movingWindow+data.nEventsPerStat-2
    np_e0 = m*data.movingWindow-1
    
    TI = pd.DataFrame(data=[],columns=['DTe','DT0','DT1','DT2','DT3','DT4','DT5','DT6',
                       'Be','B0','B1','B2','B3','B4','B5','B6',
                       'Ae','A0','A1','A2','A3','A4','A5','A6'])
    TI.loc[:,"DTe"] = tradeInfo.DateTime.iloc[np_00:np_e0:data.movingWindow]
    TI.loc[:,"Be"] = tradeInfo.SymbolBid.iloc[np_00:np_e0:data.movingWindow].tolist()
    TI.loc[:,"Ae"] = tradeInfo.SymbolAsk.iloc[np_00:np_e0:data.movingWindow].tolist()
    Return = np.zeros((len(range(np_0e,np_ee,data.movingWindow)),len(data.lookAheadVector)))
    indexOrigins = [i for i in range(np_00,np_e0,data.movingWindow)]
    
    #origins = np.array(tradeInfo.SymbolBid.iloc[np_00:np_e0:data.movingWindow])
    
    for nr in range(len(data.lookAheadVector)):
        #print("nr")
        #print(nr)
        unp_0e = int(initRange*data.movingWindow+np.floor(data.nEventsPerStat*data.lookAheadVector[nr])-2)
        unp_ee = int(np.min([m*data.movingWindow+np.floor(
                data.nEventsPerStat*data.lookAheadVector[nr])-2,nE]))
        #ends = np.array(tradeInfo.SymbolBid.iloc[unp_0e::data.movingWindow])
        #print("indexOrigins")
        #print(indexOrigins)
        indexEnds = [i for i in range(unp_0e,unp_ee,data.movingWindow)]
        #print("indexEnds")
        #print(indexEnds)
        #fill ends wth last value
        for i in range(len(indexEnds),len(indexOrigins)):
            #print("i")
            #print(i)
            indexEnds.append(tradeInfo.shape[0]-1)
        #print(indexEnds)
        #print(tradeInfo.SymbolBid.iloc[indexEnds].as_matrix())
        #print(tradeInfo.SymbolBid.iloc[indexOrigins].as_matrix())
        Return[:,nr] = tradeInfo.SymbolBid.iloc[indexEnds].as_matrix()-tradeInfo.SymbolBid.iloc[indexOrigins].as_matrix()
        
        TI.loc[:,"DT"+str(nr)] = tradeInfo.DateTime.iloc[indexEnds].tolist()
        TI.loc[:,"B"+str(nr)] = tradeInfo.SymbolBid.iloc[indexEnds].tolist()
        TI.loc[:,"A"+str(nr)] = tradeInfo.SymbolAsk.iloc[indexEnds].tolist()
    
    
    return Return,TI

def extractFeatures_v31(tradeInfo, data):
    
    tic = time.time()
    
    #print(tradeInfo.SymbolBid)
    nEvents = tradeInfo.shape[0]
    #print(nEvents)
    nEintoMovW = nEvents/data.movingWindow
    allSamps = np.floor(nEintoMovW)-np.ceil(data.nEventsPerStat/data.movingWindow)
    allSamps = allSamps.astype(int)
    initRange = int(data.nEventsPerStat/data.movingWindow)
    nSamps = allSamps-initRange
    iterateRange = range(initRange,allSamps)
    #print(nSamps)
    EMA = np.zeros((nSamps,data.lbd.shape[0]))
    bids = np.zeros((nSamps))
    variance = np.zeros((nSamps))
    maxValue = np.zeros((nSamps))
    minValue = np.zeros((nSamps))
    timeInterval = np.zeros((nSamps))
    timeSecs = np.zeros((nSamps))
    #Return_old = np.zeros((nSamps))
    features = np.zeros((nSamps,data.nFeatures))
    allBids = tradeInfo.SymbolBid.loc
    
    TI = pd.DataFrame(data=np.zeros((nSamps,6)),columns=['DT1','DT2','B1','B2','A1','A2'])
    
    secsInDay = 86400.0
    
    em = np.zeros((data.lbd.shape))+tradeInfo.SymbolBid.loc[0+tradeInfo.SymbolBid.index[0]]
    for i in range(initRange*data.movingWindow-data.movingWindow):
        em = data.lbd*em+(1-data.lbd)*tradeInfo.SymbolBid.loc[i+tradeInfo.SymbolBid.index[0]]
    #/(1-np.maximum(data.lbd**i,1e-3))
    
    parSARhigh20 = np.zeros((nSamps))
    parSARhigh2 = np.zeros((nSamps))
    oldSARh20 = tradeInfo.SymbolBid.iloc[0]
    oldSARh2 = tradeInfo.SymbolBid.iloc[0]
    parSARlow20 = np.zeros((nSamps))
    parSARlow2 = np.zeros((nSamps))
    oldSARl20 = tradeInfo.SymbolBid.iloc[0]
    oldSARl2 = tradeInfo.SymbolBid.iloc[0]
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

    for mm in iterateRange:#range(1,allSamps-data.nEventsPerStat)
        endIndex = mm*data.movingWindow+tradeInfo.SymbolBid.index[0]#np.max([np.min([nEvents,mm*data.movingWindow]),1])
        startIndex = mm*data.movingWindow-data.nEventsPerStat+tradeInfo.SymbolBid.index[0]
        thisPeriod = range(startIndex,endIndex)
        #periodLookAhead = range(endIndex-1,endIndex+data.nEventsPerStat*data.lookAhead-1)
        thisPeriodBids = allBids[thisPeriod]
        
        newBidsIndex = range(endIndex-data.movingWindow,endIndex)
        for i in newBidsIndex:
            #a=data.lbd*em/(1-data.lbd**i)+(1-data.lbd)*tradeInfo.SymbolBid.loc[i]
            em = data.lbd*em+(1-data.lbd)*tradeInfo.SymbolBid.loc[i]

        if mm%5000==0:
            toc = time.time()
            print("mm="+str(mm)+" of "+str(iterateRange[-1])+". Total time:"+str(np.floor(toc-tic))+"s")
            
        t0 = dt.datetime.strptime(tradeInfo.DateTime.loc[thisPeriod[0]],'%Y.%m.%d %H:%M:%S')
        te = dt.datetime.strptime(tradeInfo.DateTime.loc[thisPeriod[-1]],'%Y.%m.%d %H:%M:%S')
        
        maxValue[mm-initRange] = np.max(thisPeriodBids)
        minValue[mm-initRange] = np.min(thisPeriodBids)
        
        if 10 in data.features:
            HP20 = np.max([maxValue[mm-initRange],HP20])
            LP20 = np.min([minValue[mm-initRange],LP20])
            parSARhigh20[mm-initRange] = oldSARh20+AFH20*(HP20-oldSARh20)
            parSARlow20[mm-initRange] = oldSARl20-AFL20*(oldSARl20-LP20)
            if parSARhigh20[mm-initRange]<HP20:
                AFH20 = np.min([AFH20+stepAF,maxAF20])
                LP20 = np.min(thisPeriodBids)
            if parSARlow20[mm-initRange]>LP20:
                AFL20 = np.min([AFH20+stepAF,maxAF20])
                HP20 = np.max(thisPeriodBids)
            oldSARh20 = parSARhigh20[mm-initRange]
            oldSARl20 = parSARlow20[mm-initRange]
            
        if 13 in data.features:
            HP2 = np.max([maxValue[mm-initRange],HP2])
            LP2 = np.min([minValue[mm-initRange],LP2])
            parSARhigh2[mm-initRange] = oldSARh2+AFH2*(HP2-oldSARh2)
            parSARlow2[mm-initRange] = oldSARl2-AFL2*(oldSARl2-LP2)
            if parSARhigh2[mm-initRange]<HP2:
                AFH2 = np.min([AFH2+stepAF,maxAF2])
                LP2 = np.min(thisPeriodBids)
            if parSARlow2[mm-initRange]>LP2:
                AFL2 = np.min([AFH2+stepAF,maxAF2])
                HP2 = np.max(thisPeriodBids)
            oldSARh2 = parSARhigh2[mm-initRange]
            oldSARl2 = parSARlow2[mm-initRange]
        
        bids[mm-initRange] = tradeInfo.SymbolBid.loc[thisPeriod[len(thisPeriod)-1]]
        EMA[mm-initRange,:] = em
        variance[mm-initRange] = np.var(thisPeriodBids)
        timeInterval[mm-initRange] = (te-t0).seconds/data.nEventsPerStat
        timeSecs[mm-initRange] = (te.hour*60*60+te.minute*60+te.second)/secsInDay
        

    #print(dateTime)
    nF = 0
    if 0 in data.features:
        features[:,nF] = bids
    
    for i in range(data.lbd.shape[0]):
        #nF = nF+1
        if nF+1 in data.features:            
            nF += 1        
            features[:,nF] = EMA[:,i]
        
    if 8 in data.features:
        nF+=1
        logVar = 10*np.log10(variance/data.std_var+1e-10)
        features[:,nF] = logVar
    
    if 9 in data.features:
        nF = nF+1
        logTimeInt = 10*np.log10(timeInterval/data.std_time+0.01)
        features[:,nF] = logTimeInt
    
    if 10 in data.features:
        nF = nF+1  
        features[:,nF] = parSARhigh20
    
    if 11 in data.features:
        nF = nF+1
        features[:,nF] = parSARlow20
    
    if 12 in data.features:
        nF = nF+1
        features[:,nF] = timeSecs
    
    if 13 in data.features:
        nF = nF+1  
        features[:,nF] = parSARhigh2
    
    if 14 in data.features:
        nF = nF+1
        features[:,nF] = parSARlow2
    
    # Repeat non-variation features to inclue variation betwen first and second input
    if 15 in data.features:
        nF+=1
        logVar = 10*np.log10(variance/data.std_var+1e-10)
        features[:,nF] = logVar
    
    if 16 in data.features:
        nF = nF+1
        logTimeInt = 10*np.log10(timeInterval/data.std_time+0.01)
        features[:,nF] = logTimeInt
        
    if 17 in data.features:
        nF = nF+1
        features[:,nF] = maxValue-bids
    
    if 18 in data.features:
        nF = nF+1
        features[:,nF] = bids-minValue
    
    if 19 in data.features:
        nF = nF+1
        features[:,nF] = maxValue-bids
        
    if 20 in data.features:
        nF = nF+1
        features[:,nF] = bids-minValue
    
    if 21 in data.features:
        nF = nF+1
        features[:,nF] = minValue/maxValue
    
    if 22 in data.features:
        nF = nF+1
        features[:,nF] = minValue/maxValue
    
    for i in range(data.lbd.shape[0]):
        #nF = nF+1
        if nF+1 in data.features:            
            nF += 1        
            features[:,nF] = bids/EMA[:,i]
    
    for i in range(data.lbd.shape[0]):
        #nF = nF+1
        if nF+1 in data.features:
            nF += 1        
            features[:,nF] = bids/EMA[:,i]
    
    
    
    # get results and output info
    np_00 = initRange*data.movingWindow-1
    np_0e = initRange*data.movingWindow+data.nEventsPerStat-2
    np_ee = allSamps*data.movingWindow+data.nEventsPerStat-2
    np_e0 = allSamps*data.movingWindow-1
    
    nR = 10
    
    TI = pd.DataFrame(data=[],columns=['DT1','DT2','DT3','DT4','DT5','DT6','DT7','DT8','DT9','DT10','DT11'
                       ,'B1','B2','B3','B4','B5','B6','B7','B8','B9','B10','B11',
                       'A1','A2','A3','A4','A5','A6','A7','A8','A9','A10','A11'])
    TI.loc[:,"DT1"] = tradeInfo.DateTime.iloc[np_00:np_e0:data.movingWindow]
    TI.loc[:,"B1"] = tradeInfo.SymbolBid.iloc[np_00:np_e0:data.movingWindow].tolist()
    TI.loc[:,"A1"] = tradeInfo.SymbolAsk.iloc[np_00:np_e0:data.movingWindow].tolist()
    Return = np.zeros((len(range(np_0e,np_ee,data.movingWindow)),nR))
    for nr in range(nR):
    #    print(nr)
        unp_0e = int(initRange*data.movingWindow+np.floor(data.nEventsPerStat/(nr+1))-2)
        unp_ee = int(allSamps*data.movingWindow+np.floor(data.nEventsPerStat/(nr+1))-2)
        Return[:,nr] = np.array(tradeInfo.SymbolBid.iloc[unp_0e:unp_ee:data.movingWindow])-np.array(tradeInfo.SymbolBid.iloc[np_00:np_e0:data.movingWindow])
        
        TI.loc[:,"DT"+str(nr+2)] = tradeInfo.DateTime.iloc[unp_0e:unp_ee:data.movingWindow].tolist()
        TI.loc[:,"B"+str(nr+2)] = tradeInfo.SymbolBid.iloc[unp_0e:unp_ee:data.movingWindow].tolist()
        TI.loc[:,"A"+str(nr+2)] = tradeInfo.SymbolAsk.iloc[unp_0e:unp_ee:data.movingWindow].tolist()
    

    return features,Return,TI

def extractReturns_v31(tradeInfo, data):
    #tic = time.time() 
    #print(tradeInfo.SymbolBid)
    nEvents = tradeInfo.shape[0]
    nEintoMovW = nEvents/data.movingWindow
    allSamps = np.floor(nEintoMovW)-np.ceil(data.nEventsPerStat/data.movingWindow)
    allSamps = allSamps.astype(int)
    initRange = int(data.nEventsPerStat/data.movingWindow)
    nSamps = allSamps-initRange
    #iterateRange = range(initRange,allSamps)
    
    np_00 = initRange*data.movingWindow-1
    np_0e = initRange*data.movingWindow+data.nEventsPerStat-2
    np_ee = allSamps*data.movingWindow+data.nEventsPerStat-2
    np_e0 = allSamps*data.movingWindow-1
    
    TI = pd.DataFrame(data=[],columns=['DTe','DT0','DT1','DT2','DT3','DT4','DT5','DT6',
                       'Be','B0','B1','B2','B3','B4','B5','B6',
                       'Ae','A0','A1','A2','A3','A4','A5','A6'])
    TI.loc[:,"DTe"] = tradeInfo.DateTime.iloc[np_00:np_e0:data.movingWindow]
    TI.loc[:,"Be"] = tradeInfo.SymbolBid.iloc[np_00:np_e0:data.movingWindow].tolist()
    TI.loc[:,"Ae"] = tradeInfo.SymbolAsk.iloc[np_00:np_e0:data.movingWindow].tolist()
    Return = np.zeros((len(range(np_0e,np_ee,data.movingWindow)),len(data.lookAheadVector)))
    #print("Return.shape")
    #print(Return.shape)
    indexOrigins = [i for i in range(np_00,np_e0,data.movingWindow)]
    
    #origins = np.array(tradeInfo.SymbolBid.iloc[np_00:np_e0:data.movingWindow])
    
    for nr in range(len(data.lookAheadVector)):
        #print("nr")
        #print(nr)
        unp_0e = int(initRange*data.movingWindow+np.floor(data.nEventsPerStat*data.lookAheadVector[nr])-2)
        unp_ee = int(np.min([allSamps*data.movingWindow+np.floor(
                data.nEventsPerStat*data.lookAheadVector[nr])-2,nEvents]))
        #ends = np.array(tradeInfo.SymbolBid.iloc[unp_0e::data.movingWindow])
        #print("indexOrigins")
        #print(indexOrigins)
        indexEnds = [i for i in range(unp_0e,unp_ee,data.movingWindow)]
        #print("indexEnds")
        #print(indexEnds)
        #fill ends wth last value
        for i in range(len(indexEnds),len(indexOrigins)):
            #print("i")
            #print(i)
            indexEnds.append(tradeInfo.shape[0]-1)
        #print(indexEnds)
        #print(tradeInfo.SymbolBid.iloc[indexEnds].as_matrix())
        #print(tradeInfo.SymbolBid.iloc[indexOrigins].as_matrix())
        Return[:,nr] = tradeInfo.SymbolBid.iloc[indexEnds].as_matrix()-tradeInfo.SymbolBid.iloc[indexOrigins].as_matrix()
        
        TI.loc[:,"DT"+str(nr)] = tradeInfo.DateTime.iloc[indexEnds].tolist()
        TI.loc[:,"B"+str(nr)] = tradeInfo.SymbolBid.iloc[indexEnds].tolist()
        TI.loc[:,"A"+str(nr)] = tradeInfo.SymbolAsk.iloc[indexEnds].tolist()
    
    
    return Return,TI

def getAssetsFlag(AllAssets, assets):
    assetsFlag = ""
    firstAss = -1
    for i in range(len(AllAssets)):#
        #print(i)
        #print(np.argmax(np.array(data.assets)==i))
        if np.max(np.array(assets)==i):
            assetsFlag+="1"
            if firstAss==-1:
                firstAss = i
        else:
            assetsFlag+="0"
    
    return assetsFlag

def getDatesFlag(data):
            
    #connDateTest = sqlite3.connect(DBforex)
    #datesTests = pd.read_sql_query("SELECT * FROM "+ data.AllAssets[str(firstAss)]+"teDATES", connDateTest)
    #connDateTest.close()
    datesTests = pd.DataFrame(data=data.dateTest,columns=["Dates"])
    testDatesFlag = ""
    dateDT = dt.datetime.strptime(data.dateStart,"%Y.%m.%d")
    #print(dt.datetime.strftime(dateDT,"%Y.%m.%d"))
    deltaDates=dt.datetime.strptime(data.dateEnd,"%Y.%m.%d")-dateDT
    for i in range(deltaDates.days):
        a = datesTests[datesTests.Dates==dt.datetime.strftime(dateDT,"%Y.%m.%d")].shape[0]
        if a>0:
            testDatesFlag+="1"
        else:
            testDatesFlag+="0"
        dateDT += dt.timedelta(days=1)
        
    return testDatesFlag

def getFeaturesFlag(data):
    featsFlag = ""
    for i in range(len(data.AllFeatures)):#
        #print(i)
        #print(np.argmax(np.array(data.assets)==i))
        if len(data.features)>0:
            if np.max(np.array(data.features)==i):
                featsFlag+="1"
            else:
                featsFlag+="0"
                
    return featsFlag

def get_binary_representation_vector(all_options_vector,selected_options_vector):
    """
    Function that generates a binary representation of a vector, given all possible 
    values.
    Inputs: 
        - all_options_vector: vector containing the whole set of possible options.
        - selected_options_vector: vector containing the selected subset of options.
    Return: 
        - binary_rep: binary representation of selected_options_vector given 
          all_options_vector.
    """
    binary_rep = ""
    for i in all_options_vector:#
        if len(selected_options_vector)>0:
            if np.max(np.array(selected_options_vector)==i):
                binary_rep += "1"
            else:
                binary_rep += "0"
                
    return binary_rep

def printParametersFromID(ID):
    IDstr = "{0:06d}".format(int(ID))
    conn = sqlite3.connect('../DB/EVAL.sqlite')
    parameters = pd.read_sql_query("SELECT * FROM IDw WHERE ID=="+"'"+IDstr+"'", conn)
    if parameters.shape[0]>0:
        #print(parameters.keys())
        for p in parameters.keys():#range(parameters.shape[1]):
            #print(p)
            
            if parameters[p].iloc[0]!=None:
                if p=="dateTest":
                    m = len(re.findall("1",parameters[p].iloc[0]))
                    #print(m)
                    print("Number of test days:"+str(m)) 
                else:
                    print(p+":"+parameters[p].iloc[0])
            else: 
                print(p+": None")
                
    conn.close()
    return parameters

def printIDtables(DBeval):
    
    conn = sqlite3.connect(DBeval)
    print("Weights table:")
    IDw = pd.read_sql_query("SELECT * FROM IDw", conn)
    print(IDw)
    print("Results table:")
    IDr = pd.read_sql_query("SELECT * FROM IDr", conn)
    print(IDr)
    conn.close()
    return IDw,IDr

def getIDresultsRNN_v00(IDweights,data, DBeval):
    
    conn = sqlite3.connect(DBeval)
    
    try:
        #error()
        lastEntryID = pd.read_sql_query("SELECT * FROM IDr WHERE `index`>-1 ORDER BY `index` DESC LIMIT 1", conn)
        lastEntryID = lastEntryID.set_index(lastEntryID["index"])
        #print(lastEntryID)
        ID = int(lastEntryID.index[0])
        newTable = 0
    except:
        #error()
        ID = -1
        newTable = 1
        readFromTable = pd.DataFrame()
        readFromTable.index.name = "ID"
        
    assetsFlag = getAssetsFlag(data.AllAssets, data.assets)
    testDatesFlag = getDatesFlag(data)
    
    if newTable==0:
        
        assetsFlag = getAssetsFlag(data.AllAssets, data.assets)
        testDatesFlag = getDatesFlag(data)
        #print(testDatesFlag)
        #print(data.assets)
        #print(assetsFlag)
        query = ("SELECT * FROM IDr WHERE IDweights=="+"'"+IDweights+"'"+" AND assets=="+"'"+assetsFlag+"'"+
                 " AND dateStart=="+"'"+data.dateStart+"'"+" AND dateEnd="+"'"+data.dateEnd+"'"+
                 " AND dateTest=="+"'"+testDatesFlag+"'")
        
        #print(pd.read_sql_query("SELECT * FROM IDr", conn))
        readFromTable = pd.read_sql_query(query, conn)
        #print(readFromTable)
        #readFromTable=pd.read_sql_query("SELECT * FROM IDw",sqlite3.connect('../DB/EVAL.sqlite'))
        readFromTable = readFromTable.set_index(readFromTable["index"]).drop("index",1)
        readFromTable.index.name="ID"
        
    if readFromTable.shape[0]>0:
        ID = readFromTable.index[0]
        print("Results ID already exists: "+ID)
            
        #print(ID)
    else:
        ID = ID+1
        #print(ID)
        ID = "{0:06d}".format(int(float(ID)))
        print("New results ID: "+ID)
            
        entryData = {"index":ID,
                    "IDweights":IDweights,
                 "assets":assetsFlag,
                 "dateEnd":data.dateEnd,
                 "dateStart":data.dateStart,
                 "dateTest":testDatesFlag}
        #print(entryData)

        newEntry = pd.DataFrame()
        newEntry = pd.DataFrame(data=entryData,index = [ID])
        newEntry.to_sql("IDr",conn, if_exists="append")
        
    return ID

def getIDweightsRNN_v00(model, data, DBeval, tOt='tr'):
    
    conn = sqlite3.connect(DBeval)
    
    
    try:
        #error()
        lastEntryID = pd.read_sql_query("SELECT * FROM IDw WHERE `index`>-1 ORDER BY `index` DESC LIMIT 1", conn)
        #print(lastEntryID)
        lastEntryID = lastEntryID.set_index(lastEntryID["index"])
        ID = int(lastEntryID.index[0])
        newTable = 0
    except:
        print("Error while accesing table.")
        error()
        ID = -1
        newTable = 1
        readFromTable = pd.DataFrame()
        readFromTable.index.name = "ID"
     
    assetsFlag = getAssetsFlag(data.AllAssets, data.assets)
    testDatesFlag = getDatesFlag(data)
    featsFlag = getFeaturesFlag(data)
    channelsBin = get_binary_representation_vector(range(int(data.nEventsPerStat/data.movingWindow)),data.channels)

    if newTable==0:
        
        query = "SELECT * FROM IDw WHERE L=="+str(
                float(model.L))+" AND size_hidden_layer=="+str(
                float(model.size_hidden_layer))+" AND size_output_layer=="+str(
                float(model.size_output_layer))+" AND miniBatchSize=="+str(
                float(model.miniBatchSize))+" AND outputGain=="+str(
                float(model.outputGain))+" AND lR0=="+str(
                float(model.lR0))+" AND commonY=="+str(
                float(model.commonY))+" AND keepProbDO=="+str(
                float(model.keep_prob_dropout))+" AND lookAhead=="+str(
                float(data.lookAhead))+" AND divideLookAhead=="+str(
                float(data.divideLookAhead))+" AND lookAheadIndex=="+str(
                float(data.lookAheadIndex))+" AND lB=="+str(
                float(data.lB))+" AND nEventsPerStat=="+str(
                float(data.nEventsPerStat))+" AND movingWindow=="+str(
                float(data.movingWindow))+" AND nFeatures=="+str(
                float(data.nFeatures))+" AND dateTest="+"'"+testDatesFlag+"'"+(
                " AND dateStart=")+"'"+data.dateStart+"'"+" AND dateEnd="+(
                "'")+data.dateEnd+"'"+" AND assets="+"'"+assetsFlag+"'"+(
                " AND Comments="+"'"+data.comments+"'"+" AND Ver="+"'"+model.version+"'"+
                " AND featuresBin="+"'"+featsFlag+"'"+" AND channelsBin="+"'"+channelsBin+"'")
        
        readFromTable = pd.read_sql_query(query, conn)
        readFromTable = readFromTable.set_index(readFromTable["index"]).drop("index",1)
        readFromTable.index.name="ID"
        
    if readFromTable.shape[0]>0:
        print("Entry exists. ID:")
        ID = readFromTable.index[0]
        print(ID)
    elif tOt=='tr':
        print("New entry. ID:")
        ID = "{0:06d}".format(int(float(ID))+1)
        print(ID)
        dictEn = {
                  "L":int(model.L),
                  "size_hidden_layer":model.size_hidden_layer,
                  "size_output_layer":model.size_output_layer,
                  "miniBatchSize":model.miniBatchSize,
                  "lR0":model.lR0,
                  "outputGain":model.outputGain,
                  "keepProbDO":model.keep_prob_dropout,
                  "Ver":model.version,
                  "commonY":model.commonY,
                  "lookAhead":data.lookAhead,
                  "divideLookAhead":data.divideLookAhead,
                  "lookAheadIndex":data.lookAheadIndex,
                  "lB":data.lB,
                  "nEventsPerStat":data.nEventsPerStat,
                  "movingWindow":data.movingWindow,
                  "nFeatures":data.nFeatures,
                  "dateStart":data.dateStart,
                  "dateEnd":data.dateEnd,
                  "dateTest":testDatesFlag,
                  "assets":assetsFlag,                  
                  "Comments":data.comments,
                  "featuresBin":featsFlag,
                  "channelsBin":channelsBin}
            
        newEntry = pd.DataFrame()
        newEntry = pd.DataFrame(data=dictEn,index = [ID])
        newEntry.to_sql("IDw",conn, if_exists="append")
    else:
        print("Error! Table entry not found. Check parameters.")
        error()
        
        
    conn.close()
        #ID = "{0:06d}".format(int(float(ID)))
        
    
    return ID

def loadTradeInfoFromDB_v27(DB, DBDS, asset, tOt, data):
    
    conn = sqlite3.connect(DB)
    connDS = sqlite3.connect(DBDS)
    conn.create_function("REGEXP", 2, regexp)
    Dates = pd.read_sql_query("SELECT * FROM "+asset+tOt+"DATES", connDS)
    Dates = pd.to_datetime(Dates.Dates,format='%Y.%m.%d')
    #print(Dates)
    preciseDateStart = dt.datetime.strftime(Dates[(Dates.iloc[:]-dt.datetime.strptime(
            data.dateStart, '%Y.%m.%d')).abs().idxmin()],'%Y.%m.%d')
    preciseDateEnd = dt.datetime.strftime(Dates[(Dates.iloc[:]-dt.datetime.strptime(
            data.dateEnd, '%Y.%m.%d')).abs().idxmin()],'%Y.%m.%d')
    #print(preciseDateStart)
    #print(preciseDateEnd)
    #print(pd.read_sql_query("SELECT * FROM "+asset,conn))
    initIndex = pd.read_sql_query("SELECT real_index FROM "+asset+
                                  " WHERE DateTime REGEXP ('"+preciseDateStart+"') LIMIT 1", conn)
    endIndex = pd.read_sql_query("SELECT real_index FROM "+asset+
                                " DESC WHERE DateTime REGEXP ('"+preciseDateEnd+
                                "')  ORDER BY real_index DESC LIMIT 1", conn)
    #print(initIndex)
    #print(endIndex)
    tradeInfo = pd.read_sql_query("SELECT * FROM "+asset+" WHERE real_index>="+
                              str(initIndex["real_index"].iloc[0])+
                              " AND real_index<"+str(endIndex["real_index"].iloc[0]+1), conn)
    #print(tradeInfo)
    tradeInfo = tradeInfo.set_index("real_index")
    #print(tradeInfo)
    conn.close()
    
    return tradeInfo

def saveStatsInDB(DB, data, asset, means, stds):
    
    conn = sqlite3.connect(DB)
    curr = conn.cursor()
    # save stats in DB
    countFeats = 0
    IDtable = "Stats"+str(data.movingWindow)+str(data.nEventsPerStat)
    #curr.execute("DROP TABLE IF EXISTS "+IDtable)
    
    stats = {"Feature":[],
             "Asset":[],
             "Mean":[],
             "Std":[]}
    dF = pd.DataFrame(data=stats)
    dF.to_sql(IDtable, conn, if_exists="append")
    
    for nF in data.features:
        stats = {}
        stats["Feature"] = data.AllFeatures[str(nF)]
        stats["Asset"] = asset
        stats["Mean"] = means[0,countFeats]
        stats["Std"] = stds[0,countFeats]
        dF = pd.DataFrame(data=stats,index=[0])
        query = "DELETE FROM "+ IDtable+" WHERE Asset="+"'"+ asset+"'"+" AND Feature='"+data.AllFeatures[str(nF)]+"'"
        #print(query)
        curr.execute(query)
        #print(dF)
        dF.to_sql(IDtable, conn, if_exists="append",index=False)
        #save_as_matfile(asset+IDtable+data.AllFeatures[str(nF)],asset+IDtable+data.AllFeatures[str(nF)],stats)
        countFeats+=1
    
    # Add output
    stats = {}
    stats["Feature"] = "Output"
    stats["Asset"] = asset
    stats["Mean"] = means[0,-1]
    stats["Std"] = stds[0,-1]
    dF = pd.DataFrame(data=stats,index=[0])
    query = "DELETE FROM "+ IDtable+" WHERE Asset="+"'"+ asset+"'"+" AND Feature='Output'"
    #print(query)
    curr.execute(query)
    #print(dF)
    dF.to_sql(IDtable, conn, if_exists="append",index=False)
    
    #save_as_matfile(asset+IDtable+"O",asset+IDtable+"O",stats)
    
    #dF2 = pd.read_sql_query("SELECT * FROM "+IDtable,conn)
    #print(dF2)
    # save Optput in DB
    print("Stats saved in DB")
    curr.close()
    conn.close()
        
    return None

def loadStatsFromDB_v27(DBfeatures, data, asset, trainOrTest):
    
    conn = sqlite3.connect(DBfeatures)
    IDtable = "Stats"+str(data.movingWindow)+str(data.nEventsPerStat)
    initStats = {"Mean":[],
                 "Std":[]}
    stats = pd.DataFrame(data=initStats)
    
    struct = {"Feature":[],
             "Asset":[],
             "Mean":[],
             "Std":[]}
    dF = pd.DataFrame(data=struct)
    dF.to_sql(IDtable, conn, if_exists="append")
    
    # load stats per feature
    for nF in data.features:
        query = "SELECT Mean,Std FROM "+IDtable+" WHERE Asset="+"'"+ asset+"'"+" AND Feature='"+data.AllFeatures[str(nF)]+"'"
        #print(pd.read_sql_query(query,conn))
        stats = stats.append(pd.read_sql_query(query,conn))
    
    #print(pd.read_sql_query("SELECT * FROM "+IDtable,conn))
    #print("stats.shape")
    #print(stats.shape)
    # add output
    #query = "SELECT Mean,Std FROM "+IDtable+" WHERE Asset="+"'"+ asset+"'"+" AND Feature='Output'"
    #stats = stats.append(pd.read_sql_query(query,conn))
    
    IDtableOuts = "StatsOutputs_v30"+str(data.movingWindow)+str(data.nEventsPerStat)
    query = "SELECT Mean,Std FROM "+IDtableOuts+" WHERE Asset="+"'"+ asset+"'"+" AND divideLookAhead="+str(data.divideLookAhead)
    msO = pd.read_sql_query(query,conn)
    
    stats = stats.append(msO)
    #print(stats)
    #print(stats)
    #print(statsO)
    conn.close()
    
    if stats.shape[0]==0:
        print("Error! Stats don't exist. Run saveFeatures with saveStats=1 first")
        error()
    
    return stats

def saveStatsInDB_v27(DB, data, asset, means, stds, meansO, stdsO):
    # Warning! Temporal implementation. Save stats table in DB too!!
    conn = sqlite3.connect(DB)
    curr = conn.cursor()
    # save stats in DB
    countFeats = 0
    IDtable = "Stats"+str(data.movingWindow)+str(data.nEventsPerStat)
    
    #curr.execute("DROP TABLE IF EXISTS "+IDtable)
    
    stats = {"Feature":[],
             "Asset":[],
             "Mean":[],
             "Std":[]}
    dF = pd.DataFrame(data=stats)
    dF.to_sql(IDtable, conn, if_exists="append")
    
    for nF in data.features:
        stats = {}
        stats["Feature"] = data.AllFeatures[str(nF)]
        stats["Asset"] = asset
        stats["Mean"] = means[0,countFeats]
        stats["Std"] = stds[0,countFeats]
        dF = pd.DataFrame(data=stats,index=[0])
        query = "DELETE FROM "+ IDtable+" WHERE Asset="+"'"+ asset+"'"+" AND Feature='"+data.AllFeatures[str(nF)]+"'"
        #print(query)
        curr.execute(query)
        #print(dF)
        dF.to_sql(IDtable, conn, if_exists="append",index=False)
        #save_as_matfile(asset+IDtable+data.AllFeatures[str(nF)],asset+IDtable+data.AllFeatures[str(nF)],stats)
        countFeats+=1
    
    # Add output
    stats = {}
    stats["Feature"] = "Output"
    stats["Asset"] = asset
    stats["Mean"] = means[0,-1]
    stats["Std"] = stds[0,-1]
    dF = pd.DataFrame(data=stats,index=[0])
    
    IDtableOuts = "StatsOutputs_v30"+str(data.movingWindow)+str(data.nEventsPerStat)
    
    dF = pd.DataFrame(columns = ["divideLookAhead","Asset","Mean","Std"])
    dF.to_sql(IDtableOuts, conn, if_exists="append")
    
    dF.loc[:,"divideLookAhead"] = pd.Series(range(1,11)).astype(int)
    dF.loc[:,"Asset"] = asset
    dF.loc[:,"Mean"] = pd.Series(meansO[0,:]).astype(float)
    dF.loc[:,"Std"] = pd.Series(stdsO[0,:]).astype(float)
    
    
    
    #print(dF)
    query = "DELETE FROM "+ IDtableOuts+" WHERE Asset="+"'"+ asset+"'"
    curr.execute(query)
    
    dF.to_sql(IDtableOuts, conn, if_exists="append",index=False)
    
    #print(pd.read_sql_query("SELECT * FROM "+IDtableOuts,conn))
    print("Stats saved in DB")
    curr.close()
    conn.close()
    #a=p
    return None

def saveFeaturesToDB_v27(asset, Init, End, data, features, Returns, DTs, DB, logFile=""):

    conn = sqlite3.connect(DB)
    #curr = conn.cursor()
    # Save features
    for f in data.features:
        IDtable = asset+data.AllFeatures[str(f)]+Init+End+"{0:05d}".format(
                data.nEventsPerStat)+"{0:05d}".format(data.movingWindow)
        #print(IDtable)
        #curr.execute("DROP TABLE IF EXISTS "+IDtable)
        #print(features.shape)
        #print(features[f,:])
        struct = {'Value':features[:,f]}
        dataFrame = pd.DataFrame(data=struct)
        #print(dataFrame)
        dataFrame.to_sql(IDtable, conn, if_exists="replace")

    # Save output
    IDtable = asset+"Output_v30"+Init+End+"{0:05d}".format(
                data.nEventsPerStat)+"{0:05d}".format(data.movingWindow)
    #print(IDtable)
    #print(IDtable)
    #curr.execute("DROP TABLE IF EXISTS "+IDtable)
    #print(features.shape)
    #print(features[f,:])
    dictFromRets={str(i+1):Returns[:,i] for i in range(Returns.shape[1])}
    dataFrame = pd.DataFrame(data=dictFromRets)
    #print(dataFrame)
    dataFrame.to_sql(IDtable, conn, if_exists="replace")
    
    IDtable = asset+"DT_v30"+Init+End+"{0:05d}".format(
                data.nEventsPerStat)+"{0:05d}".format(data.movingWindow)
    
    DTs.to_sql(IDtable, conn, index_label="real_index",if_exists="replace")
    print("Features saved in DB")
    if logFile!="":
        file = open(logFile,"a")
        file.write("Features saved in DB"+"\n")
        file.close()
    
    conn.close()
    return None

def extractFeaturesFromDB_v27(asset, Init, End, data, DB, logFile=""):
    
    features = np.array([])
    Returns = np.array([])
    DTs = pd.DataFrame()
    conn = sqlite3.connect(DB)
    #curr = conn.cursor()
    
    IDtableO_v30 = asset+"Output_v30"+Init+End+"{0:05d}".format(
                data.nEventsPerStat)+"{0:05d}".format(data.movingWindow)
    #print(IDtable)
    dictFromRets={str(i+1):[] for i in range(10)}
    dataFrame = pd.DataFrame(data=dictFromRets)
    dataFrame.to_sql(IDtableO_v30, conn, if_exists="append")

    df_v30 = pd.read_sql("SELECT * FROM "+IDtableO_v30, conn)
    
    #IDtableO = asset+"Output"+Init+End+"{0:05d}".format(
    #            data.nEventsPerStat)+"{0:05d}".format(data.movingWindow)
    
    
    #df = pd.read_sql("SELECT * FROM "+IDtableO, conn)
    if df_v30.shape[0]>0:
        df_v30 = df_v30.drop('index', 1)
        #print(df)
        features = np.zeros((df_v30.shape[0],len(data.features)+1))
        Returns = df_v30.as_matrix()
        
        ret = df_v30[str(data.divideLookAhead)]
        #print(ret.shape)
        #print(features.shape)
        features[:,-1] = ret
        
        #df = df.drop('index', 1)
        #features = np.zeros((df.shape[0],len(data.features)+1))
        #feat = df.as_matrix()
        #print(feat.shape)
        #print(features.shape)
        #features[:,-1] = feat[:,0]
        
        IDtable = asset+"DT_v30"+Init+End+"{0:05d}".format(
                data.nEventsPerStat)+"{0:05d}".format(data.movingWindow)
        
        DTs = pd.read_sql("SELECT * FROM "+IDtable, conn).set_index("real_index")
        #print(DTs)
    counter = 0
    struct = {'Value':[]}
    for f in data.features:
        IDtable = asset+data.AllFeatures[str(f)]+Init+End+"{0:05d}".format(
                data.nEventsPerStat)+"{0:05d}".format(data.movingWindow)
        #print(IDtable)
        dataFrame = pd.DataFrame(data=struct)
        dataFrame.to_sql(IDtable, conn, if_exists="append")
    
        df = pd.read_sql("SELECT * FROM "+IDtable, conn)
        #print(df)
        #print(df.shape)
    
        if df.shape[0]>0:
            df = df.drop('index', 1)
            #print(df.as_matrix().shape)
            feat = df.as_matrix()
            #print(feat.shape)
            features[:,counter] = feat[:,0]
            counter += 1
    
   # print(features)
    #curr.execute("DROP TABLE IF EXISTS "+IDtable)
    conn.close()
    if features.shape[0]>0:
        print("Features loaded from DB")
    #print(features)
    return features, Returns, DTs

def saveFeaturesToDB_v40(asset, Init, End, data, features, DB, logFile=""):

    conn = sqlite3.connect(DB)
    #curr = conn.cursor()
    # Save features
    for f in data.features:
        IDtable = asset+data.AllFeatures[str(f)]+"_v40"+Init+End+"{0:05d}".format(
                data.nEventsPerStat)+"{0:05d}".format(data.movingWindow)
        #print(IDtable)
        #curr.execute("DROP TABLE IF EXISTS "+IDtable)
        #print(features.shape)
        #print(features[f,:])
        struct = {'Value':features[:,f]}
        dataFrame = pd.DataFrame(data=struct)
        #print(dataFrame)
        dataFrame.to_sql(IDtable, conn, if_exists="replace")


    print("Features saved in DB")

    conn.close()
    return None

def saveReturnsToDB_v40(asset, Init, End, data, Returns, DTs, DB, logFile=""):

    conn = sqlite3.connect(DB)

    # Save output
    IDtable = asset+"Output_v40"+Init+End+"{0:05d}".format(
                data.nEventsPerStat)+"{0:05d}".format(data.movingWindow)

    dictFromRets={str(i+1):Returns[:,i] for i in range(Returns.shape[1])}
    dataFrame = pd.DataFrame(data=dictFromRets)

    dataFrame.to_sql(IDtable, conn, if_exists="replace")
    
    IDtable = asset+"DT_v40"+Init+End+"{0:05d}".format(
                data.nEventsPerStat)+"{0:05d}".format(data.movingWindow)
    
    DTs.to_sql(IDtable, conn, index_label="real_index",if_exists="replace")
    print("Returns saved in DB")

    conn.close()
    return None

def extractFeaturesFromDB_v40(asset, Init, End, data, DB, logFile=""):
    
    features = np.array([])
    conn = sqlite3.connect(DB)

    counter = 0
    struct = {'Value':[]}
    
    for f in data.features:
        IDtable = asset+data.AllFeatures[str(f)]+"_v40"+Init+End+"{0:05d}".format(
                data.nEventsPerStat)+"{0:05d}".format(data.movingWindow)
        #print(IDtable)
        dataFrame = pd.DataFrame(data=struct)
        dataFrame.to_sql(IDtable, conn, if_exists="append")
    
        df = pd.read_sql("SELECT * FROM "+IDtable, conn)
        #print(df)
        #print(df.shape)
    
        if df.shape[0]>0:
            df = df.drop('index', 1)
            #print(df.as_matrix().shape)
            feat = df.as_matrix()
            #print(feat.shape)
            features[:,counter] = feat[:,0]
            counter += 1
    
    conn.close()
    if features.shape[0]>0:
        print("Features loaded from DB")

    return features

def extractResultsFromDB_v40(asset, Init, End, data, DB, logFile=""):
    
    #features = np.array([])
    Returns = pd.DataFrame()
    DTs = pd.DataFrame()
    conn = sqlite3.connect(DB)
    #curr = conn.cursor()
    
    IDtable = asset+"Output_v40"+Init+End+"{0:05d}".format(
                data.nEventsPerStat)+"{0:05d}".format(data.movingWindow)
    #print(IDtable)
    dictFromRets={str(i+1):[] for i in range(10)}
    dataFrame = pd.DataFrame(data=dictFromRets)
    dataFrame.to_sql(IDtable, conn, if_exists="append")

    df = pd.read_sql("SELECT * FROM "+IDtable, conn)

    if df.shape[0]>0:
        df = df.drop('index', 1)
        Returns = df

        IDtable = asset+"DT_v40"+Init+End+"{0:05d}".format(
                data.nEventsPerStat)+"{0:05d}".format(data.movingWindow)
        
        DTs = pd.read_sql("SELECT * FROM "+IDtable, conn).set_index("real_index")

    conn.close()
    return Returns, DTs

def loadStatsFromDB_v40(DBstats, data, asset, trainOrTest):
    
    conn = sqlite3.connect(DBstats)
    resolution = int(data.nEventsPerStat/data.movingWindow)
    IDtable = "Stats_v40_"+str(data.movingWindow)+"_"+str(data.nEventsPerStat)
    initStats = {"Mean":[],
                 "Std":[]}
    stats = pd.DataFrame(data=initStats)
    
    struct = {"Feature":[],
             "Asset":[],
             "Mean":[],
             "Std":[],
             "r":[]}
    
    dF = pd.DataFrame(data=struct)
    dF.to_sql(IDtable, conn, if_exists="append")
    
    # load stats per feature
    for r in range(resolution):
        for nF in data.features:
            query = ("SELECT Mean,Std FROM "+IDtable+" WHERE Asset="+"'"+ asset+"'"+
                     " AND Feature='"+data.AllFeatures[str(nF)]+"'"+
                     " AND r="+str(r))
            #print(pd.read_sql_query(query,conn))
            stats = stats.append(pd.read_sql_query(query,conn))
    
    #print(pd.read_sql_query("SELECT * FROM "+IDtable,conn))
    
    #print(stats)
    # add output
    #query = "SELECT Mean,Std FROM "+IDtable+" WHERE Asset="+"'"+ asset+"'"+" AND Feature='Output'"
    #stats = stats.append(pd.read_sql_query(query,conn))
    
    IDtableOuts = "StatsOutputs_v40"+str(data.movingWindow)+str(data.nEventsPerStat)
    query = "SELECT Mean,Std FROM "+IDtableOuts+" WHERE Asset="+"'"+ asset+"'"+" AND lookAhead=="+str(
                float(data.lookAheadVector[data.lookAheadIndex]))
    #print("All stats:")
    #print(pd.read_sql_query("SELECT Mean,Std FROM "+IDtableOuts+" WHERE Asset="+"'"+ asset+"'",conn))
    msO = pd.read_sql_query(query,conn)
    #print("msO")
    #print(msO)
    stats = stats.append(msO)
    #print(stats)
    #print(stats)
    #print(statsO)
    conn.close()
    
    if stats.shape[0]==0:
        print("Error! Stats don't exist. Run saveFeatures with saveStats=1 first")
        error()
    
    return stats

def saveStatsInDB_v40(DBstats, data, asset, means, stds, meansO, stdsO):
    # Warning! Temporal implementation. Save stats table in DB too!!
    conn = sqlite3.connect(DBstats)
    curr = conn.cursor()
    # save stats in DB
    resolution = int(data.nEventsPerStat/data.movingWindow)
    IDtable = "Stats_v40_"+str(data.movingWindow)+"_"+str(data.nEventsPerStat)
    
    stats = {"Feature":[],
             "Asset":[],
             "Mean":[],
             "Std":[],
             "r":[]}
    dF = pd.DataFrame(data=stats)
    dF.to_sql(IDtable, conn, if_exists="append")
    
    for r in range(resolution):
        countFeats = 0
        #curr.execute("DROP TABLE IF EXISTS "+IDtable)
        for nF in data.features:
            stats = {}
            stats["Feature"] = data.AllFeatures[str(nF)]
            stats["Asset"] = asset
            stats["Mean"] = means[r,countFeats]
            stats["Std"] = stds[r,countFeats]
            stats["r"] = r
            dF = pd.DataFrame(data=stats,index=[0])
            query = ("DELETE FROM "+ IDtable+" WHERE Asset="+"'"+ asset+"'"+
                     " AND Feature='"+data.AllFeatures[str(nF)]+"'"+
                     " AND r="+str(r))
            #print(query)
            curr.execute(query)
            #print(dF)
            dF.to_sql(IDtable, conn, if_exists="append",index=False)
            countFeats+=1
    
    #print(pd.read_sql_query("SELECT * FROM "+IDtable,conn))
    # Add output
    stats = {}
    stats["Feature"] = "Output"
    stats["Asset"] = asset
    stats["Mean"] = means[0,-1]
    stats["Std"] = stds[0,-1]
    dF = pd.DataFrame(data=stats,index=[0])
    
    IDtableOuts = "StatsOutputs_v40"+str(data.movingWindow)+str(data.nEventsPerStat)
    
    dF = pd.DataFrame(columns = ["lookAhead","Asset","Mean","Std"])
    dF.to_sql(IDtableOuts, conn, if_exists="append")
    
    dF.loc[:,"lookAhead"] = pd.Series(data.lookAheadVector).astype(float)
    dF.loc[:,"Asset"] = asset
    dF.loc[:,"Mean"] = pd.Series(meansO[0,:]).astype(float)
    dF.loc[:,"Std"] = pd.Series(stdsO[0,:]).astype(float)
    
    #print(dF)
    query = "DELETE FROM "+ IDtableOuts+" WHERE Asset="+"'"+ asset+"'"
    curr.execute(query)
    
    dF.to_sql(IDtableOuts, conn, if_exists="append",index=False)
    
    #print(pd.read_sql_query("SELECT * FROM "+IDtableOuts,conn))
    print("Stats saved in DB")
    curr.close()
    conn.close()
    #a=p
    return None

def saveFeatures_v40(data, asset, trainOrTest, 
                     DBforex, DBfeatures, DBDS, DBresults, DBstats, 
                     saveStats):
    # Warning! New version not fully implemented!!! 
    print("\n")
    print('Loading '+ asset)
    #tic = time.time()
    if trainOrTest==data.train:
        tOt = "tr"
    else:
        tOt = "te"
    #conn = sqlite3.connect(DBforex)
    #conn.create_function("REGEXP", 2, regexp)
    
    connDS = sqlite3.connect(DBDS)
    # Load separators from DB
    separators = pd.read_sql_query("SELECT * FROM "+asset+tOt+"DISP",connDS).set_index("real_index")
    separators = extractFromSeparators_v20(separators, data.dateStart, data.dateEnd, asset, DBforex)
    connDS.close()
    
    resolution = int(data.nEventsPerStat/data.movingWindow)
    allvariation = np.zeros([0,data.nFeatures,resolution])
    allRet = np.zeros([0,len(data.lookAheadVector)])
    tradeInfo = pd.DataFrame()

    conn = sqlite3.connect(DBfeatures)
    
    for s in range(0,len(separators)-1,2):#range(len(separators)-1)
        if separators.index[s+1]-separators.index[s]>=2*data.nEventsPerStat:
            print("Symbols batch {0:d} out of {1:d}".format(int(s/2),int(len(separators)/2-1)))
            print(separators.DateTime.iloc[s]+" to "+separators.DateTime.iloc[s+1])
            # get init and end dates of these separators
            sepInitDate = dt.datetime.strftime(dt.datetime.strptime(
                    separators.DateTime.iloc[s],'%Y.%m.%d %H:%M:%S'),'%y%m%d%H%M%S')
            sepEndDate = dt.datetime.strftime(dt.datetime.strptime(
                    separators.DateTime.iloc[s+1],'%Y.%m.%d %H:%M:%S'),'%y%m%d%H%M%S')
   
            # Load features from DB
            features = extractFeaturesFromDB_v40(asset, sepInitDate, sepEndDate, data, DBfeatures)
            tableRets,DTs = extractResultsFromDB_v40(asset, sepInitDate, sepEndDate, data, DBresults)
            Returns = tableRets.as_matrix()
            # If features not saved in DB
            if features.shape[0]==0:
                # If tradeInfo not yet loaded
                if tradeInfo.shape[0]==0:
                    print("loading trade info from DB...")
                    tradeInfo = loadTradeInfoFromDB_v27(DBforex, DBDS, asset, tOt, data)

                print("getting features from raw data...")
                features = extractFeatures_v40(tradeInfo.loc[separators.index[s]:separators.index[s+1]+1], data)
                print("getting outputs from raw data...")
                Returns, DTs = extractReturns_v40(tradeInfo.loc[separators.index[s]:separators.index[s+1]+1],data)
                #save_as_matfile("features","features",features)
                # Save features in DB
                if data.save_IO:
                    saveFeaturesToDB_v40(asset, sepInitDate, sepEndDate, data, features, DBfeatures)
                    saveReturnsToDB_v40(asset, sepInitDate, sepEndDate, data, Returns, DTs, DBresults)
                    
            variation = 999+np.zeros((features.shape[0],data.nFeatures,resolution))

            for r in range(resolution):
                
                variation[r+1:,:,r] = features[r+1:,:]-features[:-(r+1),:]
                variation[r+1:,data.noVarFeats,r] = features[:-(r+1),data.noVarFeats]
            
            allvariation = np.append(allvariation,variation,axis=0)
            allRet = np.append(allRet,Returns,axis=0)
            
        else:
            print("Skipping Symbols batch {0:d}. Not enough entries".format(int(s/2)))
    
    conn.close()
    # If train -> save stats in DB
    if saveStats==1 and tOt=="tr":
        
        means = np.zeros((resolution,data.nFeatures))
        stds = np.zeros((resolution,data.nFeatures))
        
        for r in range(resolution):
            nonZeros = allvariation[:,0,r]!=999
            #print(np.sum(nonZeros))
            means[r,:] = np.mean(allvariation[nonZeros,:,r],axis=0,keepdims=1)
            stds[r,:] = np.std(allvariation[nonZeros,:,r],axis=0,keepdims=1)  
        
        stdsO = np.std(allRet,axis=0,keepdims=1)
        meansO = np.mean(allRet,axis=0,keepdims=1)
        
        saveStatsInDB_v40(DBstats, data, asset, means, stds, meansO, stdsO)
        
    return None

def loadStatsFromDB_v33(DBfeatures, data, asset, trainOrTest):
    
    conn = sqlite3.connect(DBfeatures)
    IDtable = "Stats_v33"+str(data.movingWindow)+str(data.nEventsPerStat)
    initStats = {"Mean":[],
                 "Std":[]}
    stats = pd.DataFrame(data=initStats)
    
    struct = {"Feature":[],
             "Asset":[],
             "Mean":[],
             "Std":[]}
    dF = pd.DataFrame(data=struct)
    dF.to_sql(IDtable, conn, if_exists="append")
    
    # load stats per feature
    for nF in data.features:
        query = "SELECT Mean,Std FROM "+IDtable+" WHERE Asset="+"'"+ asset+"'"+" AND Feature='"+data.AllFeatures[str(nF)]+"'"
        #print(pd.read_sql_query(query,conn))
        stats = stats.append(pd.read_sql_query(query,conn))
    
    #print(pd.read_sql_query("SELECT * FROM "+IDtable,conn))
    #print("stats.shape")
    #print(stats.shape)
    # add output
    #query = "SELECT Mean,Std FROM "+IDtable+" WHERE Asset="+"'"+ asset+"'"+" AND Feature='Output'"
    #stats = stats.append(pd.read_sql_query(query,conn))
    
    IDtableOuts = "StatsOutputs_v33"+str(data.movingWindow)+str(data.nEventsPerStat)
    query = "SELECT Mean,Std FROM "+IDtableOuts+" WHERE Asset="+"'"+ asset+"'"+" AND lookAhead=="+str(
                float(data.lookAheadVector[data.lookAheadIndex]))
    #print("All stats:")
    #print(pd.read_sql_query("SELECT Mean,Std FROM "+IDtableOuts+" WHERE Asset="+"'"+ asset+"'",conn))
    msO = pd.read_sql_query(query,conn)
    #print("msO")
    #print(msO)
    stats = stats.append(msO)
    #print(stats)
    #print(stats)
    #print(statsO)
    conn.close()
    
    if stats.shape[0]==0:
        print("Error! Stats don't exist. Run saveFeatures with saveStats=1 first")
        error()
    
    return stats

def saveStatsInDB_v33(DB, data, asset, means, stds, meansO, stdsO):
    # Warning! Temporal implementation. Save stats table in DB too!!
    conn = sqlite3.connect(DB)
    curr = conn.cursor()
    # save stats in DB
    countFeats = 0
    IDtable = "Stats_v33"+str(data.movingWindow)+str(data.nEventsPerStat)
    
    #curr.execute("DROP TABLE IF EXISTS "+IDtable)
    
    stats = {"Feature":[],
             "Asset":[],
             "Mean":[],
             "Std":[]}
    dF = pd.DataFrame(data=stats)
    dF.to_sql(IDtable, conn, if_exists="append")
    
    for nF in data.features:
        stats = {}
        stats["Feature"] = data.AllFeatures[str(nF)]
        stats["Asset"] = asset
        stats["Mean"] = means[0,countFeats]
        stats["Std"] = stds[0,countFeats]
        dF = pd.DataFrame(data=stats,index=[0])
        query = "DELETE FROM "+ IDtable+" WHERE Asset="+"'"+ asset+"'"+" AND Feature='"+data.AllFeatures[str(nF)]+"'"
        #print(query)
        curr.execute(query)
        #print(dF)
        dF.to_sql(IDtable, conn, if_exists="append",index=False)
        #save_as_matfile(asset+IDtable+data.AllFeatures[str(nF)],asset+IDtable+data.AllFeatures[str(nF)],stats)
        countFeats+=1
    
    # Add output
    stats = {}
    stats["Feature"] = "Output"
    stats["Asset"] = asset
    stats["Mean"] = means[0,-1]
    stats["Std"] = stds[0,-1]
    dF = pd.DataFrame(data=stats,index=[0])
    
    IDtableOuts = "StatsOutputs_v33"+str(data.movingWindow)+str(data.nEventsPerStat)
    
    dF = pd.DataFrame(columns = ["lookAhead","Asset","Mean","Std"])
    dF.to_sql(IDtableOuts, conn, if_exists="append")
    
    dF.loc[:,"lookAhead"] = pd.Series(data.lookAheadVector).astype(float)
    dF.loc[:,"Asset"] = asset
    dF.loc[:,"Mean"] = pd.Series(meansO[0,:]).astype(float)
    dF.loc[:,"Std"] = pd.Series(stdsO[0,:]).astype(float)
    
    #print(dF)
    query = "DELETE FROM "+ IDtableOuts+" WHERE Asset="+"'"+ asset+"'"
    curr.execute(query)
    
    dF.to_sql(IDtableOuts, conn, if_exists="append",index=False)
    
    #print(pd.read_sql_query("SELECT * FROM "+IDtableOuts,conn))
    print("Stats saved in DB")
    curr.close()
    conn.close()
    #a=p
    return None

def saveFeatures_v33(data, asset, trainOrTest, 
                     DBforex, DBfeatures, DBDS, DBresults, DBstats, 
                     saveStats, newReturns=0):
    # Warning! New version not fully implemented!!! 
    print("\n")
    print('Loading '+ asset)
    #tic = time.time()
    if trainOrTest==data.train:
        tOt = "tr"
    else:
        tOt = "te"
    #conn = sqlite3.connect(DBforex)
    #conn.create_function("REGEXP", 2, regexp)
    
    connDS = sqlite3.connect(DBDS)
    # Load separators from DB
    separators = pd.read_sql_query("SELECT * FROM "+asset+tOt+"DISP",connDS).set_index("real_index")
    separators = extractFromSeparators_v20(separators, data.dateStart, data.dateEnd, asset, DBforex)
    connDS.close()
    
    allvariation = np.zeros([0,data.nFeatures])
    allRet = np.zeros([0,len(data.lookAheadVector)])
    tradeInfo = pd.DataFrame()
    
    conn = sqlite3.connect(DBfeatures)
    #curr = conn.cursor()
    resolution = int(data.nEventsPerStat/data.movingWindow)
    
    
    for s in range(0,len(separators)-1,2):#range(len(separators)-1)
        if separators.index[s+1]-separators.index[s]>=2*data.nEventsPerStat:
            print("Symbols batch {0:d} out of {1:d}".format(int(s/2),int(len(separators)/2-1)))
            print(separators.DateTime.iloc[s]+" to "+separators.DateTime.iloc[s+1])
            # get init and end dates of these separators
            sepInitDate = dt.datetime.strftime(dt.datetime.strptime(
                    separators.DateTime.iloc[s],'%Y.%m.%d %H:%M:%S'),'%y%m%d%H%M%S')
            sepEndDate = dt.datetime.strftime(dt.datetime.strptime(
                    separators.DateTime.iloc[s+1],'%Y.%m.%d %H:%M:%S'),'%y%m%d%H%M%S')
   
            # Load features from DB
            features, tableRets, DTs = extractFeaturesFromDB_v31(asset, sepInitDate, sepEndDate, data, DBfeatures)
            Returns = tableRets.as_matrix()
            # If features not saved in DB
            if features.shape[0]==0:
                # If tradeInfo not yet loaded
                if tradeInfo.shape[0]==0:
                    print("loading trade info from DB...")
                    tradeInfo = loadTradeInfoFromDB_v27(DBforex, DBDS, asset, tOt, data)
                    #print(tradeInfo)
                    # get features from raw data
                print("getting features from raw data...")
                #print(separators.index[s])
                #print(tradeInfo.loc[separators.index[s]:separators.index[s+1]+1])
                features, _, _ = extractFeatures_v31(tradeInfo.loc[separators.index[s]:separators.index[s+1]+1], data)
                #save_as_matfile("features","features",features)
                # Save features in DB
                if data.save_IO:
                    saveFeaturesToDB_v27(asset, sepInitDate, sepEndDate, data, features, Returns, DTs, DBfeatures)
            
            if newReturns:
                tableRets,DTs = extractResultsFromDB_v31(asset, sepInitDate, sepEndDate, data, DBresults)
                #print(tableRets)
                #print(DTs)
                Returns = tableRets.as_matrix()
                if tableRets.shape[0]==0:
                    if tradeInfo.shape[0]==0:
                        print("loading trade info from DB...")
                        tradeInfo = loadTradeInfoFromDB_v27(DBforex, DBDS, asset, tOt, data)
                    
                    print("getting outputs from raw data...")
                    Returns, DTs = extractReturns_v31(tradeInfo.loc[separators.index[s]:separators.index[s+1]+1],data)
                    saveResultsToDB_v31(asset, sepInitDate, sepEndDate, data, Returns, DTs, DBresults)
                
            variation = features[resolution:,:]-features[:-resolution,:]
            variation[:,data.noVarFeats] = features[:-resolution,data.noVarFeats]
            #save_as_matfile("variation","variation",variation)
            allvariation = np.append(allvariation,variation,axis=0)
            allRet = np.append(allRet,Returns,axis=0) 
            
        else:
            print("Skipping Symbols batch {0:d}. Not enough entries".format(int(s/2)))
    
    conn.close()
    # If train -> save stats in DB
    if saveStats==1 and tOt=="tr":
        #save_as_matfile(asset+"Features","feats",allvariation)
        means = np.mean(allvariation,axis=0,keepdims=1)
        stds = np.std(allvariation,axis=0,keepdims=1)  
        
        stdsO = np.std(allRet,axis=0,keepdims=1)
        meansO = np.mean(allRet,axis=0,keepdims=1)
        
        saveStatsInDB_v33(DBstats, data, asset, means, stds, meansO, stdsO)
        
    return None

def loadStatsFromDB_v32(DBstats, data, asset):
    
    conn = sqlite3.connect(DBstats)
    resolution = int(data.nEventsPerStat/data.movingWindow)
    IDtable = "Stats_v32_"+str(data.movingWindow)+"_"+str(data.nEventsPerStat)
    initStats = {"Mean":[],
                 "Std":[]}
    stats = pd.DataFrame(data=initStats)
    
    struct = {"Feature":[],
             "Asset":[],
             "Mean":[],
             "Std":[],
             "r":[]}
    
    dF = pd.DataFrame(data=struct)
    dF.to_sql(IDtable, conn, if_exists="append")
    
    # load stats per feature
    for r in range(resolution):
        for nF in data.features:
            query = ("SELECT Mean,Std FROM "+IDtable+" WHERE Asset="+"'"+ asset+"'"+
                     " AND Feature='"+data.AllFeatures[str(nF)]+"'"+
                     " AND r="+str(r))
            #print(pd.read_sql_query(query,conn))
            stats = stats.append(pd.read_sql_query(query,conn))
    
    #print(pd.read_sql_query("SELECT * FROM "+IDtable,conn))
    
    #print(stats)
    # add output
    #query = "SELECT Mean,Std FROM "+IDtable+" WHERE Asset="+"'"+ asset+"'"+" AND Feature='Output'"
    #stats = stats.append(pd.read_sql_query(query,conn))
    
    IDtableOuts = "StatsOutputs_v32"+str(data.movingWindow)+str(data.nEventsPerStat)
    query = "SELECT Mean,Std FROM "+IDtableOuts+" WHERE Asset="+"'"+ asset+"'"+" AND lookAhead=="+str(
                float(data.lookAheadVector[data.lookAheadIndex]))
    #print("All stats:")
    #print(pd.read_sql_query("SELECT Mean,Std FROM "+IDtableOuts+" WHERE Asset="+"'"+ asset+"'",conn))
    msO = pd.read_sql_query(query,conn)
    #print("msO")
    #print(msO)
    stats = stats.append(msO)
    #print(stats)
    #print(stats)
    #print(statsO)
    conn.close()
    
    if stats.shape[0]==0:
        print("Error! Stats don't exist. Run saveFeatures with saveStats=1 first")
        error()
    
    return stats

def saveStatsInDB_v32(DBstats, data, asset, means, stds, meansO, stdsO):
    # Warning! Temporal implementation. Save stats table in DB too!!
    conn = sqlite3.connect(DBstats)
    curr = conn.cursor()
    # save stats in DB
    resolution = int(data.nEventsPerStat/data.movingWindow)
    IDtable = "Stats_v32_"+str(data.movingWindow)+"_"+str(data.nEventsPerStat)
    
    stats = {"Feature":[],
             "Asset":[],
             "Mean":[],
             "Std":[],
             "r":[]}
    dF = pd.DataFrame(data=stats)
    dF.to_sql(IDtable, conn, if_exists="append")
    
    for r in range(resolution):
        countFeats = 0
        #curr.execute("DROP TABLE IF EXISTS "+IDtable)
        for nF in data.features:
            stats = {}
            stats["Feature"] = data.AllFeatures[str(nF)]
            stats["Asset"] = asset
            stats["Mean"] = means[r,countFeats]
            stats["Std"] = stds[r,countFeats]
            stats["r"] = r
            dF = pd.DataFrame(data=stats,index=[0])
            query = ("DELETE FROM "+ IDtable+" WHERE Asset="+"'"+ asset+"'"+
                     " AND Feature='"+data.AllFeatures[str(nF)]+"'"+
                     " AND r="+str(r))
            #print(query)
            curr.execute(query)
            #print(dF)
            dF.to_sql(IDtable, conn, if_exists="append",index=False)
            #save_as_matfile(asset+IDtable+data.AllFeatures[str(nF)],asset+IDtable+data.AllFeatures[str(nF)],stats)
            countFeats+=1
    
    #print(pd.read_sql_query("SELECT * FROM "+IDtable,conn))
    # Add output
    stats = {}
    stats["Feature"] = "Output"
    stats["Asset"] = asset
    stats["Mean"] = means[0,-1]
    stats["Std"] = stds[0,-1]
    dF = pd.DataFrame(data=stats,index=[0])
    
    IDtableOuts = "StatsOutputs_v32"+str(data.movingWindow)+str(data.nEventsPerStat)
    
    dF = pd.DataFrame(columns = ["lookAhead","Asset","Mean","Std"])
    dF.to_sql(IDtableOuts, conn, if_exists="append")
    
    dF.loc[:,"lookAhead"] = pd.Series(data.lookAheadVector).astype(float)
    dF.loc[:,"Asset"] = asset
    dF.loc[:,"Mean"] = pd.Series(meansO[0,:]).astype(float)
    dF.loc[:,"Std"] = pd.Series(stdsO[0,:]).astype(float)
    
    #print(dF)
    query = "DELETE FROM "+ IDtableOuts+" WHERE Asset="+"'"+ asset+"'"
    curr.execute(query)
    
    dF.to_sql(IDtableOuts, conn, if_exists="append",index=False)
    
    #print(pd.read_sql_query("SELECT * FROM "+IDtableOuts,conn))
    print("Stats saved in DB")
    curr.close()
    conn.close()
    #a=p
    return None

def saveFeatures_v32(data, asset, trainOrTest, 
                     DBforex, DBfeatures, DBDS, DBresults, DBstats, 
                     saveStats):
    # Warning! New version not fully implemented!!! 
    print("\n")
    print('Loading '+ asset)
    #tic = time.time()
    if trainOrTest==data.train:
        tOt = "tr"
    else:
        tOt = "te"
    #conn = sqlite3.connect(DBforex)
    #conn.create_function("REGEXP", 2, regexp)
    
    connDS = sqlite3.connect(DBDS)
    # Load separators from DB
    separators = pd.read_sql_query("SELECT * FROM "+asset+tOt+"DISP",connDS).set_index("real_index")
    #print(separators)
    separators = extractFromSeparators_v20(separators, data.dateStart, data.dateEnd, asset, DBforex)
    
    connDS.close()
    
    resolution = int(data.nEventsPerStat/data.movingWindow)
    allvariation = np.zeros([0,data.nFeatures])
    allvariation_v32 = np.zeros([0,data.nFeatures,resolution])
    allRet = np.zeros([0,10])
    allRet_v31 = np.zeros([0,len(data.lookAheadVector)])
    tradeInfo = pd.DataFrame()

    conn = sqlite3.connect(DBfeatures)
    
    for s in range(0,len(separators)-1,2):#range(len(separators)-1)
        if separators.index[s+1]-separators.index[s]>=2*data.nEventsPerStat:
            print("Symbols batch {0:d} out of {1:d}".format(int(s/2),int(len(separators)/2-1)))
            print(separators.DateTime.iloc[s]+" to "+separators.DateTime.iloc[s+1])
            # get init and end dates of these separators
            sepInitDate = dt.datetime.strftime(dt.datetime.strptime(
                    separators.DateTime.iloc[s],'%Y.%m.%d %H:%M:%S'),'%y%m%d%H%M%S')
            sepEndDate = dt.datetime.strftime(dt.datetime.strptime(
                    separators.DateTime.iloc[s+1],'%Y.%m.%d %H:%M:%S'),'%y%m%d%H%M%S')
   
            # Load features from DB
            features, tableRets, DTs = extractFeaturesFromDB_v31(asset, sepInitDate, sepEndDate, data, DBfeatures)
            Returns = tableRets.as_matrix()
            # If features not saved in DB
            if features.shape[0]==0:
                # If tradeInfo not yet loaded
                if tradeInfo.shape[0]==0:
                    print("loading trade info from DB...")
                    tradeInfo = loadTradeInfoFromDB_v27(DBforex, DBDS, asset, tOt, data)
                    #print(tradeInfo)
                    # get features from raw data
                print("getting features from raw data...")
                #print(separators.index[s])
                #print(tradeInfo.loc[separators.index[s]:separators.index[s+1]+1])
                features, Returns, DTs = extractFeatures_v31(tradeInfo.loc[separators.index[s]:separators.index[s+1]+1], data)
                #save_as_matfile("features","features",features)
                # Save features in DB
                if data.save_IO:
                    saveFeaturesToDB_v27(asset, sepInitDate, sepEndDate, data, features, Returns, DTs, DBfeatures)
            
            tableRets,DTs_v31 = extractResultsFromDB_v31(asset, sepInitDate, sepEndDate, data, DBresults)
            #print(tableRets)
            #print(DTs)
            Returns_v31 = tableRets.as_matrix()
            if tableRets.shape[0]==0:
                if tradeInfo.shape[0]==0:
                    print("loading trade info from DB...")
                    tradeInfo = loadTradeInfoFromDB_v27(DBforex, DBDS, asset, tOt, data)
                
                print("getting outputs from raw data...")
                Returns_v31, DTs_v31 = extractReturns_v31(tradeInfo.loc[separators.index[s]:separators.index[s+1]+1],data)
                saveResultsToDB_v31(asset, sepInitDate, sepEndDate, data, Returns_v31, DTs_v31, DBresults)
                
            variation = features[1:,:]-features[:-1,:]
            variation[:,data.noVarFeats] = features[:-1,data.noVarFeats]
            #print("variation shape "+str(variation.shape))
            #variation_v32 = np.zeros((features.shape[0]-resolution,data.nFeatures,resolution))
            variation_v32 = 999+np.zeros((features.shape[0],data.nFeatures,resolution))
            #print("variation_v32 shape "+str(variation_v32.shape))
            for r in range(resolution):
                #print(r)
                #variation_v32[:,:,r] = (features[resolution:,:]
                #                        -features[resolution-(r+1):-(r+1),:])
                
                variation_v32[r+1:,:,r] = features[r+1:,:]-features[:-(r+1),:]
                #if r!=0:
                variation_v32[r+1:,data.noVarFeats,r] = features[:-(r+1),data.noVarFeats]
                #else:
                #    variation_v32[:,data.noVarFeats,r] = features[resolution-(r+1)+1:,data.noVarFeats]
            #save_as_matfile("variation","variation",variation)
            allvariation = np.append(allvariation,variation,axis=0)
            
            allvariation_v32 = np.append(allvariation_v32,variation_v32,axis=0)
            #print(Returns.shape)
            allRet = np.append(allRet,Returns,axis=0) 
            allRet_v31 = np.append(allRet_v31,Returns_v31,axis=0)
            
        else:
            print("Skipping Symbols batch {0:d}. Not enough entries".format(int(s/2)))
    
    conn.close()
    # If train -> save stats in DB
    if saveStats==1 and tOt=="tr":
        #print(allvariation.shape)
        means = np.mean(allvariation,axis=0,keepdims=1)
        stds = np.std(allvariation,axis=0,keepdims=1) 
        
        stdsO = np.std(allRet,axis=0,keepdims=1)
        meansO = np.mean(allRet,axis=0,keepdims=1)
        
        means_v32 = np.zeros((resolution,data.nFeatures))
        stds_v32 = np.zeros((resolution,data.nFeatures))
        
        for r in range(resolution):
            nonZeros = allvariation_v32[:,0,r]!=999
            #print(np.sum(nonZeros))
            means_v32[r,:] = np.mean(allvariation_v32[nonZeros,:,r],axis=0,keepdims=1)
            stds_v32[r,:] = np.std(allvariation_v32[nonZeros,:,r],axis=0,keepdims=1)  
#            if r==0:
#                print(means-means_v32[r,:])
#                print(stds-stds_v32[r,:])
        
        #print("stdsO")
        #print(stdsO)
        #save_as_matfile("allRet","allRet",allRet)
        #saveStatsInDB_v27(DBfeatures, data, asset, means, stds, meansO, stdsO)
        
        
        stdsO_v31 = np.std(allRet_v31,axis=0,keepdims=1)
        meansO_v31 = np.mean(allRet_v31,axis=0,keepdims=1)
        #print("stdsO_v31")
        #print(stdsO_v31)
        #save_as_matfile("allRet_v31","allRet_v31",allRet_v31)
        saveStatsInDB_v32(DBstats, data, asset, means_v32, stds_v32, meansO_v31, stdsO_v31)
        
    return None

def extractFeaturesFromDB_v31(asset, Init, End, data, DB, logFile=""):
    
    features = np.array([])
    Returns = pd.DataFrame()
    DTs = pd.DataFrame()
    conn = sqlite3.connect(DB)
    #curr = conn.cursor()
    
    IDtableO_v30 = asset+"Output_v30"+Init+End+"{0:05d}".format(
                data.nEventsPerStat)+"{0:05d}".format(data.movingWindow)
    #print(IDtable)
    dictFromRets={str(i+1):[] for i in range(10)}
    dataFrame = pd.DataFrame(data=dictFromRets)
    dataFrame.to_sql(IDtableO_v30, conn, if_exists="append")

    df_v30 = pd.read_sql("SELECT * FROM "+IDtableO_v30, conn)
    
    #IDtableO = asset+"Output"+Init+End+"{0:05d}".format(
    #            data.nEventsPerStat)+"{0:05d}".format(data.movingWindow)
    
    
    #df = pd.read_sql("SELECT * FROM "+IDtableO, conn)
    if df_v30.shape[0]>0:
        df_v30 = df_v30.drop('index', 1)
        #print(df)
        features = np.zeros((df_v30.shape[0],len(data.features)))
        Returns = df_v30
        #print(df_v30)
        
        IDtable = asset+"DT_v30"+Init+End+"{0:05d}".format(
                data.nEventsPerStat)+"{0:05d}".format(data.movingWindow)
        
        DTs = pd.read_sql("SELECT * FROM "+IDtable, conn).set_index("real_index")
        #print(DTs)
    counter = 0
    struct = {'Value':[]}
    for f in data.features:
        IDtable = asset+data.AllFeatures[str(f)]+Init+End+"{0:05d}".format(
                data.nEventsPerStat)+"{0:05d}".format(data.movingWindow)
        #print(IDtable)
        dataFrame = pd.DataFrame(data=struct)
        dataFrame.to_sql(IDtable, conn, if_exists="append")
    
        df = pd.read_sql("SELECT * FROM "+IDtable, conn)
        #print(df)
        #print(df.shape)
    
        if df.shape[0]>0:
            df = df.drop('index', 1)
            #print(df.as_matrix().shape)
            feat = df.as_matrix()
            #print(feat.shape)
            features[:,counter] = feat[:,0]
            counter += 1
    
   # print(features)
    #curr.execute("DROP TABLE IF EXISTS "+IDtable)
    conn.close()
#    if features.shape[0]>0:
#        print("Features loaded from DB")
    #print(features)
    return features, Returns, DTs

def extractResultsFromDB_v31(asset, Init, End, data, DB, logFile=""):
    
    #features = np.array([])
    Returns = pd.DataFrame()
    DTs = pd.DataFrame()
    conn = sqlite3.connect(DB)
    #curr = conn.cursor()
    
    IDtableO_v30 = asset+"Output_v31"+Init+End+"{0:05d}".format(
                data.nEventsPerStat)+"{0:05d}".format(data.movingWindow)
    #print(IDtable)
    dictFromRets={str(i+1):[] for i in range(10)}
    dataFrame = pd.DataFrame(data=dictFromRets)
    dataFrame.to_sql(IDtableO_v30, conn, if_exists="append")

    df_v30 = pd.read_sql("SELECT * FROM "+IDtableO_v30, conn)
    
    #IDtableO = asset+"Output"+Init+End+"{0:05d}".format(
    #            data.nEventsPerStat)+"{0:05d}".format(data.movingWindow)
    
    
    #df = pd.read_sql("SELECT * FROM "+IDtableO, conn)
    if df_v30.shape[0]>0:
        df_v30 = df_v30.drop('index', 1)
        #print(df)
        Returns = df_v30
        #print(df_v30)
        
        IDtable = asset+"DT_v31"+Init+End+"{0:05d}".format(
                data.nEventsPerStat)+"{0:05d}".format(data.movingWindow)
        
        DTs = pd.read_sql("SELECT * FROM "+IDtable, conn).set_index("real_index")
        #print(DTs)
    else:
        print(IDtableO_v30)
    #curr.execute("DROP TABLE IF EXISTS "+IDtable)
    conn.close()
#    if Returns.shape[0]>0:
#        print("Returns loaded from DB")
    #print(features)
    return Returns, DTs

def loadStatsFromDB_v31(DBfeatures, data, asset, trainOrTest):
    
    conn = sqlite3.connect(DBfeatures)
    IDtable = "Stats_v31"+str(data.movingWindow)+str(data.nEventsPerStat)
    initStats = {"Mean":[],
                 "Std":[]}
    stats = pd.DataFrame(data=initStats)
    
    struct = {"Feature":[],
             "Asset":[],
             "Mean":[],
             "Std":[]}
    dF = pd.DataFrame(data=struct)
    dF.to_sql(IDtable, conn, if_exists="append")
    
    # load stats per feature
    for nF in data.features:
        query = "SELECT Mean,Std FROM "+IDtable+" WHERE Asset="+"'"+ asset+"'"+" AND Feature='"+data.AllFeatures[str(nF)]+"'"
        #print(pd.read_sql_query(query,conn))
        stats = stats.append(pd.read_sql_query(query,conn))
    
    #print(pd.read_sql_query("SELECT * FROM "+IDtable,conn))
    #print("stats.shape")
    #print(stats.shape)
    # add output
    #query = "SELECT Mean,Std FROM "+IDtable+" WHERE Asset="+"'"+ asset+"'"+" AND Feature='Output'"
    #stats = stats.append(pd.read_sql_query(query,conn))
    
    IDtableOuts = "StatsOutputs_v31"+str(data.movingWindow)+str(data.nEventsPerStat)
    query = "SELECT Mean,Std FROM "+IDtableOuts+" WHERE Asset="+"'"+ asset+"'"+" AND lookAhead=="+str(
                float(data.lookAheadVector[data.lookAheadIndex]))
    #print("All stats:")
    #print(pd.read_sql_query("SELECT Mean,Std FROM "+IDtableOuts+" WHERE Asset="+"'"+ asset+"'",conn))
    msO = pd.read_sql_query(query,conn)
    #print("msO")
    #print(msO)
    stats = stats.append(msO)
    #print(stats)
    #print(stats)
    #print(statsO)
    conn.close()
    
    if stats.shape[0]==0:
        print("Error! Stats don't exist. Run saveFeatures with saveStats=1 first")
        error()
    
    return stats

def saveStatsInDB_v31(DB, data, asset, means, stds, meansO, stdsO):
    # Warning! Temporal implementation. Save stats table in DB too!!
    conn = sqlite3.connect(DB)
    curr = conn.cursor()
    # save stats in DB
    countFeats = 0
    IDtable = "Stats_v31"+str(data.movingWindow)+str(data.nEventsPerStat)
    
    #curr.execute("DROP TABLE IF EXISTS "+IDtable)
    
    stats = {"Feature":[],
             "Asset":[],
             "Mean":[],
             "Std":[]}
    dF = pd.DataFrame(data=stats)
    dF.to_sql(IDtable, conn, if_exists="append")
    
    for nF in data.features:
        stats = {}
        stats["Feature"] = data.AllFeatures[str(nF)]
        stats["Asset"] = asset
        stats["Mean"] = means[0,countFeats]
        stats["Std"] = stds[0,countFeats]
        dF = pd.DataFrame(data=stats,index=[0])
        query = "DELETE FROM "+ IDtable+" WHERE Asset="+"'"+ asset+"'"+" AND Feature='"+data.AllFeatures[str(nF)]+"'"
        #print(query)
        curr.execute(query)
        #print(dF)
        dF.to_sql(IDtable, conn, if_exists="append",index=False)
        #save_as_matfile(asset+IDtable+data.AllFeatures[str(nF)],asset+IDtable+data.AllFeatures[str(nF)],stats)
        countFeats+=1
    
    # Add output
    stats = {}
    stats["Feature"] = "Output"
    stats["Asset"] = asset
    stats["Mean"] = means[0,-1]
    stats["Std"] = stds[0,-1]
    dF = pd.DataFrame(data=stats,index=[0])
    
    IDtableOuts = "StatsOutputs_v31"+str(data.movingWindow)+str(data.nEventsPerStat)
    
    dF = pd.DataFrame(columns = ["lookAhead","Asset","Mean","Std"])
    dF.to_sql(IDtableOuts, conn, if_exists="append")
    
    dF.loc[:,"lookAhead"] = pd.Series(data.lookAheadVector).astype(float)
    dF.loc[:,"Asset"] = asset
    dF.loc[:,"Mean"] = pd.Series(meansO[0,:]).astype(float)
    dF.loc[:,"Std"] = pd.Series(stdsO[0,:]).astype(float)
    
    #print(dF)
    query = "DELETE FROM "+ IDtableOuts+" WHERE Asset="+"'"+ asset+"'"
    curr.execute(query)
    
    dF.to_sql(IDtableOuts, conn, if_exists="append",index=False)
    
    #print(pd.read_sql_query("SELECT * FROM "+IDtableOuts,conn))
    print("Stats saved in DB")
    curr.close()
    conn.close()
    #a=p
    return None

def saveResultsToDB_v31(asset, Init, End, data, Returns, DTs, DBresults, logFile=""):

    conn = sqlite3.connect(DBresults)


    # Save output
    IDtable = asset+"Output_v31"+Init+End+"{0:05d}".format(
                data.nEventsPerStat)+"{0:05d}".format(data.movingWindow)
    #print(IDtable)
    #print(IDtable)
    #curr.execute("DROP TABLE IF EXISTS "+IDtable)
    #print(features.shape)
    #print(features[f,:])
    dictFromRets={str(i+1):Returns[:,i] for i in range(Returns.shape[1])}
    dataFrame = pd.DataFrame(data=dictFromRets)
    #print(dataFrame)
    dataFrame.to_sql(IDtable, conn, if_exists="replace")
    
    IDtable = asset+"DT_v31"+Init+End+"{0:05d}".format(
                data.nEventsPerStat)+"{0:05d}".format(data.movingWindow)
    
    DTs.to_sql(IDtable, conn, index_label="real_index",if_exists="replace")
    print("Results saved in DB")
    if logFile!="":
        file = open(logFile,"a")
        file.write("Features saved in DB"+"\n")
        file.close()
    
    conn.close()
    return None

def saveFeatures_v31(data, asset, trainOrTest, 
                     DBforex, DBfeatures, DBDS, DBresults, DBstats, 
                     saveStats, newReturns=0):
    # Warning! New version not fully implemented!!! 
    print("\n")
    print('Loading '+ asset)
    #tic = time.time()
    if trainOrTest==data.train:
        tOt = "tr"
    else:
        tOt = "te"
    #conn = sqlite3.connect(DBforex)
    #conn.create_function("REGEXP", 2, regexp)
    
    connDS = sqlite3.connect(DBDS)
    # Load separators from DB
    separators = pd.read_sql_query("SELECT * FROM "+asset+tOt+"DISP",connDS).set_index("real_index")
    separators = extractFromSeparators_v20(separators, data.dateStart, data.dateEnd, asset, DBforex)
    connDS.close()
    
    allvariation = np.zeros([0,data.nFeatures])
    allRet = np.zeros([0,10])
    allRet_v31 = np.zeros([0,len(data.lookAheadVector)])
    tradeInfo = pd.DataFrame()
    
    conn = sqlite3.connect(DBfeatures)
    #curr = conn.cursor()
    
    for s in range(0,len(separators)-1,2):#range(len(separators)-1)
        if separators.index[s+1]-separators.index[s]>=2*data.nEventsPerStat:
            print("Symbols batch {0:d} out of {1:d}".format(int(s/2),int(len(separators)/2-1)))
            print(separators.DateTime.iloc[s]+" to "+separators.DateTime.iloc[s+1])
            # get init and end dates of these separators
            sepInitDate = dt.datetime.strftime(dt.datetime.strptime(
                    separators.DateTime.iloc[s],'%Y.%m.%d %H:%M:%S'),'%y%m%d%H%M%S')
            sepEndDate = dt.datetime.strftime(dt.datetime.strptime(
                    separators.DateTime.iloc[s+1],'%Y.%m.%d %H:%M:%S'),'%y%m%d%H%M%S')
   
            # Load features from DB
            features, tableRets, DTs = extractFeaturesFromDB_v31(asset, sepInitDate, sepEndDate, data, DBfeatures)
            Returns = tableRets.as_matrix()
            # If features not saved in DB
            if features.shape[0]==0:
                # If tradeInfo not yet loaded
                if tradeInfo.shape[0]==0:
                    print("loading trade info from DB...")
                    tradeInfo = loadTradeInfoFromDB_v27(DBforex, DBDS, asset, tOt, data)
                    #print(tradeInfo)
                    # get features from raw data
                print("getting features from raw data...")
                #print(separators.index[s])
                #print(tradeInfo.loc[separators.index[s]:separators.index[s+1]+1])
                features, Returns, DTs = extractFeatures_v31(tradeInfo.loc[separators.index[s]:separators.index[s+1]+1], data)
                #save_as_matfile("features","features",features)
                # Save features in DB
                if data.save_IO:
                    saveFeaturesToDB_v27(asset, sepInitDate, sepEndDate, data, features, Returns, DTs, DBfeatures)
            
            if newReturns:
                tableRets,DTs_v31 = extractResultsFromDB_v31(asset, sepInitDate, sepEndDate, data, DBresults)
                #print(tableRets)
                #print(DTs)
                Returns_v31 = tableRets.as_matrix()
                if tableRets.shape[0]==0:
                    if tradeInfo.shape[0]==0:
                        print("loading trade info from DB...")
                        tradeInfo = loadTradeInfoFromDB_v27(DBforex, DBDS, asset, tOt, data)
                    
                    print("getting outputs from raw data...")
                    Returns_v31, DTs_v31 = extractReturns_v31(tradeInfo.loc[separators.index[s]:separators.index[s+1]+1],data)
                    saveResultsToDB_v31(asset, sepInitDate, sepEndDate, data, Returns_v31, DTs_v31, DBresults)
                
            variation = features[1:features.shape[0],:]-features[0:features.shape[0]-1,:]
            variation[:,data.noVarFeats] = features[:-1,data.noVarFeats]
            #save_as_matfile("variation","variation",variation)
            allvariation = np.append(allvariation,variation,axis=0)
            #print(Returns.shape)
            allRet = np.append(allRet,Returns,axis=0) 
            allRet_v31 = np.append(allRet_v31,Returns_v31,axis=0) 
            
        else:
            print("Skipping Symbols batch {0:d}. Not enough entries".format(int(s/2)))
    
    conn.close()
    # If train -> save stats in DB
    if saveStats==1 and tOt=="tr":
        #save_as_matfile(asset+"Features","feats",allvariation)
        means = np.mean(allvariation,axis=0,keepdims=1)
        stds = np.std(allvariation,axis=0,keepdims=1)  
        
        stdsO = np.std(allRet,axis=0,keepdims=1)
        meansO = np.mean(allRet,axis=0,keepdims=1)
        #print("stdsO")
        #print(stdsO)
        #save_as_matfile("allRet","allRet",allRet)
        saveStatsInDB_v27(DBfeatures, data, asset, means, stds, meansO, stdsO)
        
        if newReturns:
            stdsO_v31 = np.std(allRet_v31,axis=0,keepdims=1)
            meansO_v31 = np.mean(allRet_v31,axis=0,keepdims=1)
            #print("stdsO_v31")
            #print(stdsO_v31)
            #save_as_matfile("allRet_v31","allRet_v31",allRet_v31)
            saveStatsInDB_v31(DBstats, data, asset, means, stds, meansO_v31, stdsO_v31)
        
    return None

def delete_entry_TR(ID):
    TR = pickle.load( open( "../results/TR.p", "rb" ))
    for epoch in range(TR[ID]):
        del TR[ID+str(epoch)]
        
    del TR[ID]
    pickle.dump(TR, open( "../results/TR.p", "wb" ) )
    print("Entry deleted and TR updated")
    return None

def flush_tableResults():
    results = {}
    pickle.dump(results, open( "../results/tableResults.p", "wb" ) )
    print("Results reseted")
    
    return None

def save_as_matfile(filename,varname,var):
    
    sio.savemat("../MATLAB/"+filename+'.mat', {varname:var})
    print('MAT file saved')
    
    return None

def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y
