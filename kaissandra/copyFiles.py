# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 08:18:21 2017

@author: mgutierrez
"""

import re
import os
import pandas as pd
import datetime as dt

from kaissandra.config import Config as C

#
destiny = 'D:/SDC/py/Data_test/'
origin = 'D:/SDC/py/Data_real_trader/Data_aws_200403/'#'C:/Users/mgutierrez/AppData/Roaming/MetaQuotes/Terminal/D0E8209F77C8CF37AD8BF550E51FF075/MQL5/Files/Data/'


if os.path.isdir(destiny)==False:
    os.mkdir(destiny)

copyFrom = '2019.08.05 00:00:00'
copyUntil = '2020.12.31 23:59:59'
copyFromDT = dt.datetime.strptime(copyFrom,'%Y.%m.%d %H:%M:%S')
copyUntilDT = dt.datetime.strptime(copyUntil,'%Y.%m.%d %H:%M:%S')
assets = [1,2,3,4,7,8,10,11,12,13,14,16,17,19,27,28,29,30,31,32]

for i in assets:
    thisAsset = C.AllAssets[str(i)]
    print(thisAsset)
    directOrigin = origin+thisAsset+'/'
    directDestiny = destiny+thisAsset+'/'
    
    if os.path.isdir(directDestiny)==False:
        os.mkdir(directDestiny)
    
    newFiles = []
    if os.path.isdir(directOrigin):
        listAllDir = sorted(os.listdir(directOrigin))
        listDirDestiny = sorted(os.listdir(directDestiny))
        listDir = []
        for l in range(len(listAllDir)):
            
            thisFile = listAllDir[l]
            
            m = re.search('^'+thisAsset+'_\d+'+'.txt$',thisFile)

            if m!=None:
                #print(m.group())
                fileTime = re.search('\d+',m.group()).group()
                #print(fileTime)
                fileTimeDT = dt.datetime.strptime(fileTime,'%Y%m%d%H%M%S')
                #dt.datetime.strftime(dt.datetime.strptime(m.group(),'%Y.%m.%d %H:%M:%S'),'%y%m%d%H%M%S')
                if fileTimeDT-copyFromDT>dt.timedelta(0) and copyUntilDT-fileTimeDT>=dt.timedelta(0):
                    listDir.append(thisFile)
            
        for fileID in listDir: 
            skip = 0
            if 1:#os.path.exists(directDestiny+fileID)==False:
                try:
                    print(fileID)
                    
                    newInfo = pd.read_csv(directOrigin+fileID)#, encoding='utf_16_le'
                    #newInfo.to_csv(direct+fileID,float_format='%.5f',index=False)
                    #print("File read in ascii")
                except (UnicodeDecodeError,OSError):
                    #print("This is unicode")
                    try:
                        newInfo = pd.read_csv(directOrigin+fileID, encoding='utf_16_le')
                        #os.remove(directOrigin+fileID)
                        print("File read in unicode")
                    except :
                        print("Skipping and deleting file")
                        os.remove(directOrigin+fileID)
                        break
                except :
                    print("Skipping and deleting file")
                    os.remove(directOrigin+fileID)
                    break
                if not skip and newInfo.shape[0]>0:
                    newInfo.to_csv(directDestiny+fileID,float_format='%.5f',index=False)
                else:
                    os.remove(directOrigin+fileID)
                    print("Zero number of entries. Skipped and deleted")