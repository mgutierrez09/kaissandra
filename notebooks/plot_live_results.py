# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 10:17:00 2019

@author: mgutierrez
"""

import sys
import os

this_path = os.getcwd()
path = '\\'.join(this_path.split('\\')[:-1])+'\\'
if path not in sys.path:
    sys.path.insert(0, path)
    print(path+" added to python path")
else:
    print(path+" already in python path")
    
import pandas as pd
import datetime as dt
#import pickle
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def get_positions_filename(asset, open_dt, close_dt):
    """  """
    
    dt_open = dt.datetime.strftime(dt.datetime.strptime(
            open_dt,'%Y.%m.%d %H:%M:%S'),'%y%m%d%H%M%S')
    dt_close = dt.datetime.strftime(dt.datetime.strptime(
            close_dt,'%Y.%m.%d %H:%M:%S'),'%y%m%d%H%M%S')
    filename = 'O'+dt_open+'C'+dt_close+asset
    return filename

live = False
plot = True

#config_names = ['T0004','T0005','T0006','T0007','T0008']
#start_times = ['19_02_19_14_23_40','19_02_19_14_23_40','19_02_19_14_23_40','19_02_19_14_23_40','19_02_19_14_23_40']#,'19_02_15_17_30_37'

#config_names = ['T0009','T0010','T0011','T0005']
#start_times = ['19_02_21_20_34_40','19_02_21_20_34_40','19_02_21_20_34_40','19_02_19_14_23_40']

config_names = ['T0012','T0013','T0014']
start_times = ['19_02_23_14_35_10','19_02_23_14_35_10','19_02_23_14_35_10']

#config_names = ['T0015']
#start_times = ['19_02_24_13_07_31']

#config_names = ['T0015','T0016','T0017']
#start_times = ['19_02_24_18_27_02','19_02_24_18_27_02','19_02_24_18_27_02']

if live:
    ext = '_LI_'
    directory = 'kaissandra_live/live'
    start_time = '19_01_27_23_18_41'
else:
    ext = '_BT_'
    directory = 'back_test'
    #start_time = '19_02_15_17_30_37'#'19_02_19_14_23_40'
results_dir = '../../RNN/resultsLive/'+directory+'/trader/'
pos_dirname = results_dir
#pos_filename = start_time+ext+"positions_soll.log"
filenames = [results_dir+start_times[i]+ext+config_names[i]+"positions_soll.log" for i in range(len(config_names))]
positions_list =  [pd.read_csv(filename).sort_values(by=['Entry Time']).reset_index().drop(labels='index',axis=1) for filename in filenames]
#print(positions)

for i in range(len(config_names)):
    #print(positions.GROI.sum())
    #print(positions.ROI.sum())
    #print(positions.Profit.sum())
    #print(positions.Profit)
    config_name = config_names[i]
    print(config_name)
    positions = positions_list[i]
    
    tgroi = positions['GROI'].sum()
    troi = positions['ROI'].sum()
    tprofit = positions['Profit'].sum()
    mspread = positions['Spread'].mean()
    emspread = positions['E_spread'].mean()
    SIs = positions.agg({'GROI': lambda x: 100*np.sum(x>0)/len(x),
                          'ROI': lambda x: 100*np.sum(x>0)/len(x)})
    if len(positions[positions['GROI']>0]['GROI'])>0:
        av_win = sum(positions[positions['GROI']>0]['GROI'])/len(positions[positions['GROI']>0]['GROI'])
    else:
        av_win = 0
    if len(positions[positions['GROI']<0]['GROI'])>0:
        av_lose = sum(positions[positions['GROI']<0]['GROI'])/len(positions[positions['GROI']<0]['GROI'])
    else: 
        av_lose = 0
    if positions.shape[0]>0:
        per_under_2p = 100*positions[positions['Spread']<0.02].shape[0]/positions.shape[0]
    else:
        per_under_2p = 0

    print("Total GROI = {0:.2f}% ".format(tgroi)+"Total ROI = {0:.2f}% ".format(troi)+\
          "total profit = {0:.2f}e ".format(tprofit)+"mean spread = {0:.2f} pips ".format(100*mspread)+
          "expected mean spread = {0:.2f} pips ".format(100*emspread))
    print("Number entries "+str(positions.shape[0])+" GSP = {0:.2f}% ".format(SIs['GROI'])\
          +" NSP = {0:.2f}% ".format(SIs['ROI'])+" av win = {0:.3f}% ".format(av_win)+" av lose = {0:.3f}% ".format(av_lose))
    print("Percent below 2p {0:.2f}%".format(per_under_2p))
    print(positions['ROI'].max())
    print(positions['ROI'].min())
    #print(pd.DataFrame(prob_rois))
    grouped = positions.groupby(['Asset'])
    #print(grouped.get_group(('GBPJPY',1)).to_string())
    #grouped.aggregate(np.sum)
    #grouped['groi'].describe()
    #grouped.get_group('GBPJPY')
    SI = grouped.agg({'GROI': lambda x: np.sum(x>0)/len(x),
                 'ROI': lambda x: np.sum(x>0)/len(x)})#.rename(['GSI','NSI'])
    grouped.agg({'E_spread': lambda x: 100*np.sum(x<0.02)/len(x)})
    #print(SI)
    #print(grouped.describe())
    #for name, group in grouped:
        #print(name)
    #    print(group.to_string())
    if plot:
        plt.figure(i)
        gran = 100
        min_h = int(gran*np.floor(positions['GROI'].min()))
        max_h = int(gran*np.ceil(positions['GROI'].max()))
        bins = [i/gran for i in range(min_h,max_h,2)]
        histG = plt.hist(positions['GROI'], bins=bins)
        histR = plt.hist(positions['ROI'], bins=bins)
        plt.grid()
        plt.title(config_name)
    
        #pos_under_thr.index = range(pos_under_thr.shape[0])
        plt.figure(100)
        #plt.plot(range(positions.shape[0]),positions['GROI'].cumsum())
        plt.plot(range(positions.shape[0]),positions['ROI'].cumsum(),label=config_name)
        plt.grid()
        plt.legend()
    
        list_dates = [dt.datetime.strptime(date, '%Y.%m.%d %H:%M:%S') for date in positions['Entry Time']]
        dates = matplotlib.dates.date2num(list_dates)
        plt.figure(101)
    #    plt.plot_date(list_dates, positions['GROI'].cumsum(),fmt='-')
        plt.plot_date(list_dates, positions['ROI'].cumsum(),fmt='-',label=config_name)
        plt.gcf().autofmt_xdate()
        plt.grid()
        plt.legend()

print(positions_list[-1].sort_values(by=['ROI'],ascending=True).reset_index())