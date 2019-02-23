# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 17:02:35 2019

@author: mgutierrez
"""

import pandas as pd
from kaissandra.results2 import get_extended_results
import datetime as dt
import matplotlib

results_dir  = '../../RNN/results/'
ext = '.csv'

#list_results_names = ['100520ALR20']
#list_epochs = [40]
#list_tis = [1]
#list_mcs = [.9]
#list_mds = [.55]

#list_results_names = ['100520BSR20']
#list_epochs = [32]
#list_tis = [2]
#list_mcs = [.6]
#list_mds = [.7]

#list_results_names = ['100520ALR20','100520BSR20']
#list_epochs = [40,32]
#list_tis = [1,2]
#list_mcs = [.9,.6]
#list_mds = [.55,.7]

#list_results_names = ['100520ALR20','100520BS']
#list_epochs = [58,18]
#list_tis = [2,3]
#list_mcs = [.9,.8]
#list_mds = [.6,.6]

#list_results_names = ['100520BS']#['100520ALR20','100520BS']
#list_epochs = [18]#[58,18]
#list_tis = [3]#[2,3]
#list_mcs = [.8]#[.9,.8]
#list_mds = [.6]#[.6,.6]

#list_results_names = ['100350S','100350L']+['100327S','100500L']#
#list_epochs = [13,6]+[21,29]
#list_tis = [3,1]+[0,2]
#list_mcs = [.55,.65,.75,.7]
#list_mds = [.6,.65,.7,.7]

#list_results_names = ['100350S','100350L']
#list_epochs = [13,6]
#list_tis = [3,1]
#list_mcs = [.55,.65]
#list_mds = [.6,.65]

list_results_names = ['100540ACR20']
list_epochs = [25]
list_tis = [0]
list_mcs = [.9]
list_mds = [.6]

pip_limit = 0.025

Journal = pd.DataFrame()
for l,name in enumerate(list_results_names):
    journal_filename = results_dir+name+'/journal/J_E'+str(list_epochs[l])+'TI'+str(list_tis[l])+'MC'+str(list_mcs[l])+'MD'+str(list_mds[l])+ext
    new_journal = pd.read_csv(journal_filename,sep='\t')
    #if list_results_names[l]=='100350NROI':
    #    new_journal = new_journal[new_journal['Bet']<0]
    Journal = Journal.append(new_journal).sort_values(by=['Asset','DTi']).reset_index().drop(labels='level_0',axis=1)
    print(journal_filename)
    


# filter journal
#pip_limit = 0.02
#pos_under_2p = positions['espread']<pip_limit
print("shape")
print(Journal.shape[0])
print(journal_filename)
n_days = 33*5+1
pos_dirname = '../../RNN/results/MERGED/positions/'
pos_filename = '_'.join([entry for entry in list_results_names])
positions_summary, log = get_extended_results(Journal,
                                    5,
                                    n_days,
                                   get_positions=True,pos_dirname=pos_dirname,
                                   pos_filename=pos_filename+'.csv')
#print(positions_summary)

positions = pd.read_csv(pos_dirname+pos_filename+'.csv',sep='\t')
#print(positions)

pos_under_2p = positions['espread']<pip_limit
positions['DTo'] = positions["Do"] +" "+ positions["To"]
pos_under_thr = positions[pos_under_2p]#.sort_values(by=['DTo'])
per_under_2p = 100*sum(pos_under_2p)/positions.shape[0]
tgsr = 100*sum(positions['GROI']>0)/positions.shape[0]
gsr = 100*sum(pos_under_thr['GROI']>0)/sum(pos_under_2p)
tsr = 100*sum(positions['ROI']>0)/positions.shape[0]
sr = 100*sum(positions[pos_under_2p]['ROI']>0)/sum(pos_under_2p)
mean_spread = positions[pos_under_2p]['spread'].mean()
print("total mean GROI")
print(positions['GROI'].mean())
print("mean GROI of selected")
print(positions[pos_under_2p]['GROI'].mean())
print("mean_spread of selected")
print(mean_spread)
print("Number of pos under "+str(pip_limit))
print(positions[pos_under_2p].shape[0])
print("per under pip_limit")
print(per_under_2p)
print("total gross success rate")
print(tgsr)
print("gross success rate")
print(gsr)
print("total success rate")
print(tsr)
print("success rate")
print(sr)
print("GROI for positions under "+str(pip_limit))
print(positions[pos_under_2p]['GROI'].sum())
print("ROI for positions under "+str(pip_limit))
print(positions[pos_under_2p]['ROI'].sum())
print("positions['GROI'].sum()-pip_limit*positions['GROI'].shape[0]")
print(positions['GROI'].sum()-pip_limit*positions['GROI'].shape[0])
print("# Assets")
print(positions['Asset'][pos_under_2p].unique().shape[0])
pos_under_thr.to_csv(pos_dirname+pos_filename+str(100*pip_limit)+'pFilt.csv', index=False, sep='\t')

import matplotlib.pyplot as plt

plt.figure(0)
bins = [i/100 for i in range(-100,100,2)]
histG = plt.hist(pos_under_thr['GROI'], bins=bins)
histR = plt.hist(pos_under_thr['ROI'], bins=bins)
plt.grid()

#pos_under_thr.index = range(pos_under_thr.shape[0])
plt.figure(1)
plt.plot(range(pos_under_thr.shape[0]),pos_under_thr['GROI'].cumsum())
plt.plot(range(pos_under_thr.shape[0]),pos_under_thr['ROI'].cumsum())
plt.grid()

plt.figure(2)
#print(pos_under_thr['ROI'])
print(positions[pos_under_2p]['GROI'].shape)
cumG = plt.plot(positions['GROI'].cumsum())#, bins=bins
cumR = plt.plot(positions['ROI'].cumsum())
plt.grid()


#positions
#grouped = pos_format.groupby(['asset'])
weekly_group = pos_under_thr.groupby([pd.to_datetime(pos_under_thr['Di']).dt.strftime('%W')])['ROI']
weekly_group_G = pos_under_thr.groupby([pd.to_datetime(pos_under_thr['Di']).dt.strftime('%W')])['GROI']
asset_group = pos_under_thr.groupby(['Asset'])
weekly_sum = weekly_group.sum()
weekly_sum_G = weekly_group_G.sum()
weekly_count = weekly_group.count()
#weekly_sum.cumsum()
#print(asset_group['GROI'].sum())
print(asset_group['ROI'].sum())
plt.figure(3)
plt.plot(weekly_sum_G.cumsum())
plt.plot(range(weekly_sum.shape[0]), weekly_sum.cumsum())
plt.grid()
#print(weekly_sum_G)
print(weekly_sum)#.cumsum()
print(weekly_count)
#print(weekly_sum.cumsum())
#print(weekly_count.shape)
#print(pos_under_thr.groupby([pd.to_datetime(pos_under_thr['Di']).dt.strftime('%W')])['GROI'].sum())
#print(weekly_count)
plt.figure(4)
plt.grid()
plt.hist(weekly_count, bins=range(0,max(weekly_count),5))

#for name, group in asset_group:
#    print(name)
#    print(group.to_string())
#pd.to_datetime(positions['Di']).dt.strftime('%W')
#pd.to_datetime(positions['Di'])#.groupby('Name').resample('W-Mon', on='Date').sum().reset_index().sort_values(by='Date')



list_dates = [dt.datetime.strptime(date, '%Y.%m.%d %H:%M:%S') for date in pos_under_thr['DTo']]
dates = matplotlib.dates.date2num(list_dates)
plt.figure(5)
plt.plot_date(list_dates, pos_under_thr['GROI'].cumsum(),fmt='-')
plt.plot_date(list_dates, pos_under_thr['ROI'].cumsum(),fmt='-')
plt.gcf().autofmt_xdate()
plt.grid()