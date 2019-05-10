# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 11:21:50 2019

@author: mgutierrez
"""
import math
import datetime as dt
import pandas as pd
import numpy as np

from kaissandra.results2 import extract_result

from kaissandra.local_config import local_vars

results_filename = local_vars.results_directory
#IDresults = 'RNN01010k1-2K5ACR20wrong'#'RRNN00000K5ACNR'
#results_table_filename = results_filename+IDresults+'/performance.csv'
#TR = pd.read_csv(results_table_filename, sep='\t')
#
#func1 = lambda x:x=='mean'
#func2 = lambda x:x>=60
#extract_result(TR, {'GSP':func2}, ['eROI1.5'])

start_date = dt.date(2017, 9, 27)
start_date_monday = (start_date - dt.timedelta(days=start_date.weekday()))
end_date = dt.date(2018, 11, 9)
num_of_weeks = math.ceil((end_date - start_date_monday).days / 7.0)
print(num_of_weeks)

from kaissandra.results2 import get_extended_results,remove_outliers

results_dir  = local_vars.results_directory
ext = '.csv'


#rootname_config = 'RNN01010'
#extR = 'AC'
#sufix='R20'
#K = 5
#k_end = 2
#k_init = 0
#list_results_names=['R'+rootname_config+'k'+str(fold_idx+1)+'K'+str(K)+extR+sufix for fold_idx in range(k_init, k_end)]
#list_epochs = [11 for _ in range(k_init, k_end)]
#list_tis = ['2' for _ in range(k_init, k_end)]
#list_mcs = ['0.75' for _ in range(k_init, k_end)]#for _ in range(k_end) 
#list_mds = ['0.6' for _ in range(k_init, k_end)]

rootname_config = 'RNN01010'
extR_L = 'AL'
extR_S = 'BS'
sufix='R20'
K = 5
k_end = 2
k_init = 0
IDLs=['R'+rootname_config+'k'+str(fold_idx+1)+'K'+str(K)+extR_L+sufix for fold_idx in range(k_init, k_end)]
IDSs=['R'+rootname_config+'k'+str(fold_idx+1)+'K'+str(K)+extR_S+sufix for fold_idx in range(k_init, k_end)]
list_results_names = IDLs+IDSs
list_epochs = [9 for _ in range(k_init, k_end)]+[10 for _ in range(k_init, k_end)]
list_tis = ['2' for _ in range(k_init, k_end)]+['2' for _ in range(k_init, k_end)]
list_mcs = ['0.7' for _ in range(k_init, k_end)]+['0.6' for _ in range(k_init, k_end)]
list_mds = ['0.6' for _ in range(k_init, k_end)]+['0.6' for _ in range(k_init, k_end)]
pip_limit = 0.03
    
#Journal = pd.DataFrame()
#for l,name in enumerate(list_results_names):
#    journal_filename = results_dir+name+'/journal/J_E'+str(list_epochs[l])+'TI'+list_tis[l]+'MG'+list_mgs[l]+ext
#    new_journal = pd.read_csv(journal_filename,sep='\t')
#    Journal = Journal.append(new_journal).sort_values(by=['Asset','DTi']).reset_index().drop(labels='level_0',axis=1)
#    print(journal_filename)

Journal = pd.DataFrame()
for l,name in enumerate(list_results_names):
    journal_filename = results_dir+name+'/journal/J_E'+str(list_epochs[l])+'TI'+str(list_tis[l])+'MC'+str(list_mcs[l])+'MD'+str(list_mds[l])+ext
    new_journal = pd.read_csv(journal_filename,sep='\t')
    Journal = Journal.append(new_journal).sort_values(by=['Asset','DTi']).reset_index().drop(labels='level_0',axis=1)
    print(journal_filename)

# filter journal
#pip_limit = 0.02
#pos_under_2p = positions['espread']<pip_limit
print("shape")
print(Journal.shape[0])
print(journal_filename)
n_days = num_of_weeks*5+1
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
GROIS99, ROIS99, idx_sorted, low_arg_goi, high_arg_goi = remove_outliers(np.array(positions.GROI),
                                                                         np.array(positions.spread), thr=.99)

pos_under_2p = positions['espread']<pip_limit
#pos_under_thr_99 = positions['espread']<pip_limit
#GROIS99 = GROIS99[pos_under_2p]
#ROIS99 = ROIS99[pos_under_2p]
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
import matplotlib

plt.figure(0)
mx = int(np.ceil(max(pos_under_thr['GROI'])))
mn = int(np.floor(min(pos_under_thr['ROI'])))
bins = [i/20 for i in range(20*mn,20*mx)]
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
plt.plot(np.cumsum(positions.GROI[(positions.GROI>=low_arg_goi) & (positions.GROI<=high_arg_goi)]))
plt.plot(np.cumsum(positions.ROI[(positions.GROI>=low_arg_goi) & (positions.GROI<=high_arg_goi)]))
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
#print(asset_group['ROI'].sum())
#print(asset_group['ROI'].count())
plt.figure(3)
plt.plot(weekly_sum_G.cumsum())
plt.plot(range(weekly_sum.shape[0]), weekly_sum.cumsum())
plt.grid()
#print(weekly_sum_G)
#print(weekly_sum)#.cumsum()
#print(weekly_sum_G.cumsum()[-1])
#print(weekly_sum.cumsum()[-1])
#print(weekly_count)
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