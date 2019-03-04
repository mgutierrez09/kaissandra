# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 14:20:06 2018

@author: mgutierrez
Script that reads out trade info from csv files and saves it into a long HDF5 file.
"""

import os
import re
import h5py
import numpy as np
import pandas as pd
import datetime as dt
from kaissandra.inputs import Data, extractSeparators
from kaissandra.local_config import local_vars

def check_consecutive_trading_days(directory,prev_day, post_day):
    """
    Function that checks if two days are consecutive trading days, regardless
    of holidays or weekend in between.
    Args:
        - prev_day: datetime of previous day
        - post_day: datetime of posterious day
    Returns:
        - boolean indicating consecutive or not consecutive
    """
    consecutive = False
    # load business days
    filename = directory+'trading_days.txt'
    trading_days = pd.read_csv(filename,sep='\t')
#    print(prev_day)
#    print((trading_days.TradingDays==prev_day).max())
#    print(post_day)
#    print((trading_days.TradingDays==post_day).max())
    # check that both days are in trading days
    if (trading_days.TradingDays==prev_day).max() and (trading_days.TradingDays==post_day).max():
        # find index previous day
        idx_prev_day = (trading_days.TradingDays==prev_day).argmax()
        idx_post_day = (trading_days.TradingDays==post_day).argmax()
#        print(idx_prev_day)
#        print(idx_post_day)
#        print("trading_days[idx_prev_day] "+trading_days.loc[idx_prev_day])
#        print("trading_days[idx_post_day] "+trading_days.loc[idx_post_day])
        # check if days are consecutive
        if idx_prev_day == idx_post_day-1 or idx_prev_day == idx_post_day:
            consecutive = True
    else:
        raise ValueError("Days not in trading days. Check trading days structure")
    
    return consecutive

def check_consecutive_trading_days_from_list(list_bussines_days, prev_day, post_day):
    """
    Function that checks if two days are consecutive trading days, regardless
    of holidays or weekend in between.
    Args:
        - prev_day: datetime of previous day
        - post_day: datetime of posterious day
    Returns:
        - boolean indicating consecutive or not consecutive
    """
    consecutive = False
#    trading_days = pd.read_csv(filename,sep='\t')
#    print(prev_day)
#    print((trading_days.TradingDays==prev_day).max())
#    print(post_day)
#    print((trading_days.TradingDays==post_day).max())
    # check that both days are in trading days
    if prev_day in list_bussines_days and post_day in list_bussines_days:
        # find index previous day
        idx_prev_day = list_bussines_days.index(prev_day)
        idx_post_day = list_bussines_days.index(post_day)
#        print(idx_prev_day)
#        print(idx_post_day)
#        print("trading_days[idx_prev_day] "+trading_days.loc[idx_prev_day])
#        print("trading_days[idx_post_day] "+trading_days.loc[idx_post_day])
        # check if days are consecutive
        if idx_prev_day == idx_post_day-1 or idx_prev_day == idx_post_day:
            consecutive = True
    else:
        #print("Error. Days not in trading days. Check trading days structure")
        raise ValueError("Days not in trading days. Check trading days structure")
    
    return consecutive

def transition_train_test(data, prev_day, post_day):
    """
    Funciton that checks if two consecutive days belong to different sets of
    test or train
    """
    transition = False
    # check if one of the days belongs to test days and the other does not
    if (((prev_day in data.dateTest) and (post_day not in data.dateTest)) or
        ((post_day in data.dateTest) and (prev_day not in data.dateTest))):
        # activate transition
        transition = True
    
    return transition

def merge_separators_list(list_bussines_days, data, separators_list, thrs):
    """
    Function that merges a list with separators per file in one 
    single dataFrame structure.
    Args: 
        - separator_list: list containing separators per file
        - thrs: threshold  beween two entries in minutes to be considered 
          non-consecutive
    Returns: 
        - separators: dataFrame with merged separators
    """
    # init separators with all in-file separators but bottom
    separators = pd.DataFrame(columns=["DateTime","SymbolBid","SymbolAsk"],data=separators_list[0].iloc[:-1])
    
    # loop over separators list to merge ends of files
    for s in range(len(separators_list)-1):
        # get last separator of previous day
        prev_date = separators_list[s].DateTime.iloc[-1]
        # get first separator of posterior day
        post_date = separators_list[s+1].DateTime.iloc[0]
        # convert to datetime structures
        end_day_separator_dt = dt.datetime.strptime(prev_date, '%Y.%m.%d %H:%M:%S')
        start_day_separator_dt = dt.datetime.strptime(post_date, '%Y.%m.%d %H:%M:%S')
        # get previous and posterious days as strings
        prev_day = dt.datetime.strftime(end_day_separator_dt, '%Y.%m.%d')
        post_day = dt.datetime.strftime(start_day_separator_dt, '%Y.%m.%d')
#        print("prev_day "+str(prev_day))
#        print("post_day "+str(post_day))
        # check if previous day and posterious day are correlative business days
        if check_consecutive_trading_days_from_list(list_bussines_days,prev_day, post_day):#check_consecutive_trading_days(directory_destination,prev_day, post_day):
            # time difference between separators
            time_difference = start_day_separator_dt-end_day_separator_dt
#            print("time_difference "+str(time_difference))
#            print("time_difference.total_seconds()/60-24*60*time_difference.days "+str(time_difference.total_seconds()/60-24*60*time_difference.days))
#            print("time_difference.total_seconds()/60-24*60*time_difference.days "+str(time_difference.total_seconds()/60-24*60*time_difference.days))
#            print(time_difference.total_seconds()/60>thrs)
#            print(transition_train_test(data, prev_day, post_day))
            # check if time difference is grater than minutes difference threshold or
            # if there is a transition between training and testing dates
            if (time_difference.total_seconds()/60-24*60*time_difference.days>thrs or 
                transition_train_test(data, prev_day, post_day)):
                # append day separators to separators
                separators = separators.append(separators_list[s].iloc[-1]).append(separators_list[s+1].iloc[0])
            else:
                pass
            # append in-file separators of file s+1
            separators = separators.append(separators_list[s+1].iloc[1:-1])
        else:
            # if nt consecutive days, add to separators
            separators = separators.append(separators_list[s].iloc[-1]).append(separators_list[s+1].iloc[0])
            
    # append last entry of last separator
    separators = separators.append(pd.DataFrame(columns=["DateTime","SymbolBid","SymbolAsk"],data=separators_list[-1].iloc[-1:]))
    return separators

def find_bussines_days(data, directory_destination):
    # loop over all assets
    b_days_list = []
    for ass in data.assets:
        thisAsset = data.AllAssets[str(ass)]
        #print(thisAsset)
        directory_origin = directory_root+thisAsset+'/'#'../Data/'+thisAsset+'/'
            
        files_list = []
        # get files list, and beginning and end current dates
        if os.path.isdir(directory_origin):
            files_list_all = sorted(os.listdir(directory_origin))
            
            first_file = 1
            for file in files_list_all:
                
                m = re.search('^'+thisAsset+'_\d+'+'.txt$',file)
                if m!=None:
                    date_dt = dt.datetime.strptime(re.search('\d+',m.group()).group(), '%Y%m%d%H%M%S')
                    # check date is within init and end dates
                    ########################################
                    # WARNING!! To be tested
                    if 1:
                            
                        # save date of first file to later check if new info is older
                        if first_file:
                            first_date_s = re.search('\d+',m.group()).group()
                            first_day_dt = dt.datetime.strptime(first_date_s, '%Y%m%d%H%M%S')
                            first_file = 0
                            # save date of first file to later check if new info is older
                        files_list.append(file)
                        last_date_s = re.search('\d+',m.group()).group()
                        last_date_dt = dt.datetime.strptime(last_date_s, '%Y%m%d%H%M%S')
                        bussiness_day = dt.datetime.strftime(last_date_dt, '%Y.%m.%d')
                        if not bussiness_day in b_days_list:
                            #print("Adding "+bussiness_day)
                            b_days_list.append(bussiness_day)
    
    # add missing bdays
    filename = directory_destination+'missing_days.txt'
    missing_days = pd.read_csv(filename,sep='\t')
    #print(missing_days['Days'].tolist())
    b_days_list = sorted(b_days_list+missing_days['Days'].tolist())
    return b_days_list

# limit the build of the HDF5 to data comprised in these dates
build_partial_raw = False
build_test_db = True
int_date = '180928'
end_date = '181109'
init_date_dt = dt.datetime.strptime(int_date,'%y%m%d')
end_date_dt = dt.datetime.strptime(end_date,'%y%m%d')

directory_destination = local_vars.hdf5_directory#'D:/SDC/py/HDF5/'
if build_partial_raw and not build_test_db:
    hdf5_file_name = 'tradeinfo_F'+int_date+'T'+end_date+'.hdf5'
    directory_root = 'D:/SDC/py/Data/'
    separators_directory_name = 'separators_F'+int_date+'T'+end_date+'/'
elif build_test_db and not build_partial_raw:
    hdf5_file_name = 'tradeinfo_test.hdf5'
    directory_root = local_vars.data_test_dir
    separators_directory_name = 'separators_test/'
    dateTest = ['2018.11.12','2018.11.13','2018.11.14','2018.11.15','2018.11.16',
                '2018.11.19','2018.11.20','2018.11.21','2018.11.22','2018.11.23',
                '2018.11.26','2018.11.27','2018.11.28','2018.11.29','2018.11.30',
                '2018.12.03','2018.12.04','2018.12.05','2018.12.06','2018.12.07',
                '2018.12.10','2018.12.11','2018.12.12','2018.12.13','2018.12.14',
                '2018.12.17','2018.12.18','2018.12.19','2018.12.20','2018.12.21',
                '2018.12.24','2018.12.25','2018.12.26','2018.12.27','2018.12.28',
                '2018.12.31','2019.01.01','2019.01.02','2019.01.03','2019.01.04',
                '2019.01.07','2019.01.08','2019.01.09','2019.01.10','2019.01.11',
                '2019.01.14','2019.01.15','2019.01.16','2019.01.17','2019.01.18',
                '2019.01.21','2019.01.22','2019.01.23','2019.01.24','2019.01.25',
                '2019.01.28','2019.01.29','2019.01.30','2019.01.31','2019.02.01',
                '2019.02.04','2019.02.05','2019.02.06','2019.02.07','2019.02.08',
                '2019.02.11','2019.02.12','2019.02.13','2019.02.14','2019.02.15',
                '2019.02.18','2019.02.19','2019.02.20','2019.02.21','2019.02.22']
elif not build_test_db and not build_partial_raw:
    hdf5_file_name = 'tradeinfo.hdf5'
    separators_directory_name = 'separators/'
    directory_root = 'D:/SDC/py/Data/'
    dateTest = ([                                                 '2018.03.09',
             '2018.03.12','2018.03.13','2018.03.14','2018.03.15','2018.03.16',
             '2018.03.19','2018.03.20','2018.03.21','2018.03.22','2018.03.23',
             '2018.03.26','2018.03.27','2018.03.28','2018.03.29','2018.03.30',
             '2018.04.02','2018.04.03','2018.04.04','2018.04.05','2018.04.06',
             '2018.04.09','2018.04.10','2018.04.11','2018.04.12','2018.04.13',
             '2018.04.16','2018.04.17','2018.04.18','2018.04.19','2018.04.20',
             '2018.04.23','2018.04.24','2018.04.25','2018.04.26','2018.04.27',
             '2018.04.30','2018.05.01','2018.05.02','2018.05.03','2018.05.04',
             '2018.05.07','2018.05.08','2018.05.09','2018.05.10','2018.05.11',
             '2018.05.14','2018.05.15','2018.05.16','2018.05.17','2018.05.18',
             '2018.05.21','2018.05.22','2018.05.23','2018.05.24','2018.05.25',
             '2018.05.28','2018.05.29','2018.05.30','2018.05.31','2018.06.01',
             '2018.06.04','2018.06.05','2018.06.06','2018.06.07','2018.06.08',
             '2018.06.11','2018.06.12','2018.06.13','2018.06.14','2018.06.15',
             '2018.06.18','2018.06.19','2018.06.20','2018.06.21','2018.06.22',
             '2018.06.25','2018.06.26','2018.06.27','2018.06.28','2018.06.29',
             '2018.07.02','2018.07.03','2018.07.04','2018.07.05','2018.07.06',
             '2018.07.09','2018.07.10','2018.07.11','2018.07.12','2018.07.13',
             '2018.07.30','2018.07.31','2018.08.01','2018.08.02','2018.08.03',
             '2018.08.06','2018.08.07','2018.08.08','2018.08.09','2018.08.10']+
            ['2018.08.13','2018.08.14','2018.08.15','2018.08.16','2018.08.17',
             '2018.08.20','2018.08.21','2018.08.22','2018.08.23','2018.08.24',
             '2018.08.27','2018.08.28','2018.08.29','2018.08.30','2018.08.31',
             '2018.09.03','2018.09.04','2018.09.05','2018.09.06','2018.09.07',
             '2018.09.10','2018.09.11','2018.09.12','2018.09.13','2018.09.14',
             '2018.09.17','2018.09.18','2018.09.19','2018.09.20','2018.09.21',
             '2018.09.24','2018.09.25','2018.09.26','2018.09.27']+['2018.09.28',
             '2018.10.01','2018.10.02','2018.10.03','2018.10.04','2018.10.05',
             '2018.10.08','2018.10.09','2018.10.10','2018.10.11','2018.10.12',
             '2018.10.15','2018.10.16','2018.10.17','2018.10.18','2018.10.19',
             '2018.10.22','2018.10.23','2018.10.24','2018.10.25','2018.10.26',
             '2018.10.29','2018.10.30','2018.10.31','2018.11.01','2018.11.02',
             '2018.11.05','2018.11.06','2018.11.07','2018.11.08','2018.11.09'])
else:
    raise ValueError("Not supported")
trusted_source = False
data = Data(dateTest = dateTest)
# create directory if not exists
if not os.path.exists(directory_destination+separators_directory_name):
    os.mkdir(directory_destination+separators_directory_name)
# thresholds for separators
bidThresDay = 0.0
bidThresNight = 0.0
minThresDay = 15
minThresNight = 20

# reset file
reset_file = False
if reset_file:
    fw = h5py.File(directory_destination+hdf5_file_name,'w')
    fw.close()
    
# open file for read and write
f = h5py.File(directory_destination+hdf5_file_name,'a')

# get gussines days
list_bussines_days = find_bussines_days(data, directory_destination)
    
# loop over all assets
for ass in data.assets:
    thisAsset = data.AllAssets[str(ass)]
    print(thisAsset)
    directory_origin = directory_root+thisAsset+'/'#'../Data/'+thisAsset+'/'
    # extend the threshold margin for GOLD since it always starts at 01:00 am
    if thisAsset == 'GOLD':
        inter_day_thres = 120
    else:
        inter_day_thres = 120
        
    files_list = []
    # get files list, and beginning and end current dates
    if os.path.isdir(directory_origin):
        lastM = None
        files_list_all = sorted(os.listdir(directory_origin))
        
        first_file = 1
        for file in files_list_all:
            
            m = re.search('^'+thisAsset+'_\d+'+'.txt$',file)
            if m!=None:
                date_dt = dt.datetime.strptime(re.search('\d+',m.group()).group(), '%Y%m%d%H%M%S')
                # check date is within init and end dates
                ########################################
                # WARNING!! To be tested
                if not build_partial_raw or (date_dt>=init_date_dt and
                    date_dt<end_date_dt+dt.timedelta(1)):
                        
                    # save date of first file to later check if new info is older
                    if first_file:
                        first_date_s = re.search('\d+',m.group()).group()
                        first_day_dt = dt.datetime.strptime(first_date_s, '%Y%m%d%H%M%S')
                        first_file = 0
                        # save date of first file to later check if new info is older
                    files_list.append(file)
                    last_date_s = re.search('\d+',m.group()).group()
                    last_date_dt = dt.datetime.strptime(last_date_s, '%Y%m%d%H%M%S')
    
#    if thisAsset=='EURCAD':
#        del f[thisAsset]
        ##### TODO remove separators as well! #### 
    
    if thisAsset not in f:
        # create group, its attributes and its datasets
        group = f.create_group(thisAsset)
        
        group.create_dataset("DateTime", (0,), maxshape=(None,),dtype='S19')
        group.create_dataset("SymbolBid", (0,), maxshape=(None,),dtype=float)
        group.create_dataset("SymbolAsk", (0,), maxshape=(None,),dtype=float)
        
        group.attrs.create("dateStart", "", shape=(1,), dtype='S14')
        group.attrs.create("dateEnd", "", shape=(1,), dtype='S14')
        
        new_files_newer = files_list
        new_files_older = []
        
    else:
        # get current init and end dates
        group = f[thisAsset]
        curr_start_date_s = group.attrs.get("dateStart")
        curr_end_date_s = group.attrs.get("dateEnd")
        try:
            curr_start_date_dt = dt.datetime.strptime(curr_start_date_s[0].decode("utf-8") , '%Y%m%d%H%M%S')
            curr_end_date_dt = dt.datetime.strptime(curr_end_date_s[0].decode("utf-8") , '%Y%m%d%H%M%S')
            # init file lists
            new_files_older = []
            new_files_newer = []
            for file in files_list:
                m = re.search('^'+thisAsset+'_\d+'+'.txt$',file)
                this_date_dt = dt.datetime.strptime(re.search('\d+',m.group()).group(), '%Y%m%d%H%M%S')
                # check if 
                if this_date_dt<curr_start_date_dt:
                    # there is new older info
                    new_files_older.append(file)
                elif this_date_dt>curr_end_date_dt:
                    # there is new newer info
                    new_files_newer.append(file)
                else:
                    # this info is already included
                    pass
        except ValueError:
            # if dates are wrong, start with all files
            print("ERROR in attributes. Reseting group.")
            group = f.create_group(thisAsset)
        
            group.create_dataset("DateTime", (0,), maxshape=(None,),dtype='S19')
            group.create_dataset("SymbolBid", (0,), maxshape=(None,),dtype=float)
            group.create_dataset("SymbolAsk", (0,), maxshape=(None,),dtype=float)
            
            group.attrs.create("dateStart", "", shape=(1,), dtype='S14')
            group.attrs.create("dateEnd", "", shape=(1,), dtype='S14')
            
            new_files_newer = files_list
            new_files_older = []
            
    # get data sets in this asset group
    DateTime = group["DateTime"]
    SymbolBid = group["SymbolBid"]
    SymbolAsk = group["SymbolAsk"]
    
    # init separators
    separators_filename = separators_directory_name+thisAsset+'_separators.txt'
    if reset_file:
        old_separators = pd.DataFrame(columns=["DateTime","SymbolBid","SymbolAsk"])
        old_separators.index.name = 'Pointer'
    else:
        # load old separators
        if os.path.exists(directory_destination+separators_filename):
            old_separators = pd.read_csv(directory_destination+separators_filename, index_col='Pointer')
            
        else:
            old_separators = pd.DataFrame(columns=["DateTime","SymbolBid","SymbolAsk"])
            old_separators.index.name = 'Pointer'

    # init pandas data frame where files are added to
    tradeInfo = pd.DataFrame(columns=["DateTime","SymbolBid","SymbolAsk"])
    # init separators list
    list_separators_older = []
    # init temp hdf5 file
    # create temporal file, its group and its datasets
    if len(new_files_older)>0:
        ft = h5py.File(directory_destination+'temp.hdf5','w')
        group_temp = ft.create_group(thisAsset)
        
        group_temp.create_dataset("DateTime", (0,), maxshape=(None,),dtype='S19')
        group_temp.create_dataset("SymbolBid", (0,), maxshape=(None,),dtype=float)
        group_temp.create_dataset("SymbolAsk", (0,), maxshape=(None,),dtype=float)
        
        DateTime_temp = group_temp["DateTime"]
        SymbolBid_temp = group_temp["SymbolBid"]
        SymbolAsk_temp = group_temp["SymbolAsk"]
        
        # init separator pointer to file to zero
        pointer_sep = 0
        
    # add new files that are older
    for file in new_files_older:
        print("Copying "+file+" in HDF5 file")
        # read file and save it in a pandas data frame
        tradeInfo = tradeInfo.append(pd.read_csv(directory_origin+file), ignore_index=True)
        
        if tradeInfo.shape[0]>0:
            if not trusted_source:
                # extract separators
                this_separators = extractSeparators(tradeInfo,minThresNight,minThresNight,
                                                       bidThresDay,bidThresNight,[])
                # reference index according to general pointer
                this_separators.index = this_separators.index+pointer_sep
                list_separators_older.append(this_separators)
            # save data in a temp file
            # resize data sets
            size_dataset = DateTime_temp.shape[0]
            # resize DateTime
            DateTime_temp.resize((size_dataset+tradeInfo.shape[0],))
            # build numpy helper vector
            charar = np.chararray((tradeInfo.shape[0],),itemsize=19)
            charar[:] = tradeInfo.DateTime.iloc[:]
            # add new newer info at the end of the data set
            DateTime_temp[size_dataset:] = charar
            # resize Symbolbid dataset
            SymbolBid_temp.resize((size_dataset+tradeInfo.shape[0],))
            # add new newer info at the end of the data set
            SymbolBid_temp[size_dataset:] = tradeInfo.SymbolBid.iloc[:]
            # resize SymbolAsk dataset
            SymbolAsk_temp.resize((size_dataset+tradeInfo.shape[0],))
            # add new newer info at the end of the data set
            SymbolAsk_temp[size_dataset:] = tradeInfo.SymbolAsk.iloc[:]
            # update pointer to file
            pointer_sep += tradeInfo.shape[0]
        else:
            print("WARNING: Empty file. Skipped.")
        # reset tradeInfo
        tradeInfo = pd.DataFrame(columns=["DateTime","SymbolBid","SymbolAsk"])
    # end of for file in new_files_older:
    # Add temporal information to the hdf5 group and datasets
    if len(new_files_older)>0:
        # new sizes
        size_dataset = DateTime.shape[0]
        size_increase = DateTime_temp.shape[0]
        # resize DateTime dataset
        DateTime.resize((size_dataset+size_increase,))
        # shift info to the end of the file
        DateTime[-size_dataset:] = DateTime[:size_dataset]
        # add new old info at the beginning of the data set
        DateTime[:size_increase] = DateTime_temp
        # resize SymbolBid dataset
        SymbolBid.resize((size_dataset+size_increase,))
        # shift info to the end of the file
        SymbolBid[-size_dataset:] = SymbolBid[:size_dataset]
        # add new old info at the beginning of the data set
        SymbolBid[:size_increase] = SymbolBid_temp
        # resize SymbolAsk dataset
        SymbolAsk.resize((size_dataset+size_increase,))
        # shift info to the end of the file
        SymbolAsk[-size_dataset:] = SymbolAsk[:size_dataset]
        # add new old info at the beginning of the data set
        SymbolAsk[:size_increase] = SymbolAsk_temp
        # close temporal file
        ft.close()
        # build new separators if source is trustworthy
        # update starting date attribute
        if trusted_source:
            new_separators_older = pd.DataFrame()
            # read fist new file
            tradeInfo = pd.read_csv(directory_origin+new_files_older[0])
            # add first entry as lit
            new_separators_older = new_separators_older.append(tradeInfo.iloc[0])
            # read last new file
            tradeInfo = pd.read_csv(directory_origin+new_files_older[-1])
            # add last entry as bottom
            new_separators_older = new_separators_older.append(tradeInfo.iloc[-1])
            # update separators list
            list_separators_older.append(new_separators_older)
        # add old separators' bottom and update indexes
        if old_separators.shape[0]>0:
            # update indexes
            old_separators.index = old_separators.index+pointer_sep
            # add old separators' bottom
            list_separators_older.append(old_separators.iloc[:1].append(old_separators.iloc[-1:]))
        # merge separators
        new_older_separators = merge_separators_list(list_bussines_days, data, list_separators_older, inter_day_thres)
        # save new separators
        if old_separators.shape[0]>0:
            # if separators exist, remove new older bottom sep and old newer lit sep
            separators = new_older_separators.iloc[:-1].append(old_separators.iloc[1:])
        else:
            separators = new_older_separators
        # save saparators to file
        separators.to_csv(directory_destination+separators_filename, index=True, index_label='Pointer')
        
    # add new date start to attributes
    group.attrs.modify("dateStart", first_date_s.encode('utf-8'))

    ### newer new trade info ###
    # init separators
    if reset_file:
        old_separators = pd.DataFrame(columns=["DateTime","SymbolBid","SymbolAsk"])
        old_separators.index.name = 'Pointer'
    else:
        # load old separators
        if os.path.exists(directory_destination+separators_filename):
            old_separators = pd.read_csv(directory_destination+separators_filename, index_col='Pointer')
            
        else:
            old_separators = pd.DataFrame(columns=["DateTime","SymbolBid","SymbolAsk"])
            old_separators.index.name = 'Pointer'
    
    # init separators and add old separators' bottom
    if old_separators.shape[0]>0:
        list_separators_newer = [old_separators.iloc[:1].append(old_separators.iloc[-1:])]
        # init separator pointer to file to last pointer plus one
        pointer_sep = old_separators.index[-1]+1
    else:
        list_separators_newer = []
        # init separator pointer to file to zero
        pointer_sep = 0
    # save init pointer separator for later saving
    init_pointer_sep = pointer_sep
    # init file index
    file_newer_index = 0
    # add new files that are newer
    for file in new_files_newer:
        print("Copying "+file+" in HDF5 file")
        # read file and save it in a pandas data frame
        tradeInfo = tradeInfo.append(pd.read_csv(directory_origin+file), ignore_index=True)
        if tradeInfo.shape[0]>0:
            
            # get separators if the source is not trusted
            if not trusted_source:
                # update index
                file_newer_index += 1
                # get eparators from latest info
                this_separators = extractSeparators(tradeInfo,minThresNight,minThresNight,
                                                       bidThresDay,bidThresNight,[])
                # reference index according to general pointer
                this_separators.index = this_separators.index+pointer_sep
                # update list
                list_separators_newer.append(this_separators)
            # resize data sets
            size_dataset = DateTime.shape[0]
            # resize DateTime
            DateTime.resize((size_dataset+tradeInfo.shape[0],))
            # build numpy helper vector
            charar = np.chararray((tradeInfo.shape[0],),itemsize=19)
            charar[:] = tradeInfo.DateTime.iloc[:]
            # add new newer info at the end of the data set
            DateTime[size_dataset:] = charar
            # resize Symbolbid dataset
            SymbolBid.resize((size_dataset+tradeInfo.shape[0],))
            # add new newer info at the end of the data set
            SymbolBid[size_dataset:] = tradeInfo.SymbolBid.iloc[:]
            # resize SymbolAsk dataset
            SymbolAsk.resize((size_dataset+tradeInfo.shape[0],))
            # add new newer info at the end of the data set
            SymbolAsk[size_dataset:] = tradeInfo.SymbolAsk.iloc[:]
            # update pointer to file
            pointer_sep += tradeInfo.shape[0]
            
        else:
            print("WARNING: Empty file. Skipped.")
        # reset tradeInfo
        tradeInfo = pd.DataFrame(columns=["DateTime","SymbolBid","SymbolAsk"])
    
    # end of file in new_files_newer:
    if len(new_files_newer)>0:
        # build new separators if source is trustworthy
        if trusted_source:
            # update starting date attribute
            new_separators_newer = pd.DataFrame()
            # read fist new file
            tradeInfo = pd.read_csv(directory_origin+new_files_newer[0])
            # add first entry as lit
            new_separators_newer = new_separators_newer.append(tradeInfo.iloc[0])
            # read last new file
            tradeInfo = pd.read_csv(directory_origin+new_files_newer[-1])
            # add last entry as bottom
            new_separators_newer = new_separators_newer.append(tradeInfo.iloc[-1])
            # update the index pointer
            new_separators_newer.index = [init_pointer_sep,pointer_sep]
            # build separators list
            list_separators_newer.append(new_separators_newer)
        # merge separators
        new_newer_separators = merge_separators_list(list_bussines_days, data, list_separators_newer, inter_day_thres)
        # save new separators
        if old_separators.shape[0]>0:
            separators = old_separators.iloc[:-1].append(new_newer_separators.iloc[1:])
        else:
            separators = new_newer_separators
        # save saparators to file
        separators.to_csv(directory_destination+separators_filename, index=True, index_label='Pointer')
    # update ending date attribute
    group.attrs.modify("dateEnd", last_date_s.encode('utf-8'))
    # flush info to disk
    f.flush()
# remove and close
if os.path.exists(directory_destination+'temp.hdf5'):
    os.remove(directory_destination+'temp.hdf5')
f.close()
if build_partial_raw:
    os.remove(directory_destination+hdf5_file_name)
    print("Raw file "+directory_destination+hdf5_file_name+" deleted")