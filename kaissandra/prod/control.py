# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 11:29:50 2019

@author: mgutierrez

This sub-package deals with all controlling methods and functions to control
the proper behavior of Kaissandra's production package
"""
import os
import time
import datetime as dt
import sys

def control(running_assets, timeout=15):
    """ Master function to manage all controlling functions such as connection
    control or log control 
    Args:
        - running_assets (list): list of assets being tracked by trader 
        - timeout (int): max timeout in minutes before reseting networks """
    directory_MT5 = local_vars.directory_MT5_IO
    directory_io = local_vars.io_live_dir
    reset_command = 'RESET'
    reset = False
    AllAssets = Data().AllAssets
    timeouts = [time.time() for _ in range(len(running_assets))]
    # get last file in asset channel if not empty, empty string otherwise
    list_last_file = [sorted(os.listdir(directory_MT5+AllAssets[str(ass_id)]+"/"))[-1] \
                      if len(sorted(os.listdir(directory_MT5+AllAssets[str(ass_id)]+"/")))>0 \
                      else '' for ass_id in running_assets]
    while 1:
        # control connection
        list_last_file, timeouts, reset = control_connection(AllAssets, running_assets, 
                                                      timeout, directory_io,
                                                      reset_command, directory_MT5, 
                                                      list_last_file, timeouts, 
                                                      reset)
        time.sleep(5)
    
def control_connection(AllAssets, running_assets, timeout, directory_io,
                           reset_command, directory_MT5, list_last_file, timeouts, reset):
    """ Controls the connection and arrival of new info from trader and 
    sends reset command in case connection is lost """
    for ass_idx, ass_id in enumerate(running_assets):
        thisAsset = AllAssets[str(ass_id)]
        directory_MT5_IO_ass = directory_MT5+thisAsset+"/"
        directory_io_ass = directory_io+thisAsset+"/"
        listAllFiles = sorted(os.listdir(directory_MT5_IO_ass))
        # avoid error in case listAllFiles is empty and replace with empty
        # string if so
        if len(listAllFiles)>0:
            newLastFile = listAllFiles[-1]
        else:
            newLastFile = ''
        if newLastFile!=list_last_file[ass_idx]:
            # reset timeout
            timeouts[ass_idx] = time.time()
            # update last file list
            list_last_file[ass_idx] = newLastFile
            # reset reset flag
            reset = False
#        else:
#            print(thisAsset+" timeout NOT reset")
    min_to = min([time.time()-to for to in timeouts])
    print("\r"+dt.datetime.strftime(dt.datetime.now(),'%y.%m.%d %H:%M:%S')+
          " Min TO = {0:.2f} mins".format(min_to/60), sep=' ', end='', flush=True)
    if min_to>timeout*60 and not reset:
        # Reset networks
        reset = True
        send_command(directory_io_ass, reset_command)
    return list_last_file, timeouts, reset
            
if __name__=='__main__':
    # add kaissandra to path
    this_path = os.getcwd()
    path = '\\'.join(this_path.split('\\')[:-2])+'\\'
    if path not in sys.path:
        sys.path.insert(0, path)
        print(path+" added to python path")
    else:
        print(path+" already added to python path")
    import re
    # extract args
    timeout = 15 # default timeout 15 mins
#    if m!=None:
#        logid = re.search('\d+',m.group()).group()
#        log_ids[ass_idx] = logid
    for arg in sys.argv:
        m = re.search('^timeout=\d+$',arg)
        if m!=None:
            timeout = float(m.group().split('=')[-1])

                
    print("Timeout={0} mins".format(timeout))

from kaissandra.inputs import Data
from kaissandra.local_config import local_vars
from kaissandra.prod.communication import send_command 

if __name__=='__main__':
    # launch control
    control([1,2,3,4,7,8,10,11,12,13,14,15,16,17,19,27,28,29,30,31,32], timeout=timeout)