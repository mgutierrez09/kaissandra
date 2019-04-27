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

from kaissandra.inputs import Data
from kaissandra.local_config import local_vars


def send_command(directory_MT5_ass, command, msg=''):
    """
    Send command for opening position to MT5 software   
    """
    success = 0
    # load network output
    while not success:
        try:
            fh = open(directory_MT5_ass+command,"w", encoding='utf_16_le')
            if msg!='':
                fh.write(msg)
            fh.close()
            success = 1
            #stop_timer(ass_idx)
        except PermissionError:
            print("Error writing")
    print(" "+directory_MT5_ass+command+" command sent")
    
def shutdown():
    """  """
    io_dir = local_vars.io_dir
    AllAssets = Data().AllAssets
    for asset_key in AllAssets:
        asset = AllAssets[asset_key]
        print(asset)
        try:
            fh = open(io_dir+asset+'/SD',"w")
            fh.close()
        except FileNotFoundError:
            print("FileNotFoundError")
    
def pause():
    """  """
    io_dir = local_vars.io_dir
    AllAssets = Data().AllAssets
    for asset_key in AllAssets:
        asset = AllAssets[asset_key]
        print(asset)
        try:
            fh = open(io_dir+asset+'/PA',"w")
            fh.close()
        except FileNotFoundError:
            print("Asset not running")

def resume():
    """  """
    io_dir = local_vars.io_dir
    AllAssets = Data().AllAssets
    for asset_key in AllAssets:
        asset = AllAssets[asset_key]
        print(asset)
        try:
            fh = open(io_dir+asset+'/RE',"w")
            fh.close()
        except FileNotFoundError:
            print("FileNotFoundError")

def close_positions():
    """ Close all positions from py """
    directory_MT5 = local_vars.directory_MT5_IO
    command = "LC"
    AllAssets = Data().AllAssets
    for asset_key in AllAssets:
        thisAsset = AllAssets[asset_key]
        directory_MT5_ass = directory_MT5+thisAsset+"/"
        if os.path.exists(directory_MT5_ass):
            #print("Sent command to "+directory_MT5_ass)
            send_command(directory_MT5_ass, command)
            
def close_position(thisAsset):
    """ Close all positions from py """
    directory_MT5 = local_vars.directory_MT5_IO
    command = "LC"
    directory_MT5_ass = directory_MT5+thisAsset+"/"
    if os.path.exists(directory_MT5_ass):
        #print("Sent command to "+directory_MT5_ass)
        send_command(directory_MT5_ass, command)

def control(running_assets, timeout=15):
    """ Master function to manage all controlling functions such as connection
    control or log control 
    Args:
        - running_assets (list): list of assets being tracked by trader 
        - timeout (int): max timeout in minutes before reseting networks """
    directory_MT5 = local_vars.directory_MT5_IO
    directory_io = local_vars.io_dir
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

def reset_networks():
    """  """
    directory_io = local_vars.io_dir
    command = "RESET"
    AllAssets = Data().AllAssets
    for asset_key in AllAssets:
        thisAsset = AllAssets[asset_key]
        directory_io_ass = directory_io+thisAsset+"/"
        if os.path.exists(directory_io_ass):
            #print("Sent command to "+directory_MT5_ass)
            send_command(directory_io_ass, command)