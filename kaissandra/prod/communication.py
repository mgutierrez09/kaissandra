# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 11:20:05 2019

@author: mgutierrez
"""
import sys
import os

def send_command(directory_MT5_ass, command, msg=''):
    """
    Send command for opening position to MT5 software   
    """
    success = 0
    # load network output
    while not success:
        try:
            fh = open(directory_MT5_ass+command,"w")
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
    io_dir = local_vars.io_live_dir
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
    io_dir = local_vars.io_live_dir
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
    io_dir = local_vars.io_live_dir
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
        
def reset_networks():
    """  """
    directory_io = local_vars.io_live_dir
    command = "RESET"
    AllAssets = Data().AllAssets
    for asset_key in AllAssets:
        thisAsset = AllAssets[asset_key]
        directory_io_ass = directory_io+thisAsset+"/"
        if os.path.exists(directory_io_ass):
            #print("Sent command to "+directory_MT5_ass)
            send_command(directory_io_ass, command)

if __name__=='__main__':
    # add kaissandra to path
    
    this_path = os.getcwd()
    path = '\\'.join(this_path.split('\\')[:-2])+'\\'
    if path not in sys.path:
        sys.path.insert(0, path)
        print(path+" added to python path")
    else:
        print(path+" already added to python path")
    from kaissandra.inputs import Data
    from kaissandra.local_config import local_vars