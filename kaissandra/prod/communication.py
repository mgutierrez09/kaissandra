# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 11:20:05 2019

@author: mgutierrez
"""
import sys
import os
import re

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
    
def shutdown(id=None):
    """  """
    io_dir = local_vars.io_live_dir
    AllAssets = Config.AllAssets
    for asset_key in AllAssets:
        asset = AllAssets[asset_key]
        print(asset)
        try:
            fh = open(io_dir+asset+'/SD',"w")
            fh.close()
            
        except FileNotFoundError:
            print("FileNotFoundError")
    if id!=None:
        close_session(id)

def close_session(id):
    """  """
    from kaissandra.prod.api import API
    API().close_session(id)
    
def pause():
    """  """
    io_dir = local_vars.io_live_dir
    AllAssets = Config.AllAssets
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
    AllAssets = Config.AllAssets
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
    AllAssets = Config.AllAssets
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
    AllAssets = Config.AllAssets
    for asset_key in AllAssets:
        thisAsset = AllAssets[asset_key]
        directory_io_ass = directory_io+thisAsset+"/"
        if os.path.exists(directory_io_ass):
            #print("Sent command to "+directory_MT5_ass)
            send_command(directory_io_ass, command)

from kaissandra.config import Config
from kaissandra.local_config import local_vars
    
if __name__=='__main__':
    # add kaissandra to path
    
    this_path = os.getcwd()
    path = '\\'.join(this_path.split('\\')[:-2])+'\\'
    if path not in sys.path:
        sys.path.insert(0, path)
        print(path+" added to python path")
    else:
        print(path+" already added to python path")
    
    
    #from kaissandra.prod.config import Config as CC
    
    for arg in sys.argv:
        print(arg)
        if re.search('^shutdown',arg)!=None:
            if re.search('--id=',arg)!=None:
                id = int(arg.split('=')[-1])
                print("session id found: "+str(id))
                shutdown(id=id)
            else:
                shutdown()
        elif re.search('^reset_networks',arg)!=None:
            reset_networks()
        elif re.search('^close_positions',arg)!=None:
            close_positions()
        elif re.search('^pause',arg)!=None:
            pause()
        elif re.search('^resume',arg)!=None:
            resume()
        elif re.search('^close_session',arg)!=None:
            if re.search('--id=',arg)!=None:
                id = int(arg.split('=')[-1])
            else:
                raise ValueError("session id must be specified")
            close_session(id)
