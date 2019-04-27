# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 11:10:31 2019

@author: mgutierrez
"""
from kaissandra.local_config import local_vars
from kaissandra.inputs import Data
from kaissandra.serial import renew_mt5_dir

def test_open_position(ass_idx, direction, lots, deadline):
    """  """
    thisAsset = Data().AllAssets[str(ass_idx)]
    directory_MT5 = local_vars.directory_MT5_IO
    directory_MT5_ass = directory_MT5+thisAsset+"/"
    command = "TT"
    msg = str(direction)+","+str(lots)+","+str(deadline)
    send_command(directory_MT5_ass, command, msg=msg)
    
def send_command(directory, command, msg=''):
    """
    Send command for opening position to MT5 software   
    """
    success = 0
    # load network output
    while not success:
        try:
            fh = open(directory+command,"w", encoding='utf_16_le')
            if msg!='':
                fh.write(msg)
            fh.close()
            success = 1
            #stop_timer(ass_idx)
        except PermissionError:
            print("Error writing TT")
    print(command+" command sent")
    
def test_close_position(ass_idx):
    """  """
    thisAsset = Data().AllAssets[str(ass_idx)]
    directory_MT5 = local_vars.directory_MT5_IO
    directory_MT5_ass = directory_MT5+thisAsset+"/"
    command = "LC"
    send_command(directory_MT5_ass, command)

def test_renew_mt5_dir(running_assets):
    """  """
    start_file_ids, log_ids = renew_mt5_dir(Data().AllAssets, running_assets)
    print(start_file_ids)
    print(log_ids)