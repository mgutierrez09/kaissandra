# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 17:18:27 2018

@author: mgutierrez

Module for control of system flow in online mode
"""

from kaissandra.inputs import Data
from kaissandra.local_config import local_vars

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