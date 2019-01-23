# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 17:18:27 2018

@author: mgutierrez
"""

from inputs import Data

if __name__=='__main__':
    
    io_dir = '../RNN/IOlive/'
    AllAssets = Data().AllAssets
    for asset_key in AllAssets:
        asset = AllAssets[asset_key]
        print(asset)
        try:
            fh = open(io_dir+asset+'/SD',"w")
            fh.close()
        except FileNotFoundError:
            print("FileNotFoundError")
    