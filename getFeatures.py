# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 12:41:50 2018

@author: mgutierrez
"""

import numpy as np
import time
import h5py
from inputs import Data, load_separators, get_features_results_stats_from_raw


ticTotal = time.time()
# create data structure
data=Data(movingWindow=100,nEventsPerStat=1000,dateEnd='2018.06.01',comments="",
             dateTest = ['2017.11.27','2017.11.28','2017.11.29','2017.11.30','2017.12.01',
                         '2017.12.04','2017.12.05','2017.12.06','2017.12.07','2017.12.08']+
                ['2018.03.09','2018.03.12','2018.03.13','2018.03.14','2018.03.15','2018.03.16','2018.03.19','2018.03.20',
                 '2018.03.21','2018.03.22','2018.03.23','2018.03.26','2018.03.27','2018.03.28','2018.03.29','2018.03.30',
                 '2018.04.02','2018.04.03','2018.04.04','2018.04.05','2018.04.06','2018.04.09','2018.04.10','2018.04.11',
                 '2018.04.12','2018.04.13','2018.04.16','2018.04.17','2018.04.18','2018.04.19','2018.04.20',
                 '2018.04.23','2018.04.24','2018.04.25','2018.04.26','2018.04.27',
                 '2018.04.30','2018.05.01','2018.05.02','2018.05.03','2018.05.04',
                 '2018.05.07','2018.05.08','2018.05.09','2018.05.10','2018.05.11',
                 '2018.05.14','2018.05.15','2018.05.16','2018.05.17','2018.05.18',
                 '2018.05.21','2018.05.22','2018.05.23','2018.05.24','2018.05.25',
                 '2018.05.28','2018.05.29','2018.05.30','2018.05.31'])

# init booleans
save_stats = True  

# init file directories
hdf5_directory = 'D:/SDC/py/HDF5/'#'../HDF5/'#
# define files and directories names
filename_prep_IO = hdf5_directory+'IO_mW'+str(data.movingWindow)+'_nE'+str(data.nEventsPerStat)+'_nF'+str(data.nFeatures)+'_180531.hdf5'#
filename_raw = hdf5_directory+'tradeinfo_180531.hdf5'
separators_directory = hdf5_directory+'separators_180531/'

# reset file
#reset = False
#if reset:
#    f_w = h5py.File(filename_prep_IO,'w')
#    f_w.close()

# reset only one asset
reset_asset = ''

# init hdf5 files
f_prep_IO = h5py.File(filename_prep_IO,'a')
f_raw = h5py.File(filename_raw,'r')
# init total number of samples
m = 0
# max number of input channels
nChannels = int(data.nEventsPerStat/data.movingWindow)
# index asset
ass_idx = 0
# loop over all assets
for ass in data.assets:
    thisAsset = data.AllAssets[str(ass)]
    print(str(ass)+". "+thisAsset)
    tic = time.time()
    # open file for read
    
    group_raw = f_raw[thisAsset]
    #bid_means[ass_idx] = np.mean(group_raw["SymbolBid"])
    # load separators
    separators = load_separators(data, thisAsset, separators_directory, from_txt=1)
    
    if thisAsset==reset_asset:
        print(separators)
        del f_prep_IO[thisAsset]
    # crate asset_group if does not exist
    if thisAsset not in f_prep_IO:
        # init total stats
        ass_group = f_prep_IO.create_group(thisAsset)
    else:
        # retrive ass group if exists
        ass_group = f_prep_IO[thisAsset]
    # init or load total stats
    stats = {}
    if save_stats:
        
        stats["means_t_in"] = np.zeros((nChannels,data.nFeatures))
        stats["stds_t_in"] = np.zeros((nChannels,data.nFeatures))
        stats["means_t_out"] = np.zeros((1,len(data.lookAheadVector)))
        stats["stds_t_out"] = np.zeros((1,len(data.lookAheadVector)))
        stats["m_t_in"] = 0
        stats["m_t_out"]  = 0
    else:
        stats["means_t_in"] = ass_group.attrs.get("means_t_in")
        stats["stds_t_in"] = ass_group.attrs.get("stds_t_in")
        stats["means_t_out"] = ass_group.attrs.get("means_t_out")
        stats["stds_t_out"] = ass_group.attrs.get("stds_t_out")
        stats["m_t_in"] = ass_group.attrs.get("m_t_in")
        stats["m_t_out"] = ass_group.attrs.get("m_t_out")
            
    # loop over separators
    for s in range(0,len(separators)-1,2):
        # number of events within this separator chunk
        nE = separators.index[s+1]-separators.index[s]+1
        #print(nE)
        # check if number of events is not enough to build two features and one return
        if nE>=2*data.nEventsPerStat:
            print("\ts {0:d} of {1:d}".format(int(s/2),int(len(separators)/2-1))+
                  ". From "+separators.DateTime.iloc[s]+" to "+separators.DateTime.iloc[s+1])
            #print("\t"+separators.DateTime.iloc[s]+" to "+separators.DateTime.iloc[s+1])
            # calculate features, returns and stats from raw data
            IO_prep, stats = get_features_results_stats_from_raw(
                    data, thisAsset, separators, f_prep_IO, group_raw, 
                    stats, hdf5_directory, s, save_stats)
                
        else:
            print("\ts {0:d} of {1:d}. Not enough entries. Skipped.".format(int(s/2),int(len(separators)/2-1)))
    # end of for s in range(0,len(separators)-1,2):
    
    # update asset index
    ass_idx += 1
    # save stats in attributes
    if save_stats:
        # normalize stats
        means_t_in = stats["means_t_in"]/stats["m_t_in"]
        stds_t_in = stats["stds_t_in"]/stats["m_t_in"]
        means_t_out = stats["means_t_out"]/stats["m_t_out"]
        stds_t_out = stats["stds_t_out"]/stats["m_t_out"]
        #save total stats as attributes
        ass_group.attrs.create("means_t_in", means_t_in, dtype=float)
        ass_group.attrs.create("stds_t_in", stds_t_in, dtype=float)
        ass_group.attrs.create("means_t_out", means_t_out, dtype=float)
        ass_group.attrs.create("stds_t_out", stds_t_out, dtype=float)
        ass_group.attrs.create("m_t_in", stats["m_t_in"], dtype=int)
        ass_group.attrs.create("m_t_out", stats["m_t_out"], dtype=int)
        # print number of IO samples
        print("\tStats saved. m_t_in="+str(stats["m_t_in"])+", m_t_out="+str(stats["m_t_out"]))
        
    # update total number of samples
    m += stats["m_t_out"]
    # flush content file
    f_prep_IO.flush()
    
    print("\tTime for "+thisAsset+":"+str(np.floor(time.time()-tic))+"s"+
          ". Total time:"+str(np.floor(time.time()-ticTotal))+"s")
    
    #print("Total time:"+str(np.floor(time.time()-ticTotal))+"s")
# end of for ass in data.assets:

# create number of samps attribute 
if save_stats:
    print("total number of samps m="+str(m))
    f_prep_IO.attrs.create('m', m, dtype=int)
# close files
f_prep_IO.close()
f_raw.close()