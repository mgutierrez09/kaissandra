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
import numpy as np
import logging
import logging.handlers
from multiprocessing import Process, Queue

from kaissandra.log import config_logger_online, worker_configurer_online
from kaissandra.config import Config as C
from kaissandra.local_config import local_vars
import kaissandra.prod.communication as ct

def listener_process(queue, configurer):
    configurer()
    while True:
        try:
            record = queue.get()
            if record is None:  # We send this as a sentinel to tell the listener to quit.
                break
            logger = logging.getLogger(record.name)
            logger.handle(record)  # No level or filter logic applied - just do it!
        except Exception:
            import sys, traceback
            print('Whoops! Problem:', file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
    print("EXIT Log queue")

def control(running_assets, timeout=15, queues=[], queues_prior=[], send_info_api=False):
    """ Master function to manage all controlling functions such as connection
    control or log control 
    Args:
        - running_assets (list): list of assets being tracked by trader 
        - timeout (int): max timeout in minutes before reseting networks """
    
    
    directory_MT5 = local_vars.directory_MT5_IO
    directory_io = local_vars.io_live_dir
    reset_command = 'RESET'
    reset = False
    AllAssets = C.AllAssets
    timeouts = [time.time() for _ in range(len(running_assets))]
    # get last file in asset channel if not empty, empty string otherwise
    list_last_file = [sorted(os.listdir(directory_MT5+AllAssets[str(ass_id)]+"/"))[-1] \
                      if len(sorted(os.listdir(directory_MT5+AllAssets[str(ass_id)]+"/")))>0 \
                      else '' for ass_id in running_assets]
    
    list_num_files = [{'max':-1,'curr':0, 'time':dt.datetime.now()} for i in running_assets]
    # crate log queue and lauch it in a separate process
    log_queue = Queue(-1)
    listener = Process(target=listener_process, args=(log_queue, config_logger_online))
    listener.start()
    
    token_header = ct.build_token_header(ct.post_token())
    # launch queue listeners as independent processes
    kwargs = {'send_info_api':send_info_api, 'token_header':token_header, 'priority':False}
    kwargs_prior = {'send_info_api':send_info_api, 'token_header':token_header, 'priority':True}
    for q, queue in enumerate(queues):
        queue_prior = queues_prior[q]
        disp = Process(target=listen_trader_connection, args=[queue, log_queue, worker_configurer_online, running_assets[q]], kwargs=kwargs)
        disp.start()
        disp = Process(target=listen_trader_connection, args=[queue_prior, log_queue, worker_configurer_online, running_assets[q]], kwargs=kwargs_prior)
        disp.start()
    
    # monitor connections
    watchdog_counter = 0
    ass_idx = 0
    run = True
    print("Control running...")
    while run:
        # control connection
        list_last_file, list_num_files, timeouts, reset = control_broker_connection(AllAssets, running_assets, 
                                                      timeout, directory_io,
                                                      reset_command, directory_MT5, 
                                                      list_last_file, list_num_files, timeouts, 
                                                      reset, log_queue)
        
        # loop over assets
#        for ass_idx, ass_id in enumerate(running_assets):
#            # check for new log info to send
#            thisAsset = AllAssets[str(ass_id)]
#            directory_io_ass = directory_io+thisAsset+"/"
#            if os.path.exists(directory_io_ass+"TRADERLOG"):
#                pass
#            elif os.path.exists(directory_io_ass+"NETWORKLOG"):
#                pass
        ct.send_account_status(token_header)
        ct.send_positions_status(token_header)
        asset = AllAssets[str(running_assets[ass_idx])]
        MSG = " Current files in dir: "+str(list_num_files[ass_idx]['curr'])+\
            ". Max: "+str(list_num_files[ass_idx]['max'])+". Time: "+list_num_files[ass_idx]['time'].strftime("%d.%m.%Y %H:%M:%S")
        print(asset+MSG)
        ct.send_trader_log(MSG, asset, token_header)
        ass_idx = np.mod(ass_idx+1,len(running_assets))
        ct.check_for_warnings()
        time.sleep(5)
        
        watchdog_counter += 1
        # check parameters every minute
        if watchdog_counter==1:
            try:
                ct.check_params()
                token = ct.get_token()
                print(token)
                # wake up server
                watchdog_counter = 0
                # send number of files in Broker communication directory of one asset
                
            except ConnectionError:
                print("WARNING! Connection Error in control() of control.py.")
            
        if os.path.exists(directory_io+'SD'):
            os.remove(directory_io+'SD')
            # shutting down queues
            log_queue.put(None)
#            for q, queue in enumerate(queues):
#                queue_prior = queues_prior[q]
#                queue.put({'FUNC':'SD'})
#                queue_prior.put({'FUNC':'SD'})
            run = False
        
    print("EXIT control")
            
def listen_trader_connection(queue, log_queue, configurer, ass_id, send_info_api=False, token_header=None, priority=False):
    """ Local connection with trader through a queue """
    
    configurer(log_queue)
    name = C.AllAssets[str(ass_id)]
    print("Reading queue")
    assets_opened = {}
    run = True
    while run:
        info = queue.get()         # Read from the queue
        if priority:
            print("From priority queue: ")
            
        else:
            print("From regular queue: ")
        print(info)
        # send log to server
        if send_info_api and info['FUNC'] == 'LOG':
            
#                print(info['MSG'])
            logger = logging.getLogger(name)
            #level = logging.INFO
            message = info['MSG']
            logger.info(message)
            if info['ORIGIN'] == 'NET':
                ct.send_network_log(info['MSG'], info['ASS'], token_header)
            elif info['ORIGIN'] == 'TRADE':
                ct.send_trader_log(info['MSG'], info['ASS'], token_header)
            elif info['ORIGIN'] == 'MONITORING':
                ct.send_monitoring_log(info['MSG'], info['ASS'], token_header)
            else:
                print("WARNING! Info origing "+info["ORIGIN"]+" unknown. Skipped")
        elif send_info_api and info['FUNC'] == 'POS':
            params = info['PARAMS']
            if info["EVENT"] == "OPEN":
                position_json = ct.send_open_position(params, info["SESS_ID"], 
                                                      token_header)
                if 'id' in position_json:
                    assets_opened[position_json['asset']] = position_json['id']
                else:
                    print("WARNING! id NOT in position_json")
            elif info["EVENT"] == "EXTEND":
                ct.send_extend_position(params, info["ASSET"], token_header)
#                if info["ASSET"] in assets_opened:
#                    pos_id = assets_opened[info["ASSET"]]
#                    ct.send_extend_position(params, pos_id, token_header)
#                    
#                else:
#                    print("WARNING! "+info["ASSET"]+" not in assets_opened. send_extend_position skipped.")
            elif info["EVENT"] == "NOTEXTEND":
                ct.send_not_extend_position(params, info["ASSET"], token_header)
#                if info["ASSET"] in assets_opened:
#                    pos_id = assets_opened[info["ASSET"]]
#                    ct.send_not_extend_position(params, pos_id, token_header)
#                else:
#                    print("WARNING! "+info["ASSET"]+" not in assets_opened. send_not_extend_position skipped.")
            elif info["EVENT"] == "CLOSE":
                dirfilename = info["DIRFILENAME"]
                ct.send_close_position(params, info["ASSET"], dirfilename, token_header)
#                if info["ASSET"] in assets_opened:
#                    pos_id = assets_opened[info["ASSET"]]
#                    dirfilename = info["DIRFILENAME"]
#                    ct.send_close_position(params, pos_id, dirfilename, 
#                                           token_header)
#                else:
#                    print("WARNING! "+info["ASSET"]+" not in assets_opened. send_close_position skipped.")
            else:
                print("WARNING! EVENT "+info["EVENT"]+" unsupported. Ignored")
                
        elif info['FUNC'] == 'SD':
            run = False
#        except Exception as e:
#            print("WARNING! Error in when reading queue in listen_trader_connection of control.py: "+str(e))
    print("EXIT queue")
        
    
def control_broker_connection(AllAssets, running_assets, timeout, directory_io,
                           reset_command, directory_MT5, list_last_file, list_num_files, 
                           timeouts, reset, log_queue):
    """ Controls the connection and arrival of new info from trader and 
    sends reset command in case connection is lost """
    for ass_idx, ass_id in enumerate(running_assets):
        thisAsset = AllAssets[str(ass_id)]
        directory_MT5_IO_ass = directory_MT5+thisAsset+"/"
        directory_io_ass = directory_io+thisAsset+"/"
        listAllFiles = sorted(os.listdir(directory_MT5_IO_ass))
        # track max delay in ticks processing
        if len(listAllFiles)>list_num_files[ass_idx]['max']:
            max_num = len(listAllFiles)
            occured = dt.datetime.now()
        else:
            max_num = list_num_files[ass_idx]['max']
            occured = list_num_files[ass_idx]['time']
        list_num_files[ass_idx] = {'max':max_num,'curr':len(listAllFiles),'time':occured}
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
    print(dt.datetime.strftime(dt.datetime.now(),'%y.%m.%d %H:%M:%S')+
          " Min TO = {0:.2f} mins".format(min_to/60))
#    print("\r"+dt.datetime.strftime(dt.datetime.now(),'%y.%m.%d %H:%M:%S')+
#          " Min TO = {0:.2f} mins".format(min_to/60), sep=' ', end='', flush=True)
#    if min_to>timeout*60 and not reset:
#        # Reset networks
#        reset = True
#        ct.send_command(directory_io_ass, reset_command)
    return list_last_file, list_num_files, timeouts, reset
            
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
    

if __name__=='__main__':
    control([1,2], timeout=timeout)#[1,2,3,4,7,8,10,11,12,13,14,16,17,19,27,28,29,30,31,32]