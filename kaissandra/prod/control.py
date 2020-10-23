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
from threading import Thread

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
            import traceback
            print('Whoops! Problem:', file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
    print("EXIT Log queue")

def control(running_assets, timeout=15, queues=[], queues_prior=[], send_info_api=False, from_main=False, test=False):
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
    for ass_id in running_assets:
        if not os.path.exists(directory_MT5+AllAssets[str(ass_id)]+"/"):
            os.makedirs(directory_MT5+AllAssets[str(ass_id)]+"/")
    # get last file in asset channel if not empty, empty string otherwise
    list_last_file = [sorted(os.listdir(directory_MT5+AllAssets[str(ass_id)]+"/"))[-1] \
                      if len(sorted(os.listdir(directory_MT5+AllAssets[str(ass_id)]+"/")))>0 \
                      else '' for ass_id in running_assets]
    
    list_num_files = [{'max':-1,'curr':0, 'time':dt.datetime.now()} for i in running_assets]
    # crate log queue and lauch it in a separate process
    if len(queues)>0:
        log_queue = Queue(-1)
        listener = Process(target=listener_process, args=(log_queue, config_logger_online))
        listener.start()
    if send_info_api:
        token = ct.post_token()
        token_header = ct.build_token_header(token)
    else:
        token = None
        token_header = None
    # launch queue listeners as independent processes
    kwargs = {'send_info_api':send_info_api, 'token_header':token_header, 'priority':False}
    kwargs_prior = {'send_info_api':send_info_api, 'token_header':token_header, 'priority':True}
    for q, queue in enumerate(queues):
        queue_prior = queues_prior[q]
        disp = Process(target=listen_trader_connection, args=[queue, log_queue, worker_configurer_online, running_assets[q]], kwargs=kwargs)
        disp.start()
        disp = Process(target=listen_trader_connection, args=[queue_prior, log_queue, worker_configurer_online, running_assets[q]], kwargs=kwargs_prior)
        disp.start()
        time.sleep(10)
    
    # monitor connections
    watchdog_counter = 0
    ass_idx = 0
    run = True
    print("Control running...")
    logger = logging.getLogger("CONTROL")
    while run:
        # control connection
        list_last_file, list_num_files, timeouts, reset = control_broker_connection(AllAssets, running_assets, 
                                                      timeout, directory_io,
                                                      reset_command, directory_MT5, 
                                                      list_last_file, list_num_files, timeouts, 
                                                      reset, from_main, token_header, logger, send_info_api)
        
        asset = AllAssets[str(running_assets[ass_idx])]
        if not test:
            MSG = " Current files in dir: "+str(list_num_files[ass_idx]['curr'])+\
                ". Max: "+str(list_num_files[ass_idx]['max'])+". Time: "+list_num_files[ass_idx]['time'].strftime("%d.%m.%Y %H:%M:%S")
            print(asset+MSG)
            if send_info_api:
                ct.send_trader_log(MSG, asset, token_header)
        ass_idx = np.mod(ass_idx+1,len(running_assets))
        
        time.sleep(1)
        
        watchdog_counter += 1
        # check parameters every minute
        if watchdog_counter==5:
            try:
                ct.check_for_warnings(token_header=token_header, send_info_api=send_info_api)
                if send_info_api:
                    ct.check_params(token_header=token_header)
                    ct.send_account_status(token_header)
                    ct.send_positions_status(token_header)
                watchdog_counter = 0
            except ConnectionError:
                print("WARNING! Connection Error in control() of control.py.")
            
            if os.path.exists(directory_io+'SD'):
                os.remove(directory_io+'SD')
                # shutting down queues
                if len(queues)>0:
                    log_queue.put(None)
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
        run, assets_opened = arrange_message_and_send(info, assets_opened, name, token_header, send_info_api)
#        # send log to server
#        if send_info_api and info['FUNC'] == 'LOG':
#            print(info)
#            logger = logging.getLogger(name)
#            message = info['MSG']
#            logger.info(message)
#            if info['ORIGIN'] == 'NET':
#                ct.send_network_log(info['MSG'], info['ASS'], token_header)
#            elif info['ORIGIN'] == 'TRADE':
#                ct.send_trader_log(info['MSG'], info['ASS'], token_header)
#            elif info['ORIGIN'] == 'MONITORING':
#                ct.send_monitoring_log(info['MSG'], info['ASS'], token_header)
#            else:
#                print("WARNING! Info origing "+info["ORIGIN"]+" unknown. Skipped")
#        elif send_info_api and info['FUNC'] == 'POS':
#            print(info)
#            params = info['PARAMS']
#            if info["EVENT"] == "OPEN":
#                position_json = ct.send_open_position(params, info["SESS_ID"], 
#                                                      token_header)
#                if 'id' in position_json:
#                    assets_opened[position_json['asset']] = position_json['id']
#                else:
#                    print("WARNING! id NOT in position_json")
#            elif info["EVENT"] == "EXTEND":
#                ct.send_extend_position(params, info["ASSET"], info["STRATEGY"], token_header)
#            elif info["EVENT"] == "NOTEXTEND":
#                ct.send_not_extend_position(params, info["ASSET"], info["STRATEGY"], token_header)
#            elif info["EVENT"] == "CLOSE":
#                dirfilename = info["DIRFILENAME"]
#                ct.send_close_position(params, info["ASSET"], info["STRATEGY"], dirfilename, token_header)
#            else:
#                print("WARNING! EVENT "+info["EVENT"]+" unsupported. Ignored")
#                
#        elif send_info_api and info['FUNC'] == 'CONFIG':
#            ct.confirm_config_info(info['CONFIG'], info["ASSET"], info["ORIGIN"], token_header)
#        elif info['FUNC'] == 'SD':
#            print(info)
#            run = False
#        except Exception as e:
#            print("WARNING! Error in when reading queue in listen_trader_connection of control.py: "+str(e))
    print("EXIT queue")
        
def arrange_message_and_send(info, assets_opened, name, token_header, send_info_api):
    """  """
    run = True
    if send_info_api and info['FUNC'] == 'LOG':
        print(info)
        logger = logging.getLogger(name)
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
        print(info)
        params = info['PARAMS']
        if info["EVENT"] == "OPEN":
            position_json = ct.send_open_position(params, info["SESS_ID"], 
                                                  token_header)
            if 'id' in position_json:
                assets_opened[position_json['asset']] = position_json['id']
            else:
                print("WARNING! id NOT in position_json")
        elif info["EVENT"] == "EXTEND":
            ct.send_extend_position(params, info["ASSET"], info["STRATEGY"], token_header)
        elif info["EVENT"] == "NOTEXTEND":
            ct.send_not_extend_position(params, info["ASSET"], info["STRATEGY"], token_header)
        elif info["EVENT"] == "CLOSE":
            dirfilename = info["DIRFILENAME"]
            ct.send_close_position(params, info["ASSET"], info["STRATEGY"], dirfilename, token_header)
        else:
            print("WARNING! EVENT "+info["EVENT"]+" unsupported. Ignored")
            
    elif send_info_api and info['FUNC'] == 'CONFIG':
        ct.confirm_config_info(info['CONFIG'], info["ASSET"], info["ORIGIN"], token_header)
    elif info['FUNC'] == 'SD':
        print(info)
        run = False
    return run, assets_opened

def control_broker_connection(AllAssets, running_assets, timeout, directory_io,
                           reset_command, directory_MT5, list_last_file, list_num_files, 
                           timeouts, reset, from_main, token_header, logger, send_info_api):
    """ Controls the connection and arrival of new info from trader and 
    sends reset command in case connection is lost """
    for ass_idx, ass_id in enumerate(running_assets):
        try:
            thisAsset = AllAssets[str(ass_id)]
            directory_MT5_IO_ass = directory_MT5+thisAsset+"/"
            #directory_io_ass = directory_io+thisAsset+"/"
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
        except:
            print("WARNING! Error in control_broker_connection. Skipped")
        # check for logs in disk if from main
        if send_info_api:
            try:
                if from_main:
                    ass_dir = local_vars.local_log_comm+thisAsset+'/'
                    list_logs = sorted(os.listdir(ass_dir))
                    for file in list_logs:
                        fh = open(ass_dir+file,"r")
                        list_info = fh.read()[:-1].split(",")
                        #print(list_info)
                        fh.close()
                        os.remove(ass_dir+file)
                        # split message
                        info = get_dict_from_list(list_info)
                        #print(info)
                        #send log
                        Thread(target=arrange_message_and_send, args=(info, None, thisAsset, token_header, send_info_api)).start()
                        #_, _ = arrange_message_and_send(info, None, thisAsset, token_header, send_info_api)
    #                    message = info['MSG']
    #                    logger.info(message)
    #                    ct.send_log(info, token_header=token_header)
            except Exception:
                import traceback
                print('Whoops! Problem:', file=sys.stderr)
                traceback.print_exc(file=sys.stderr)
                message = "Error in control_broker_connection of kaissandra.prod.control"
                logger.exception(message)
                if os.path.exists(ass_dir+file):
                    os.remove(ass_dir+file)
    
        
    reset = False
    return list_last_file, list_num_files, timeouts, reset

def get_dict_from_list(list_info):
    """  """
    info = {}
    for i in range(0,len(list_info)-1,2):
        info[list_info[i]] = list_info[i+1]
    return info
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

from kaissandra.log import config_logger_online, worker_configurer_online
from kaissandra.config import Config as C
from kaissandra.local_config import local_vars
import kaissandra.prod.communication as ct  

if __name__=='__main__':
    for ass_idx, ass_id in enumerate(local_vars.ASSETS):
        thisAsset = C.AllAssets[str(ass_id)]
        ass_dir = local_vars.local_log_comm+thisAsset+'/'
        if not os.path.exists(ass_dir):
            os.makedirs(ass_dir)
    control(local_vars.ASSETS, send_info_api=local_vars.API, from_main=True)#