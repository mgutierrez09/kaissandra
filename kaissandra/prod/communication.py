# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 11:20:05 2019

@author: mgutierrez
"""
import sys
import os
import re
import time
import requests
import datetime as dt
import numpy as np
import logging
import logging.handlers

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
    #print(" "+directory_MT5_ass+command+" command sent")
    return None
    
def shutdown(id=None):
    """  """
    io_dir = LC.io_live_dir
    
    
    AllAssets = Config.AllAssets
    for asset_key in AllAssets:
        asset = AllAssets[asset_key]
        
        try:
            fh = open(io_dir+asset+'/SD',"w")
            fh.close()
        except FileNotFoundError:
            #print("FileNotFoundError")
            pass
    if id!=None:
        close_session(id)
    
        
    return None

def shutdown_control():
    # shutdown control
    io_dir = LC.io_live_dir
    try:
        fh = open(io_dir+'SD',"w")
        fh.close()
    except FileNotFoundError:
        #print("FileNotFoundError")
        pass

def hibernate():
    """  """
    io_dir = LC.io_live_dir
    
    
    AllAssets = Config.AllAssets
    for asset_key in AllAssets:
        asset = AllAssets[asset_key]
        
        try:
            fh = open(io_dir+asset+'/HIBER',"w")
            fh.close()
        except FileNotFoundError:
            #print("FileNotFoundError")
            pass

#def close_session(id):
#    """  """
#    from kaissandra.prod.api import API
#    API().close_session(id)
    
def pause():
    """  """
    io_dir = LC.io_live_dir
    # shutdown control
    AllAssets = Config.AllAssets
    for asset_key in AllAssets:
        asset = AllAssets[asset_key]
        
        try:
            fh = open(io_dir+asset+'/PA',"w")
            fh.close()
            print(asset)
        except FileNotFoundError:
            #print("Asset not running")
            pass

def check_params(config_name='', token_header=None):
    """  """
    try:
        io_dir = LC.io_live_dir
        AllAssets = Config.AllAssets
        cl_command = False # close command
        sd_command = False # shutdown command
        hi_command = False # hibernate command
        if config_name=='':
            # get config from server and pass it to live session
            #token = post_token()
            #if token_header:
            json = get_config_session(token_header=token_header)
#            else:
#                token = post_token()
#                print("WARNING! Unable to build token in check_params in kaissandra.prod.communication. Skipped.")
#                return None
            if 'config' in json:
                config = json['config']
                if 'config_name' in config:
                    save_config(config, from_server=True)
            else:
                config = {}
            if 'commands' in json:
                commands = json['commands']
                if len(commands)>0:
                    print("Command/s found: ")
                    print(commands)
                if 'CL' in commands:
                    cl_command = True
                if 'SD' in commands:
                    sd_command = True
                if 'HI' in commands:
                    hi_command = True
            if 'who' not in json or json['who']==[]:
                who = [AllAssets[a] for a in AllAssets]
            else:
                who = json['who']
                print(who)
        for asset_key in AllAssets:
            asset = AllAssets[asset_key]
            
            try:
                
                # if parameters come locally
    #            if config_name!='':
    #                # let live session load the file from disk
    #                fh = open(io_dir+asset+'/PARAM',"w")
    #                fh.write(config_name)
    #                fh.close()
                if config_name=='' and config and len(config)>0 and asset in who:
    #                print(asset+": config saved")
                    
                    fh = open(io_dir+asset+'/PARAM',"w")
                    fh.write(config['config_name']+'remote')
                    fh.close()
                    # reset server config json to empty
                # pass config_name through disk to trader to load configuration file
                elif config_name!='' and asset in who:
                    fh = open(io_dir+asset+'/PARAM',"w")
                    fh.write(config_name)
                    fh.close()
                
                if cl_command and asset in who:
                    close_position(asset)
    #                print(asset+": cl_command")
            except FileNotFoundError:
                print("WARNING! "+io_dir+asset+"/ not fount")
                #print("Asset not running")
        # execute command
        if sd_command:
            # command is shutdown
            shutdown()
            print("shutdown")
        if hi_command:
            hibernate()
            print("From kaissandra.prod.communication.check_params: hibernate")
        if (config_name=='' and config and len(config)>0) or cl_command or sd_command or hi_command:
            if not token_header:
                token_header = build_token_header(post_token())
            set_config_session({'config':{},'commands':[],'who':[]}, token_header)
    except Exception:
        import sys, traceback
        print('Whoops! Problem:', file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        logger = logging.getLogger("COMMUNICATION")
        message = "Error in check_params of kaissandra.communication"
        logger.exception(message)
        
    return None

def resume():
    """  """
    io_dir = LC.io_live_dir
    AllAssets = Config.AllAssets
    for asset_key in AllAssets:
        asset = AllAssets[asset_key]
        
        try:
            fh = open(io_dir+asset+'/RE',"w")
            fh.close()
            print(asset)
        except FileNotFoundError:
            pass
            #print("FileNotFoundError")
    return None

def close_positions():
    """ Close all positions from py """
    directory_MT5 = LC.directory_MT5_IO
    command = "LC"
    AllAssets = Config.AllAssets
    for asset_key in AllAssets:
        thisAsset = AllAssets[asset_key]
        directory_MT5_ass = directory_MT5+thisAsset+"/"
        if os.path.exists(directory_MT5_ass):
            #print("Sent command to "+directory_MT5_ass)
            send_command(directory_MT5_ass, command)
    return None
            
def close_position(thisAsset):
    """ Close all positions from py """
    directory_MT5 = LC.directory_MT5_IO
    command = "LC"
    directory_MT5_ass = directory_MT5+thisAsset+"/"
    if os.path.exists(directory_MT5_ass):
        #print("Sent command to "+directory_MT5_ass)
        send_command(directory_MT5_ass, command)
    return None
        
def reset_networks():
    """  """
    directory_io = LC.io_live_dir
    command = "RESET"
    AllAssets = Config.AllAssets
    for asset_key in AllAssets:
        thisAsset = AllAssets[asset_key]
        directory_io_ass = directory_io+thisAsset+"/"
        if os.path.exists(directory_io_ass):
            #print("Sent command to "+directory_MT5_ass)
            send_command(directory_io_ass, command)
    return None
            
def send_close_command(asset):
    """ Send command for closeing position to MT5 software """
    directory_MT5_ass2close = LC.directory_MT5_IO+asset+"/"
    # load network output
    try:
        fh = open(directory_MT5_ass2close+"LC","w")
        fh.close()
    except (PermissionError, FileNotFoundError):
        print("Error writing LC")
    return None

def send_close_commands_all():
    """  """
    AllAssets = Config.AllAssets
    for asset_key in AllAssets:
        thisAsset = AllAssets[asset_key]
        print(thisAsset)
        send_close_command(thisAsset)
    return None
        
def get_account_status():
    """ Get account status from broker """
    #success = 0
    ass_idx = np.random.randint(len(LC.ASSETS))
    try:
        ##### WARNING! #####
        dirfilename = LC.directory_MT5_account+Config.AllAssets[str(LC.ASSETS[ass_idx])]+'/Status.txt'
        if os.path.exists(dirfilename):
            # load network output
            #while not success:
            #try:
            fh = open(dirfilename,"r")
            info_close = fh.read()[:-1]
            # close file
            fh.close()
            #success = 1
            #stop_timer(ass_idx)
#            except PermissionError:
#                print("Error reading Account status")
                    #time.sleep(.1)
            info_str = info_close.split(',')
            #print(info_close)
            balance = float(info_str[0])
            leverage = float(info_str[1])
            equity = float(info_str[2])
            profits = float(info_str[3])
            margin = float(info_str[4])
            free_margin = float(info_str[5])
            
            print("Margin {4:.2f} Free Margin {5:.2f} Balance {0:.2f} Leverage {1:.2f} Equity {2:.2f} Profits {3:.2f}"\
              .format(balance,leverage,equity,profits,margin,free_margin))
            
            status = {'balance':balance, 'leverage':leverage, 
                      'equity':equity, 'profits':profits, 
                      'margin':margin,'free_margin':free_margin,
                      'error':False}
            return status
        else:
            print("WARNING! Account Status file not found. Turning to default")        
            balance = 500.0
            leverage = 30
            equity = balance
            profits = 0.0
            margin = 0.0
            free_margin = 0.0
        
        
    except:
        print("WARNING! Error in get_account_status. Skipped")
        balance = 500.0
        leverage = 30
        equity = balance
        profits = 0.0
        margin = 0.0
        free_margin = 0.0
        
    status = {'balance':balance, 'leverage':leverage, 'equity':equity, 
              'profits':profits, 'margin':margin,'free_margin':free_margin, 'error':True}
    return status

def check_for_warnings(token_header=None, send_info_api=True):
    """ Check for warning messages from broker """
    AllAssets = Config.AllAssets
    try:
        for asset_key in AllAssets:
            thisAsset = AllAssets[asset_key]
            dirfilename = LC.directory_MT5_comm+thisAsset+'/WARNING'
            if os.path.exists(dirfilename):
                # load network output
                success = 0
                while not success:
                    try:
                        fh = open(dirfilename,"r")
                        info = fh.read()[:-1]
                        # close file
                        fh.close()
                        success = 1
                        #stop_timer(ass_idx)
                    except PermissionError:
                        print("Error reading position status")
                msgLog = "\n\nWARNING FROM BROKER in "+thisAsset+": "+info+"\n\n"
                print(thisAsset+msgLog)
                if not token_header and send_info_api:
                    token_header= build_token_header(post_token())
                if send_info_api:
                    send_monitoring_log(msgLog, thisAsset, token_header)
                
                        
    except Exception:
        import sys, traceback
        print('Whoops! Problem:', file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        logger = logging.getLogger("COMMUNICATION")
        message = "Error in check_for_warnings of kaissandra.communication"
        logger.exception(message)
    return None

def get_positions_status():
    """ Get account status from broker """
    
    ##### WARNING! #####
    AllAssets = Config.AllAssets
    max_strategies_per_asset = 30
    status = {}
    try:
        for asset_key in AllAssets:
            thisAsset = AllAssets[asset_key]
            for str_idx in range(max_strategies_per_asset):
                dirfilename = LC.directory_MT5_comm+thisAsset+"/POSINFO"+str(str_idx)+".txt"
                if os.path.exists(dirfilename):
                    # load network output
                    success = 0
                    while not success:
                        try:
                            fh = open(dirfilename,"r")
                            info_close = fh.read()[:-1]
                            # close file
                            fh.close()
                            success = 1
                            #stop_timer(ass_idx)
                        except PermissionError:
                            print("Error reading position status")
                            time.sleep(.1)
                    info_str = info_close.split(',')
                    #print(info_str)
                    dti = info_str[8]
                    pos_id = int(info_str[0])
                    volume = float(info_str[1])
                    open_price = float(info_str[2])
                    current_price = float(info_str[3])
                    current_profit = float(info_str[4])
                    swap = float(info_str[5])
                    deadline = int(info_str[6])
                    direction = int(info_str[7])
                    
                    print(dti+" "+thisAsset+"_"+str(str_idx)+": pos_id {0:d} volume {1:.2f} open price {2:.2f} current price {3:.2f} swap {5:.2f} dir {7:d} dealine in {6:d} current profit {4:.2f}"\
                      .format(pos_id, volume, open_price, current_price, current_profit, swap, deadline, direction))
                    
                    status[thisAsset+"_"+str(str_idx)] = {
                                         'dti':dti,
                                         'pos_id':pos_id, 
                                         'volume':volume, 
                                         'open_price':open_price, 
                                         'current_price':current_price,
                                         'current_profit':current_profit,
                                         'swap':swap,
                                         'deadline':deadline,
                                         'direction':direction}
    except Exception:
        import sys, traceback
        print('Whoops! Problem:', file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        logger = logging.getLogger("COMMUNICATION")
        message = "Error in get_positions_status of kaissandra.communication"
        logger.exception(message)
        
    print("Total open positions: "+str(len(status)))
    return status


def post_token():
    """ POST request to create new token if expired or retrieve current one """
    #print(LC.URL+'tokens')
    try:
        response = requests.post(LC.URL+'tokens',auth=(CC.USERNAME,CC.PASSWORD), verify=True)
        if response.status_code == 200:
            token = response.json()['token']
            return token
        else:
            print(response.text)
        return None
    except Exception:
        import sys, traceback
        print('Whoops! Problem:', file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        logger = logging.getLogger("COMMUNICATION")
        message = "Error in post_token of kaissandra.communication"
        logger.exception(message)
        return None

def build_token_header(token):
    """ Build header for token """
    if token:
        return {'Authorization': 'Bearer '+token}
    else:
        return {'Authorization': 'Bearer '+""}

def send_log(info, token_header=None):
    """  """
    try:
        if not token_header:
            token_header = build_token_header(get_token())
        if info['ORIGIN'] == 'NET':
            send_network_log(info['MSG'], info['ASS'], token_header)
        elif info['ORIGIN'] == 'TRADE':
            send_trader_log(info['MSG'], info['ASS'], token_header)
        elif info['ORIGIN'] == 'MONITORING':
            send_monitoring_log(info['MSG'], info['ASS'], token_header)
        else:
            print("WARNING! Info origing "+info["ORIGIN"]+" unknown. Skipped")
    except Exception:
        import sys, traceback
        print('Whoops! Problem:', file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        logger = logging.getLogger("COMMUNICATION")
        message = "Error in send_trader_log of kaissandra.communication"
        logger.exception(message)
    return None
                        
def send_trader_log(message, asset, token_header):
    """ Send trader log to server api """
    url_ext = 'logs/traders'
    try:
        response = requests.post(LC.URL+url_ext, json={'Message':message,
                                                       'Asset':asset,
                                                       'Name':CC.TRADERNAME},
                                headers=token_header, verify=True, timeout=10)
#        print(response.json())
    except Exception:
        import sys, traceback
        print('Whoops! Problem:', file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        logger = logging.getLogger("COMMUNICATION")
        message = "Error in send_trader_log of kaissandra.communication"
        logger.exception(message)
    return None
        
def send_monitoring_log(message, asset, token_header):
    """ Send trader log to server api """
    url_ext = 'logs/monitor'
    try:
        response = requests.post(LC.URL+url_ext, json={'Message':message,
                                                       'Asset':asset,
                                                       'Name':CC.TRADERNAME},
                                headers=token_header, verify=True, timeout=10)
#        print(response.json())
    except Exception:
        import sys, traceback
        print('Whoops! Problem:', file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        logger = logging.getLogger("COMMUNICATION")
        message = "Error in send_monitoring_log of kaissandra.communication"
        logger.exception(message)
    return None
        
def send_global_log(message, token_header):
    """ Send trader log to server api """
    url_ext = 'logs/global'
    try:
        response = requests.post(LC.URL+url_ext, json={'Message':message,
                                                       'Name':CC.TRADERNAME},
                                headers=token_header, verify=True, timeout=10)
#        print(response.json())
    except Exception:
        import sys, traceback
        print('Whoops! Problem:', file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        logger = logging.getLogger("COMMUNICATION")
        message = "Error in send_global_log of kaissandra.communication"
        logger.exception(message)
    return None
        
def send_network_log(message, asset, token_header):
    """ Send network log to server api """
    url_ext = 'logs/networks'
#        self.futureSession.post(LC.URL+url_ext, json={'Message':message},
#                                headers=self.build_token_header(), verify=False, timeout=10)
    try:
        response = requests.post(LC.URL+url_ext, json={'Message':message,
                                                       'Asset':asset
                                                       },
        headers=token_header, verify=True, timeout=10)
#        print(response.json())
    except Exception:
        import sys, traceback
        print('Whoops! Problem:', file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        logger = logging.getLogger("COMMUNICATION")
        message = "Error in send_network_log of kaissandra.communication"
        logger.exception(message)
    return None
            
def set_config_session(config, token_header):
    """ Send trader log to server api """
    url_ext = 'traders/sessions/set_session_config'
    try:
        response = requests.put(LC.URL+url_ext, json=config,
                                headers=token_header, verify=True, timeout=10)
        print(response.json())
    except Exception:
        import sys, traceback
        print('Whoops! Problem:', file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        logger = logging.getLogger("COMMUNICATION")
        message = "Error in set_config_session of kaissandra.communication"
        logger.exception(message)
        
def get_config_session(token_header=None):
    """ Send trader log to server api """
    url_ext = 'traders/sessions/get_session_config'
    try:
        if not token_header:
            token_header = build_token_header(post_token())
        response = requests.get(LC.URL+url_ext, 
                                headers=token_header, 
                                verify=True, timeout=10)
        #print(response.json())
        return response.json()
    except Exception:
        import sys, traceback
        print('Whoops! Problem:', file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        logger = logging.getLogger("COMMUNICATION")
        message = "Error in get_config_session of kaissandra.communication"
        logger.exception(message)
        return None
    
def confirm_config_info(config, asset, origin, token_header):
    """ Send confirmation with config info to server """
    url_ext = 'traders/sessions/'+asset+'/confirm_config'
    try:
        response = requests.put(LC.URL+url_ext, json=config,
                                headers=token_header, verify=True, timeout=10)
        print(response.json())
    except Exception:
        import sys, traceback
        print('Whoops! Problem:', file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        logger = logging.getLogger("COMMUNICATION")
        message = "Error in confirm_config_info of kaissandra.communication"
        logger.exception(message)
#    except:
#        print("WARNING! Error in confirm_config_info in kaissandra.prod.communication.")
    
def build_and_set_config(config_name='TESTPARAMUPDATE5'):
    """  """
    config = retrieve_config(config_name)
    token = get_token()
    token_header = {'Authorization': 'Bearer '+token}
    set_config_session({'config':config}, token_header)
    return None
    

def send_open_position(params, session_id, token_header):
    """ Send open position to server api"""
    try:
        url_ext = 'traders/sessions/'+str(session_id)+'/positions/open'
        response = requests.post(LC.URL+url_ext, json=params, headers=
                                                 token_header, verify=True)
        #print("Status code: "+str(response.status_code))
        if response.status_code == 200:
    #        print(response.json())
            return response.json()['Position'][0]
        else:
            print(response.text)
            return {}
    except Exception:
        import sys, traceback
        print('Whoops! Problem:', file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        logger = logging.getLogger("COMMUNICATION")
        message = "Error in send_open_position of kaissandra.communication"
        logger.exception(message)
        return None
    
def send_extend_position(params, pos_id, str_idx, token_header):
    """ Send extend position commnad to server api"""
    try:
        url_ext = 'traders/positions/'+str(pos_id)+'/'+str(str_idx)+'/extend'
        response = requests.post(LC.URL+url_ext, json=params, headers=
                                 token_header, verify=True)
        #print("Status code: "+str(response.status_code))
        if response.status_code == 200:
    #        print(response.json())
            return True
        else:
            print(response.text)
            return False
    except Exception:
        import sys, traceback
        print('Whoops! Problem:', file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        logger = logging.getLogger("COMMUNICATION")
        message = "Error in send_extend_position of kaissandra.communication"
        logger.exception(message)
        return False
    
def send_not_extend_position(params, pos_id, str_idx, token_header):
    """ Send extend position commnad to server api"""
    url_ext = 'traders/positions/'+str(pos_id)+'/'+str(str_idx)+'/notextend'
    try:
        response = requests.post(LC.URL+url_ext, json=params, headers=
                                 token_header, verify=True)
        #print("Status code: "+str(response.status_code))
        if response.status_code == 200:
    #        print(response.json())
            return True
        else:
            print(response.text)
            return False
    except Exception:
        import sys, traceback
        print('Whoops! Problem:', file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        logger = logging.getLogger("COMMUNICATION")
        message = "Error in send_not_extend_position of kaissandra.communication"
        logger.exception(message)
    return None
    
def send_close_position(params, pos_id, str_idx, dirfilename, token_header):
    """ Send close position command to server api """
    
    url_ext = 'traders/positions/'+str(pos_id)+'/'+str(str_idx)+'/close'
    url_file = 'traders/positions/'+str(pos_id)+'/'+str(str_idx)+'/upload'
    try:
        files={'file': open(dirfilename,'rb')}
        response_file = requests.post(LC.URL+url_file, files=files, 
                                      headers=token_header, verify=True)
        response = requests.put(LC.URL+url_ext, json=params, 
                                headers=token_header, verify=True)
        
        #print("Status code: "+str(response.status_code))
        if response.status_code == 200:
    #        print(response.json())
            if response_file.status_code != 200:
                print("WARNING! File not saved in DB")
            else:
                print(response.text)
    except Exception:
        import sys, traceback
        print('Whoops! Problem:', file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        logger = logging.getLogger("COMMUNICATION")
        message = "Error in send_close_position of kaissandra.communication"
        logger.exception(message)

def get_token():
    """  """
    try:
        response = requests.post(LC.URL+'tokens',auth=(CC.USERNAME,CC.PASSWORD), verify=True)
        if response.status_code == 200:
            token = response.json()['token']
            return token
        else:
            return None
    except Exception:
        import sys, traceback
        print('Whoops! Problem:', file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        logger = logging.getLogger("COMMUNICATION")
        message = "Error in get_token of kaissandra.communication"
        logger.exception(message)
        return None
    
def send_account_status(token_header):
    """ Send trader log to server api """
    url_ext = 'traders/account/status'
    status = get_account_status()
    json = status
    json['tradername'] = CC.TRADERNAME
    try:
        response = requests.put(LC.URL+url_ext, json=status,
                                headers=token_header, verify=True, timeout=10)
#        print(response.json())
    except Exception:
        import sys, traceback
        print('Whoops! Problem:', file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        logger = logging.getLogger("COMMUNICATION")
        message = "Error in send_account_status of kaissandra.communication"
        logger.exception(message)
    return None
        
def send_positions_status(token_header):
    """ Send trader log to server api """
    url_ext = 'traders/positions/status'
    status = get_positions_status()
    json = status
    #json['tradername'] = CC.TRADERNAME
    try:
        response = requests.put(LC.URL+url_ext, json=status,
                                headers=token_header, verify=True, timeout=10)
#        print(response.json())
    except Exception:
        import sys, traceback
        print('Whoops! Problem:', file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        logger = logging.getLogger("COMMUNICATION")
        message = "Error in send_positions_status of kaissandra.communication"
        logger.exception(message)
    return None

def send_reset_positions(token_header):
    """ Send trader log to server api """
    url_ext = 'traders/reset_positions'
    try:
        response = requests.post(LC.URL+url_ext,
                                headers=token_header, verify=True, timeout=10)
        print(response.json())
    except Exception:
        import sys, traceback
        print('Whoops! Problem:', file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        logger = logging.getLogger("COMMUNICATION")
        message = "Error in send_reset_positions of kaissandra.communication"
        logger.exception(message)
        
def get_number_positions(token_header):
    """ Send trader log to server api """
    url_ext = 'traders/number_positions'
    try:
        response = requests.get(LC.URL+url_ext,
                                headers=token_header, verify=True, timeout=10)
        print(response.json())
    except Exception:
        import sys, traceback
        print('Whoops! Problem:', file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        logger = logging.getLogger("COMMUNICATION")
        message = "Error in get_number_positions of kaissandra.communication"
        logger.exception(message)
        
def get_user_id(token_header, username):
    """ get user id """
    url_ext = 'users/id'
    try:
        response = requests.get(LC.URL+url_ext, json={'username':username},
                                headers=token_header, verify=True, timeout=10)
        print(response.json())
    except Exception:
        import sys, traceback
        print('Whoops! Problem:', file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        logger = logging.getLogger("COMMUNICATION")
        message = "Error in get_user_id of kaissandra.communication"
        logger.exception(message)
        
def set_budget(token_header, id, budget):
    """ get user id """
    url_ext = 'users/'+str(id)+'/set_budget'
    try:
        response = requests.put(LC.URL+url_ext, json={'budget':budget},
                                headers=token_header, verify=True, timeout=10)
#        print(response.json())
    except Exception:
        import sys, traceback
        print('Whoops! Problem:', file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        logger = logging.getLogger("COMMUNICATION")
        message = "Error in set_budget of kaissandra.communication"
        logger.exception(message)
        
def open_session(config_name, sessiontype, sessiontest):
    """ POST request to open a session """
    sessionname = dt.datetime.strftime(dt.datetime.utcnow(),'%y%m%d%H%M%S')+\
                  '_'+config_name
    params_session = {'sessionname':sessionname,
                      'sessiontype':sessiontype,
                      'sessiontest':sessiontest}
    try:
        url_ext = 'traders/'+str(CC.TRADERNAME)+'/sessions'
        response = requests.post(LC.URL+url_ext, json=params_session, 
                                 headers=build_token_header(post_token()), 
                                 verify=True)
        print("Status code open_session: "+str(response.status_code))
        if response.status_code == 200:
#            print(response.json())
#                self.session_json = response.json()['Session'][0]
            return response.json()['Session'][0]
        else:
            print(response.text)
        return None
    except Exception:
        import sys, traceback
        print('Whoops! Problem:', file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        logger = logging.getLogger("COMMUNICATION")
        message = "Error in open_session of kaissandra.communication"
        logger.exception(message)
        return None
    
def close_session(session_json):
    """  PUT request to close session """
    id = session_json['id']
    url_ext = 'traders/sessions/'+str(id)+'/close'
    try:
        response = requests.put(LC.URL+url_ext, 
                                headers=build_token_header(post_token()), 
                                verify=True)
        print("Status code close_session: "+str(response.status_code))
        if response.status_code == 200:
            return True
        else:
            print(response.text)
        return False
    except Exception:
        import sys, traceback
        print('Whoops! Problem:', file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        logger = logging.getLogger("COMMUNICATION")
        message = "Error in close_session of kaissandra.communication"
        logger.exception(message)
        return False
    
def delete_all_positions():
    """  """
    url_ext = 'traders/reset_positions'
    try:
        response = requests.post(LC.URL+url_ext, 
                                headers=build_token_header(post_token()), 
                                verify=True)
        print("Status code close_session: "+str(response.status_code))
        if response.status_code == 200:
            return True
        else:
            print(response.text)
        return False
    except Exception:
        import sys, traceback
        print('Whoops! Problem:', file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        logger = logging.getLogger("COMMUNICATION")
        message = "Error in close_session of kaissandra.communication"
        logger.exception(message)
        return False
        
from kaissandra.config import retrieve_config, Config, save_config
from kaissandra.local_config import local_vars as LC
from kaissandra.prod.config import Config as CC
    
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
