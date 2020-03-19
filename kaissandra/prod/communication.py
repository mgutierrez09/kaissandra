# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 11:20:05 2019

@author: mgutierrez
"""
import sys
import os
import re
import requests

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
    io_dir = LC.io_live_dir
    AllAssets = Config.AllAssets
    for asset_key in AllAssets:
        asset = AllAssets[asset_key]
        
        try:
            fh = open(io_dir+asset+'/SD',"w")
            fh.close()
            print(asset)
        except FileNotFoundError:
            #print("FileNotFoundError")
            pass
    if id!=None:
        close_session(id)

def close_session(id):
    """  """
    from kaissandra.prod.api import API
    API().close_session(id)
    
def pause():
    """  """
    io_dir = LC.io_live_dir
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

def check_params(config_name=''):
    """  """
    io_dir = LC.io_live_dir
    AllAssets = Config.AllAssets
    for asset_key in AllAssets:
        asset = AllAssets[asset_key]
        
        try:
            fh = open(io_dir+asset+'/PARAM',"w")
            # if parameters come locally
            if config_name!='':
                fh.write(config_name)
            fh.close()
            #print(asset)
        except FileNotFoundError:
            pass
            #print("Asset not running")
    print("Params checked")
    # retrieve previous event if asynch
#    try:
#        
#        url_ext = 'traders/sessions/'+str(self.session_json['id'])+'/get_params'
#        
#        response = requests.get(LC.URL+url_ext, headers=token, verify=True)
#        #print("Status code: "+str(response.status_code))
#        if response.status_code == 200:
#            #print(response.json())
#            #self.positions_json_list.append(response.json()['params'])#[0]
#            return True
#        else:
#            return False
#    
#    except:
#        print("WARNING! Error when enquiring parameters. Skiiped")
#        return False


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
            
def close_position(thisAsset):
    """ Close all positions from py """
    directory_MT5 = LC.directory_MT5_IO
    command = "LC"
    directory_MT5_ass = directory_MT5+thisAsset+"/"
    if os.path.exists(directory_MT5_ass):
        #print("Sent command to "+directory_MT5_ass)
        send_command(directory_MT5_ass, command)
        
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
            
def post_token():
    """ POST request to create new token if expired or retrieve current one """
    print(LC.URL+'tokens')
    response = requests.post(LC.URL+'tokens',auth=(CC.USERNAME,CC.PASSWORD), verify=True)
    if response.status_code == 200:
        token = response.json()['token']
        return token
    else:
        print(response.text)
    return None

def build_token_header(token):
    """ Build header for token """
    return {'Authorization': 'Bearer '+token}
            
def send_trader_log(message, asset, token_header):
    """ Send trader log to server api """
    url_ext = 'logs/traders'
    try:
        response = requests.post(LC.URL+url_ext, json={'Message':message,
                                                       'Asset':asset,
                                                       'Name':CC.TRADERNAME},
                                headers=token_header, verify=True, timeout=10)
        print(response.json())
    except:
        print("WARNING! Error in send_network_log die to timeout.")
        
def send_monitoring_log(message, asset, token_header):
    """ Send trader log to server api """
    url_ext = 'logs/monitor'
    try:
        response = requests.post(LC.URL+url_ext, json={'Message':message,
                                                       'Asset':asset,
                                                       'Name':CC.TRADERNAME},
                                headers=token_header, verify=True, timeout=10)
        print(response.json())
    except:
        print("WARNING! Error in send_network_log die to timeout.")
        
def send_global_log(message, token_header):
    """ Send trader log to server api """
    url_ext = 'logs/global'
    try:
        response = requests.post(LC.URL+url_ext, json={'Message':message,
                                                       'Name':CC.TRADERNAME},
                                headers=token_header, verify=True, timeout=10)
        print(response.json())
    except:
        print("WARNING! Error in send_network_log die to timeout.")
        
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
        print(response.json())
    except:
        print("WARNING! Error in send_network_log die to timeout.")
            
def set_config_session(config, token_header):
    """ Send trader log to server api """
    url_ext = 'traders/sessions/set_session_config'
    try:
        response = requests.put(LC.URL+url_ext, json=config,
                                headers=token_header, verify=True, timeout=10)
        print(response.json())
    except:
        print("WARNING! Error in send_network_log die to timeout.")
        
def get_config_session(token_header):
    """ Send trader log to server api """
    url_ext = 'traders/sessions/get_session_config'
    try:
        response = requests.get(LC.URL+url_ext, 
                                headers=token_header, verify=True, timeout=10)
        print(response.json())
        return response.json()
    except:
        print("WARNING! Error in send_network_log die to timeout.")
        return None
    
def build_and_set_config(config_name='TESTPARAMUPDATE5'):
    """  """
    config=retrieve_config(config_name)
    token = get_token()
    token_header = {'Authorization': 'Bearer '+token}
    set_config_session(config, token_header)
    

def send_open_position(params, session_id, token_header):
    """ Send open position to server api"""
    url_ext = 'traders/sessions/'+str(session_id)+'/positions/open'
    response = requests.post(LC.URL+url_ext, json=params, headers=
                                             token_header, verify=True)
    print("Status code: "+str(response.status_code))
    if response.status_code == 200:
        print(response.json())
        return response.json()['Position'][0]
    else:
        print(response.text)
        return {}
    
def send_extend_position(params, pos_id, token_header):
    """ Send extend position commnad to server api"""
    url_ext = 'traders/positions/'+str(pos_id)+'/extend'
    response = requests.post(LC.URL+url_ext, json=params, headers=
                             token_header, verify=True)
    print("Status code: "+str(response.status_code))
    if response.status_code == 200:
        print(response.json())
        return True
    else:
        print(response.text)
        return False
    
def send_not_extend_position(params, pos_id, token_header):
    """ Send extend position commnad to server api"""
    url_ext = 'traders/positions/'+str(pos_id)+'/notextend'
    response = requests.post(LC.URL+url_ext, json=params, headers=
                             token_header, verify=True)
    print("Status code: "+str(response.status_code))
    if response.status_code == 200:
        print(response.json())
        return True
    else:
        print(response.text)
        return False
    
def send_close_position(params, pos_id, dirfilename, token_header):
    """ Send close position command to server api """
    url_ext = 'traders/positions/'+str(pos_id)+'/close'
    url_file = 'traders/positions/'+str(pos_id)+'/upload'
    files={'file': open(dirfilename,'rb')}
    response = requests.put(LC.URL+url_ext, json=params, 
                            headers=token_header, verify=True)
    response_file = requests.post(LC.URL+url_file, files=files, 
                                  headers=token_header, verify=True)
    print("Status code: "+str(response.status_code))
    if response.status_code == 200:
        print(response.json())
        if response_file.status_code != 200:
            print("WARNING! File not saved in DB")
        else:
            print(response.text)

def get_token():
    """  """
    try:
        response = requests.post(LC.URL+'tokens',auth=(CC.USERNAME,CC.PASSWORD), verify=True)
        if response.status_code == 200:
            token = response.json()['token']
            return token
        else:
            return None
    except:
        print("WARNING! Error in get_token of communication.py")
        return None
    
def get_account_status():
    """ Get account status from broker """
    success = 0
    ##### WARNING! #####
    dirfilename = LC.directory_MT5_account+'Status.txt'
    if os.path.exists(dirfilename):
        # load network output
        while not success:
            try:
                fh = open(dirfilename,"r")
                info_close = fh.read()[:-1]
                # close file
                fh.close()
                success = 1
                #stop_timer(ass_idx)
            except PermissionError:
                print("Error writing TT")
        info_str = info_close.split(',')
        #print(info_close)
        balance = float(info_str[0])
        leverage = float(info_str[1])
        equity = float(info_str[2])
        profits = float(info_str[3])
    else:
        print("WARNING! Account Status file not found. Turning to default")        
        balance = 500.0
        leverage = 30
        equity = balance
        profits = 0.0
#    print("Balance {0:.2f} Leverage {1:.2f} Equity {2:.2f} Profits {3:.2f}"\
#          .format(balance,leverage,equity,profits))
    status = {'balance':balance, 'leverage':leverage, 'equity':equity, 'profits':profits}
    return status
    
def send_account_status(token_header):
    """ Send trader log to server api """
    url_ext = 'traders/status'
    status = get_account_status()
    json = status
    json['tradername'] = CC.TRADERNAME
    try:
        response = requests.put(LC.URL+url_ext, json=status,
                                headers=token_header, verify=True, timeout=10)
        print(response.json())
    except:
        print("WARNING! Error in send_account_status of communication.py")
        
def send_reset_positions(token_header):
    """ Send trader log to server api """
    url_ext = 'traders/reset_positions'
    try:
        response = requests.post(LC.URL+url_ext,
                                headers=token_header, verify=True, timeout=10)
        print(response.json())
    except:
        print("WARNING! Error in send_account_status of communication.py")
        
def get_number_positions(token_header):
    """ Send trader log to server api """
    url_ext = 'traders/number_positions'
    try:
        response = requests.get(LC.URL+url_ext,
                                headers=token_header, verify=True, timeout=10)
        print(response.json())
    except:
        print("WARNING! Error in send_account_status of communication.py")
        
def get_user_id(token_header, username):
    """ get user id """
    url_ext = 'users/id'
    try:
        response = requests.get(LC.URL+url_ext, json={'username':username},
                                headers=token_header, verify=True, timeout=10)
        print(response.json())
    except:
        print("WARNING! Error in send_account_status of communication.py")
        
def set_budget(token_header, id, budget):
    """ get user id """
    url_ext = 'users/'+str(id)+'/set_budget'
    try:
        response = requests.put(LC.URL+url_ext, json={'budget':budget},
                                headers=token_header, verify=True, timeout=10)
        print(response.json())
    except:
        print("WARNING! Error in send_account_status of communication.py")
        
def send_close_command(asset):
    """ Send command for closeing position to MT5 software """
    directory_MT5_ass2close = LC.directory_MT5_IO+asset+"/"
    # load network output
    try:
        fh = open(directory_MT5_ass2close+"LC","w")
        fh.close()
    except (PermissionError, FileNotFoundError):
        print("Error writing LC")

def send_close_commands_all():
    """  """
    AllAssets = Config.AllAssets
    for asset_key in AllAssets:
        thisAsset = AllAssets[asset_key]
        print(thisAsset)
        send_close_command(thisAsset)
#def parameters_enquiry(session_id, token_header):
#    """  """
#    url_ext = 'traders/sessions/'+str(session_id)+'/get_params'
#    response = requests.get(LC.URL+url_ext, headers=token_header, verify=True)
#    if response.status_code == 200:
#        return True
#    else:
#        return False

from kaissandra.config import retrieve_config, Config
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
