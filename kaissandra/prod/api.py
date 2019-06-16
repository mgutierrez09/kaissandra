# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 11:58:56 2019

@author: mgutierrez
"""

import requests
import os
import datetime as dt
from kaissandra.config import CommConfig as CC
from kaissandra.config import Config
from kaissandra.local_config import local_vars as LC

class API():
    """ Class that handles API calls to server """
    token = None
    token_timer = 0
    trader_json = None
    strategies_json_list = []
    networks_json_list = []
    assets_json_list = []
    session_json = None
    positions_json_list = []
    
    def __init__(self):
        """ """
        self.post_token()
    
    def intit_all(self, config_trader, ass_idx, sessiontype):
        """ Init trader, strategy, networks, session and assets """
        balance, leverage, equity, profits = self.read_budget()
        params_trader = {'tradername':CC.TRADERNAME,
                       'machine':CC.MACHINE,
                       'magicnumber':CC.MAGICNUMBER,
                       'budget':balance}
        if not self.set_trader(params_trader):
            print("WARNING! Request set trader failed")
        
        params_strategies = []
        params_networks = []
        for s in range(len(config_trader['list_name'])):
#            print(len(config_trader['list_name']))
            #print(config_trader['config_list'])
#            print(len(config_trader['config_list']))
#            print(config_trader['config_list'])
#            print(config_trader['config_list'][s])
#            print(config_trader['config_list'][s][0])
            symbols = ['BID' if i['feats_from_bids'] else 'ASK' for i in config_trader['config_list'][s]]
#            print(symbols)
#            print(config_trader['config_list'][s][0]['outputGain'])
#            print(config_trader['list_spread_ranges'][s]['th'][0][1])
#            print(config_trader['config_list'][s][0]['combine_ts']['params_combine'][0]['alg'])
#            print(config_trader['list_spread_ranges'][s]['sp'])
            params_strategy = {'strategyname':config_trader['list_name'][s],
                                     'phaseshift':config_trader['phase_shifts'][s],
                                     'poslots':config_trader['list_max_lots_per_pos'][s],
                                     'nexs':config_trader['nExSs'][s],
                                     'oraclename':config_trader['netNames'][s],
                                     'symbol':symbols[s],
                                     'extthr':config_trader['list_lim_groi_ext'][s],
                                     'og':config_trader['config_list'][s][0]['outputGain'],
                                     'mw':config_trader['mWs'][s],
                                     'diropts':config_trader['list_spread_ranges'][s]['dir'],
                                     'spreadthr':config_trader['list_spread_ranges'][s]['sp'][0],
                                     'mcthr':config_trader['list_spread_ranges'][s]['th'][0][0],
                                     'mdthr':config_trader['list_spread_ranges'][s]['th'][0][1],
                                     'combine':str(config_trader['config_list'][s][0]['combine_ts']['if_combine']),
                                     'combineparams':config_trader['config_list'][s][0]['combine_ts']['params_combine'][0]['alg']}
            if not self.set_strategy(params_strategy):
                print("WARNING! Request set strategy failed")
            params_strategies.append(params_strategy)
            
            params_networks.append([])
            for n in range(len(config_trader['IDweights'][s])):
                weightsfile = config_trader['IDweights'][s][n]
                epoch = config_trader['IDepoch'][s][n]
#                print("epoch")
#                print(epoch)
                params_network = {'strategy':params_strategy['strategyname'],
                                  'networkname':weightsfile+str(epoch),
                                 'weightsfile':weightsfile ,
                                  'epoch':epoch}
                if not self.set_network(params_network):
                    print("WARNING! Request set network failed")
                params_networks[s].append(params_network)
        
        sessionname = dt.datetime.strftime(dt.datetime.utcnow(),'%y%m%d%H%M%S')
        params_session = {'sessionname':sessionname,
                          'sessiontype':sessiontype}
        if not self.open_session(params_session):
            print("WARNING! Request open session failed")
        
        assets = ','.join([Config.AllAssets[str(id)] for id in ass_idx])
        if not self.set_assets({'assets':assets}):
            print("WARNING! Request set assets failed")
            
        
    def read_budget(self):
        """  """
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
            return 0.0, 0.0, 0.0, 0.0
        return balance, leverage, equity, profits
    
    def build_token_header(self):
        """ Build header for token """
        return {'Authorization': 'Bearer '+self.token}
    
    def post_token(self):
        """ POST request to create new token if expired or retrieve current one """
        response = requests.post(CC.URL+'tokens',auth=(CC.USERNAME,CC.PASSWORD))
        if response.status_code == 200:
            self.token = response.json()['token']
            return True
        else:
            print(response.text)
        return False
    
    def set_trader(self, params, req_type="POST"):
        """ POST/PUT request to create new trader or change parameters of 
        existing one """
        if not self.token:
            return False
        if req_type == "POST":
            response = requests.post(CC.URL+'traders', json=params, headers=
                                     self.build_token_header())
        elif req_type == "PUT":
            response = requests.put(CC.URL+'traders', json=params, headers=
                                     self.build_token_header())
        else:
            return False
        print("Status code: "+str(response.status_code))
        if response.status_code == 200:
            print(response.json())
            self.trader_json = response.json()['trader'][0]
            return True
        else:
            print(response.text)
        return False
    
    def set_strategy(self, params, req_type="POST"):
        """ POST/PUT request to create new strategy or change parameters of 
        existing one """
        if not self.token or not self.trader_json or 'id' not in self.trader_json:
            return False
        url_ext = 'traders/'+str(self.trader_json['id'])+'/strategies'
        if req_type == "POST":
            response = requests.post(CC.URL+url_ext, json=params, headers=
                                     self.build_token_header())
        elif req_type == "PUT":
            response = requests.put(CC.URL+url_ext, json=params, headers=
                                     self.build_token_header())
        else:
            return False
        print("Status code: "+str(response.status_code))
        if response.status_code == 200:
            print(response.json())
            self.strategies_json_list.append(response.json()['Strategy'][0])
            return True
        else:
            print(response.text)
        return False
    
    def set_network(self, params, req_type="POST"):
        """ POST/PUT request to create new network or change parameters of 
        existing one """
        if not self.token or self.strategies_json_list == []:
            return False
        url_ext = 'traders/networks'
        if req_type == "POST":
            response = requests.post(CC.URL+url_ext, json=params, headers=
                                     self.build_token_header())
        elif req_type == "PUT":
            response = requests.put(CC.URL+url_ext, json=params, headers=
                                     self.build_token_header())
        else:
            return False
        print("Status code: "+str(response.status_code))
        if response.status_code == 200:
            print(response.json())
            self.networks_json_list.append(response.json()['Network'][0])
            return True
        else:
            print(response.text)
        return False
    
    def set_assets(self, params):
        """ POST request to set assets """
        if not self.token or not self.trader_json or 'id' not in self.trader_json:
            return False
        url_ext = 'traders/'+str(self.trader_json['id'])+'/assets'
        response = requests.post(CC.URL+url_ext, json=params, headers=
                                     self.build_token_header())
        print("Status code: "+str(response.status_code))
        if response.status_code == 200:
            print(response.json())
            self.assets_json_list = response.json()['Assets']
            return True
        else:
            print(response.text)
        return False
    
    def open_session(self, params):
        """ POST request to open a session """
        if not self.token or not self.trader_json or 'id' not in self.trader_json:
            return False
        url_ext = 'traders/'+str(self.trader_json['id'])+'/sessions'
        response = requests.post(CC.URL+url_ext, json=params, headers=
                                     self.build_token_header())
        print("Status code: "+str(response.status_code))
        if response.status_code == 200:
            print(response.json())
            self.session_json = response.json()['Session'][0]
            return True
        else:
            print(response.text)
        return False
    
    def close_session(self):
        """  PUT request to close session """
        if not self.session_json:
            return False
        url_ext = 'traders/sessions/'+str(self.session_json['id'])+'/close'
        response = requests.put(CC.URL+url_ext, headers=
                                     self.build_token_header())
        print("Status code: "+str(response.status_code))
        if response.status_code == 200:
            print(response.json())
            self.session_json = response.json()['Session'][0]
            return True
        else:
            print(response.text)
        return False
    
    def open_position(self, params):
        """ POST request to open a position """
        if not self.token or not self.session_json or 'id' not in self.session_json:
            return False
        url_ext = 'traders/sessions/'+str(self.session_json['id'])+'/positions/open'
        response = requests.post(CC.URL+url_ext, json=params, headers=
                                     self.build_token_header())
        print("Status code: "+str(response.status_code))
        if response.status_code == 200:
            print(response.json())
            self.positions_json_list.append(response.json()['Position'][0])
            return True
        else:
            print(response.text)
        return False
    
    def extend_position(self, assetname):
        """ PUT request to extend a position """
        #assetname = params['Asset']
        id = [pos['id'] for pos in self.positions_json_list if pos['asset']==assetname and not pos['closed']][0]
        url_ext = 'traders/positions/'+str(id)+'/extend'
        response = requests.put(CC.URL+url_ext, headers=
                                     self.build_token_header())
        print("Status code: "+str(response.status_code))
        if response.status_code == 200:
            print(response.json())
            id_list = [i for i in range(len(self.positions_json_list)) if \
                       self.positions_json_list[i]['asset']==assetname and not \
                       self.positions_json_list[i]['closed']][0]
            self.positions_json_list[id_list] = response.json()['Position'][0]
            return True
        else:
            print(response.text)
        return False
    
    def close_postition(self, assetname, params):
        """ PUT request to extend a position """
        #assetname = params['Asset']
        id = [pos['id'] for pos in self.positions_json_list if pos['asset']==assetname and not pos['closed']][0]
        url_ext = 'traders/positions/'+str(id)+'/close'
        print(url_ext)
        response = requests.put(CC.URL+url_ext, json=params, headers=
                                     self.build_token_header())
        print("Status code: "+str(response.status_code))
        if response.status_code == 200:
            print(response.json())
            id_list = [i for i in range(len(self.positions_json_list)) if \
                       self.positions_json_list[i]['asset']==assetname and not\
                       self.positions_json_list[i]['closed']][0]
            self.positions_json_list[id_list] = response.json()['Position'][0]
            return True
        else:
            print(response.text)
        return False