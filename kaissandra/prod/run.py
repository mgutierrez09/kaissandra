# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 11:40:34 2017

@author: Miguel Angel Gutierrez Estevez

This script merges the sequential execution of the entire chain: Fetcher/Dispatcher/live RNN/Trader
"""
import time
import os
import numpy as np
import pandas as pd
import datetime as dt
import pickle
import re
import tensorflow as tf
from itertools import count, filterfalse

entry_time_column = 'Entry Time'#'Entry Time
exit_time_column = 'Exit Time'#'Exit Time
entry_bid_column = 'Bi'
entry_ask_column = 'Ai'
exit_ask_column = 'Ao'
exit_bid_column = 'Bo'


nFiles = 100
extension = ".txt"
deli = "_"
flag_cl_name = "CL"
flag_ma_name = "MA" # manual close of position
flag_sl_name = "SL"
flag_tp_name = "TP"
# TODO: Save entire output vector
columnsResultInfo = ["Asset","Entry Time","Exit Time","GROI","Spread","ROI","Bet",
                     "Outcome","Diff","Bi","Ai","Bo","Ao","P_mc","P_md",
                     "P_mg"]
    
#start_time = dt.datetime.strftime(dt.datetime.now(),'%y_%m_%d_%H_%M_%S')

class Results:
        
    def __init__(self, IDresults, IDepoch, list_t_indexs, numberNetworks,
                 list_w_str, start_time, dir_results_trader, config_name=''):
        """ Init Results attributs """
        self.total_GROI = 0.0
        self.GROIs_week = np.array([])
        self.GROIs = np.array([])
        
        self.total_ROI = 0.0
        self.ROIs_week = np.array([])
        self.ROIs = np.array([])
        
        self.sum_GROI = 0.0
        self.sum_GROIs_week = np.array([])
        self.sum_GROIs = np.array([])
        
        self.sum_ROI = 0.0
        self.sum_ROIs_week = np.array([])
        self.sum_ROIs = np.array([])
        
        self.total_earnings = 0.0
        self.earnings_week = np.array([])
        self.earnings = np.array([])
        
        self.dts_close = []
        self.number_entries = np.array([])
        
        self.n_entries = 0
        self.net_successes = np.array([])
        self.total_losses = np.array([])
        self.total_wins = np.array([])
        self.gross_successes = np.array([])
        self.number_stoplosses = np.array([])
        
        self.sl_levels = np.array([5, 10, 15, 20])
        self.n_slots = 10
        #numberNetworks = len(IDresults)
        self.results_file_name = '_'.join([IDresults[i]+'E'+str(IDepoch[i])+'T'+
                         str(list_t_indexs[i])+'W'+list_w_str[i]
                         for i in range(numberNetworks)])
        self.results_dir_and_file = dir_results_trader+start_time+config_name+"_"+\
                                        self.results_file_name+".p"
        self.dir_positions = dir_results_trader+'positions/'+start_time+config_name+'/'
        
    def update_weekly_results(self, GROI, earnings, ROI, n_entries, stoplosses,
                              tGROI_live, tROI_live, net_successes, average_loss,
                              average_win, gross_successes):
        """ Update weekly results """
        self.n_entries += n_entries
        
        self.sum_GROI += 100*tGROI_live
        self.sum_GROIs_week = np.append(self.sum_GROIs_week, 100*tGROI_live)
        
        self.sum_ROI += 100*tROI_live
        self.sum_ROIs_week = np.append(self.sum_ROIs_week, 100*tROI_live)
        
        self.total_GROI += 100*GROI
        self.GROIs_week = np.append(self.GROIs_week, 100*GROI)
        
        self.total_ROI += 100*ROI
        self.ROIs_week = np.append(self.ROIs_week, 100*ROI)
        
        self.total_earnings += earnings
        self.earnings_week = np.append(self.earnings_week, earnings)
        
        self.number_entries = np.append(self.number_entries, n_entries)
        self.net_successes = np.append(self.net_successes, net_successes)
        self.total_losses = np.append(self.total_losses, np.abs(average_loss))
        self.total_wins = np.append(self.total_wins, np.abs(average_win))
        self.gross_successes = np.append(self.gross_successes, gross_successes)
        self.number_stoplosses = np.append(self.number_stoplosses, stoplosses)
        
    def update_outputs(self, date_time, GROI, ROI, nett_win):
        """ Update output lists """
        self.dts_close.append(date_time)
        self.GROIs = np.append(self.GROIs,GROI)
        self.ROIs = np.append(self.ROIs,ROI)
        self.earnings = np.append(self.GROIs,nett_win)
        
    def save_results(self):
        """ Save results in disk """
        pass
#        results_dict = {'dts_close':self.dts_close,
#                    'GROIs':self.GROIs,
#                    'ROIs':self.ROIs,
#                    'earnings':self.earnings}
        #pickle.dump( results_dict, open( self.results_dir_and_file, "wb" ))
    
    def update_meta_pos(self):
        """ Update metadata of position. Metadata consists of:
            - Entry time event
            - Direction
            - Extended time events
            - P_mc/P_md of events
            - Profitability values of events
            - Strategy name of events
            - t_index of events
            - Exit time event
            - Reason exit: {deadline reached, self-banned, banned by others,
              direction change, ...} """
        pass
    
    def save_pos_evolution(self, filename, dts, bids, asks):
        """ Save evolution of the position from opening till close """
        # format datetime for filename
        
        df = pd.DataFrame({'DateTime':dts,
                           'SymbolBid':bids,
                           'SymbolAsk':asks})
        df.to_csv(self.dir_positions+filename+'.txt', index=False)
        return self.dir_positions+filename+'.txt'
    
    def save_pos_evolution_live(self, filename, df):
        """  """
        df.to_csv(self.dir_positions+filename+'.txt', index=False)
        return self.dir_positions+filename+'.txt'
    
#class stats:
#    
#    def __init__():

def get_positions_filename(asset, open_dt, close_dt):
    """  """
    dt_open = dt.datetime.strftime(dt.datetime.strptime(
            open_dt,'%Y.%m.%d %H:%M:%S'),'%y%m%d%H%M%S')
    dt_close = dt.datetime.strftime(dt.datetime.strptime(
            close_dt,'%Y.%m.%d %H:%M:%S'),'%y%m%d%H%M%S')
    filename = 'O'+dt_open+'C'+dt_close+asset
    return filename

class Position:
    """
    Class containing market position info.
    """

    def __init__(self, journal_entry, strategy):
        
        
        self.asset = journal_entry['Asset']
        self.entry_time = journal_entry[entry_time_column]
        self.entry_bid = int(np.round(journal_entry[entry_bid_column]*100000))/100000
        self.entry_ask = int(np.round(journal_entry[entry_ask_column]*100000))/100000
        self.bet = int(journal_entry['Bet'])
###############################################################################   
############################# WARNING!! ##################################################        
###############################################################################        
        self.direction = np.sign(self.bet)#np.sign(np.random.randn()).astype(int)#
        self.level = int(np.abs(self.bet)-1)
        self.p_mc = float(journal_entry['P_mc'])
        self.p_md = float(journal_entry['P_md'])
        self.e_spread = float(journal_entry['E_spread'])
        self.deadline = int(journal_entry['Deadline'])
        self.strategy = strategy
        self.strategy_index = int(journal_entry['strategy_index'])
        self.profitability = journal_entry['profitability']
#        self.idx_mc = strategy._get_idx(self.p_mc)
#        self.idx_md = strategy._get_idx(self.p_md)

class Strategy():
    
    thresholds_mc = [.5,.6,.7,.8,.9]
    thresholds_md = [.5,.6,.7,.8,.9]
    
    def __init__(self, direct='',thr_sl=1000, thr_tp=1000, fix_spread=False, 
                 fixed_spread_pips=2, max_lots_per_pos=.1, 
                 flexible_lot_ratio=False, lb_mc_op=0.6, lb_md_op=0.6, 
                 lb_mc_ext=0.6, lb_md_ext=0.6, ub_mc_op=1, ub_md_op=1, 
                 ub_mc_ext=1, ub_md_ext=1,if_dir_change_close=False, 
                 if_dir_change_extend=False, name='',t_indexs=[3],entry_strategy='spread_ranges',
                 IDr='',epoch='11',weights=[0,1],info_spread_ranges={},
                 priorities=[0],lim_groi_ext=-.02):
        
        self.name = name
        self.dir_origin = direct
        self.thr_sl=thr_sl
        self.thr_tp=thr_tp
        
        self.fix_spread = fix_spread
        self.pip = 0.0001
        self.fixed_spread_ratio = fixed_spread_pips
        
        self.max_lots_per_pos = max_lots_per_pos#np.inf
        self.flexible_lot_ratio = flexible_lot_ratio
        
        self.lb_mc_op = lb_mc_op
        self.lb_md_op = lb_md_op
        self.lb_mc_ext = lb_mc_ext
        self.lb_md_ext = lb_md_ext
        self.ub_mc_op = ub_mc_op
        self.ub_md_op = ub_md_op
        self.ub_mc_ext = ub_mc_ext
        self.ub_md_ext = ub_md_ext
        
        self.if_dir_change_close = if_dir_change_close
        self.if_dir_change_extend = if_dir_change_extend
        
        # load GRE
        self.entry_strategy = entry_strategy
        self.IDr = IDr
        if IDr:
            self.IDgre = IDr+'R20INT'
        else:
            self.IDgre = None
        self.epoch = epoch
        self.t_indexs = t_indexs
        self.weights = weights
        self.priorities = priorities
        # retrocompatinility in info_spread_ranges
        if 'mar' not in info_spread_ranges:
            info_spread_ranges['mar'] = [(0.0,0.0) for _ in range(3)]
        self.info_spread_ranges = info_spread_ranges
        self.lim_groi_ext = lim_groi_ext
        
        self._load_GRE()
        
    def _load_GRE(self):
        """ Load strategy efficiency matrix GRE """
        # shape GRE: (model.seq_len+1, len(thresholds_mc), len(thresholds_md), 
        #int((model.size_output_layer-1)/2))
        if self.entry_strategy=='gre':
            assert(np.sum(self.weights)==1)
            allGREs = pickle.load( open( self.dir_origin+self.IDr+
                                        "/GRE_e"+self.epoch+".p", "rb" ))
            # fill the gaps in the GRE matrix
            allGREs = self._fill_up_GRE(allGREs)
            GRE = allGREs[self.t_indexs, :, :, :]/self.pip
#            print("GRE level 1:")
#            print(GRE[:,:,0])
#            print("GRE level 2:")
#            print(GRE[:,:,1])
            if os.path.exists(self.dir_origin+self.IDr+"/GREex_e"+self.epoch+".p"):
                allGREs = pickle.load( open( self.dir_origin+self.IDr+
                                            "/GREex_e"+self.epoch+".p", "rb" ))
                # fill the gaps in the GRE matrix
                allGREs = self._fill_up_GRE(allGREs)
                GREex = allGREs[self.t_indexs, :, :, :]/self.pip
#                print("GREex level 1:")
#                print(GREex[:,:,0])
#                print("GREex level 2:")
#                print(GREex[:,:,1])
            else: 
                GREex = 1-self.weights[0]*GRE
            
            self.GRE = self.weights[0]*GRE+self.weights[1]*GREex
            
#            if test:
#                self.GRE = self.GRE+20
#            print("GRE combined level 1:")
#            print(self.GRE[:,:,0])
#            print("GRE combined level 2:")
#            print(self.GRE[:,:,1])
        elif self.entry_strategy=='gre_v2':
            # New GRE implementation
            [GRE, model] = pickle.load( open( LC.gre_directory+self.IDgre+".p", "rb" ))
            self.GRE = GRE
            self.gre_model = model
        else:
            self.GRE = None
    
    def _fill_up_GRE(self, allGREs):
        """
        Fill uo the gaps in GRE matrix
        """
        for t in range(allGREs.shape[0]):
            for idx_mc in range(len(self.thresholds_mc)):
                min_md = allGREs[t,idx_mc,:,:]
                for idx_md in range(len(self.thresholds_md)):
                    min_mc = allGREs[t,:,idx_md,:]    
                    for l in range(int((5-1)/2)):
                        if allGREs[t,idx_mc,idx_md,l]==0:
                            if idx_md==0:
                                if idx_mc==0:
                                    if l==0:
                                        # all zeros, nothing to do
                                        pass
                                    else:
                                        allGREs[t,idx_mc,idx_md,l] = max(min_mc[idx_mc,:l])
                                else:
                                    allGREs[t,idx_mc,idx_md,l] = max(min_mc[:idx_mc,l])
                            else:
                                allGREs[t,idx_mc,idx_md,l] = max(min_md[:idx_md,l])
        return allGREs
    
    def _get_idx(self, p):
        """  """
        
        if p>0.5 and p<=0.6:
            idx = 0
        elif p>0.6 and p<=0.7:
            idx = 1
        elif p>0.7 and p<=0.8:
            idx = 2
        elif p>0.8 and p<=0.9:
            idx = 3
        elif p>0.9:
            idx = 4
        else:
            print("WARNING: p<0.5")
            idx = 0
            
        return idx
    
    def get_profitability(self, t, p_mc, p_md, level):
        """ get profitability for a t_index, prob and output level """
        if self.entry_strategy=='gre':
            return self.GRE[t, self._get_idx(p_mc), self._get_idx(p_md), level]
        elif self.entry_strategy=='gre_v2':
            return self.gre_model.predict(np.array([[p_mc, p_md, level]]))[0]
        else:
            # here it represents a priority
            return self.priorities[t]
        
 
class Trader:
    
    def __init__(self, running_assets, ass2index_mapping, strategies,
                 log_file, results_dir="", max_opened_positions=None, 
                 start_time='', config_name='',net2strategy=[], 
                 queue=None, queue_prior=None, session_json=None, token_header=None):
        
        self.list_opened_positions = [[] for _ in range(len(strategies))]
        self.map_ass_idx2pos_idx = [np.array([-1 for i in range(len(C.AllAssets))]) for _ in range(len(strategies))]
        self.pos_idx2map_ass_str = [(-1, -1, -1) for _ in range(max_opened_positions)]
        self.list_pos_idx = []
        self.list_count_events = [[] for _ in range(len(strategies))]
        self.list_dd_info = [[] for _ in range(len(strategies))] # track double down positions
        self.list_count_all_events = [[] for _ in range(len(strategies))]
        self.list_stop_losses = [[] for _ in range(len(strategies))]
        self.list_take_profits = [[] for _ in range(len(strategies))]
        self.list_lots_per_pos = [[] for _ in range(len(strategies))]
        self.list_lots_entry = [[] for _ in range(len(strategies))]
        self.list_last_bid = [[] for _ in range(len(strategies))]
        self.list_EM = [[] for _ in range(len(strategies))]
        self.list_last_ask = [[] for _ in range(len(strategies))]
        self.list_last_dt = [[] for _ in range(len(strategies))]
        self.list_sl_thr_vector = [[] for _ in range(len(strategies))]
        self.list_deadlines = [[] for _ in range(len(strategies))]
        self.positions_tracker = [[] for _ in range(len(strategies))]
        self.list_symbols_tracking = [[] for _ in range(len(strategies))]
        
        self.list_is_asset_banned = [False for _ in running_assets]
        self.max_opened_positions = max_opened_positions
        
#        self.journal_idx = 0
#        self.sl_thr_vector = np.array([5, 10, 15, 20, 25, 30])
        
#        init_budget, leverage, _, _ = self.get_account_status()
        status = self.get_account_status()#
        #print("Init budget: "+str(init_budget)+" Leverage: "+str(leverage))
        self.budget = status['balance']#init_budget
        self.init_budget = status['balance']
        self.LOT = 100000.0
        
        self.pip = 0.0001
        self.leverage = status['leverage']#
        #self.budget_in_lots = self.leverage*self.budget/self.LOT
        self.available_budget = self.budget*self.leverage
        self.available_bugdet_in_lots = self.available_budget/self.LOT
        print("Available budget in lots: "+str(self.available_bugdet_in_lots))
        self.budget_in_lots = self.available_bugdet_in_lots
        self.gross_earnings = 0.0
        self.nett_earnigs = 0.0
        
        self.tROI_live = 0.0
        self.tGROI_live = 0.0
        
        self.net_successes = 0 
        self.average_loss = 0.0
        self.average_win = 0.0
        self.gross_successes = 0
        
        # counters
        self.n_entries = 0
        self.n_pos_opened = 0
        self.stoplosses = 0
        self.takeprofits = 0
        self.n_pos_extended = 0
        self.n_pos_currently_open = 0
        
        self.margins = [0.0 for _ in running_assets]
        
        # log
        self.save_log = 1
        self.results_dir_trader = results_dir+'trader/'
        
        self.running_assets = running_assets
        self.ass2index_mapping = ass2index_mapping
        
        self.strategies = strategies
        if len(net2strategy)==0:
            self.net2strategy = [i for i in range(len(strategies))]
        else:
            self.net2strategy = net2strategy
        
        if start_time=='':
            
            #raise ValueError("Depricated. String start_time cannot be empty")
            pass
        else:
            if run_back_test:
                tag = '_BT_'
            else:
                tag = '_LI_'
            self.log_file_trader = self.results_dir_trader+start_time+tag+config_name+"trader.log"
            self.log_file = log_file
            self.log_positions_soll = self.results_dir_trader+start_time+tag+config_name+"positions_soll.csv"
            self.log_positions_ist = self.results_dir_trader+start_time+tag+config_name+"positions_ist.csv"
            self.log_summary = self.results_dir_trader+start_time+tag+config_name+"summary.log"
            self.dir_positions = results_dir+'/positions/'+start_time+config_name+'/'
            self.budget_file = self.results_dir_trader+start_time+tag+config_name+"budget.log"
            self.ban_currencies_dir = LC.io_live_dir+'/ban'+config_name+'/'
            
        
            self.start_time = start_time
            try:
                if not os.path.exists(self.results_dir_trader):
                    os.makedirs(self.results_dir_trader)
                if not os.path.exists(self.dir_positions):
                        os.makedirs(self.dir_positions)
                if not os.path.exists(self.ban_currencies_dir):
                        os.makedirs(self.ban_currencies_dir)
            except FileExistsError:
                print("WARNING! File already exists. Moving on")
            # little pause to garantee no 2 processes access at the same time
            time.sleep(np.random.rand())
            # results tracking
            if not os.path.exists(self.log_positions_soll):
                resultsInfoHeaderSoll = "Asset,Entry Time,Exit Time,Position,"+\
                    "Bi,Ai,Bo,Ao,ticks_d,GROI,Spread,ROI,strategy,Profit,E_spread,stoploss,stGROI,stROI"
                write_log(resultsInfoHeaderSoll, self.log_positions_soll)
                if not run_back_test:
                    resultsInfoHeaderIst = "Asset,Entry Time,Exit Time,Position,"+\
                    "Bi,Ai,Bo,Ao,ticks_d,GROI,Spread,ROI,Profit,Equity,Swap,strategy"
                    write_log(resultsInfoHeaderIst, self.log_positions_ist)
                write_log(str(self.available_bugdet_in_lots), self.budget_file)
        # flow control
        self.EXIT = 0
        self.rewind = 0
        self.approached = 0
        self.swap_pending = 0
        #self.api = api
        self.queue = queue
        self.queue_prior = queue_prior
        self.session_json = session_json
        self.token_header = token_header
    
    def get_account_status(self):
        """ Get account status from broker """
#        success = 0
#        ##### WARNING! #####
#        dirfilename = LC.directory_MT5_account+'Status.txt'
#        if os.path.exists(dirfilename):
#            # load network output
#            while not success:
#                try:
#                    fh = open(dirfilename,"r")
#                    info_close = fh.read()[:-1]
#                    # close file
#                    fh.close()
#                    success = 1
#                    #stop_timer(ass_idx)
#                except PermissionError:
#                    print("Error writing TT")
#            info_str = info_close.split(',')
#            #print(info_close)
#            balance = float(info_str[0])
#            leverage = float(info_str[1])
#            equity = float(info_str[2])
#            profits = float(info_str[3])
#        else:
#            print("WARNING! Account Status file not found. Turning to default")
#            if not hasattr(self, 'budget'):
#                balance = 500.0
#                leverage = 30
#                equity = balance
#                profits = 0.0
#            else:
#                balance = self.budget
#                leverage = 30
#                equity = balance
#                profits = 0.0
#        print("Balance {0:.2f} Leverage {1:.2f} Equity {2:.2f} Profits {3:.2f}"\
#              .format(balance,leverage,equity,profits))
        status = {'error':True}
        while status['error']:
            status = ct.get_account_status()
        return status
#        
#    def _get_thr_sl_vector(self):
#        '''
#        '''
#        return self.next_candidate.entry_bid*(1-self.next_candidate.direction*
#                                              self.sl_thr_vector)
    
    def add_new_candidate(self, position):
        '''
        '''
        self.next_candidate = position
    
    def add_position(self, idx, lots, datetime, bid, ask, deadline):
        """  """
        str_idx = self.next_candidate.strategy_index
        self.list_opened_positions[str_idx].append(self.next_candidate)
        self.list_count_events[str_idx].append(0)
        self.list_count_all_events[str_idx].append(0)
        self.list_lots_per_pos[str_idx].append(lots)
        self.list_lots_entry[str_idx].append(lots)
        self.list_last_bid[str_idx].append([bid])
        self.list_EM[str_idx].append([bid])
        self.list_last_ask[str_idx].append([ask])
        self.list_last_dt[str_idx].append([datetime])
        self.list_symbols_tracking[str_idx].append(pd.DataFrame(columns=['DateTime','SymbolBid','SymbolAsk']))
        self.map_ass_idx2pos_idx[str_idx][idx] = len(self.list_count_events[str_idx])-1
        self.list_stop_losses[str_idx].append(self.next_candidate.entry_bid*\
                                (1-self.next_candidate.direction*\
                                self.next_candidate.strategy.thr_sl*self.pip))
        self.list_take_profits[str_idx].append(self.next_candidate.entry_bid*\
                                (1+self.next_candidate.direction*\
                                self.next_candidate.strategy.thr_tp*self.pip))
        self.list_deadlines[str_idx].append(deadline)
        self.list_dd_info[str_idx].append([{'entry_bid':self.next_candidate.entry_bid,
                                            'entry_ask':self.next_candidate.entry_ask,
                                            'entry_time':self.next_candidate.entry_time,
                                            'checkpoint':0,
                                            'every':double_down['every'],
                                            'lots':lots}])
        return None
        
    def remove_position(self, idx, s):
        """ Remove a position from lists after closing """
        self.list_opened_positions[s] = self.list_opened_positions[s]\
            [:self.map_ass_idx2pos_idx[s][idx]]+self.list_opened_positions[s]\
            [self.map_ass_idx2pos_idx[s][idx]+1:]
        self.list_count_events[s] = self.list_count_events[s]\
            [:self.map_ass_idx2pos_idx[s][idx]]+self.list_count_events[s]\
            [self.map_ass_idx2pos_idx[s][idx]+1:]
        self.list_count_all_events[s] = self.list_count_all_events[s]\
            [:self.map_ass_idx2pos_idx[s][idx]]+self.list_count_all_events[s]\
            [self.map_ass_idx2pos_idx[s][idx]+1:]
        self.list_stop_losses[s] = self.list_stop_losses[s]\
            [:self.map_ass_idx2pos_idx[s][idx]]+self.list_stop_losses[s]\
            [self.map_ass_idx2pos_idx[s][idx]+1:]
        self.list_take_profits[s] = self.list_take_profits[s]\
            [:self.map_ass_idx2pos_idx[s][idx]]+self.list_take_profits[s]\
            [self.map_ass_idx2pos_idx[s][idx]+1:]
        self.list_lots_per_pos[s] = self.list_lots_per_pos[s]\
            [:self.map_ass_idx2pos_idx[s][idx]]+self.list_lots_per_pos[s]\
            [self.map_ass_idx2pos_idx[s][idx]+1:]
        self.list_lots_entry[s] = self.list_lots_entry[s]\
            [:self.map_ass_idx2pos_idx[s][idx]]+self.list_lots_entry[s]\
            [self.map_ass_idx2pos_idx[s][idx]+1:]
        self.list_last_bid[s] = self.list_last_bid[s]\
            [:self.map_ass_idx2pos_idx[s][idx]]+self.list_last_bid[s]\
            [self.map_ass_idx2pos_idx[s][idx]+1:]
        self.list_EM[s] = self.list_EM[s]\
            [:self.map_ass_idx2pos_idx[s][idx]]+self.list_EM[s]\
            [self.map_ass_idx2pos_idx[s][idx]+1:]
        self.list_last_ask[s] = self.list_last_ask[s]\
            [:self.map_ass_idx2pos_idx[s][idx]]+self.list_last_ask[s]\
            [self.map_ass_idx2pos_idx[s][idx]+1:]
        self.list_last_dt[s] = self.list_last_dt[s]\
            [:self.map_ass_idx2pos_idx[s][idx]]+self.list_last_dt[s]\
            [self.map_ass_idx2pos_idx[s][idx]+1:]
        self.list_deadlines[s] = self.list_deadlines[s]\
            [:self.map_ass_idx2pos_idx[s][idx]]+self.list_deadlines[s]\
            [self.map_ass_idx2pos_idx[s][idx]+1:]
        self.positions_tracker[s] = self.positions_tracker[s]\
            [:self.map_ass_idx2pos_idx[s][idx]]+self.positions_tracker[s]\
            [self.map_ass_idx2pos_idx[s][idx]+1:]
        self.list_symbols_tracking[s] = self.list_symbols_tracking[s]\
            [:self.map_ass_idx2pos_idx[s][idx]]+self.list_symbols_tracking[s]\
            [self.map_ass_idx2pos_idx[s][idx]+1:]
        self.list_dd_info[s] = self.list_dd_info[s]\
            [:self.map_ass_idx2pos_idx[s][idx]]+self.list_dd_info[s]\
            [self.map_ass_idx2pos_idx[s][idx]+1:]
        
        mask = self.map_ass_idx2pos_idx[s]>self.map_ass_idx2pos_idx[s][idx]
        self.map_ass_idx2pos_idx[s][idx] = -1
        self.map_ass_idx2pos_idx[s] = self.map_ass_idx2pos_idx[s]-mask*1#np.maximum(,-1)
        # remove position
        list_remove = []
        print("pre remove_position():")
        print("self.pos_idx2map_ass_str")
        print(self.pos_idx2map_ass_str)
        print("self.list_pos_idx")
        print(self.list_pos_idx)
        print("(idx, s)")
        print((idx, s))
        for i, pos_idx in enumerate(self.list_pos_idx):
            if self.pos_idx2map_ass_str[pos_idx][:2] == (idx, s):
                self.pos_idx2map_ass_str[pos_idx] = (-1, -1, -1)
                list_remove.append(i)
        
        print("list_remove")
        print(list_remove)
        for i in range(len(list_remove)):
            self.list_pos_idx.pop(list_remove[-(i+1)])
        print("post remove_position():")
        print("self.pos_idx2map_ass_str")
        print(self.pos_idx2map_ass_str)
        print("self.list_pos_idx")
        print(self.list_pos_idx)
        
    def update_position(self, idx):
        """ Update position info due to extension """
        str_idx = self.next_candidate.strategy_index
        # reset counter
        self.list_count_events[str_idx][self.map_ass_idx2pos_idx[str_idx][idx]] = 0
        self.list_deadlines[str_idx][self.map_ass_idx2pos_idx[str_idx][idx]] = self.next_candidate.deadline
        
        entry_bid = self.list_opened_positions[str_idx][self.map_ass_idx2pos_idx[str_idx][idx]].entry_bid
        entry_ask = self.list_opened_positions[str_idx][self.map_ass_idx2pos_idx[str_idx][idx]].entry_ask
        direction = self.list_opened_positions[str_idx][self.map_ass_idx2pos_idx[str_idx][idx]].direction
        bet = self.list_opened_positions[str_idx][self.map_ass_idx2pos_idx[str_idx][idx]].bet
        entry_time = self.list_opened_positions[str_idx][self.map_ass_idx2pos_idx[str_idx][idx]].entry_time
        p_mc = self.list_opened_positions[str_idx][self.map_ass_idx2pos_idx[str_idx][idx]].p_mc
        p_md = self.list_opened_positions[str_idx][self.map_ass_idx2pos_idx[str_idx][idx]].p_md
        strategy = self.list_opened_positions[str_idx][self.map_ass_idx2pos_idx[str_idx][idx]].strategy
        profitability = self.list_opened_positions[str_idx][self.map_ass_idx2pos_idx[str_idx][idx]].profitability
        
        self.list_opened_positions[str_idx][self.map_ass_idx2pos_idx[str_idx][idx]] = self.next_candidate
        
        self.list_opened_positions[str_idx][self.map_ass_idx2pos_idx[str_idx][idx]].entry_bid = entry_bid
        self.list_opened_positions[str_idx][self.map_ass_idx2pos_idx[str_idx][idx]].entry_ask = entry_ask
        self.list_opened_positions[str_idx][self.map_ass_idx2pos_idx[str_idx][idx]].direction = direction
        self.list_opened_positions[str_idx][self.map_ass_idx2pos_idx[str_idx][idx]].bet = bet
        self.list_opened_positions[str_idx][self.map_ass_idx2pos_idx[str_idx][idx]].entry_time = entry_time
        self.list_opened_positions[str_idx][self.map_ass_idx2pos_idx[str_idx][idx]].p_mc = p_mc
        self.list_opened_positions[str_idx][self.map_ass_idx2pos_idx[str_idx][idx]].p_md = p_md
        self.list_opened_positions[str_idx][self.map_ass_idx2pos_idx[str_idx][idx]].strategy = strategy
        self.list_opened_positions[str_idx][self.map_ass_idx2pos_idx[str_idx][idx]].profitability = profitability
    
    def is_opened_asset(self, idx):
        """ Check if Asset is opened for any strategy """
        for s in range(len(self.strategies)):
            if self.map_ass_idx2pos_idx[s][idx]>=0:
                return True
        return False
    
    def is_opened_strategy(self, idx):
        """ Check if Asset is opened for a strategy """
        str_idx = self.next_candidate.strategy_index
        if self.map_ass_idx2pos_idx[str_idx][idx]>=0:
            return True
        else:
            return False
    
#    def is_opened(self, idx):
#        """  """
#        if self.map_ass_idx2pos_idx[idx]>=0:
#            return True
#        else:
#            return False
    
    def count_events(self, idx, n_events):
        """  """
        for s in range(len(self.strategies)):
            if self.map_ass_idx2pos_idx[s][idx]>=0:
                self.list_count_events[s][self.map_ass_idx2pos_idx[s][idx]] += n_events
                self.list_count_all_events[s][self.map_ass_idx2pos_idx[s][idx]] += n_events
    
    def direction_map(self, candidate_direction, strategy_direction):
        """ Map strategy direction to condition for opening """
        if strategy_direction=='COMB':
            condition = True
        elif strategy_direction=='BIDS' and candidate_direction<0:
            condition =True
        elif strategy_direction=='ASKS' and candidate_direction>0:
            condition = True
        else:
            condition = False
#        if test:
#            condition = not condition
        return condition
    
    def check_contition_for_opening(self, tactic, ass_idx):
        """  """
        reason = ''
        this_strategy = self.next_candidate.strategy
        e_spread = self.next_candidate.e_spread
        margin = 2
        if this_strategy.fix_spread and this_strategy.entry_strategy=='fixed_thr':
            condition_open = (self.next_candidate.p_mc>=this_strategy.lb_mc_op and \
                              self.next_candidate.p_md>=this_strategy.lb_md_op and \
                              self.next_candidate.p_mc<this_strategy.ub_mc_op and \
                              self.next_candidate.p_md<this_strategy.ub_md_op)
        elif not this_strategy.fix_spread and this_strategy.entry_strategy=='fixed_thr':
            condition_open = (self.next_candidate!= None and \
                              e_spread<this_strategy.fixed_spread_ratio and \
                              self.next_candidate.p_mc>=this_strategy.lb_mc_op and 
                              self.next_candidate.p_md>=this_strategy.lb_md_op and \
                              self.next_candidate.p_mc<this_strategy.ub_mc_op and 
                              self.next_candidate.p_md<this_strategy.ub_md_op)
        elif not this_strategy.fix_spread and (this_strategy.entry_strategy=='gre' or this_strategy.entry_strategy=='gre_v2'):
            cond_prof = self.next_candidate.profitability>margin*e_spread#+margin
            cond_bet = self.direction_map(self.next_candidate.direction, 
                                   self.next_candidate.strategy.info_spread_ranges['dir'])
            condition_open = cond_prof and cond_bet
            if not cond_prof:
                reason += str(self.next_candidate.profitability)+'<'+str(margin*e_spread)+' '
            if not cond_bet:
                reason += 'bet'
        elif this_strategy.entry_strategy=='spread_ranges':
            cond_pmc = self.next_candidate.p_mc>=this_strategy.info_spread_ranges['th'][tactic][0]+this_strategy.info_spread_ranges['mar'][tactic][0]
            cond_pmd = self.next_candidate.p_md>=this_strategy.info_spread_ranges['th'][tactic][1]+this_strategy.info_spread_ranges['mar'][tactic][1]+self.margins[ass_idx]
            cond_spread = e_spread<=this_strategy.info_spread_ranges['sp'][tactic]
            cond_bet = self.direction_map(self.next_candidate.direction, 
                                   self.next_candidate.strategy.info_spread_ranges['dir'])
            
            n_open_pos = len(self.list_opened_positions)
            if self.max_opened_positions!= None and n_open_pos >= self.max_opened_positions:
                cond_pos_can_be_allocated = False
            else:
                cond_pos_can_be_allocated = True
            # get string explanation for condition not opened
            condition_open = cond_pmc and\
                cond_pmd and\
                cond_spread and\
                cond_bet and\
                cond_pos_can_be_allocated
            if not cond_pmc:
                reason += 'pmc_'
            if not cond_pmd:
                reason += 'pmd_'
            if not cond_spread:
                reason += 'spread_'
            if not cond_bet:
                reason += 'wrongdir_'
            if not cond_pos_can_be_allocated:
                reason+='maxpos_'
        else:
            #print("ERROR: fix_spread cannot be fixed if GRE is in use")
            raise ValueError("fix_spread cannot be fixed if GRE is in use")
            
        return condition_open, reason
    
    def check_same_direction(self, ass_id):
        """ Check that extension candidate is in the same direction as current
        position. """
        str_idx = self.next_candidate.strategy_index
        return self.list_opened_positions[str_idx][self.map_ass_idx2pos_idx[str_idx][ass_id]]\
                                .direction==self.next_candidate.direction
                                
    def check_same_strategy(self, ass_id):
        """ Check if candidate and current position share same strategy """
        str_idx = self.next_candidate.strategy_index
        return self.list_opened_positions[str_idx][self.map_ass_idx2pos_idx[str_idx][ass_id]]\
                                .strategy.name==self.next_candidate.strategy.name
    
    def check_remain_samps(self, ass_id):
        """ Check that the number of samples for extantion is larger than the 
        remaining ones. This situation can happen in multi-network environments
        if the number of events of one network is smaller than others. """
        str_idx = self.next_candidate.strategy_index
        samps_extension = self.next_candidate.deadline
        samps_remaining = self.list_deadlines[str_idx][self.map_ass_idx2pos_idx[str_idx][ass_id]]-\
            self.list_count_events[str_idx][self.map_ass_idx2pos_idx[str_idx][ass_id]]
        return samps_remaining<samps_extension
    
    def get_remain_samps(self, ass_id):
        """  """
        str_idx = self.next_candidate.strategy_index
        return self.list_deadlines[str_idx][self.map_ass_idx2pos_idx[str_idx][ass_id]]-\
            self.list_count_events[str_idx][self.map_ass_idx2pos_idx[str_idx][ass_id]]
    
    def check_primary_condition_for_extention(self, ass_id):
        """  """
        if not crisis_mode:
            return self.check_same_direction(ass_id) and self.check_remain_samps(ass_id)
        else:
            return self.check_remain_samps(ass_id)
        
        
    def check_secondary_condition_for_extention(self, ass_id, ass_idx, curr_GROI, tactic):
        """  """
        this_strategy = self.next_candidate.strategy
        margin = 0.5
        reason = ''
        if this_strategy.entry_strategy=='fixed_thr':
            condition_extension = (self.next_candidate.p_mc>=this_strategy.lb_mc_ext and 
                              self.next_candidate.p_md>=this_strategy.lb_md_ext and
                              self.next_candidate.p_mc<this_strategy.ub_mc_ext and 
                              self.next_candidate.p_md<this_strategy.ub_md_ext)
        elif this_strategy.entry_strategy=='gre' or this_strategy.entry_strategy=='gre_v2':
            cond_prof = self.next_candidate.profitability>margin
            cond_groi = 100*curr_GROI>=this_strategy.lim_groi_ext
            condition_extension = cond_prof and cond_groi
            if not cond_prof:
                reason += str(self.next_candidate.profitability)+'<'+str(margin)+' '
            if not cond_groi:
                reason += 'groi'
        elif this_strategy.entry_strategy=='spread_ranges':
            if not crisis_mode:
                cond_pmc = self.next_candidate.p_mc>=this_strategy.info_spread_ranges['th'][0][0]+this_strategy.info_spread_ranges['mar'][0][0]
                cond_pmd = self.next_candidate.p_md>=this_strategy.info_spread_ranges['th'][0][1]+this_strategy.info_spread_ranges['mar'][0][1]
            else:
                cond_pmc = self.next_candidate.p_mc>=this_strategy.info_spread_ranges['th'][tactic][0]+this_strategy.info_spread_ranges['mar'][tactic][0]
                cond_pmd = self.next_candidate.p_md>=this_strategy.info_spread_ranges['th'][tactic][1]+this_strategy.info_spread_ranges['mar'][tactic][1]
            cond_groi = 100*curr_GROI>=this_strategy.lim_groi_ext
            cond_bet = self.direction_map(self.next_candidate.direction, 
                                   this_strategy.info_spread_ranges['dir'])
            condition_extension = cond_pmc and\
                cond_pmd and \
                cond_groi and cond_bet \
                and not force_no_extesion
            if not cond_pmc:
                reason+='pmc'
            if not cond_pmd:
                reason+='pmd'
            if not cond_groi:
                reason+='groi'
            if not cond_bet:
                reason+='bet'
            if force_no_extesion:
                reason+='bet'
        else:
            raise ValueError("Wrong entry strategy")            
        
        return condition_extension, reason, cond_bet

#    def update_stoploss_open_pos(self, idx, bid):
#        # update stoploss
#        this_strategy = self.next_candidate.strategy
#        if self.list_opened_positions[self.map_ass_idx2pos_idx[idx]].direction == 1:
#            self.list_stop_losses[self.map_ass_idx2pos_idx[idx]] = (max(
#                    self.list_stop_losses[self.map_ass_idx2pos_idx[idx]],
#                    bid*(1-self.list_opened_positions[
#                    self.map_ass_idx2pos_idx[idx]].direction*this_strategy.thr_sl*
#                    this_strategy.fixed_spread_ratio)))
#        else:
#            self.list_stop_losses[self.map_ass_idx2pos_idx[idx]] = (min(
#                    self.list_stop_losses[self.map_ass_idx2pos_idx[idx]],
#                    bid*(1-self.list_opened_positions[self.map_ass_idx2pos_idx[
#                    idx]].direction*this_strategy.thr_sl*
#                    this_strategy.fixed_spread_ratio)))
    
    def is_stoploss_reached(self, lists, datetime, ass_id, bid, em, event_idx, results):
        """ check stop loss reachead """
        str_idx = self.next_candidate.strategy_index
        direction = self.list_opened_positions[str_idx][self.map_ass_idx2pos_idx[str_idx][ass_id]].direction
        if (direction*
            (bid-self.list_stop_losses[str_idx][self.map_ass_idx2pos_idx[str_idx][ass_id]])<=0):
            # exit position due to stop loss
            asset = self.list_opened_positions[str_idx][self.\
                                               map_ass_idx2pos_idx[str_idx][ass_id]].asset
            self.stoplosses += 1
            if verbose_trader:
                logMsg =  (" Exit position due to stop loss @event idx "+
                       str(event_idx)+" bid="+str(bid)+" sl="+
                       str(self.list_stop_losses[str_idx][self.map_ass_idx2pos_idx[str_idx][ass_id]]))
                out = asset+logMsg
                print("\r"+out)
                self.write_log(out)
                send_log_info(self.queue, asset, {"FUNC":"LOG","ORIGIN":"TRADE","ASS":asset,"MSG":logMsg})
#                if log_thu_control:
#                    self.queue.put({"FUNC":"LOG","ORIGIN":"TRADE","ASS":asset,"MSG":logMsg})
            stoploss_flag = True
            self.ban_currencies(lists, asset, datetime, results, direction)
        else:
            stoploss_flag = False
        return stoploss_flag
    
#    def is_takeprofit_reached(self, this_pos, take_profit, takeprofits, bid, event_idx):
#        """ check if take-profit threshold has been reached """
#        # check take profit reachead
#        if this_pos.direction*(bid-take_profit)>=0:
#            # exit position due to stop loss
#            exit_pos = 1
#            takeprofits += 1
#            if verbose_trader:
#                out = "Exit position due to take profit @event idx "+str(event_idx)+\
#                    ". tp="+str(take_profit)
#                print("\r"+out)
#                self.write_log(out)
#                self.queue.put({"FUNC":"LOG","ORIGIN":"TRADE","MSG":out})
#        else:
#            exit_pos = 0
#        return exit_pos, takeprofits
    
    def get_rois(self, idx, date_time='', roi_ratio=1, ass='', s=-1):
        """ Get current GROI and ROI of a given asset idx """
        if s==-1:
            str_idx = self.next_candidate.strategy_index
        else:
            str_idx = s
        strategy_name = self.list_opened_positions[str_idx][self.map_ass_idx2pos_idx[str_idx][idx]].strategy.name
        direction = self.list_opened_positions[str_idx][self.map_ass_idx2pos_idx[str_idx][idx]].direction
#        Ti = self.list_opened_positions[str_idx][self.map_ass_idx2pos_idx[str_idx][idx]].entry_time
        bet = self.list_opened_positions[str_idx][self.map_ass_idx2pos_idx[str_idx][idx]].bet
        if run_back_test:
            Ao = self.list_last_ask[str_idx][self.map_ass_idx2pos_idx[str_idx][idx]][-1]
            Bo = self.list_last_bid[str_idx][self.map_ass_idx2pos_idx[str_idx][idx]][-1]
        else:
            try:
                Ao = self.list_symbols_tracking[str_idx][self.map_ass_idx2pos_idx[str_idx][idx]].SymbolAsk.iloc[-1]
                Bo = self.list_symbols_tracking[str_idx][self.map_ass_idx2pos_idx[str_idx][idx]].SymbolBid.iloc[-1]
            except:
                print("WARNING! Error when reading last symbol from list_symbols_tracking. "+
                      "Reading info from list_last_bid instead.")
                Ao = self.list_last_ask[str_idx][self.map_ass_idx2pos_idx[str_idx][idx]][-1]
                Bo = self.list_last_bid[str_idx][self.map_ass_idx2pos_idx[str_idx][idx]][-1]
        GROIs = []
        ROIs = []
        spreads = []
        infos = []
        lotss = []
        for sub_position in self.list_dd_info[str_idx][self.map_ass_idx2pos_idx[str_idx][idx]]:
            Ti = sub_position['entry_time']#dt.datetime.strftime(sub_position['entry_time'], '%Y.%m.%d %H:%M:%S')
            Bi = sub_position['entry_bid']
            Ai = sub_position['entry_ask']
            lots = sub_position['lots']
#        Bi = self.list_opened_positions[str_idx][self.map_ass_idx2pos_idx[str_idx][idx]].entry_bid
#        Ai = self.list_opened_positions[str_idx][self.map_ass_idx2pos_idx[str_idx][idx]].entry_ask
        
        
            if direction>0:
                GROI_live = roi_ratio*(Ao-Ai)/Ai
                spread = (Ao-Bo)/Ai
                
            else:
                GROI_live = roi_ratio*(Bi-Bo)/Ao
                spread = (Ao-Bo)/Ao
            
            if type(self.next_candidate)!=type(None):
                this_strategy = self.next_candidate.strategy
                if this_strategy.fix_spread:
                    ROI_live = GROI_live-roi_ratio*this_strategy.fixed_spread_ratio
                else:
                    ROI_live = GROI_live-roi_ratio*spread
            else:
                if self.last_fix_spread:
                    ROI_live = GROI_live-roi_ratio*self.last_fixed_spread_ratio
                else:
                    ROI_live = GROI_live-roi_ratio*spread
            
            info = (ass+","+Ti+","+date_time+","+str(bet)+","+
                      str(Bi)+","+str(Ai)+","+str(Bo)+","+
                      str(Ao)+","+"0"+","+str(100*GROI_live)+","+
                      str(100*spread)+","+str(100*ROI_live)+","+strategy_name)
            # update lists
            GROIs.append(GROI_live)
            ROIs.append(ROI_live)
            spreads.append(spread)
            infos.append(info)
            lotss.append(lots)
        
        return GROIs, ROIs, spreads, lotss, Bo, Ao, infos
        
    
    def close_position(self, date_time, ass, idx, results, s,
                       lot_ratio=None, partial_close=False, from_sl=0, 
                       DTi_real='', groiist=None, roiist=None, swap=0, 
                       returnist=None):
        """ Close position """
        
        list_idx = self.map_ass_idx2pos_idx[s][idx]
        
        # if it's full close, get the raminings of lots as lots ratio
        if not partial_close:
            lot_ratio = 1.0
        
        roi_ratio = lot_ratio*self.list_lots_per_pos[s]\
            [list_idx]/self.list_lots_entry[s]\
            [list_idx]
        if np.isnan(roi_ratio):
            raise AssertionError("np.isnan(roi_ratio)")
        # get returns
        GROIs, ROIs, spreads, lotss, Bo, Ao, infos = self.get_rois(idx, 
                                                          date_time=date_time,
                                                          roi_ratio=roi_ratio,
                                                          ass=ass, s=s)
        direction = self.list_opened_positions[s][self.map_ass_idx2pos_idx[s][idx]].direction
        status = self.get_account_status()
        self.available_bugdet_in_lots = status['free_margin']*status['leverage']/self.LOT#+lots2add#self.get_current_available_budget()
        self.available_budget = self.available_bugdet_in_lots*self.LOT 
        for i in range(len(GROIs)):
            GROI_live = GROIs[i]
            ROI_live = ROIs[i]
            if not groiist:
                groiist = 100*GROI_live
                roiist = 100*ROI_live
                returnist = self.list_lots_entry[s][list_idx]*ROI_live*self.LOT
            if DTi_real=='':
                DTi_real = date_time
            
#            lots2add = self.list_lots_per_pos[s][list_idx]*(lot_ratio+roiist/100)
            
            # update available budget file
#            self.update_current_available_budget()
            
            self.budget_in_lots += self.list_lots_per_pos[s][list_idx]*ROI_live
            
            
            nett_win = self.list_lots_entry[s][list_idx]*ROI_live*self.LOT
            gross_win = self.list_lots_entry[s][list_idx]*GROI_live*self.LOT
            
            self.gross_earnings += gross_win
            self.nett_earnigs += returnist
            
            if ROI_live>0:
                self.net_successes += 1
                self.average_win += ROI_live
            else:
                self.average_loss += ROI_live
                
            if GROI_live>0:
                self.gross_successes += 1
                
            #ROI_live = ROI_live-spread/entry_bid
            self.tROI_live += ROI_live
            self.tGROI_live += GROI_live
            
            e_spread = self.list_opened_positions[s][list_idx].e_spread
            # check if stoploss for server info
            if from_sl == 0:
                slfalg = False
            else:
                slfalg = True
            # check if close comes from external ban
            if not from_sl and self.list_count_events[s][list_idx]!=\
                self.list_deadlines[s][list_idx]:
                    from_sl = 2
            # write output to trader summary
            info_close = infos[i]+","+str(nett_win)+","+str(e_spread*100*self.pip)+","+\
                str(from_sl)+","+str(100*self.tGROI_live)+","+str(100*self.tROI_live)
            write_log(info_close, self.log_positions_soll)
            
            # save position evolution
            pos_filename = get_positions_filename(ass, self.list_opened_positions[s]\
                                                  [list_idx].entry_time, 
                                                  date_time)
            if run_back_test:
    #            pos_filename = get_positions_filename(ass, self.list_last_dt[list_idx][0], 
    #                                              self.list_last_dt[list_idx][-1])
                dirfilename = results.save_pos_evolution(pos_filename, self.list_last_dt[s][list_idx],
                                           self.list_last_bid[s][list_idx], 
                                           self.list_last_ask[s][list_idx])
            else:
                ##last_dt = self.list_symbols_tracking[self.map_ass_idx2pos_idx[idx]].DateTime.iloc[-1]
                #pos_filename = get_positions_filename(ass, DTi_real, date_time)
                dirfilename = results.save_pos_evolution_live(pos_filename, self.list_symbols_tracking[s][list_idx])
            
            self.track_position('close', date_time, idx=idx, groi=GROI_live, filename=pos_filename, s=s)
            # update output lists
            results.update_outputs(date_time, 100*GROI_live, 100*ROI_live, nett_win)
                        
            if partial_close:
                partial_string = ' Partial'
            else:
                partial_string = ' Full'
                self.n_pos_currently_open -= 1
            logMsg = " "+( date_time+partial_string+" close"+" dir {0:d}"\
                  .format(direction)+
                  " GROI {2:.3f}% Spread {1:.3f}% ROI = {0:.3f}%".format(
                          100*ROI_live,100*spreads[i],100*GROI_live)+
                          " TGROI {1:.3f}% TROI = {0:.3f}%".format(
                          100*self.tROI_live,100*self.tGROI_live)+
                          " Earnings {0:.2f}".format(self.nett_earnigs)+
                          ". Remaining open "+str(self.n_pos_currently_open))
            out =ass+logMsg
            if verbose_trader:
                self.write_log(out)
                print("\r"+out)
            if send_info_api:
                send_log_info(self.queue, ass, {"FUNC":"LOG","ORIGIN":"TRADE","ASS":ass,"MSG":logMsg})
    #            self.queue.put({"FUNC":"LOG","ORIGIN":"TRADE","ASS":ass,"MSG":logMsg})
            # compare budget with real one
            if not run_back_test:
#                status = ct.get_account_status()
#                balance, leverage, equity, profits = #self.get_account_status()
                logMsg = " "+date_time+" equity "+str(status['equity'])+" Balance "+str(status['balance'])
                
                out = ass+logMsg
                self.write_log(out)
                print("\r"+out)
                if send_info_api:
                    send_log_info(self.queue, ass, {"FUNC":"LOG","ORIGIN":"TRADE","ASS":ass,"MSG":logMsg})
    #                self.queue.put({"FUNC":"LOG","ORIGIN":"TRADE","ASS":ass,"MSG":logMsg})
                self.budget = status['balance']
            else:
                self.budget += returnist
            if send_info_api:
                self.send_close_pos_api(date_time, ass, Bo, Ao, 100*spreads[i], 
                                        100*GROI_live, 100*ROI_live, returnist, 
                                        pos_filename, dirfilename, DTi_real, groiist, 
                                        roiist, slfalg, swap, s)
            assert(lot_ratio<=1.00 and lot_ratio>0)
        
        if not partial_close:
            self.remove_position(idx, s)
        else:
            # decrease the lot ratio in case the position is not fully closed
            self.list_lots_per_pos[s][list_idx] = \
                self.list_lots_per_pos[s][list_idx]*(1-lot_ratio)
        
        return None
    
    def get_current_available_budget(self):
        """ get available budget from shared file among traders """
        fh = open(self.budget_file,"r")
        # read output
        av_bugdet_in_lots = float(fh.read())
        fh.close()
        print("get_current_available_budget: "+str(av_bugdet_in_lots))
        return av_bugdet_in_lots
    
#    def update_current_available_budget(self):
#        """ update available budget from shared file among traders """
#        fh = open(self.budget_file,"w")
#        fh.write(str(self.available_bugdet_in_lots))
#        fh.close()
#        print("update_current_available_budget: "+str(self.available_bugdet_in_lots))
#        return None
    
    def open_position(self, idx, lots, DateTime, e_spread, bid, ask, deadline):
        """ Open position """
        # update available budget
        str_idx = self.next_candidate.strategy_index
        status = self.get_account_status()
        self.available_bugdet_in_lots = status['free_margin']*status['leverage']/self.LOT#+lots2add#self.get_current_available_budget()
        self.available_budget = self.available_bugdet_in_lots*self.LOT
        # update available budget file
#        self.update_current_available_budget()
        #self.available_bugdet_in_lots -= lots
        self.n_entries += 1
        self.n_pos_opened += 1
        self.n_pos_currently_open += 1
        
        # update vector of opened positions
        self.add_position(idx, lots, DateTime, bid, ask, deadline)
        # track position
        self.track_position('open', DateTime)
        
        thisAsset = self.list_opened_positions[str_idx][-1].asset
        logMsg = (DateTime+" Open "+
              " Lots {0:.2f}".format(lots)+" "+str(self.list_opened_positions[str_idx][-1].bet)+
              " p_mc={0:.2f}".format(self.list_opened_positions[str_idx][-1].p_mc)+
              " p_md={0:.2f}".format(self.list_opened_positions[str_idx][-1].p_md)+
              " spread={0:.3f} ".format(e_spread)+" total opened {0:d}".format(self.n_pos_currently_open)+
              " strategy "+self.list_opened_positions[str_idx][-1].strategy.name)
        
        if self.next_candidate.strategy.entry_strategy == 'gre_v2':
            logMsg += logMsg+' prof '+str(self.next_candidate.profitability)
        if verbose_trader:
            out = thisAsset+logMsg
            print("\r"+out)
            self.write_log(out)
            send_log_info(self.queue, thisAsset, {"FUNC":"LOG","ORIGIN":"TRADE","ASS":thisAsset,"MSG":logMsg})
#            if log_thu_control:
#                self.queue.put({"FUNC":"LOG","ORIGIN":"TRADE","ASS":thisAsset,"MSG":logMsg})
        
        # Send open position command to api
        if send_info_api:
            self.send_open_pos_api(DateTime, bid, ask, e_spread, lots)
        
        return None
    
    def find_position(self):
        """  """
#        out = "find_position"
#        print("\r"+out)
#        self.write_log(out)
        if len(self.list_pos_idx)==0:
            pos_idx = 0
        elif not 0 in self.list_pos_idx:
            pos_idx = 0
        else:
            pos_idx = next(filterfalse(set(self.list_pos_idx).__contains__, count(1)))
#        out = "self.list_pos_idx"
#        print("\r"+out)
#        self.write_log(out)
#        out = str(self.list_pos_idx)
#        print("\r"+out)
#        self.write_log(out)
#        out = "pos_idx"
#        print("\r"+out)
#        self.write_log(out)
#        out = str(pos_idx)
#        print("\r"+out)
#        self.write_log(out)
        return pos_idx
    
    def double_down_position(self, ass_id, str_idx, amount_dd, dt, bid, ask, directory_MT5_ass):
        """  """
        self.list_dd_info[str_idx][self.map_ass_idx2pos_idx[str_idx][ass_id]].append({'entry_bid':bid,
                                                                               'entry_ask':ask,
                                                                               'entry_time':dt,
                                                                               'lots':amount_dd})
        dd_idx = len(self.list_dd_info[str_idx][self.map_ass_idx2pos_idx[str_idx][ass_id]])-1
#        self.available_budget -= lots*self.LOT
#        self.available_bugdet_in_lots -= lots
#        self.update_current_available_budget()
        self.n_pos_opened += 1
        self.n_pos_currently_open += 1
        self.n_entries += 1
        pos_idx = self.find_position()
        self.list_pos_idx.append(pos_idx)
        self.pos_idx2map_ass_str[pos_idx] = (ass_id, str_idx, dd_idx)
        self.send_open_command(directory_MT5_ass, ass_id, self.list_pos_idx[-1], lots=amount_dd)
#        out = (dt+" Double Down "+#asset+
#              " Lots {0:.2f}".format(amount_dd))
#        print(out)
#        self.write_log(out)
        status = self.get_account_status()
        self.available_bugdet_in_lots = status['free_margin']*status['leverage']/self.LOT#+lots2add#self.get_current_available_budget()
        self.available_budget = self.available_bugdet_in_lots*self.LOT 
        return None
    
    def check_condition_double_down(self, currGROI, ass_id, str_idx, thisAsset, 
                                    dt, bid, ask, spread, 
                                    condition_dd, reason_dd, directory_MT5_ass):
        """  """
        if condition_dd:
            if double_down['on']:
                if currGROI/self.pip<=-(self.list_dd_info[str_idx][self.map_ass_idx2pos_idx[str_idx][ass_id]][0]['checkpoint']+
                                    self.list_dd_info[str_idx][self.map_ass_idx2pos_idx[str_idx][ass_id]][0]['every']):
                    
                    times_dd_supposed = np.floor((-currGROI/self.pip-self.list_dd_info[str_idx]
                        [self.map_ass_idx2pos_idx[str_idx][ass_id]][0]['checkpoint'])/
                        self.list_dd_info[str_idx][self.map_ass_idx2pos_idx[str_idx][ass_id]][0]['every'])
                    times_dd = min(double_down['max'], times_dd_supposed)
                    lots_per_pos = self.assign_lots(dt)
                    slots_requested = times_dd*double_down['amount']
                    status = self.get_account_status()
                    self.available_bugdet_in_lots = status['free_margin']*status['leverage']/self.LOT#self.get_current_available_budget()
                    slots_available = np.floor(double_down['amount']*self.available_bugdet_in_lots/lots_per_pos)
                    slots_assign = min(slots_requested, slots_available)
                    n_dds = len(self.list_dd_info[str_idx][self.map_ass_idx2pos_idx[str_idx][ass_id]])
    #                print("slots_requested")
    #                print(slots_requested)
    #                print("slots_available")
    #                print(slots_available)
    #                print("slots_assign")
    #                print(slots_assign)
                    amount_dd = slots_assign*lots_per_pos
                    if slots_available>0 and self.n_pos_currently_open<self.max_opened_positions:
                        
                        self.list_dd_info[str_idx][self.map_ass_idx2pos_idx[str_idx][ass_id]][0]['checkpoint'] = \
                            self.list_dd_info[str_idx][self.map_ass_idx2pos_idx[str_idx][ass_id]][0]['checkpoint']+\
                            times_dd_supposed*self.list_dd_info[str_idx][self.map_ass_idx2pos_idx[str_idx][ass_id]][0]['every']
                        self.list_dd_info[str_idx][self.map_ass_idx2pos_idx[str_idx][ass_id]][0]['every'] = \
                            2*self.list_dd_info[str_idx][self.map_ass_idx2pos_idx[str_idx][ass_id]][0]['every']
                        self.double_down_position(ass_id, 
                                                  str_idx,
                                                  amount_dd,
                                                  dt,
                                                  bid,
                                                  ask, 
                                                  directory_MT5_ass)
                        logMsg = (dt+" Double Down "+#asset+
                              " Lots {0:.2f}".format(amount_dd)+" "+
                              " spread={0:.3f}".format(spread)+
                              " "+str(n_dds)+"-th time. Multiply by "+str(times_dd)+
                              ". Next Checkpoint "+
                              str(self.list_dd_info[str_idx][self.map_ass_idx2pos_idx[str_idx][ass_id]][0]['checkpoint'])+
                              " Every "+
                              str(self.list_dd_info[str_idx][self.map_ass_idx2pos_idx[str_idx][ass_id]][0]['every']))
#                        print(out)
#                        trader.write_log(out)
#                        logMsg = "DOUBLING DOWN! "+str(amount_dd)+" lots. New checkpoint "+\
#                            str(self.list_dd_info[str_idx][self.map_ass_idx2pos_idx[str_idx][ass_id]][0]['checkpoint'])
                        print(thisAsset+" "+logMsg)
                        self.write_log(thisAsset+" "+logMsg)
                        if send_info_api:
                            send_log_info(self.queue, thisAsset, {"FUNC":"LOG","ORIGIN":"TRADE","ASS":thisAsset,"MSG":logMsg})
                        
                        #a=p
                    elif slots_available==0:
                        logMsg = "NOT ENOUGH BUDGET TO DOUBLE DOWN!"
                        print(logMsg)
                        self.write_log(logMsg)
                        if send_info_api:
                            send_log_info(self.queue, thisAsset, {"FUNC":"LOG","ORIGIN":"TRADE","ASS":thisAsset,"MSG":logMsg})
                    else:
                        logMsg = "NOT DOUBLE DOWN BECAUSE MAX NUMBER OF POSITIONS REACHED!"
                        print(logMsg)
                        self.write_log(logMsg)
                        if send_info_api:
                            send_log_info(self.queue, thisAsset, {"FUNC":"LOG","ORIGIN":"TRADE","ASS":thisAsset,"MSG":logMsg})
                else:
                    logMsg = "NOT DD DOUBLE DOWN since next DD is at "+\
                        str(-self.list_dd_info[str_idx][self.map_ass_idx2pos_idx[str_idx][ass_id]][0]['checkpoint']-double_down['every'])+" pips"
                    print(logMsg)
                    self.write_log(logMsg)
                    if send_info_api:
                        send_log_info(self.queue, thisAsset, {"FUNC":"LOG","ORIGIN":"TRADE","ASS":thisAsset,"MSG":logMsg})
        else:
            logMsg = "NOT DD DOUBLE DOWN due to "+reason_dd
            print(logMsg)
            self.write_log(logMsg)
            if send_info_api:
                send_log_info(self.queue, thisAsset, {"FUNC":"LOG","ORIGIN":"TRADE","ASS":thisAsset,"MSG":logMsg})
    
    def send_open_pos_api(self, DateTime, bid, ask, e_spread, lots):
        """ Send command to API for position opening """
        str_idx = self.next_candidate.strategy_index
        params = {'asset':self.list_opened_positions[str_idx][-1].asset,
                  'dtisoll':DateTime,#.replace(' ', '_', 1)
                  'bi':bid,
                  'ai':ask,
                  'espread':e_spread,
                  'lots':lots,
                  'direction':self.list_opened_positions[str_idx][-1].bet,
                  'strategyname':self.list_opened_positions[str_idx][-1].strategy.name,
                  'p_mc':self.list_opened_positions[str_idx][-1].p_mc,
                  'p_md':self.list_opened_positions[str_idx][-1].p_md,
                  'strategyidx':str_idx}
        
        try:
            if log_thu_control:
                self.queue_prior.put({"FUNC":"POS","EVENT":"OPEN","SESS_ID":self.session_json['id'],"PARAMS":params})
            else:
                position_json = ct.send_open_position(params, self.session_json['id'], 
                                                      self.token_header)
                print(position_json)
        except:
            print("WARNING! Error in send_open_pos_api in kaissandra.prod.run. Skipped.")
        
        
    def send_extend_pos_api(self, DateTime, thisAsset, groi, p_mc, p_md, 
                            direction, strategy, roi, ticks):
        """ Send command to API for position extension """
        str_idx = self.next_candidate.strategy_index
        params = {'groi':groi,
                  'dt':DateTime,
                  'p_mc':p_mc,
                  'p_md':p_md,
                  'tickscounter':ticks,
                  'direction':direction,
                  'strategyname':strategy,
                  'roi':roi}
        
        try:
            if log_thu_control:
                self.queue_prior.put({"FUNC":"POS","EVENT":"EXTEND","ASSET":thisAsset,"PARAMS":params,"STRATEGY":str_idx})
            else:
                ct.send_extend_position(params, thisAsset, str_idx, self.token_header)
        except:
            print("WARNING! Error in send_extend_pos_api in kaissandra.prod.run. Skipped.")
        
        
    def send_not_extend_pos_api(self, DateTime, thisAsset, groi, p_mc, p_md, 
                                direction, strategy, roi, ticks):
        """ Send command to API for position extension """
        str_idx = self.next_candidate.strategy_index
        params = {'asset':thisAsset,
                  'groi':groi,
                  'dt':DateTime,
                  'p_mc':p_mc,
                  'p_md':p_md,
                  'tickscounter':ticks,
                  'direction':direction,
                  'strategyname':strategy,
                  'roi':roi}
        
        try:
            if log_thu_control:
                self.queue_prior.put({"FUNC":"POS","EVENT":"NOTEXTEND","ASSET":thisAsset,"PARAMS":params,"STRATEGY":str_idx})
            else:
                ct.send_not_extend_position(params, thisAsset, str_idx, self.token_header)
        except:
            print("WARNING! Error in send_not_extend_pos_api in kaissandra.prod.run. Skipped.")
        
        
    def send_close_pos_api(self, DateTime, thisAsset, bid, ask, spread, groisoll, 
                           roisoll, returns, filename, dirfilename, dtiist, groiist, 
                           roiist, slfalg, swap, str_idx):
        """ Send command to API for position closing """
        params = {'dtosoll':DateTime,
                  'dtiist':dtiist,
                  'bo':bid,
                  'ao':ask,
                  'spread':spread,
                  'groisoll':groisoll,
                  'roisoll':roisoll,
                  'groiist':groiist,
                  'roiist':roiist,
                  'returns':returns,
                  'filename':filename,
                  'slfalg':slfalg,
                  'swap':swap
                }
        
        try:
            if log_thu_control:
                self.queue_prior.put({"FUNC":"POS","EVENT":"CLOSE","DIRFILENAME":dirfilename,"ASSET":thisAsset,"PARAMS":params,"STRATEGY":str_idx})
            else:
                ct.send_close_position(params, thisAsset, str_idx, dirfilename, self.token_header)
        except:
            print("WARNING! Error in send_close_pos_api in kaissandra.prod.run. Skipped.")
        
    
    def track_position(self, event, DateTime, idx=None, groi=0.0, 
                       filename='', s=-1):
        """ track position.
        Args:
            - event (str): {open, extend, close} """
        if s==-1:
            str_idx = self.next_candidate.strategy_index
        else:
            str_idx = s
        if event=='open':
            pos_info = {'id':0,
                        'n_ext':0,
                        'dts':[DateTime],
                        '@tick#':[0],
                        'grois':[groi],
                        'p_mcs':[self.list_opened_positions[str_idx][-1].p_mc],
                        'p_mds':[self.list_opened_positions[str_idx][-1].p_md],
                        'levels':[self.list_opened_positions[str_idx][-1].bet],
                        'strategy':[self.list_opened_positions[str_idx][-1].strategy.name]}
            #print(pos_info)
            self.positions_tracker[str_idx].append(pos_info)
        elif event=='extend':
            #print(self.map_ass_idx2pos_idx[idx])
            #print(self.positions_tracker)
            pos_info = self.positions_tracker[str_idx][self.map_ass_idx2pos_idx[str_idx][idx]]
            p_mc = self.list_opened_positions[str_idx][self.map_ass_idx2pos_idx[str_idx][idx]].p_mc
            p_md = self.list_opened_positions[str_idx][self.map_ass_idx2pos_idx[str_idx][idx]].p_md
            bet = self.list_opened_positions[str_idx][self.map_ass_idx2pos_idx[str_idx][idx]].bet
            strategy_name = self.list_opened_positions[str_idx][self.map_ass_idx2pos_idx[str_idx][idx]].strategy.name
            tick_counts = self.list_count_all_events[str_idx][self.map_ass_idx2pos_idx[str_idx][idx]]
            pos_info['n_ext'] += 1
            pos_info['dts'].append(DateTime)
            pos_info['p_mcs'].append(p_mc)
            pos_info['p_mds'].append(p_md)
            pos_info['levels'].append(bet)
            pos_info['grois'].append(groi)
            pos_info['@tick#'].append(tick_counts)
            pos_info['strategy'].append(strategy_name)
        elif event=='close':
            pos_info = self.positions_tracker[str_idx][self.map_ass_idx2pos_idx[str_idx][idx]]
            if run_back_test:
                n_ticks = len(self.list_last_bid[str_idx][self.map_ass_idx2pos_idx[str_idx][idx]])
            else:
                n_ticks = self.list_symbols_tracking[str_idx][self.map_ass_idx2pos_idx[str_idx][idx]].shape[0]
            pos_info['dts'].append(DateTime)
            pos_info['p_mcs'].append(None)
            pos_info['p_mds'].append(None)
            pos_info['levels'].append(None)
            pos_info['grois'].append(groi)
            pos_info['@tick#'].append(n_ticks)
            pos_info['strategy'].append(None)
            # save in disk
            #print(pos_info)
            pickle.dump( pos_info, open( self.dir_positions+filename+'.p', "wb" ))
        
    def assign_lots(self, date_time):#, date_time, ass, idx
        """  """
        this_strategy = self.next_candidate.strategy
        if not this_strategy.flexible_lot_ratio:
            # update available budget
#            status = self.get_account_status()
#            self.available_bugdet_in_lots = status['balance']/self.LOT #self.get_current_available_budget()
            open_lots = this_strategy.max_lots_per_pos
            # check if there's enough bugdet available
#            if self.available_bugdet_in_lots>0:
#                open_lots = min(this_strategy.max_lots_per_pos, 
#                                self.available_bugdet_in_lots)
#            else:
#                open_lots = this_strategy.max_lots_per_pos
        else:
            # lots ratio to asssign to new asset
            pass
#            open_lots = min(self.budget_in_lots/(len(self.list_opened_positions)+1),
#                            this_strategy.max_lots_per_pos)
#            margin = 0.0001
#            # check if necessary lots for opening are available
#            if open_lots>self.available_bugdet_in_lots+margin:
#                close_lots = (open_lots-self.available_bugdet_in_lots)/\
#                    len(self.list_opened_positions)
#                print("close_lots "+str(close_lots))
#                if close_lots==np.inf:
#                    raise ValueError("close_lots==np.inf")
#                # loop over open positions to close th close_lot_ratio ratio to allocate the new position
#                for pos in range(len(self.list_opened_positions)):
#                    # lots ratio to be closed for this asset
#                    ass = self.list_opened_positions[pos].asset.decode("utf-8")
##                    print(ass)
##                    print("self.list_lots_per_pos[pos] "+str(self.list_lots_per_pos[pos]))
#                    close_lot_ratio = close_lots/self.list_lots_per_pos[pos]
##                    print("close_lot_ratio "+str(close_lot_ratio))
#                    idx = running_assets[ass2index_mapping[ass]]
#                    self.close_position(date_time, ass, idx, lot_ratio = 
#                                        close_lot_ratio, partial_close = True)
            status = self.get_account_status()
            self.available_bugdet_in_lots = status['free_margin']*status['leverage']/self.LOT
            # make sure the available resources are smaller or equal than slots to open
            open_lots = min(this_strategy.max_lots_per_pos,self.available_bugdet_in_lots)
        #print("open_lots corrected "+str(open_lots))
            
        return open_lots
    
#    def select_new_entry(self, inputs, thisAsset):
#        """ get new entry from inputs comming from network
#        Arg:
#            - inputs (list): [n_inputs][list, network_index][t_index][input] """
#        #print(inputs)
#        e_spread = inputs[0][0][0][1]
#        DateTime = inputs[0][0][0][2]
#        Bi = inputs[0][0][0][3]
#        Ai = inputs[0][0][0][4]
#        e_spread_pip = e_spread/self.pip
#        s_prof = -10000 # -infinite
#        for nn in range(len(inputs)):
#            network_index = inputs[nn][-1]
#            for i in range(len(inputs[nn])-1):
#                
#                deadline = inputs[nn][i][0][5]
#                
#                for t in range(len(inputs[nn][i])):
#                    soft_tilde = inputs[nn][i][t][0]
#                    #t_index = inputs[nn][i][t][6]
#                    #print("nn "+str(network_index)+" i "+str(i)+" t "+str(t_index))
#                    # get probabilities
#                    max_bit_md = int(np.argmax(soft_tilde[1:3]))
#                    if not max_bit_md:
#                        #Y_tilde = -1
#                        Y_tilde = np.argmax(soft_tilde[3:5])-2
#                    else:
#                        #Y_tilde = 1
#                        Y_tilde = np.argmax(soft_tilde[6:])+1
#                        
#                    p_mc = soft_tilde[0]
#                    p_md = np.max([soft_tilde[1],soft_tilde[2]])
#                    profitability = self.strategies[network_index].get_profitability(
#                            t, p_mc, p_md, int(np.abs(Y_tilde)-1))
#                    #print("profitability: "+str(profitability))
#                    if profitability>s_prof:
#                        s_prof = profitability
#                        s_deadline = deadline
#                        s_p_mc = p_mc
#                        s_p_md = p_md
#                        s_Y_tilde = Y_tilde
#                        s_network_index = network_index
#                        s_t = t
#                    # end of for t in range(len(inputs[nn][i])):
#            # end of for i in range(len(inputs[nn])-1):
#        # end of for nn in range(len(inputs)):
#        # add profitabilities
#        #print("s_prof: "+str(s_prof))
#        new_entry = {}
#        new_entry[entry_time_column] = DateTime
#        new_entry['Asset'] = thisAsset
#        new_entry['Bet'] = s_Y_tilde
#        new_entry['P_mc'] = s_p_mc
#        new_entry['P_md'] = s_p_md
#        new_entry[entry_bid_column] = Bi
#        new_entry[entry_ask_column] = Ai
#        new_entry['E_spread'] = e_spread_pip
#        new_entry['Deadline'] = s_deadline
#        new_entry['network_index'] = s_network_index
#        new_entry['profitability'] = s_prof
#        new_entry['t'] = s_t
#        
#        return new_entry
    
    def select_next_entry(self, inputs, thisAsset):
        """  """
        e_spread = inputs[0][0][0][1]
        DateTime = inputs[0][0][0][2]
        Bi = inputs[0][0][0][3]
        Ai = inputs[0][0][0][4]
        e_spread_pip = e_spread/self.pip
        #s_prof = -10000 # -infinite
        #print("Number of Networks:")
        #print(len(inputs))
        for nn in range(len(inputs)):
            #network_index = inputs[nn][-1]
            
            for i in range(len(inputs[nn])-1):
                
                deadline = inputs[nn][i][0][5]
                network_name = inputs[nn][i][0][6]
                #print("network_name")
                #print(network_name)
                
                for t in range(len(inputs[nn][i])):
                    soft_tilde = inputs[nn][i][t][0]
                    #t_index = inputs[nn][i][t][6]
                    #print("nn "+str(network_index)+" i "+str(i)+" t "+str(t_index))
                    # get probabilities
                    max_bit_md = int(np.argmax(soft_tilde[1:3]))
#                    if not max_bit_md:
#                        #Y_tilde = -1
#                        Y_tilde = np.argmax(soft_tilde[3:5])-2
#                    else:
#                        #Y_tilde = 1
#                        Y_tilde = np.argmax(soft_tilde[6:])+1
                        
                    if not max_bit_md:
                        #Y_tilde = -1
                        # Revert direction if crisis mode
                        if not crisis_mode:
                            Y_tilde = np.argmax(soft_tilde[3:5])-2
                        else:
                            Y_tilde = np.argmax(soft_tilde[6:])+1
                    else:
                        #Y_tilde = 1
                        if not crisis_mode:
                            Y_tilde = np.argmax(soft_tilde[6:])+1
                        else:
                            Y_tilde = np.argmax(soft_tilde[3:5])-2
                    
                    # change direction if test
#                    if test:
#                        Y_tilde = -1*Y_tilde
#                        print("Y_tilde")
#                        print(Y_tilde)
                        
                    p_mc = soft_tilde[0]
                    p_md = np.max([soft_tilde[1],soft_tilde[2]])
                    if network_name in self.net2strategy:
                        strategy_index = self.net2strategy.index(network_name)
                        #print("str_index in trader")
                        if self.strategies[strategy_index].entry_strategy!='spread_ranges':
                            profitability = self.strategies[strategy_index].get_profitability(
                                    t, p_mc, p_md, int(np.abs(Y_tilde)-1))
                        else:
                            profitability = 0.0
                        yield {entry_time_column:DateTime,
                               'Asset':thisAsset,
                               'Bet':Y_tilde,
                               'P_mc':p_mc,
                               'P_md':p_md,
                               entry_bid_column:Bi,
                               entry_ask_column:Ai,
                               'E_spread':e_spread_pip,
                               'Deadline':deadline,
                               'strategy_index':strategy_index,
                               'profitability':profitability,
                               't':t}
                    else:
                        yield []
    
    def update_margin(self, ass_idx, vi_struct, DateTime, thisAsset):
        """ Update margin """
#        margins[ass_idx] = 0.16*idx_struct['VIs'][0]-0.04
        # step-like relationship
        if vi_struct['VIs'][0]>=0.5 and self.margins[ass_idx]!=0.04:# and margin<0.12:
            self.margins[ass_idx] = 0.04
            logMsg = DateTime+" "+thisAsset+" VI="+str(vi_struct['VIs'][0])+". Margin changed to 0.04"
            print(logMsg)
            self.write_log(logMsg)
            if send_info_api:
                send_log_info(self.queue, thisAsset, {"FUNC":"LOG","ORIGIN":"TRADE","ASS":thisAsset,"MSG":logMsg})
        if vi_struct['VIs'][0]<0.5 and self.margins[ass_idx]!=0.0:# and margin<0.12:
            self.margins[ass_idx] = 0.0
            logMsg = DateTime+" "+thisAsset+" VI="+str(vi_struct['VIs'][0])+" Margin changed to 0.0"
            print(logMsg)
            self.write_log(logMsg)
            if send_info_api:
                send_log_info(self.queue, thisAsset, {"FUNC":"LOG","ORIGIN":"TRADE","ASS":thisAsset,"MSG":logMsg})
    
    def check_new_inputs(self, inputs, thisAsset, results, vi_struct, directory_MT5_ass=''):
        """
        <DocString>
        """
        ass_idx = self.ass2index_mapping[thisAsset]
        ass_id = self.running_assets[ass_idx]
        
        #new_entry = self.select_new_entry(inputs, thisAsset)
        for new_entry in self.select_next_entry(inputs, thisAsset):
            if not type(new_entry)==list:
                # update margin
                if margin_adapt:
                    self.update_margin(ass_idx, vi_struct, new_entry[entry_time_column], thisAsset)
                    margin_str = "Margin {0:.3f}".format(self.margins[ass_idx])+" VI {0:.3f} ".format(vi_struct['VIs'][0])
                else:
                    margin_str = ""
                # get number of tactics
                tactics = []
                if self.strategies[new_entry['strategy_index']].entry_strategy=='spread_ranges':
                    #n_tactics = len(self.strategies[new_entry['strategy_index']].info_spread_ranges['th'])
                    #print(self.strategies[new_entry['strategy_index']].info_spread_ranges['th'])
                    fixed_margins = self.strategies[new_entry['strategy_index']].info_spread_ranges['mar']
                    for t, tupl in enumerate(self.strategies[new_entry['strategy_index']].info_spread_ranges['th'][::-1]):
                        if new_entry['P_mc']>=tupl[0]+fixed_margins[-(t+1)][0] and new_entry['P_md']>=tupl[1]+fixed_margins[-(t+1)][1]+self.margins[ass_idx]:
#                            print("tupl")
#                            print(tupl)
##                            print("(new_entry['P_mc'],new_entry['P_md'])")
##                            print((new_entry['P_mc'],new_entry['P_md']))
#                            print("-(t+1)")
#                            print(-(t+1))
#                            print("margins[-(t+1)]")
#                            print(margins[-(t+1)])
                            
#                            print("len(self.strategies[new_entry['strategy_index']].info_spread_ranges['th'])")
#                            print(len(self.strategies[new_entry['strategy_index']].info_spread_ranges['th']))

                            tactics.append(len(self.strategies[new_entry['strategy_index']].info_spread_ranges['th'])-t-1)
                            
                            break
                    
                if len(tactics)==0:
                    tactics = [0]
                
                # loop over tactics of one strategy
                for tactic in tactics:
                    
                    strategy_name = self.strategies[new_entry['strategy_index']].name
                    position = Position(new_entry, self.strategies[new_entry['strategy_index']])
                    
                    self.add_new_candidate(position)
                    str_idx = self.next_candidate.strategy_index
                    
                    
                    
                    # check for opening/extension in order of expected returns
                    logMsg = (" New entry @ "+new_entry[entry_time_column]+" "+
                           "P_mc {0:.3f} ".format(new_entry['P_mc'])+
                           "P_md {0:.3f} ".format(new_entry['P_md'])+
                           margin_str+
                           #"prof. {0:.2f} ".format(new_entry['profitability'])+
                           "Bet {0:d} ".format(new_entry['Bet'])+
                           "E_spread {0:.3f} ".format(new_entry['E_spread'])+
                           "Strategy "+strategy_name)
                    if verbose_trader:
                        out = new_entry['Asset']+logMsg
                        print("\r"+out)
                        self.write_log(out)
                    if send_info_api:
                        send_log_info(self.queue, thisAsset, {"FUNC":"LOG","ORIGIN":"TRADE","ASS":thisAsset,"MSG":logMsg})
#                        self.queue.put({"FUNC":"LOG","ORIGIN":"TRADE","ASS":thisAsset,"MSG":logMsg})
                    
                    
#                    tactic = len(self.strategies[new_entry['strategy_index']].info_spread_ranges['th'])-t-1
#                    print(this_strategy.info_spread_ranges['th'][tactic][1]+this_strategy.info_spread_ranges['mar'][tactic][1])
#                    print("tactic")
#                    print(tactic)
#                    print("self.next_candidate.strategy.info_spread_ranges['th'][tactic]")
#                    print(self.next_candidate.strategy.info_spread_ranges['th'][tactic])
                    
                    # check if asset is banned
                    if not self.list_is_asset_banned[ass_idx]:
                        # open market
                        if not self.is_opened_strategy(ass_id):
                            # check if condition for opening is met
                            condition_open, reason = self.check_contition_for_opening(tactic, ass_idx)
                            if condition_open:
                                # assign budget
                                lots = self.assign_lots(new_entry[entry_time_column])
                                status = self.get_account_status()
                                self.available_bugdet_in_lots = status['free_margin']*status['leverage']/self.LOT 
                                # check if there is enough budget
                                if self.available_bugdet_in_lots>=lots:
                                    # add postion to str,ass,dd mapping
                                    pos_idx = self.find_position()
                                    self.list_pos_idx.append(pos_idx)
                                    self.pos_idx2map_ass_str[pos_idx] = (ass_id, str_idx, 0)
#                                    print("pos_idx")
#                                    print(pos_idx)
#                                    
#                                    print("self.list_pos_idx")
#                                    print(self.list_pos_idx)
#                                    
#                                    print("self.pos_idx2map_ass_str[pos_idx]")
#                                    print(self.pos_idx2map_ass_str[pos_idx])
                                    if not run_back_test:
                                        self.send_open_command(directory_MT5_ass, ass_id, self.list_pos_idx[-1], lots=lots)
                                    self.open_position(ass_id, lots, 
                                                       new_entry[entry_time_column], 
                                                       self.next_candidate.e_spread, 
                                                       self.next_candidate.entry_bid, 
                                                       self.next_candidate.entry_ask, 
                                                       self.next_candidate.deadline)
                                else: # no opening due to budget lack
                                    if verbose_trader:
                                        logMsg = " Not enough budget"
                                        out = thisAsset+logMsg
                                        print("\r"+out)
                                        self.write_log(out)
                                    if send_info_api:
                                        send_log_info(self.queue, thisAsset, {"FUNC":"LOG","ORIGIN":"TRADE","ASS":thisAsset,"MSG":logMsg})
#                                        self.queue.put({"FUNC":"LOG","ORIGIN":"TRADE","ASS":thisAsset,"MSG":logMsg})
                                    # check swap of resources
                                    if self.next_candidate.strategy.entry_strategy=='gre':
            #                             and self.check_resources_swap()
                                        # TODO: Check propertly function of swapinng
                                        # lauch swap of resourves
                                        pass
                                        ### WARNING! With asynched traders no swap
                                        ### possible
                                        #self.initialize_resources_swap(directory_MT5_ass)
                                # break loop over tactics
                                continue
                            else:
                                logMsg = " "+new_entry[entry_time_column]+" not opened "+\
                                          " due to "+reason
                                if verbose_trader:
                                    out = thisAsset+logMsg
                                    print("\r"+out)
                                    self.write_log(out)
#                                if send_info_api:
#                                    self.queue.put({"FUNC":"LOG","ORIGIN":"TRADE","ASS":thisAsset,"MSG":logMsg})
                        else: # position is opened
                            str_idx = self.next_candidate.strategy_index
                            direction = self.list_opened_positions[str_idx][self.map_ass_idx2pos_idx[str_idx][ass_id]].bet
                            curr_GROI, curr_ROI, spreads, lotss, Bo, Ao, infos = self.get_rois(ass_id, date_time='', roi_ratio=1)
                            logMsg = " "+new_entry[entry_time_column]+" "+\
                                           " deadline in "+str(self.get_remain_samps(ass_id))+\
                                           " Dir {0:d} ".format(direction)+\
                                            " current GROI = {0:.2f}%".format(100*curr_GROI[0])+\
                                            " current ROI = {0:.2f}%".format(100*curr_ROI[0])
                            if verbose_trader:
                                out = thisAsset+logMsg
                                print("\r"+out)
                                self.write_log(out)
                            if send_info_api:
                                send_log_info(self.queue, thisAsset, {"FUNC":"LOG","ORIGIN":"TRADE","ASS":thisAsset,"MSG":logMsg})
#                                self.queue.put({"FUNC":"LOG","ORIGIN":"TRADE","ASS":thisAsset,"MSG":logMsg})
                            # check for extension
                            if self.check_primary_condition_for_extention(ass_id):
                                # TEMP! In crisi_mode and different directions -> close!
                                if not self.check_same_direction(ass_id) and crisis_mode:
                                    logMsg = new_entry[entry_time_column]+" "\
                                            " CLOSING DUE TO CRISIS MODE!"
                                    out = thisAsset+" "+logMsg
                                    if verbose_trader:
                                        print("\r"+out)
                                        self.write_log(out)
                                    if run_back_test:
                                        self.close_position(new_entry[entry_time_column], 
                                                            thisAsset, ass_id, results, str_idx)
                                    else:
                                        self.send_close_command(thisAsset, str_idx)
                                    if send_info_api:
                                        send_log_info(self.queue, thisAsset, {"FUNC":"LOG","ORIGIN":"TRADE","ASS":thisAsset,"MSG":logMsg})
#                                        self.queue.put({"FUNC":"LOG","ORIGIN":"TRADE","ASS":thisAsset,"MSG":logMsg})
                                else:
                                    extention, reason, cond_bet = self.check_secondary_condition_for_extention(ass_id, ass_idx, curr_GROI[0], tactic)
                                    if extention:
                                        # include third condition for thresholds
                                        # extend deadline
                                        if not run_back_test:
                                            # find all positions with this ass_idx and str_idx
                                            # and send extention command
#                                            print("Check pos_idx for extention")
#                                            print("(ass_id, str_idx)")
#                                            print((ass_id, str_idx))
#                                            print("self.pos_idx2map_ass_str")
#                                            print(self.pos_idx2map_ass_str)
#                                            print("self.list_pos_idx")
#                                            print(self.list_pos_idx)
                                            for pos_idx in self.list_pos_idx:
                                                if self.pos_idx2map_ass_str[pos_idx][:2] == (ass_id, str_idx):
                                                    print("sending ex command. Pos ID "+str(pos_idx))
                                                    dd_idx = self.pos_idx2map_ass_str[pos_idx][2]
                                                    lots = self.list_dd_info[str_idx][self.map_ass_idx2pos_idx[str_idx][ass_id]][dd_idx]['lots']
                                                    self.send_open_command(directory_MT5_ass, ass_id, pos_idx, lots=lots)
                                        
                                        self.n_pos_extended += 1
                                        # track position
                                        self.track_position('extend', new_entry[entry_time_column], idx=ass_id, groi=curr_GROI[0])
                                        # build out
                                        logMsg = " "+new_entry[entry_time_column]+" "+\
                                               " Extended "+str(self.list_deadlines[str_idx][\
                                                   self.map_ass_idx2pos_idx[str_idx][ass_id]])+" samps"+\
                                               " bet "+str(new_entry['Bet'])+\
                                               " p_mc={0:.2f}".format(new_entry['P_mc'])+\
                                               " p_md={0:.2f}".format(new_entry['P_md'])+\
                                               " spread={0:.3f}".format(new_entry['E_spread'])+\
                                               " strategy "+strategy_name
                                        # print output
                                        if verbose_trader:
                                            out = thisAsset+logMsg
                                            print("\r"+out)
                                            self.write_log(out)
                                            
                                        # Check if Double Down
                                        condition_dd, reason_dd = self.check_contition_for_opening(tactic, ass_idx)
                                        #if condition_dd:
                                        self.check_condition_double_down(curr_GROI[0], ass_id, str_idx, 
                                                                         thisAsset, new_entry[entry_time_column], 
                                                                         Bo, Ao, new_entry['E_spread'], condition_dd, 
                                                                         reason_dd, directory_MT5_ass)
                                        self.update_position(ass_id)
                                        # send position extended command to api
                                        if send_info_api:
                                            send_log_info(self.queue, thisAsset, {"FUNC":"LOG","ORIGIN":"TRADE","ASS":thisAsset,"MSG":logMsg})
#                                            if log_thu_control:
#                                                self.queue.put({"FUNC":"LOG","ORIGIN":"TRADE","ASS":thisAsset,"MSG":logMsg})
                                            self.send_extend_pos_api(new_entry[entry_time_column], 
                                                                         thisAsset, 100*curr_GROI[0], 
                                                                         new_entry['P_mc'], new_entry['P_md'], 
                                                                         int(new_entry['Bet']), strategy_name, 
                                                                         100*curr_ROI[0], 
                                                                         self.list_count_all_events[str_idx][self.map_ass_idx2pos_idx[str_idx][ass_id]])
                                        
                                            
                                    elif not crisis_mode or not cond_bet: #  # if candidate for extension does not meet requirements
                                        logMsg = " "+new_entry[entry_time_column]+" not extended "+\
                                              " due to "+reason
                                        if verbose_trader:
                                            out = thisAsset+logMsg
                                            print("\r"+out)
                                            self.write_log(out)
                                        if send_info_api:
                                            self.send_not_extend_pos_api(new_entry[entry_time_column], 
                                                                         thisAsset, 100*curr_GROI[0], 
                                                                         new_entry['P_mc'], new_entry['P_md'], 
                                                                         int(new_entry['Bet']), strategy_name,
                                                                         100*curr_ROI[0], 
                                                                         self.list_count_all_events[str_idx][self.map_ass_idx2pos_idx[str_idx][ass_id]])
#                                            self.queue.put({"FUNC":"LOG","ORIGIN":"TRADE","ASS":thisAsset,"MSG":logMsg})
                                    else:
                                        # crisis and no extention. Close Pos!
                                        logMsg = new_entry[entry_time_column]+" "\
                                                " CLOSING DUE TO CRISIS MODE!"
                                        out = thisAsset+" "+logMsg
                                        if verbose_trader:
                                            print("\r"+out)
                                            self.write_log(out)
                                        if run_back_test:
                                            self.close_position(new_entry[entry_time_column], 
                                                                thisAsset, ass_id, results, str_idx)
                                        else:
                                            self.send_close_command(thisAsset, str_idx)
                                        if send_info_api:
                                            send_log_info(self.queue, thisAsset, {"FUNC":"LOG","ORIGIN":"TRADE","ASS":thisAsset,"MSG":logMsg})
#                                            self.queue.put({"FUNC":"LOG","ORIGIN":"TRADE","ASS":thisAsset,"MSG":logMsg})
                                
                            else: # if direction is different
                                this_strategy = self.next_candidate.strategy
                                close_pos = False
                                #if this_strategy.if_dir_change_close:
                                # TODO: Deeper analysis on when to close due to direction change
                                if this_strategy.entry_strategy=='spread_ranges':
                                    if this_strategy.if_dir_change_close and not self.check_same_direction(ass_id) and \
                                    self.check_same_strategy(ass_id) and \
                                    self.next_candidate.p_mc>=this_strategy.info_spread_ranges['th'][0][0]+this_strategy.info_spread_ranges['mar'][0][0] and \
                                    self.next_candidate.p_md>=this_strategy.info_spread_ranges['th'][0][1]+this_strategy.info_spread_ranges['mar'][0][1]:
                                        close_pos = True
                                elif this_strategy.entry_strategy=='gre_v2':
                                    if this_strategy.if_dir_change_close and not self.check_same_direction(ass_id) and \
                                    self.check_same_strategy(ass_id) and \
                                    self.next_candidate.profitability>self.list_opened_positions[str_idx][self.map_ass_idx2pos_idx[str_idx][ass_id]].profitability:
                                        print("self.next_candidate.profitability")
                                        print(self.next_candidate.profitability)
                                        print("self.list_opened_positions[self.map_ass_idx2pos_idx[ass_id]].profitability")
                                        print(self.list_opened_positions[str_idx][self.map_ass_idx2pos_idx[str_idx][ass_id]].profitability)
                                        close_pos = True
                                if close_pos:
                                    # close position due to direction change
                                    out = "WARNING! "+new_entry[entry_time_column]+" "+thisAsset\
                                    +" closing due to direction change!"
                                    if verbose_trader:
                                        print("\r"+out)
                                        self.write_log(out)
                                    if run_back_test:
                                        self.close_position(new_entry[entry_time_column], 
                                                            thisAsset, ass_id, results, str_idx)
                                    else:
                                        self.send_close_command(thisAsset, str_idx)
        #                                # TODO: study the option of not only closing 
                            # end of extention options
                    else: # asset banned
                        self.track_banned_asset(new_entry, ass_idx)
                        
                # end of for tactics    
            else:
                pass
                #print("Network not in Trader. Skipped")
        # end of for entries
        return None
    
    def write_log(self, log):
        """
        Write in log file
        """
        if self.save_log:
            file = open(self.log_file_trader,"a")
            file.write(log+"\n")
            file.close()
        return None
    
    def send_open_command(self, directory_MT5_ass, ass_id, str_idx, lots=-1):
        """ Send command for opening position to broker """
        if lots==-1:
            lots = self.next_candidate.strategy.max_lots_per_pos
        success = 0
        # load network output
        while not success:
            try:
                fh = open(directory_MT5_ass+"TT"+str(str_idx),"w")
                fh.write(str(self.next_candidate.direction)+","+
                         str(lots)+
                         ","+str(self.next_candidate.deadline)+","+
                         str(self.next_candidate.strategy.thr_sl)+","+
                         str(str_idx))
                fh.close()
                success = 1
                #stop_timer(ass_idx)
            except PermissionError:
                print("Error writing TT")
    
#    def send_extend_command(self, directory_MT5_ass, ass_idx, str_idx):
#        """ Send command for extending position to broker """
#        success = 0
#        # load network output
#        while not success:
#            try:
#                fh = open(directory_MT5_ass+"EX"+str(ass_idx),"w")
##                fh.write(str(self.next_candidate.direction)+","+
##                         str(self.next_candidate.strategy.max_lots_per_pos)+
##                         ","+str(self.next_candidate.deadline)+","+
##                         str(self.next_candidate.strategy.thr_sl)+","+
##                         str(str_idx))
#                fh.close()
#                success = 1
#                #stop_timer(ass_idx)
#            except PermissionError:
#                print("Error writing EX")
    
    def send_close_command(self, asset, str_idx):
        """ Send command for closeing position to MT5 software """
        directory_MT5_ass2close = LC.directory_MT5_IO+asset+"/"
        success = 0
        # load network output
        while not success:
            try:
                fh = open(directory_MT5_ass2close+"LC"+str(str_idx),"w")
                fh.write(str(str_idx))
                fh.close()
                success = 1
            except PermissionError:
                print("Error writing LC")
                
#    def check_resources_swap(self):
#        """
#        <DocString>
#        """
#        min_op = 0
#        min_profitability = self.list_opened_positions[min_op].profitability
#        for op in range(1,len(self.list_opened_positions)):
#            if self.list_opened_positions[op].profitability<min_profitability:
#                min_profitability = self.list_opened_positions[op].profitability
#                min_op = op
#        if self.next_candidate.profitability>min_profitability:    
#            self.swap_pos = min_op
#            out = "Swap "+self.next_candidate.asset+" with "\
#                +self.list_opened_positions[op].asset
#            self.write_log(out)
#            print(out)
#            return True
#        else:
#            return False
    
#    def initialize_resources_swap(self, directory_MT5_ass):
#        """
#        <DocString>
#        """
#        out = self.next_candidate.entry_time+" "+\
#            self.next_candidate.asset+" initialize swap"
#        self.write_log(out)
#        print(out)
#        #send close command if running live
#        if not run_back_test:
#            directory_MT5_ass2close = LC.directory_MT5_IO+\
#                self.list_opened_positions[self.swap_pos].asset+"/"
#            self.send_close_command(directory_MT5_ass2close)
#        # activate swap pending flag
#        self.swap_pending = 1
#        self.new_swap_candidate = self.next_candidate
#        self.swap_directory_MT5_ass = directory_MT5_ass
#        # if back test, finlaize swap
#        if run_back_test:
#            self.finalize_resources_swap()
        
    
#    def finalize_resources_swap(self):
#        """
#        <DocString>
#        """
#        out = self.new_swap_candidate.entry_time+" "+\
#            self.new_swap_candidate.asset+" finlaize swap"
#        self.write_log(out)
#        print(out)
#        # open new position
#        lots = self.assign_lots(self.new_swap_candidate.entry_time)
#        ass_idx = self.running_assets[self.ass2index_mapping\
#                                      [self.new_swap_candidate.asset]]
#        # check if there is enough budget
#        if self.available_bugdet_in_lots>=lots:
#            # send open command to MT5 if running live
#            if not run_back_test:
#                self.send_open_command(self.swap_directory_MT5_ass)
#            # open position
#            self.open_position(ass_idx, lots, self.new_swap_candidate.entry_time, 
#                               self.new_swap_candidate.e_spread, 
#                               self.new_swap_candidate.entry_bid, 
#                               self.new_swap_candidate.entry_ask, 
#                               self.new_swap_candidate.deadline)
#            self.swap_pending = 0
    
    def update_symbols_tracking(self, l, s, buffer):
        """ Update bids and asks list with new value from live """
        # update bid and ask lists if exist
        #if l>-1:
        self.list_symbols_tracking[s][l] = self.list_symbols_tracking[s][l].append(buffer)
            
    def update_list_last(self, l, s, datetime, bid, ask):
        """ Update bids and asks list with new value from backtest """
        # update bid and ask lists if exist
        if l>-1:
            self.list_last_bid[s][l].append(bid)
            self.list_last_ask[s][l].append(ask)
            self.list_last_dt[s][l].append(datetime)
            w = 1-1/20
            em = self.list_EM[s][l][-1]
            self.list_EM[s][l].append(w*em+(1-w)*bid)
            
#    def save_pos_evolution(self, asset, list_idx):
#        """ Save evolution of the position from opening till close """
#        # format datetime for filename
#        dt_open = dt.datetime.strftime(dt.datetime.strptime(
#                        self.list_last_dt[list_idx][0],'%Y.%m.%d %H:%M:%S'),
#                '%y%m%d%H%M%S')
#        dt_close = dt.datetime.strftime(dt.datetime.strptime(
#                        self.list_last_dt[list_idx][-1],'%Y.%m.%d %H:%M:%S'),
#                '%y%m%d%H%M%S')
#        filename = 'O'+dt_open+'C'+dt_close+asset+'.txt'
#        direct = self.dir_positions
#        df = pd.DataFrame({'DateTime':self.list_last_dt[list_idx],
#                           'SymbolBid':self.list_last_bid[list_idx],
#                           'SymbolAsk':self.list_last_ask[list_idx],
#                           'EmBid':self.list_EM[list_idx]})
#        df.to_csv(direct+filename, index=False)
    
    def ban_currencies(self, lists, thisAsset, DateTime, results, direction, 
                       dtiist='', groiist=None, roiist=None):
        """ Ban currency pairs related to ass_idx asset. WARNING! Assets 
        involving GOLD are not supported """
        # WARNING! Ban of only the asset closing stoploss. Change and for or
        # for ban on all assets sharing one currency
        ass_idx = 0
        message = thisAsset+','+str(direction)
        for ass_key in C.AllAssets:
            asset = C.AllAssets[ass_key]
            ass_id = int(ass_key)
            m1 = re.search(thisAsset[:3],asset)
            m2 = re.search(thisAsset[3:],asset)
            if spread_ban:
                condition = ((m1!=None and m1.span()[1]-m1.span()[0]==3) or \
                             (m2!=None and m2.span()[1]-m2.span()[0]==3))
            else:
                condition = ((m1!=None and m1.span()[1]-m1.span()[0]==3) and \
                             (m2!=None and m2.span()[1]-m2.span()[0]==3))
            if condition:# and self.is_opened(ass_id)
                # check if asset is controlled by this trader
                if ass_id in self.running_assets:
                    # check if position is opened
                    if self.is_opened_strategy(ass_id):
                        otherDirection = self.list_opened_positions\
                             [self.map_ass_idx2pos_idx[ass_id]].direction
                        
                        # check if directions coincide
                        if self.assets_same_direction(ass_id, thisAsset, 
                                                      direction, 
                                                      asset, otherDirection):
                            # ban assets this trader is controlling
                            ass_idx = self.running_assets.index(ass_id)
                            out = asset+" banned "
                            print(out)
                            self.write_log(out)
                            
#                            list_idx = self.map_ass_idx2pos_idx[ass_id]
                            for s in range(len(self.strategies)):
                                list_idx = self.map_ass_idx2pos_idx[s][ass_id]
                                if list_idx>-1:
                                    self.close_position(DateTime, asset, ass_id, results, s, from_sl=1, 
                                                        DTi_real=dtiist, groiist=groiist, 
                                                        roiist=roiist)
#                            if run_back_test:
#                                pass
##                                bid = self.list_last_bid[list_idx][-1]
#                            else:
#                                bid = self.list_symbols_tracking[list_idx].SymbolBid.iloc[-1]
#                            lists = flush_asset(lists, ass_idx, bid)
                            self.ban_asset(DateTime, asset, ass_idx)
                        else:
                            out = asset+" NOT banned due to different directions"
                            print(out)
                            self.write_log(out)
                    else:
                        out = asset+" NOT banned due to not opened"
                        print(out)
                        self.write_log(out)
                else:
                    # send ban message to traders controlling the other assets
                    self.send_ban_command(asset,message)
                    out = "Ban command sent from "+thisAsset+" to "+asset
                    print(out)
                    self.write_log(out)
                
            ass_idx += 1
            
        return lists
    
    def send_ban_command(self, asset_banned, message):
        """ Send ban command to an asset """
        file = open(self.ban_currencies_dir+self.start_time+asset_banned,"w")
        file.write(message+"\n")
        file.close()
        return None
    
    def assets_same_direction(self, ass_id, thisAsset, thisDirection, 
                              otherAsset, otherDirection):
        """  """
        m1 = re.search(thisAsset[:3],otherAsset)
        m2 = re.search(thisAsset[3:],otherAsset)
        condition = ((m1!=None and m1.span()[1]-m1.span()[0]==3 and \
                     thisDirection*otherDirection>0) or \
                     (m2!=None and m2.span()[1]-m2.span()[0]==3 and \
                     thisDirection*otherDirection<0))
        
        return condition
    
    def ban_asset(self, DateTime, thisAsset, ass_idx):
        """  """
        self.list_is_asset_banned[ass_idx] = True
        if not hasattr(self, 'list_dict_banned_assets'):
            self.list_dict_banned_assets = [None for _ in self.list_is_asset_banned]
        if not crisis_mode:
            counter = 20
        else:
            counter = 1
        tracing_dict = {'lastDateTime':DateTime,
                            'counter':counter}
        self.list_dict_banned_assets[ass_idx] = tracing_dict
        logMsg = " "+DateTime+" "+\
                " ban counter set to "\
                +str(self.list_dict_banned_assets[ass_idx]['counter'])
        if verbose_trader:
            out = thisAsset+logMsg
            print("\r"+out)
            self.write_log(out)
        if send_info_api:
            send_log_info(self.queue, thisAsset, {"FUNC":"LOG","ORIGIN":"TRADE","ASS":thisAsset,"MSG":logMsg})
#            self.queue.put({"FUNC":"LOG","ORIGIN":"TRADE","ASS":thisAsset,"MSG":logMsg})
        
    def track_banned_asset(self, entry, ass_idx):
        """ """
        if self.list_dict_banned_assets[ass_idx]['lastDateTime'] != entry[entry_time_column]:
            self.list_dict_banned_assets[ass_idx]['lastDateTime'] = entry[entry_time_column]
            self.list_dict_banned_assets[ass_idx]['counter'] -= 1
            logMsg = entry[entry_time_column]+" "+\
                " ban counter set to "\
                +str(self.list_dict_banned_assets[ass_idx]['counter'])
            out = entry['Asset']+logMsg
            if verbose_trader:
                print("\r"+out)
                self.write_log(out)
                if send_info_api:
#                    self.queue.put({"FUNC":"LOG","ORIGIN":"TRADE","ASS":entry['Asset'],"MSG":logMsg})
                    send_log_info(self.queue, entry['Asset'], {"FUNC":"LOG","ORIGIN":"TRADE","ASS":entry['Asset'],"MSG":logMsg})
            if self.list_dict_banned_assets[ass_idx]['counter'] == 0:
                self.lift_ban_asset(ass_idx)
                out = "Ban lifted"
                if verbose_trader:
                    print("\r"+out)
                self.write_log(out)
                if send_info_api:
                    send_log_info(self.queue, entry['Asset'], {"FUNC":"LOG","ORIGIN":"TRADE","ASS":entry['Asset'],"MSG":out})
#                    self.queue.put({"FUNC":"LOG","ORIGIN":"TRADE","ASS":entry['Asset'],"MSG":out})
        else:
            logMsg = entry[entry_time_column]+" "+\
                " ban counter already reduced for this DT"
            out = entry['Asset']+logMsg
            if verbose_trader:
                print("\r"+out)
            self.write_log(out)
            if send_info_api:
#                self.queue.put({"FUNC":"LOG","ORIGIN":"TRADE","ASS":entry['Asset'],"MSG":logMsg})
                send_log_info(self.queue, entry['Asset'], {"FUNC":"LOG","ORIGIN":"TRADE","ASS":entry['Asset'],"MSG":logMsg})
    
    def lift_ban_asset(self, ass_idx):
        """  """
        self.list_is_asset_banned[ass_idx] = False
        
    def update_parameters(self, config, thisAsset):
        """ Update parameters from local config file """
        try:
            
            if 'list_max_lots_per_pos' in config:
                # update lots
                self.update_lots(config['list_max_lots_per_pos'])
            if 'list_thr_sl' in config:
                # update stoplosses
                self.update_stoploss(config['list_thr_sl'])
            if 'max_opened_positions' in config:
                # update max_opened_positions
                self.max_opened_positions = config['max_opened_positions']
                print("max_opened_positions updated:")
                print(config['max_opened_positions'])
            if 'list_spread_ranges' in config:
                # update spread ranges
                self.update_spread_ranges(config['list_spread_ranges'])
            if send_info_api:
                send_log_info(self.queue, thisAsset, {"FUNC":"LOG",
                                                      "ORIGIN":"TRADE",
                                                      "ASS":thisAsset,
                                                      "MSG":"PARAMETERS UPDATED:"})
                send_log_info(self.queue_prior, thisAsset, {"FUNC":"CONFIG", 
                                          "CONFIG":config, 
                                          "ASSET":thisAsset, 
                                          "ORIGIN":"PARAM_UPDATE"})
#                if log_thu_control:
#                    self.queue.put({"FUNC":"LOG","ORIGIN":"TRADE","ASS":thisAsset,"MSG":"PARAMETERS UPDATED:"})
#                    self.queue_prior.put({"FUNC":"CONFIG", 
#                                          "CONFIG":config, 
#                                          "ASSET":thisAsset, 
#                                          "ORIGIN":"PARAM_UPDATE"})
#                else:
#                    ct.confirm_config_info(config, thisAsset, "PARAM_UPDATE", self.token_header)
        except:
            print("WARNING!! Error while reading conig file. Skipped")
        
    def update_lots(self, list_max_lots_per_pos):
        """ Update lots per position """
        for s in range(len(self.strategies)):
            self.strategies[s].max_lots_per_pos = list_max_lots_per_pos[s]
        print("list_max_lots_per_pos updated:")
        print(list_max_lots_per_pos)
        
            
    def update_stoploss(self, list_thr_sl):
        """ Update stoploss threshold """
        for s in range(len(self.strategies)):
            self.strategies[s].thr_sl = list_thr_sl[s]
        print("list_thr_sl updated:")
        print(list_thr_sl)
        
    def update_spread_ranges(self, list_spread_ranges):
        """ Update spread ranges for all strategies in trader """
        for s in range(len(self.strategies)):
            self.strategies[s].info_spread_ranges = list_spread_ranges[s]
            print(list_spread_ranges[s])

def write_log(log_message, log_file):
        """
        Write in log file
        """
        file = open(log_file,"a")
        file.write(log_message+"\n")
        file.close()
        return None

def runRNNliveFun(tradeInfoLive, listFillingX, init, listFeaturesLive, listParSarStruct,
                  listEM,listAllFeatsLive,list_X_i, means_in,phase_shift,stds_in, 
                  stds_out,AD, thisAsset, netName,listCountPos,list_weights_matrix,
                  list_time_to_entry,list_list_soft_tildes, list_Ylive,list_Pmc_live,
                  list_Pmd_live,list_Pmg_live,EOF,countOuts,t_indexes, c, results_network, 
                  results_file, model, config, log_file, nonVarIdx, list_inv_out, queue):
    """
    <DocString>
    """
##################################################################################################################################
########################################################## WARNING!!!!! ##########################################################
##################################################################################################################################
    thr_mc = 0
    thr_levels = 5
    lb_level = 5
    first_nonzero = 0
    nEventsPerStat = config['nEventsPerStat']
    movingWindow = config['movingWindow']
    feature_keys_manual = config['feature_keys_manual']
    nFeatures = len(feature_keys_manual)
    channels = config['channels']
    noVarFeats = config['noVarFeatsManual']
    lookAheadIndex = config['lookAheadIndex']
    n_bits_mg = config['n_bits_outputs'][-1]
    seq_len = config['seq_len']
    outputGain = config['outputGain']
#    thr_md = 0.5
    #print("\r"+tradeInfoLive.DateTime.iloc[-1]+" "+thisAsset+netName+" New input")
    nChannels = int(nEventsPerStat/movingWindow)
    file = 0
    #deli = "_"
    
    countIndex = 0
    countOut = 0
    count_t = 0
    
    output = []# init as an empy position
    
    # sub channel
    sc = c%phase_shift
    #rc = int(np.floor(c/phase_shift))
    #time_stamp[0] = tradeInfoLive.DateTime.iloc[-1]
    if listFillingX[sc]:
        logMsg = " "+tradeInfoLive.DateTime.iloc[-1]+" "+netName+"C"+\
            str(c)+"F"+str(file)
        if verbose_RNN:# and not simulate
            out = thisAsset+logMsg
            print("\r"+out)
            write_log(out, log_file)
        if send_info_api:
            send_log_info(queue, thisAsset, {"FUNC":"LOG","ORIGIN":"NET","ASS":thisAsset,"MSG":logMsg})
#            queue.put({"FUNC":"LOG","ORIGIN":"NET","ASS":thisAsset,"MSG":logMsg})
                
    # launch features extraction
    if init[sc]==False:
        logMsg = " "+tradeInfoLive.DateTime.iloc[-1]+" "+netName+\
                " Features inited"
        if verbose_RNN:
            out = thisAsset+logMsg
            print("\r"+out)
            write_log(out, log_file)
        if send_info_api:
            send_log_info(queue, thisAsset, {"FUNC":"LOG","ORIGIN":"NET","ASS":thisAsset,"MSG":logMsg})
#            queue.put({"FUNC":"LOG","ORIGIN":"NET","ASS":thisAsset,"MSG":logMsg})
        listFeaturesLive[sc],listParSarStruct[sc],listEM[sc] = \
            init_features_live(config, tradeInfoLive)
        init[sc] = True

    listFeaturesLive[sc],listParSarStruct[sc],listEM[sc] = get_features_live\
        (config,tradeInfoLive,listFeaturesLive[sc],listParSarStruct[sc],listEM[sc])
    # check if variations can be calculated or have to wait
    #allFeats = np.append(allFeats,features,axis=1)
    listAllFeatsLive[sc] = np.append(listAllFeatsLive[sc],listFeaturesLive[sc],
                                     axis=1)
    
    if listAllFeatsLive[sc].shape[1]>nChannels:
        
        # Shift old samples forward and leave space for new one
        if not list_inv_out:
            list_X_i[sc][0,:-1,:] = list_X_i[sc][0,1:,:]
        else:
            list_X_i[sc][0,1:,:] = list_X_i[sc][0,:-1,:]
        variationLive = np.zeros((nFeatures,len(channels)))
        for r in range(len(channels)):
            variationLive[:,r] = listAllFeatsLive[sc][:,-1]-\
                listAllFeatsLive[sc][:,-(channels[r]+2)]
            variationLive[nonVarIdx,0] = listAllFeatsLive[sc]\
                [noVarFeats,-(channels[r]+2)]
            varNormed = np.minimum(np.maximum((variationLive.T-\
                means_in[channels[r],:])/stds_in[channels[r],:],-10),10)
            # copy new entry in last pos of X
            if not list_inv_out:
                list_X_i[sc][0,-1,r*nFeatures:(r+1)*nFeatures] = varNormed
            else:
                list_X_i[sc][0,0,r*nFeatures:(r+1)*nFeatures] = varNormed
            
        #delete old infos
        listAllFeatsLive[sc] = listAllFeatsLive[sc][:,-100:]
        listCountPos[sc]+=1
        
        if listFillingX[sc]:
            logMsg = " "+tradeInfoLive.DateTime.iloc[-1]+" "+\
                    netName+" Filling X sc "+str(sc)+\
                    " t "+str(listCountPos[sc])+" of "+str(seq_len-1)
            if verbose_RNN:
                out = thisAsset+logMsg
                print("\r"+out)
                write_log(out, log_file)
            if send_info_api:
                send_log_info(queue, thisAsset, {"FUNC":"LOG","ORIGIN":"NET","ASS":thisAsset,"MSG":logMsg})
#                queue.put({"FUNC":"LOG","ORIGIN":"NET","ASS":thisAsset,"MSG":logMsg})
                
            if listCountPos[sc]>=seq_len-1:
                logMsg = " "+tradeInfoLive.DateTime.iloc[-1]+" "+\
                        netName+" Filling X sc "+str(sc)+\
                        " done. Waiting for output..."
                if verbose_RNN:
                    out = thisAsset+logMsg
                    print("\r"+out)
                    write_log(out, log_file)
                if send_info_api:
                    send_log_info(queue, thisAsset, {"FUNC":"LOG","ORIGIN":"NET","ASS":thisAsset,"MSG":logMsg})
#                    queue.put({"FUNC":"LOG","ORIGIN":"NET","ASS":thisAsset,"MSG":logMsg})
                listFillingX[sc] = False
        else:
########################################### Predict ###########################            
#            if verbose_RNN:
#                out = tradeInfoLive.DateTime.iloc[-1]+" "+thisAsset+netName
#                print("\r"+out, sep=' ', end='', flush=True)
                #write_log(out, log_file)
            
            soft_tilde = model.predict(list_X_i[sc])
            # loop over t indexes
            for t_index in range(len(t_indexes)):
                
                # if only one t_index mode
                if t_indexes[t_index]<seq_len:
                    soft_tilde_t = soft_tilde[:,t_indexes[t_index],:]
                else:
                    # if MRC mode
                    for t in range(seq_len):
                        # get weight from AD
                        mc_idx = (np.floor(soft_tilde[0,t,0]*10)-lb_level)*thr_levels
                        if mc_idx>=0:
                            idx_AD = int(mc_idx+(np.floor(np.max(soft_tilde\
                                                    [0,t,1:3])*10)-lb_level))
                            w = AD[t, max(idx_AD,first_nonzero), 0]
                        else:
                            w = AD[t, first_nonzero, 0]# temp. First non-zero level is 6
                        #print(idx_AD)
    #                      print(AD.shape)
    #                      print(first_nonzero)
    #                   update t-th matrix row from lower row
                        if t<seq_len-1:
                            list_weights_matrix[sc][t,:] = list_weights_matrix\
                                [sc][t+1,:]
                            # update t matrix entry with weight
    #                       print(w)
                                    
                        list_weights_matrix[sc][t,t] = w
                            
                        # expand dimensions of one row to fit network output
                        weights = np.expand_dims(np.expand_dims(
                                list_weights_matrix[sc][0,:]/np.sum(
                                    list_weights_matrix[sc][0,:]),axis=0),axis=2)
                    soft_tilde_t = np.sum(weights*soft_tilde, axis=1)
                                
    ###########################################################################
                if 0:#test:
                    soft_tilde_t = np.ones(soft_tilde_t.shape)
    ################################# Send prediciton to trader ###############
                # get condition to 
                condition = soft_tilde_t[0,0]>=thr_mc
          
                # set countdown to enter market
                if condition:
                    list_time_to_entry[sc][t_index].append(0)#model.seq_len-t_indexes[t_index]-1
                    list_list_soft_tildes[sc][t_index].append(soft_tilde_t[0,:])
#                    if verbose_RNN:
#                        out = thisAsset+netName+" ToBe sent DateTime "+\
#                            tradeInfoLive.DateTime.iloc[-1]+\
#                            " P_mc "+str(soft_tilde_t[0,0])+\
#                            " P_md "+str(np.max(soft_tilde_t[0,1:3]))
#                        print("\r"+out)
#                        write_log(out, log_file)
                else:
                    Bi = int(np.round(tradeInfoLive.SymbolBid.iloc[-1]*100000))/100000
                    Ai = int(np.round(tradeInfoLive.SymbolAsk.iloc[-1]*100000))/100000
                    logMsg = netName+\
                        tradeInfoLive.DateTime.iloc[-1]+\
                        " Bi "+str(Bi)+" Ai "+str(Ai)+\
                        " P_mc "+str(soft_tilde_t[0,0])+\
                        " P_md "+str(np.max(soft_tilde_t[0,1:3]))
                    if verbose_RNN:
                        out = thisAsset+logMsg
                        print("\r"+out)
                        write_log(out, log_file)
#                    if send_info_api and log_thu_control:
#                        queue.put({"FUNC":"LOG","ORIGIN":"NET","ASS":thisAsset,"MSG":logMsg})
                    #print("Prediction in market change")
                    # Snd prediction to trading robot
                if len(list_time_to_entry[sc][t_index])>0 and \
                    list_time_to_entry[sc][t_index][0]==0:
                    count_t += 1 
                    e_spread = (tradeInfoLive.SymbolAsk.iloc[-1]-\
                                tradeInfoLive.SymbolBid.iloc[-1])/\
                                tradeInfoLive.SymbolAsk.iloc[-1]
    #                tradeManager([list_list_soft_tildes[sc][0], e_spread])
                    output.append([list_list_soft_tildes[sc][t_index][0], \
                                   e_spread, tradeInfoLive.DateTime.iloc[-1],\
                                   tradeInfoLive.SymbolBid.iloc[-1], \
                                   tradeInfoLive.SymbolAsk.iloc[-1], \
                                   nEventsPerStat, netName, t_indexes[t_index]])
                    
                    logMsg = netName+" Sent DateTime "+\
                              tradeInfoLive.DateTime.iloc[-1]+\
                              " P_mc "+str(list_list_soft_tildes\
                                           [sc][t_index][0][0])+\
                              " P_md "+str(np.max(list_list_soft_tildes\
                                                  [sc][t_index][0][1:3]))
                    if verbose_RNN:
                        out = thisAsset+logMsg
                        print(out)
                        write_log(out, log_file)
#                    if send_info_api:
#                        queue.put({"FUNC":"LOG","ORIGIN":"NET","ASS":thisAsset,"MSG":logMsg})
                    list_time_to_entry[sc][t_index] = list_time_to_entry[sc]\
                        [t_index][1:]
                    list_list_soft_tildes[sc][t_index] = list_list_soft_tildes\
                        [sc][t_index][1:]
                    
                list_time_to_entry[sc][t_index] = [list_time_to_entry[sc]\
                                   [t_index][i]-1\
                                   for i in range(len(list_time_to_entry[sc]\
                                   [t_index]))]
    ###########################################################################
                            
    ################################################# Evaluation ##############
                
                
                prob_mc = np.array([soft_tilde_t[0,0]])
                            
                if prob_mc>=thr_mc:
                    #Y_tilde_idx = np.argmax(soft_tilde_t[0,3:])#
                    Y_tilde_check = np.array([(-1)**(1-np.argmax(soft_tilde_t[0,1:3]))]).astype(int)
                else:
                    pass
                    #Y_tilde_idx = int((n_bits_mg-1)/2) # zero index
                    Y_tilde_check = np.array([0])
#                Y_tilde = np.array([Y_tilde_idx-(n_bits_mg-1)/2]).astype(int)
                Y_tilde = Y_tilde_check
#                if np.sign(Y_tilde)!=np.sign(Y_tilde_check):
#                    out = "WARNING! Sign Y_tilde!=sign Y_tilde_check"
#                    print("\r"+out)
#                    write_log(out, log_file)
#                    print("Y_tilde")
#                    print(Y_tilde)
#                    print("Y_tilde_check")
#                    print(Y_tilde_check)
#                    print(soft_tilde_t[0,:])
#                    a=p
                prob_md = np.array([np.max([soft_tilde_t[0,1],soft_tilde_t[0,2]])])
                prob_mg = soft_tilde_t[0:,np.argmax(soft_tilde_t[0,3:])]
                
                # Check performance. Evaluate prediction
                list_Ylive[sc][t_index] = np.append(list_Ylive[sc][t_index],Y_tilde,axis=0)
                list_Pmc_live[sc][t_index] = np.append(list_Pmc_live[sc]\
                                                       [t_index],prob_mc,axis=0)
                list_Pmd_live[sc][t_index] = np.append(list_Pmd_live[sc]\
                                                       [t_index],prob_md,axis=0)
                list_Pmg_live[sc][t_index] = np.append(list_Pmg_live[sc]\
                                                       [t_index],prob_mg,axis=0)
                
                # wait for output to come
                if listCountPos[sc]>nChannels+seq_len-1:#t_t=2:listCountPos[sc]>nChannels+model.seq_len-1
                    if test or not modular:
                        std_out = stds_out[0,lookAheadIndex]
                    else:
                        std_out = stds_out[lookAheadIndex]
                    Output_i=(tradeInfoLive.SymbolBid.iloc[-2]-EOF.SymbolBid.iloc[c]
                                 )/std_out
                    
                    countOut+=1
                                
                    Y = (np.minimum(np.maximum(np.sign(Output_i)*np.round(
                            abs(Output_i)*outputGain),-(n_bits_mg
                               -1)/2),(n_bits_mg-1)/2)).astype(int)
                    look_back_index = -(min(nChannels+1,list_Ylive[sc][t_index].shape[0]))#model.seq_len+3#+2#-nChannels-t_indexes[t_index]-1
                    #print("look_back_index")
                    #print(look_back_index)
                    # compare prediction with actual output 
                    pred = list_Ylive[sc][t_index][look_back_index]
                    p_mc = list_Pmc_live[sc][t_index][look_back_index]
                    p_md = list_Pmd_live[sc][t_index][look_back_index]
                    p_mg = list_Pmg_live[sc][t_index][look_back_index]
                    # delete older entries if real time test is running
                    #if 1:#not simulate:
                    list_Ylive[sc][t_index] = list_Ylive[sc][t_index]\
                        [look_back_index:]
                    list_Pmc_live[sc][t_index] = list_Pmc_live[sc][t_index]\
                        [look_back_index:]
                    list_Pmd_live[sc][t_index] = list_Pmd_live[sc][t_index]\
                        [look_back_index:]
                    list_Pmg_live[sc][t_index] = list_Pmg_live[sc][t_index]\
                        [look_back_index:]
                    countOuts[t_index][c]+=1
                    
                    # if prediction is different than 0, evaluate
                    if p_mc>thr_mc:#pred!=0:
                        
                        # entry ask and bid
                        Ai = int(np.round(EOF.SymbolAsk.iloc[c]*100000))/100000
                        Bi = int(np.round(EOF.SymbolBid.iloc[c]*100000))/100000
                        # exit ask and bid
                        Ao = int(np.round(tradeInfoLive.SymbolAsk.iloc[-2]*100000))/100000
                        Bo = int(np.round(tradeInfoLive.SymbolBid.iloc[-2]*100000))/100000
                        # long position prediction
                        #print(pred)
                        if pred>0:
                            GROI = 100*(Ao-Ai)/Ai
                            ROI = 100*(Bo-Ai)/Ai
                            
                        else:
                            GROI = 100*(Bi-Bo)/Ao
                            ROI = 100*(Bi-Ao)/Ao
                            
                        spread = GROI-ROI
                        newEntry = {}
                        # update new entry and save
                        newEntry['Asset'] = thisAsset
                        newEntry["Entry Time"] = EOF.DateTime.iloc[c]
                        newEntry["Exit Time"] = tradeInfoLive.DateTime.iloc[-2]
                        newEntry["GROI"] = GROI
                        newEntry["Spread"] = spread
                        newEntry["ROI"] = ROI
                        newEntry["Bet"] = int(pred)
                        newEntry["Outcome"] = int(Y)
                        newEntry["Diff"] = int(np.abs(np.sign(pred)-np.sign(Y)))
                        newEntry["P_mc"] = p_mc
                        newEntry["P_md"] = p_md
                        newEntry["P_mg"] = p_mg
                        newEntry['Bi'] = Bi
                        newEntry['Ai'] = Ai
                        newEntry['Bo'] = Bo
                        newEntry['Ao'] = Ao
                        
                        
                        #resultInfo = pd.DataFrame(columns = columnsResultInfo)
                        resultInfo = pd.DataFrame(newEntry,index=[0])[pd.DataFrame(
                                     columns = columnsResultInfo).columns.tolist()]
                        #if p_mc>.5:
                        resultInfo.to_csv(results_network[t_index]+results_file[t_index],mode="a",
                                              header=False,index=False,sep='\t',
                                              float_format='%.5f')
                        # print entry
                        out = netName+resultInfo.to_string(index=False,\
                                                               header=False)
                        if verbose_RNN:
                            print("\r"+out)
                            write_log(out, log_file)
                        if send_info_api:
                            send_log_info(queue, thisAsset, {"FUNC":"LOG","ORIGIN":"NET","ASS":thisAsset,"MSG":out})
#                            queue.put({"FUNC":"LOG","ORIGIN":"NET","ASS":thisAsset,"MSG":out})
                    # end of if pred!=0:
                # end of if listCountPos[sc]>nChannels+model.seq_len+t_index-1:
            # end of for t_index in t_indexes:
        # end of if listFillingX[sc]:/else:
        countIndex+=1
    # end of else if fillingX:
    EOF.iloc[c] = tradeInfoLive.iloc[-1]
    
    return output

def flush_asset(lists, ass_idx, bid):
    """ Flush asset """
    
    phase_shifts = lists['phase_shifts']
    nChans = lists['nChans']
    list_t_indexs = lists['list_t_indexs']
    lBs = lists['lBs']
    nExSs = lists['nExSs']
    mWs = lists['mWs']
    list_n_feats = lists['list_n_feats']
    nNets = len(phase_shifts)
    average_over = np.array([0.1, 0.5, 1, 5, 10, 50, 100])
    lbd_shape = average_over.shape
    
    lists['inits'][ass_idx] = [[False for ps in range(phase_shifts[nn])] \
         for nn in range(nNets)]
    lists['listFillingXs'][ass_idx] = [[True for ps in range(phase_shifts[nn])] \
         for nn in range(nNets)]
    lists['listCountPoss'][ass_idx] = [[0 for ps in range(phase_shifts[nn])] \
         for nn in range(nNets)]
    lists['countOutss'][ass_idx] = [[np.zeros((int(nChans[nn]*\
        phase_shifts[nn]))).astype(int) for t in list_t_indexs[nn]] 
        for nn in range(nNets)]
    lists['EOFs'][ass_idx] = [pd.DataFrame(columns=['DateTime','SymbolBid','SymbolAsk'], 
        index=range(int(nChans[nn]*phase_shifts[nn]))) 
        for nn in range(nNets)]
    lists['list_list_X_i'][ass_idx] = [[np.zeros((1, int((lBs[nn]-nExSs[nn])/\
         mWs[nn]+1), list_n_feats[nn])) for ps in range(phase_shifts[nn])] 
         for nn in range(nNets)]
    lists['list_listAllFeatsLive'][ass_idx] = [[np.zeros((list_n_feats[nn],0)) \
         for ps in range(phase_shifts[nn])] for nn in range(nNets)]
    lists['list_listFeaturesLive'][ass_idx] = [[None for ps in range(phase_shifts[nn])] \
         for nn in range(nNets)]
    lists['list_listParSarStruct'][ass_idx] = [[None for ps in range(phase_shifts[nn])] \
         for nn in range(nNets)]
    lists['list_listEM'][ass_idx] = [[np.zeros((lbd_shape))+bid \
         for ps in range(phase_shifts[nn])] for nn in range(nNets)]
    lists['list_list_Ylive'][ass_idx] = [[[np.zeros((0,)) for t in list_t_indexs[nn]] \
         for ps in range(phase_shifts[nn])] for nn in range(nNets)]
    lists['list_list_Pmc_live'][ass_idx] = [[[np.zeros((0,)) for t in list_t_indexs[nn]] \
         for ps in range(phase_shifts[nn])] for nn in range(nNets)]
    lists['list_list_Pmd_live'][ass_idx] = [[[np.zeros((0,)) for t in list_t_indexs[nn]] \
         for ps in range(phase_shifts[nn])] for nn in range(nNets)]
    lists['list_list_Pmg_live'][ass_idx] = [[[np.zeros((0,)) for t in list_t_indexs[nn]] \
         for ps in range(phase_shifts[nn])] for nn in range(nNets)]
    # init condition vector to open market
    #condition = np.zeros((model.seq_len))
    lists['list_list_time_to_entry'][ass_idx] = [[[[]  for t in list_t_indexs[nn]] \
         for ps in range(phase_shifts[nn])] for nn in range(nNets)] # list tracking times to entry the market
    lists['list_list_list_soft_tildes'][ass_idx] = [[[[]  for t in list_t_indexs[nn]] \
         for ps in range(phase_shifts[nn])] for nn in range(nNets)]
    # upper diagonal matrix containing latest weight values
    lists['list_list_weights_matrix'][ass_idx] = [[np.zeros((int((lBs[nn]-nExSs[nn])/\
         mWs[nn]+1),int((lBs[nn]-nExSs[nn])/mWs[nn]+1))) for ps in range(phase_shifts[nn])] \
         for nn in range(nNets)]
    print("Flushed")
    return lists

def dispatch(lists, list_models, tradeInfo, AllAssets, ass_id, ass_idx, first_info_nets_fetched, log_file, queue):
    #AllAssets, assets, running_assets, nCxAxN, buffSizes, simulation, delays, PA, verbose
    '''
    inputs: AllAssets
            assets
            nCxAss: matrix containning number of channels per asset and per network
            buffSizes: matrix containning max size of buffer for each channel
    '''

    #print("Dispatcher running and ready.")

    thisAsset = AllAssets[str(ass_id)]
    nNets = len(lists['phase_shifts'])
    
    outputs = []
    new_outputs = 0
    # add trade info to all networks of this asset
    for nn in range(nNets):
        # shift each strategy by one buffer
        if first_info_nets_fetched[ass_idx][nn]:
            # loop over buffers to add new info
            for ch in range(int(lists['nCxAxN'][ass_idx,nn])):
                
                # add info to buffer
                if lists['buffersCounter'][ass_idx][nn][ch]>=0:
                    
                    lists['buffers'][ass_idx][nn][ch] = lists['buffers'][ass_idx][nn][ch].append(
                            tradeInfo,ignore_index=True)
                lists['buffersCounter'][ass_idx][nn][ch] += tradeInfo.shape[0]
                
                # check if buffer reached max
                if lists['buffers'][ass_idx][nn][ch].shape[0]==lists['buffSizes'][ass_idx,nn]:
                    
                    ### Dispatch buffer to corresponding network ###
                    output = runRNNliveFun(lists['buffers'][ass_idx][nn][ch],
                                           lists['listFillingXs'][ass_idx][nn],
                                           lists['inits'][ass_idx][nn],
                                           lists['list_listFeaturesLive'][ass_idx][nn],
                                           lists['list_listParSarStruct'][ass_idx][nn],
                                           lists['list_listEM'][ass_idx][nn],
                                           lists['list_listAllFeatsLive'][ass_idx][nn],
                                           lists['list_list_X_i'][ass_idx][nn],
                                           lists['list_means_in'][ass_idx][nn],
                                           lists['phase_shifts'][nn],
                                           lists['list_stds_in'][ass_idx][nn],
                                           lists['list_stds_out'][ass_idx][nn],
                                           lists['ADs'][nn],
                                           thisAsset,
                                           lists['netNames'][nn],
                                           lists['listCountPoss'][ass_idx][nn],
                                           lists['list_list_weights_matrix'][ass_idx][nn],
                                           lists['list_list_time_to_entry'][ass_idx][nn],
                                           lists['list_list_list_soft_tildes'][ass_idx][nn],
                                           lists['list_list_Ylive'][ass_idx][nn],
                                           lists['list_list_Pmc_live'][ass_idx][nn],
                                           lists['list_list_Pmd_live'][ass_idx][nn],
                                           lists['list_list_Pmg_live'][ass_idx][nn],
                                           lists['EOFs'][ass_idx][nn],
                                           lists['countOutss'][ass_idx][nn],
                                           lists['list_t_indexs'][nn],
                                           ch, 
                                           lists['resultsDir'][nn],
                                           lists['results_files'][nn], 
                                           list_models[nn],
                                           lists['list_unique_configs'][nn], 
                                           log_file, 
                                           lists['list_nonVarIdx'][nn],
                                           lists['list_inv_out'][nn],
                                           queue)
                    if len(output)>0:
                        outputs.append([output,nn])
                        new_outputs = 1
                    
                    #reset buffer
                    lists['buffers'][ass_idx][nn][ch] = pd.DataFrame()
                    lists['buffersCounter'][ass_idx][nn][ch] = 0
                    # update file extension
                    lists['bufferExt'][ass_idx][nn][ch] = (lists['bufferExt'][ass_idx][nn][ch]+1)%2
                                
                elif lists['buffers'][ass_idx][nn][ch].shape[0]>lists['buffSizes'][ass_idx,nn]:
                    print(thisAsset+" buffer size "+str(lists['buffers'][ass_idx][nn][ch].shape[0]))
                    #print("Error. Buffer cannot be greater than max buff size")
                    raise ValueError("Buffer cannot be greater than max buff size")
        else:
            print(thisAsset+" nn "+str(nn)+" not inited yet")
#    if len(outputs)>1:
#        print("WARNING! Outputs length="+str(len(outputs))+
#              ". No support for multiple outputs at the same time yet.") 
    return outputs, new_outputs

def renew_mt5_dir(AllAssets, running_assets):
    """ Renew MT5 directory for synch between py and trader """
    # init list with files index for each running asset to zero
    directory_MT5_IO = LC.directory_MT5_IO
    directory_MT5_log = LC.directory_MT5_log
    file_ids = [0 for _ in range(len(running_assets))]
    log_ids = ['' for _ in range(len(running_assets))]
    nonsynched_assets = [i for i in running_assets]
    waitings = [True for _ in running_assets]
    buffers = [[] for _ in running_assets]
    its = 0
    run = True
    # loop over assets
    while len(nonsynched_assets)>0:
        
        for ass_idx, ass_id in enumerate(running_assets):
            if ass_id in nonsynched_assets:
                thisAsset = AllAssets[str(ass_id)]
                logMsg = " Synching buffer with broker"
                print(thisAsset+logMsg)
#                queue.put({"FUNC":"LOG","ORIGIN":"NET","ASS":thisAsset,"MSG":logMsg})
                # this asset directory
                directory_MT5_IO_ass = directory_MT5_IO+thisAsset+"/"
                directory_MT5_log_ass = directory_MT5_log+thisAsset+"/"
                # check if IO asset directory exists
                if os.path.exists(directory_MT5_IO_ass):
                    # list of all files in MT5 directory
                    listAllDir = sorted(os.listdir(directory_MT5_IO_ass))
                    dates_mod_files = [os.path.getmtime(directory_MT5_IO_ass+file) for file in listAllDir]
                    indexes_ordered = sorted(range(len(dates_mod_files)), key=lambda k: dates_mod_files[k])
                    #for file in listAllDir:
                    for idx in indexes_ordered:
                        file = listAllDir[idx]
#                        print("Reading file "+file)
                        try:
                            # read and delete
                            buffer = pd.read_csv(directory_MT5_IO_ass+file)
#                            print(file+" read")
                            buffers[ass_idx].append(buffer)
                            # try to delete file
                            os.remove(directory_MT5_IO_ass+file)
                        except:
                            #print(thisAsset+" Warning. Error when deleting "+file)
                            # check file belongs to Symbols stream
                            m = re.search('^'+thisAsset+'_\d+'+'.txt$',file)
                            # if does, update its file id
                            if m!=None:
                                fileid = re.search('\d+',m.group()).group()
                                file_ids[ass_idx] = int(fileid)
                                # delete asset from nonsynched assets list
                                if ass_id in nonsynched_assets:
                                    nonsynched_assets.remove(ass_id)
                                    its += 1
                                else:
                                    #while 1:
                                    print(thisAsset+" WARNING! ass_id "+str(ass_id)+
                                              " not in nonsynched_assets "+str(nonsynched_assets)+
                                              " iterations "+str(its)+" file_ids "+str(file_ids)+
                                              " running_assets "+str(running_assets)+
                                              " len(nonsynched_assets) "+str(len(nonsynched_assets)))
                                        #time.sleep(2)
                                print(thisAsset+" Buffer synched with broker: Starting file "+file)
                                waitings[ass_idx] = False
                                
                else:
                    #create directory
                    os.makedirs(directory_MT5_IO_ass)
                    print(directory_MT5_IO_ass+" Directiory created")
                    
            # create log directory
            if not os.path.exists(directory_MT5_log_ass):
                os.makedirs(directory_MT5_log_ass)
                print(directory_MT5_log_ass+" Directiory created")
            else:
                pass
    #            listAllDir = sorted(os.listdir(directory_MT5_log_ass))
    #            for file in listAllDir:
    #                try:
    #                    # try to delete file
    #                    os.remove(directory_MT5_log_ass+file)
    #                except:
    #                    print(thisAsset+" Warning. Error when deleting "+file)
    #                    # check file belongs to log files
    #                    m = re.search('^'+thisAsset+'_\d+'+'.log$',file)
    #                    # if does, update its file id
    #                    if m!=None:
    #                        logid = re.search('\d+',m.group()).group()
    #                        log_ids[ass_idx] = logid
        if sum(waitings)>0:
            for w in range(len(waitings)):
                if waitings[w]:
                    thisAsset = AllAssets[str(running_assets[w])]
                    io_ass_dir = LC.io_live_dir+thisAsset+"/"
                    if os.path.exists(io_ass_dir+'SD'):
                        print(thisAsset+" Shutting down")
                        os.remove(io_ass_dir+'SD')
                        run = False
                        waitings = [False for _ in running_assets]
                        nonsynched_assets = []
            time.sleep(2)
            
    return file_ids, log_ids, run, buffers

def renew_directories(AllAssets, running_assets, clean_mt5):
    """ Renew MT5 directories """
    for ass_id in running_assets:
        thisAsset = AllAssets[str(ass_id)]
        
        if clean_mt5:
            directory_MT5_IO_ass = LC.directory_MT5_IO+thisAsset+"/"
            
            if os.path.exists(directory_MT5_IO_ass):
                # list of all files in MT5 directory
                listAllDir = sorted(os.listdir(directory_MT5_IO_ass))
                #for file in listAllDir:
                for file in listAllDir:
#                        print("Reading file "+file)
                    try:
                        # delte file
                        os.remove(directory_MT5_IO_ass+file)
                    except:
                        pass
    
#            if os.path.exists(directory_MT5_ass):
#                try:
#                    shutil.rmtree(directory_MT5_ass)
#                    
#                    time.sleep(1)
#                except:
#                    print(thisAsset+" Warning. Error when renewing MT5 directory")
#                    # TODO: Synch fetcher with current file if MT5 is recording
#                
            #try:
        io_ass_dir = LC.io_live_dir+thisAsset+"/"
        if os.path.exists(io_ass_dir):
            try:
                shutil.rmtree(io_ass_dir)
                print(io_ass_dir+" Directiory renewed")
                time.sleep(1)
            except:
                print(thisAsset+" Warning. Error when renewing IO directory") 
        
        
            #except:
                #print(directory_MT5_ass+" Warning. Error when creating MT5 directory")
        
        if not os.path.exists(io_ass_dir):
            os.makedirs(io_ass_dir)
            print(io_ass_dir+" Directiory created")


def save_snapshot(asset, lists, trader, command='SD'):
    """ Save session snapshot for later resume """
    snapshot_dir_ass = LC.snapshot_live_dir+asset+'/'
    # save network image
    #pickle.dump( lists, open( backup_dir_ass+"netImage.p", "wb" ))
    
    if command=='SD':
        trader_dict = {}
    else:
        trader_dict = vars(trader)
        del trader_dict['queue']
        del trader_dict['queue_prior']
    
    pickle.dump( {'trader':trader_dict,
                  'network':lists}, open( snapshot_dir_ass+"snapshot.p", "wb" ))
    print(asset+" SNAPSHOT SAVED")
    print("Snapshot saved in "+snapshot_dir_ass)
    # save trader image
#    list_trader_attrs = [a for a in dir(trader) if not a.startswith('__') and not callable(getattr(trader, a))]

def load_snapshot():
    """  """
    pass

def send_log_info(queue, thisAsset, dict_info, bias=199, only_queue=False):
    """  """
    if log_thu_control:
        queue.put(dict_info)
    elif not only_queue:
        log_file = LC.local_log_comm+thisAsset+'/'+str(np.random.randint(99)+bias).zfill(5)
        string = ""
        for key in dict_info:
            string += key+","+str(dict_info[key])+","
        # delete last comma
        string = string[:-1]
        write_log(string, log_file)
        
def init_vi(assets):
    """ Init Volatility Index structure """
    vi_struct = {}
    
    w1 = 1-1/1
#    w10 = 1-1/10
#    w20 = 1-1/20
#    w100 = 1-1/100
#    w1000 = 1-1/1000
#    w10000 = 1-1/10000
    ws = [w1]
    vi_struct['ws'] = ws
    vi_struct['window_size'] = 5000
    vi_struct['emas_volat'] = [[-1 for _ in ws] for _ in assets]#[[-1 for _ in assets] for _ in ws]
    vi_struct['means_volat'] = [0 for _ in ws]
    vi_struct['events_per_ass_counter'] = [-1 for _ in assets]
    vi_struct['VIs'] = [0 for _ in ws]
    vi_struct['max_volat'] = [0.0 for _ in assets]
    vi_struct['min_volat'] = [9999999999999999.99 for _ in assets] # infty
    vi_struct['track_idx'] = [0 for i in assets]
    vi_struct['track_last_asks'] = [np.zeros((vi_struct['window_size'])) for i in assets]
    
    return vi_struct

def update_vi(vi_struct, ass_idx, asks):
    """ Get Volatility Index """
    track_idx = vi_struct['track_idx']
    track_last_asks = vi_struct['track_last_asks']
    window_size = vi_struct['window_size']
    max_volat = vi_struct['max_volat']
    min_volat = vi_struct['min_volat']
    emas_volat = vi_struct['emas_volat']
    
    n_new_samps = len(asks)
    
    prev_asks = track_last_asks[ass_idx][track_idx[ass_idx]:track_idx[ass_idx]+n_new_samps]
    max_prev_asks = max(prev_asks)
#    print("max_prev_asks")
#    print(max_prev_asks)
    min_prev_asks = min(prev_asks)
#    print("min_prev_asks")
#    print(min_prev_asks)
    
    track_last_asks[ass_idx][track_idx[ass_idx]:track_idx[ass_idx]+n_new_samps] = asks
    
    if track_last_asks[ass_idx][-1]>0:
        max_buff = max(asks)
#        print("max_buff")
#        print(max_buff)
        min_buff = min(asks)
#        print("min_buff")
#        print(min_buff)
        
        
        if max_buff>max_volat[ass_idx]:
#            print("ask>max_volat")
            max_volat[ass_idx] = max_buff
        if max_prev_asks==max_volat[ass_idx]:
#            print("prev_ask==max_volat")
            max_volat[ass_idx] = np.max(track_last_asks[ass_idx])
        if min_buff<min_volat[ass_idx]:
#            print("ask<min_volat")
            min_volat[ass_idx] = min_buff
        if min_prev_asks==min_volat[ass_idx]:
#            print("prev_ask==min_volat")
            min_volat[ass_idx] = np.min(track_last_asks[ass_idx])
#        max_volat = np.max(track_last_asks[ass_id])
#        min_volat = np.min(track_last_asks[ass_id])
        vi_struct['max_volat'] = max_volat
        vi_struct['min_volat'] = min_volat
        #window_asks = np.array(track_last_asks[ass_id])
#        window_asks = track_last_asks[ass_id]
        volat = max_volat[ass_idx]/min_volat[ass_idx]-1
        # update max volume
#                if volat>max_volats[ass_id]:
        for i in range(len(emas_volat[ass_idx])):
            if emas_volat[ass_idx][i] == -1:
                emas_volat[ass_idx][i] = volat
            # update volatility tracking
            emas_volat[ass_idx][i] = volat#ws[i]*emas_volat[ass_idx][i]+(1-ws[i])*volat
        
        VIs = [100*ema for ema in emas_volat[ass_idx]]
#        print("\r"+" VI1 {0:.4f}".
#              format(VIs[0]))#, sep=' ', end='', flush=True
        vi_struct['VIs'] = VIs
    
    track_idx[ass_idx] = (track_idx[ass_idx]+n_new_samps) % window_size
#    print("track_idx[ass_idx]")
#    print(track_idx[ass_idx])
    #events_per_ass_counter[ass_idx] += 1
    
    vi_struct['emas_volat'] = emas_volat
    
    #struct['events_per_ass_counter'] = events_per_ass_counter
    vi_struct['track_idx'] = track_idx
    
    return vi_struct

def fetch(lists, list_models, trader, directory_MT5, AllAssets, 
          running_assets, log_file, results, queue, queue_prior):
    """ Fetch info coming from MT5 """
    print("Fetcher lauched")
    
    #nAssets = len(running_assets)
    # renew MT5 directories
    fileExt, list_log_ids, run, buffers = renew_mt5_dir(AllAssets, running_assets)
    
    
    #fileExt = [0 for ass in range(nAssets)]
    
    first_info_fetched = False    
    nNets = len(lists['phase_shifts'])
    first_info_nets_fetched = [[False for _ in range(nNets)] for _ in running_assets]
    nn_counter = [0 for _ in running_assets]
    #nMaxFilesInDir = 0
    tic = time.time()
    delayed_stop_run = False
    count10s = [0 for _ in running_assets]
    # init VI structure
    if margin_adapt:
        vi_struct = init_vi(running_assets)
    else:
        vi_struct = {}
    while run:
#        tic = time.time()
        for ass_idx, ass_id in enumerate(running_assets):
            count10s[ass_idx] = np.mod(count10s[ass_idx]+1, 1000)
            
            thisAsset = AllAssets[str(ass_id)]
            
            #test_multiprocessing(ass_idx)
            
#            disp = Process(target=test_multiprocessing, args=[ass_idx])
#            disp.start()
            
            directory_MT5_ass = directory_MT5+thisAsset+"/"
            # Fetching buffers
            fileID = thisAsset+deli+str(fileExt[ass_idx]).zfill(2)+extension
            success = 0
            io_ass_dir = LC.io_live_dir+thisAsset+"/"
            if len(buffers[ass_idx])==0:
                try: # no info in laying behind
                    buffer = pd.read_csv(directory_MT5_ass+fileID)#
                    #print(thisAsset+" new buffer received")
                    os.remove(directory_MT5_ass+fileID)
                    from_file = True
                    #nFilesDir = len(os.listdir(directory_MT5_ass))
                    #start_timer(ass_idx)
                    if not first_info_fetched:
                        logMsg = " First info fetched"
                        out = thisAsset+logMsg
                        print(out)
                        send_log_info(queue, thisAsset, {"FUNC":"LOG","ORIGIN":"MONITORING","ASS":thisAsset,"MSG":logMsg})
    #                    if log_thu_control:
    #                        queue.put({"FUNC":"LOG","ORIGIN":"MONITORING","ASS":thisAsset,"MSG":logMsg})
    #                    else:
    #                        log_file = LC.local_log_comm+thisAsset+'/'+str(np.random.randint(99)+99).zfill(5)
    #                        write_log("FUNC,"+"LOG,"+"ORIGIN,"+"MONITORING,"+"ASS,"+thisAsset+",MSG,"+logMsg, log_file)
                        #print(buffer)
                        first_info_fetched = True
                    if not first_info_nets_fetched[ass_idx][nn_counter[ass_idx]]:
                        first_info_nets_fetched[ass_idx][nn_counter[ass_idx]] = True
                        print(thisAsset+" nn "+str(nn_counter[ass_idx])+" set true")
                        nn_counter[ass_idx] = np.mod(nn_counter[ass_idx]+1, nNets)
    #                print(fileID+" size: "+str(buffer.shape[0]))
    #                print(buffer)
                    if buffer.shape[0]==10:
                        success = 1
                    else:
                        logMsg = " WARNING! Buffer size not 10. Discarded"
                        print(thisAsset+logMsg)
                        send_log_info(queue, thisAsset, {"FUNC":"LOG","ORIGIN":"MONITORING","ASS":thisAsset,"MSG":logMsg})
    #                    if log_thu_control:
    #                        queue.put({"FUNC":"LOG","ORIGIN":"MONITORING","ASS":thisAsset,"MSG":logMsg})
    #                    else:
    #                        log_file = LC.local_log_comm+thisAsset+'/'+str(np.random.randint(99)+99).zfill(5)
    #                        write_log("FUNC,"+"LOG,"+"ORIGIN,"+"MONITORING,"+"ASS,"+thisAsset+",MSG,"+logMsg, log_file)
                        
                        fileExt, _, _, _ = renew_mt5_dir(AllAssets, running_assets)
                        
                except (FileNotFoundError,PermissionError,OSError):
                    # reset coming from Broker
                    if os.path.exists(directory_MT5_ass+'0RESET'):
                        print("RESET from broker found.")
                        os.remove(directory_MT5_ass+'0RESET')
                        fileExt, _, _, _ = renew_mt5_dir(AllAssets, running_assets)
                    if os.path.exists(io_ass_dir+'PA'):
                        print(thisAsset+" PAUSED. Waiting for RE command...")
                        os.remove(io_ass_dir+'PA')
                        while not os.path.exists(io_ass_dir+'RE'):
                            time.sleep(np.random.randint(6)+5)
                        os.remove(io_ass_dir+'RE')
                    elif os.path.exists(io_ass_dir+'RE'):
                        print("WARNING! RESUME command found. Send first PAUSE command")
                        os.remove(io_ass_dir+'RE')
                    elif os.path.exists(io_ass_dir+'RESET'):
                        print("RESET command found.")
                        os.remove(io_ass_dir+'RESET')
                        lists = flush_asset(lists, ass_idx, 0.0)
                    time.sleep(.02)
            else:
                
                buffer = buffers[ass_idx][0]
                buffers[ass_idx].pop(0)
#                print("len(buffers[ass_idx])")
#                print(len(buffers[ass_idx]))
                from_file = False
                if not first_info_fetched:
                    logMsg = " First info fetched"
                    out = thisAsset+logMsg
                    print(out)
                    send_log_info(queue, thisAsset, {"FUNC":"LOG","ORIGIN":"MONITORING","ASS":thisAsset,"MSG":logMsg})
#                    if log_thu_control:
#                        queue.put({"FUNC":"LOG","ORIGIN":"MONITORING","ASS":thisAsset,"MSG":logMsg})
#                    else:
#                        log_file = LC.local_log_comm+thisAsset+'/'+str(np.random.randint(99)+99).zfill(5)
#                        write_log("FUNC,"+"LOG,"+"ORIGIN,"+"MONITORING,"+"ASS,"+thisAsset+",MSG,"+logMsg, log_file)
                    #print(buffer)
                    first_info_fetched = True
                if not first_info_nets_fetched[ass_idx][nn_counter[ass_idx]]:
                    first_info_nets_fetched[ass_idx][nn_counter[ass_idx]] = True
                    print(thisAsset+" nn "+str(nn_counter[ass_idx])+" set true")
                    nn_counter[ass_idx] = np.mod(nn_counter[ass_idx]+1, nNets)
#                print(fileID+" size: "+str(buffer.shape[0]))
#                print(buffer)
                if buffer.shape[0]==10:
                    success = 1
                else:
                    logMsg = " WARNING! Buffer size not 10. Discarded"
                    print(thisAsset+logMsg)
                    send_log_info(queue, thisAsset, {"FUNC":"LOG","ORIGIN":"MONITORING","ASS":thisAsset,"MSG":logMsg})
#                    if log_thu_control:
#                        queue.put({"FUNC":"LOG","ORIGIN":"MONITORING","ASS":thisAsset,"MSG":logMsg})
#                    else:
#                        log_file = LC.local_log_comm+thisAsset+'/'+str(np.random.randint(99)+99).zfill(5)
#                        write_log("FUNC,"+"LOG,"+"ORIGIN,"+"MONITORING,"+"ASS,"+thisAsset+",MSG,"+logMsg, log_file)
                    
                    fileExt, _, _, buffers = renew_mt5_dir(AllAssets, running_assets)
            
            # check shut down command
#            if count10s[ass_idx]==56:
#                print("CHECKING SD")
            if count10s[ass_idx]==56 and os.path.exists(io_ass_dir+'SD'):
                logMsg = " Shutting down"
                out = thisAsset+logMsg
                print(out)
                send_log_info(queue, thisAsset, {"FUNC":"LOG","ORIGIN":"MONITORING","ASS":thisAsset,"MSG":logMsg})
#                if log_thu_control:
#                    queue.put({"FUNC":"LOG","ORIGIN":"MONITORING","ASS":thisAsset,"MSG":logMsg})
#                else:
#                    log_file = LC.local_log_comm+thisAsset+'/'+str(np.random.randint(99)+99).zfill(5)
#                    write_log("FUNC,"+"LOG,"+"ORIGIN,"+"MONITORING,"+"ASS,"+thisAsset+",MSG,"+logMsg, log_file)
                os.remove(io_ass_dir+'SD')
                for p in range(len(trader.pos_idx2map_ass_str)):
                    if trader.pos_idx2map_ass_str[p][-1]>=0:
                        trader.send_close_command(thisAsset, p)
                
                if trader.n_pos_currently_open>0:
                    delayed_stop_run = True
                else:
                    send_log_info(queue, thisAsset, {"FUNC":"SD"})
                    send_log_info(queue_prior, thisAsset, {"FUNC":"SD"}, only_queue=True)
#                    if log_thu_control:
#                        queue.put({"FUNC":"SD"})
#                        queue_prior.put({"FUNC":"SD"})
                    
                    run = False
                    # close session
#                    if send_info_api:
#                        close_session(trader.session_json)
                    # save network lists dictionary
                    save_snapshot(thisAsset, lists, trader, command='SD')
                time.sleep(5*np.random.rand(1)+1)
                
            
            if count10s[ass_idx]==6 and send_info_api and os.path.exists(io_ass_dir+'PARAM'):
                print("PARAM found")
                # check first for local info
                with open(io_ass_dir+'PARAM', 'r') as f:
                    config_name = f.read()
                    f.close()
                try:
                    os.remove(io_ass_dir+'PARAM')
                except:
                    print("WARNING! Error while deleting PARAM. Continuing")
                    
                if config_name=='':
                    pass
                else:
                    # update from local
                    config = retrieve_config(config_name)
                    print(thisAsset)
                    trader.update_parameters(config, thisAsset)
#                    logMsg = " Parameters updated"
#                    queue.put({"FUNC":"LOG","ORIGIN":"MONITORING","ASS":thisAsset,"MSG":logMsg})
                
            # check hibernate command
            if count10s[ass_idx]==106 and os.path.exists(io_ass_dir+'HIBER'):
                logMsg = " HIBERNATING"
                out = thisAsset+logMsg
                print(out)
                send_log_info(queue, thisAsset, {"FUNC":"LOG","ORIGIN":"MONITORING","ASS":thisAsset,"MSG":logMsg})
#                if log_thu_control:
#                    queue.put({"FUNC":"LOG","ORIGIN":"MONITORING","ASS":thisAsset,"MSG":logMsg})
#                else:
#                    log_file = LC.local_log_comm+thisAsset+'/'+str(np.random.randint(99)+99).zfill(5)
#                    write_log("FUNC,"+"LOG,"+"ORIGIN,"+"MONITORING,""ASS,"+thisAsset+",MSG,"+logMsg, log_file)
                os.remove(io_ass_dir+'HIBER')
                send_log_info(queue, thisAsset, {"FUNC":"SD"})
                send_log_info(queue_prior, thisAsset, {"FUNC":"SD"}, only_queue=True)
#                if log_thu_control:
#                    queue.put({"FUNC":"SD"})
#                    queue_prior.put({"FUNC":"SD"})
#                else:
#                    log_file = LC.local_log_comm+thisAsset+'/'+str(np.random.randint(99)+99).zfill(5)
#                    write_log("FUNC,"+"SD", log_file)
                # save network lists dictionary
                save_snapshot(thisAsset, lists, trader, command='HI')
                run = False
                
                
            # update file extension
            if success:
                if from_file:
                    fileExt[ass_idx] = (fileExt[ass_idx]+1)%nFiles
                # update list
                
                
                for s in range(len(trader.strategies)):
                    list_idx = trader.map_ass_idx2pos_idx[s][ass_id]
                    if list_idx>-1:
                        trader.update_symbols_tracking(list_idx, s, buffer)
                trader.count_events(ass_id, buffer.shape[0])
                # dispatch
                outputs, new_outputs = dispatch(lists, list_models, buffer, AllAssets, 
                                                ass_id, ass_idx, first_info_nets_fetched,
                                                log_file, queue)
                # update VIs
                if margin_adapt:
                    vi_struct = update_vi(vi_struct, ass_idx, list(buffer['SymbolAsk']))
                ################# Trader ##################
                if new_outputs and not trader.swap_pending:
                    
                    trader.check_new_inputs(outputs, thisAsset, results, vi_struct, 
                                            directory_MT5_ass=directory_MT5_ass)
            
            # check for closing
            flag_cl = 0
            # check if position closed
            if os.path.exists(directory_MT5_ass+flag_cl_name) or \
               os.path.exists(directory_MT5_ass+flag_ma_name):
#            if trader.is_opened_asset(ass_id) and \
#            (os.path.exists(directory_MT5_ass+flag_cl_name) or 
#             os.path.exists(directory_MT5_ass+flag_ma_name)):
                
                success = 0
                while not success:
                    try:
                        if os.path.exists(directory_MT5_ass+flag_cl_name):
                            fh = open(directory_MT5_ass+flag_cl_name,"r")
                        else:
                            fh = open(directory_MT5_ass+flag_ma_name,"r")
                        # read output
                        out = fh.read()
                        info_close = out[:-1]
                        
                        #print(info_close)
                        # close file
                        fh.close()
#                        print("out")
#                        print(out)
                        print("info_close")
                        print(info_close)
                        if trader.is_opened_asset(ass_id):
                            flag_cl = 1
                        else:
                            print("WARNING! CL/MA found but no position open. Skipped.")
                        if len(info_close)>1:
                            if os.path.exists(directory_MT5_ass+flag_cl_name):
                                os.remove(directory_MT5_ass+flag_cl_name)
                            else:
                                os.remove(directory_MT5_ass+flag_ma_name)
                            success = 1
                        else:
                            pass
                            #print("Error in reading file. Length "+str(len(info_close)))
                    except (FileNotFoundError,PermissionError,OSError):
                        pass
                
                
            # check for stoploss closing
            flag_sl = 0
            if trader.is_opened_asset(ass_id) and os.path.exists(directory_MT5_ass+flag_sl_name):
                success = 0
                while not success:
                    try:
                        #print(dirOr+flag_name)
                        fh = open(directory_MT5_ass+flag_sl_name,"r")
                        # read output
                        info_close = fh.read()[:-1]
                        
                        # close file
                        fh.close()
                        
                        flag_sl = 1
                        if len(info_close)>1:
                            os.remove(directory_MT5_ass+flag_sl_name)
                            success = 1
                        else:
                            out = "Error in reading file. Length "+str(len(info_close))
                            print(out)
                            write_log(out, log_file)
                    except (FileNotFoundError,PermissionError,OSError):
                        pass
                    
            # check for stoploss closing
            flag_tp = 0
            if trader.is_opened_asset(ass_id) and os.path.exists(directory_MT5_ass+flag_tp_name):
                success = 0
                while not success:
                    try:
                        #print(dirOr+flag_name)
                        fh = open(directory_MT5_ass+flag_tp_name,"r")
                        # read output
                        info_close = fh.read()[:-1]
                        
                        # close file
                        fh.close()
                        
                        flag_tp = 1
                        if len(info_close)>1:
                            os.remove(directory_MT5_ass+flag_tp_name)
                            success = 1
                        else:
                            out = "Error in reading file. Length "+str(len(info_close))
                            print(out)
                            write_log(out, log_file)
                    except (FileNotFoundError,PermissionError,OSError):
                        pass

            # check if asset has been banned from outside
            if os.path.exists(trader.ban_currencies_dir+trader.start_time+thisAsset):
                fh = open(trader.ban_currencies_dir+trader.start_time+thisAsset,"r")
                message = fh.read()
                fh.close()
                info = message.split(",")
                otherAsset = info[0]
                otherDirection = int(info[1])
                logMsg = " flag ban from found: "+message[:-1]
                out = thisAsset+logMsg
                print(out)
                trader.write_log(out)
                send_log_info(queue, thisAsset, {"FUNC":"LOG","ORIGIN":"TRADE","ASS":thisAsset,"MSG":logMsg})
#                if log_thu_control:
#                    queue.put({"FUNC":"LOG","ORIGIN":"TRADE","ASS":thisAsset,"MSG":logMsg})
#                else:
#                    log_file = LC.local_log_comm+thisAsset+'/'+str(np.random.randint(99)+99).zfill(5)
#                    write_log("FUNC,"+"LOG,"+"ORIGIN,"+"TRADE,"+"ASS,"+thisAsset+",MSG,"+logMsg, log_file)
                # for now, ban only if asset is opened AND they go in same direction
                if trader.is_opened_asset(ass_id):
                    thisDirection = trader.list_opened_positions\
                        [trader.map_ass_idx2pos_idx[ass_id]].direction
                    if trader.assets_same_direction(ass_id, thisAsset, thisDirection, 
                                  otherAsset, otherDirection):
                        lists = flush_asset(lists, ass_idx, 0.0)
                        logMsg = " flushed"
                        out = thisAsset+logMsg
                        print(out)
                        trader.write_log(out)
                        for p in range(len(trader.pos_idx2map_ass_str)):
                            if trader.pos_idx2map_ass_str[p][-1]==0:
                                trader.send_close_command(thisAsset, p)
                        send_log_info(queue, thisAsset, {"FUNC":"LOG","ORIGIN":"TRADE","ASS":thisAsset,"MSG":logMsg})
#                        if log_thu_control:
#                            queue.put({"FUNC":"LOG","ORIGIN":"TRADE","ASS":thisAsset,"MSG":logMsg})
#                        else:
#                            log_file = LC.local_log_comm+thisAsset+'/'+str(np.random.randint(99)+99).zfill(5)
#                            write_log("FUNC,"+"LOG,"+"ORIGIN,"+"TRADE,"+"ASS,"+thisAsset+",MSG,"+logMsg, log_file)
                    else:
                        logMsg = " NOT flushed due to different directions"
                        out = thisAsset+logMsg
                        print(out)
                        trader.write_log(out)
                        send_log_info(queue, thisAsset, {"FUNC":"LOG","ORIGIN":"TRADE","ASS":thisAsset,"MSG":logMsg})
#                        if log_thu_control:
#                            queue.put({"FUNC":"LOG","ORIGIN":"TRADE","ASS":thisAsset,"MSG":logMsg})
#                        else:
#                            log_file = LC.local_log_comm+thisAsset+'/'+str(np.random.randint(99)+99).zfill(5)
#                            write_log("FUNC,"+"LOG,"+"ORIGIN,"+"TRADE,"+"ASS,"+thisAsset+",MSG,"+logMsg, log_file)
                else:
                    logMsg = " NOT flushed due to not opened"
                    out = thisAsset+logMsg
                    print(out)
                    trader.write_log(out)
                    send_log_info(queue, thisAsset, {"FUNC":"LOG","ORIGIN":"TRADE","ASS":thisAsset,"MSG":logMsg})
#                    if log_thu_control:
#                        queue.put({"FUNC":"LOG","ORIGIN":"TRADE","ASS":thisAsset,"MSG":logMsg})
                os.remove(trader.ban_currencies_dir+trader.start_time+thisAsset)
                
                
            # close position
            if flag_cl:
                # info close: [thisSymbol,toc,TimeCurrent(),position,Bi,Ai,bid,ask,dif_ticks,GROI,spread,ROI,real_profit,equity,swap]
                # update postiions vector
                info_split = info_close.split(",")

#                bid = float(info_split[6])
#                ask = float(info_split[7])
#                DateTime = info_split[2]
                #str_idx = int(info_split[15])
                # update bid and ask lists if exist
                #trader.update_symbols_tracking(list_idx, DateTime, bid, ask)
#                list_idx = trader.map_ass_idx2pos_idx[str_idx][ass_id]
                swap = float(info_split[14])
                returnist = float(info_split[12])
                pos_idx = int(info_split[15])
#                print("\nCL found. info=")
#                print(info_split)
#                print("pos_idx")
#                print(pos_idx)
#                print("trader.pos_idx2map_ass_str[pos_idx]")
#                print(trader.pos_idx2map_ass_str[pos_idx])
                # this is the lead position and no double down
                if trader.pos_idx2map_ass_str[pos_idx][-1]==0:
                    str_idx = trader.pos_idx2map_ass_str[pos_idx][1]
                    trader.close_position(info_split[2], thisAsset, ass_id, results, str_idx,
                                          DTi_real=info_split[1], groiist=float(info_split[9]), 
                                          roiist=float(info_split[11]), swap=swap, 
                                          returnist=returnist)
                elif trader.pos_idx2map_ass_str[pos_idx][-1]>0:
                    logMsg = " WARNING! Double downed position. Not closing thru here."
                    out = thisAsset+logMsg
                    print(out)
                    trader.write_log(out)
                else:
                    logMsg = " WARNING! Position already closed. Skipped."
                    out = thisAsset+logMsg
                    print(out)
                    trader.write_log(out)
                
                write_log(info_close, trader.log_positions_ist)
                # open position if swap process is on
                if trader.swap_pending:
                    trader.finalize_resources_swap()
                if delayed_stop_run:
                    send_log_info(queue, thisAsset, {"FUNC":"SD"})
                    send_log_info(queue_prior, thisAsset, {"FUNC":"SD"}, only_queue=True)
#                    if log_thu_control:
#                        queue.put({"FUNC":"SD"})
#                        queue_prior.put({"FUNC":"SD"})
#                    if send_info_api:
#                        close_session(trader.session_json)
                    save_snapshot(thisAsset, lists, trader, command='SD')
                    run = False
                
            elif flag_sl:
                # update positions vector
                info_split = info_close.split(",")
                list_idx = trader.map_ass_idx2pos_idx[ass_id]
#                bid = float(info_split[6])
#                ask = float(info_split[7])
                DateTime = info_split[2]
                direction = int(info_split[3])
                str_idx = int(info_split[15])
                trader.stoplosses += 1
                logMsg = " Exit position due to STOPLOSS "+" sl="+\
                       str(trader.list_stop_losses[trader.map_ass_idx2pos_idx[ass_id]])
                out = (thisAsset+logMsg)
                print("\r"+out)
                write_log(out, trader.log_file)
                if log_thu_control:
                    queue.put({"FUNC":"LOG","ORIGIN":"TRADE","ASS":thisAsset,"MSG":logMsg})
                
                #if not simulate:
                write_log(info_close, trader.log_positions_ist)
                # ban asset
                lists = trader.ban_currencies(lists, thisAsset, DateTime, 
                                              results, direction, 
                                              dtiist=info_split[1], 
                                              groiist=float(info_split[9]), 
                                              roiist=float(info_split[11]))
                
            elif flag_tp:
                # update positions vector
                info_split = info_close.split(",")
#                list_idx = trader.map_ass_idx2pos_idx[ass_id]
#                bid = float(info_split[6])
#                ask = float(info_split[7])
                DateTime = info_split[2]
                direction = int(info_split[3])
                
                trader.takeprofits += 1
                logMsg = " Exit position due to TAKEPROFIT"
                out = (thisAsset+logMsg)
                print("\r"+out)
                write_log(out, trader.log_file)
                if log_thu_control:
                    queue.put({"FUNC":"LOG","ORIGIN":"TRADE","ASS":thisAsset,"MSG":logMsg})
                
                #if not simulate:
                write_log(info_close, trader.log_positions_ist)
                
    
    # end of while run
    budget = get_intermediate_results(trader, AllAssets, running_assets, tic, results)
    return budget

#def update_params():
#    """ Update parameters coming from server """
    

def back_test(DateTimes, SymbolBids, SymbolAsks, Assets, nEvents,
              traders, list_results, running_assets, ass2index_mapping, lists,
              list_models, AllAssets, log_file, queue):
    """
    <DocString>
    """
    nAssets = len(running_assets)
    print("Back test launched")
    # number of events per file
    n_files = 10
    
    init_row = ['d',0.0,0.0]
    fileIDs = [0 for ass in range(nAssets)]
    buffers = [pd.DataFrame(data=[init_row for i in range(n_samps_buffer)],
            columns=['DateTime','SymbolBid','SymbolAsk']) for ass in range(nAssets)]
    
    sampsBuffersCounter = [0 for ass in range(nAssets)]
    
    event_idx = 0
    
    tic = time.time()
    shutdown = False
    while event_idx<nEvents:
        
        outputs = []
        thisAsset = Assets[event_idx].decode("utf-8")
        ass_idx = ass2index_mapping[thisAsset]
        ass_id = running_assets[ass_idx]
        
        DateTime = DateTimes[event_idx].decode("utf-8")
        bid = int(np.round(SymbolBids[event_idx]*100000))/100000
        ask = int(np.round(SymbolAsks[event_idx]*100000))/100000
        # update bid and ask lists if exist
        
        new_outputs = 0
#        print("\r"+DateTime+" "+thisAsset, sep=' ', end='', flush=True)
        # Run RNN
        # add new entry to buffer
        (buffers[ass_idx]).iloc[sampsBuffersCounter[ass_idx]] = [DateTime, bid, ask]
        sampsBuffersCounter[ass_idx] = (sampsBuffersCounter[ass_idx]+1)%n_samps_buffer
        str_idx=0
        for idx, trader in enumerate(traders):
            list_idx = trader.map_ass_idx2pos_idx[str_idx][ass_id]
            trader.update_list_last(list_idx, DateTime, bid, ask)
            # check if asset has been banned from outside
            if os.path.exists(trader.ban_currencies_dir+trader.start_time+thisAsset):
                time.sleep(.01)
                fh = open(trader.ban_currencies_dir+trader.start_time+thisAsset,"r")
                message = fh.read()
                fh.close()
                info = message.split(",")
                otherAsset = info[0]
                otherDirection = int(info[1])
                logMsg = " flag ban from found: "+message[:-1]
                out = thisAsset+logMsg
                print(out)
                trader.write_log(out)
                if log_thu_control:
                    queue.put({"FUNC":"LOG","ORIGIN":"TRADE","ASS":thisAsset,"MSG":logMsg})
                # for now, ban only if asset is opened AND they go in same direction
                if trader.is_opened_strategy(ass_id):
                    thisDirection = trader.list_opened_positions\
                        [trader.map_ass_idx2pos_idx[ass_id]].direction
                    if trader.assets_same_direction(ass_id, thisAsset, thisDirection, 
                                  otherAsset, otherDirection):
                        lists = flush_asset(lists, ass_idx, 0.0)
                        logMsg = " flushed"
                        out = thisAsset+logMsg
                        print(out)
                        trader.write_log(out)
                        trader.close_position(DateTime, thisAsset, ass_id, list_results[idx], DTi_real=DateTime)
                        if log_thu_control:
                            queue.put({"FUNC":"LOG","ORIGIN":"TRADE","ASS":thisAsset,"MSG":logMsg})
                    else:
                        out = thisAsset+" NOT flushed due to different directions"
                        print(out)
                        trader.write_log(out)
                else:
                    log_mess = " NOT flushed due to not opened"
                    out = thisAsset+log_mess
                    print(out)
                    trader.write_log(out)
                    if log_thu_control:
                        queue.put({"FUNC":"LOG","ORIGIN":"TRADE","ASS":thisAsset,"MSG":out})
                os.remove(trader.ban_currencies_dir+trader.start_time+thisAsset)

        if sampsBuffersCounter[ass_idx]==0:
            outputs, new_outputs = dispatch(lists, list_models, buffers[ass_idx], AllAssets, 
                                            ass_id, ass_idx, 
                                            log_file, queue)

            ####### Update counters and buffers ##########
            fileIDs[ass_idx] = (fileIDs[ass_idx]+1)%n_files
            # reset buffer
            buffers[ass_idx] = pd.DataFrame(data=[init_row for i in range(n_samps_buffer)],
                   columns=['DateTime','SymbolBid','SymbolAsk'])
        
        
        ################# Trader ##################
        for idx, trader in enumerate(traders):
            if new_outputs:
                #print(outputs)
                trader.check_new_inputs(outputs, thisAsset, list_results[idx])
            
            ################ MT5 simulator ############
            # check if a position is already opened
            if trader.is_opened(ass_id):
                
                trader.count_events(ass_id, 1)
                stoploss_flag  = trader.is_stoploss_reached(lists, DateTime, ass_id, bid, 
                                                            trader.list_EM[list_idx][-1], 
                                                            event_idx, list_results[idx])
                
                # check for closing
                if (not stoploss_flag and 
                    trader.list_count_events[trader.map_ass_idx2pos_idx[ass_id]]==
                    trader.list_deadlines[trader.map_ass_idx2pos_idx[ass_id]]):
                    # close position
                    trader.close_position(DateTime, thisAsset, ass_id, list_results[idx])
        
        ###################### Check for control commands ##############
        io_ass_dir = LC.io_live_dir+thisAsset+"/"
        # check shut down command
        if os.path.exists(io_ass_dir+'SD'):
            print(thisAsset+" Shutting down")
            os.remove(io_ass_dir+'SD')
            shutdown = True
            for idx, trader in enumerate(traders):
                if trader.is_opened(ass_id):
                    trader.close_position(DateTime, thisAsset, ass_id, list_results[idx])
            if log_thu_control:
                queue.put({"FUNC":"SD"})
            break
        elif os.path.exists(io_ass_dir+'PA'):
            
            print(thisAsset+" PAUSED. Waiting for RE command...")
            os.remove(io_ass_dir+'PA')
            while not os.path.exists(io_ass_dir+'RE'):
                time.sleep(np.random.randint(6)+5)
            os.remove(io_ass_dir+'RE')
        elif os.path.exists(io_ass_dir+'RE'):
            print("WARNING! RESUME command found. Send first PAUSE command")
            os.remove(io_ass_dir+'RE')
        elif os.path.exists(io_ass_dir+'RESET'):
            print("RESET command found.")
            os.remove(io_ass_dir+'RESET')
            lists = flush_asset(lists, ass_idx, bid)
        # Enquire parameters from Server
        elif send_info_api and os.path.exists(io_ass_dir+'PARAM'):
            print("PARAM found")
            # check first for local info
            with open(io_ass_dir+'PARAM', 'r') as f:
                config_name = f.read()
                f.close()
            try:
                os.remove(io_ass_dir+'PARAM')
            except:
                print("WARNING! Error while deleting PARAM. Continuing")
                
            if config_name=='':
                pass
            else:
                # update from local
                config = retrieve_config(config_name)
                print(thisAsset)
                trader.update_parameters(config, thisAsset)
            
        ###################### End of Trader ###########################
        event_idx += 1
        # A pause to avoid communication congestion with server
#        time.sleep(.05)
    # end of while events
    for idx, trader in enumerate(traders):
        # get intermediate results
        get_intermediate_results(trader, AllAssets, running_assets, tic, list_results[idx])
#    if send_info_api:
#        close_session(trader.session_json)
    return shutdown

def get_intermediate_results(trader, AllAssets, running_assets, tic, results):
    """  """
    # get statistics
    #t_entries = trader.n_pos_opened+trader.n_pos_extended
    perEntries = 0
    if trader.n_entries>0:
        per_net_success = trader.net_successes/trader.n_entries
        average_loss = np.abs(trader.average_loss)/(trader.n_entries-trader.net_successes)
        per_gross_success = trader.gross_successes/trader.n_entries
        perSL = trader.stoplosses/trader.n_entries
        ROI_per_entry = 100*trader.tROI_live/trader.n_entries
    else:
        per_net_success = 0.0
        average_loss = 0.0
        per_gross_success = 0.0
        perSL = 0.0
        ROI_per_entry = 0.0
        
    GROI = trader.gross_earnings/trader.init_budget
    earnings = trader.budget-trader.init_budget
    ROI = earnings/trader.init_budget
    #budget = trader.budget
    
    out = ("\n"+str([AllAssets[str(ass)] for ass in running_assets])[1:-1]+
           ":\nGROI = {0:.3f}% ".format(100*GROI)+"ROI = {0:.3f}%".format(100*ROI)+
           " Sum GROI = {0:.3f}%".format(100*trader.tGROI_live)+
           " Sum ROI = {0:.3f}%".format(100*trader.tROI_live)+
          " Final budget {0:.2f}E".format(trader.budget)+
          " Earnings {0:.2f}E".format(earnings)+
          " per earnings {0:.3f}%".format(100*(
                  trader.budget-trader.init_budget)/trader.init_budget)+
          " ROI per position {0:.3f}%".format(ROI_per_entry))
    write_log(out, trader.log_file)
    write_log(out, trader.log_summary)
    print(out)
    out = ("Number entries "+str(trader.n_entries)+
           " per entries {0:.2f}%".format(100*perEntries)+
           " per net success "+"{0:.3f}%".format(100*per_net_success)+
          " per gross success "+"{0:.3f}%".format(100*per_gross_success)+
          " av loss {0:.3f}%".format(100*average_loss)+
          " per sl {0:.3f}%".format(100*perSL))
    write_log(out, trader.log_file)
    write_log(out, trader.log_summary)
    print(out)
    out = "This period DONE. Time: "+"{0:.2f}".format((time.time()-tic)/60)+" mins"
    write_log(out, trader.log_file)
    write_log(out, trader.log_summary)
    print(out)
    
    results.update_weekly_results(GROI, earnings, ROI, trader.n_entries, trader.stoplosses,
                                  trader.tGROI_live, trader.tROI_live, trader.net_successes,
                                  average_loss, trader.average_win, trader.gross_successes)
    # TODO: update badget
#    init_budget = trader.budget
        
    out = ("\nTotal GROI = {0:.3f}% ".format(results.total_GROI)+
           "Total ROI = {0:.3f}% ".format(results.total_ROI)+
           "Sum GROI = {0:.3f}% ".format(results.sum_GROI)+
           "Sum ROI = {0:.3f}%".format(results.sum_ROI)+
           " Accumulated earnings {0:.2f}E\n".format(results.total_earnings))
    print(out)
    write_log(out, trader.log_summary)
    write_log(out, trader.log_file)
    
    return None

def init_network_structures(lists, nNets, nAssets):
    """  """
    phase_shifts = lists['phase_shifts']
    nChans = lists['nChans']
    list_t_indexs = lists['list_t_indexs']
    lBs = lists['lBs']
    nExSs = lists['nExSs']
    mWs = lists['mWs']
    list_n_feats = lists['list_n_feats']
    
    inits = [[[False for ps in range(phase_shifts[nn])] for nn in range(nNets)] 
               for ass in range(nAssets)]
    listFillingXs = [[[True for ps in range(phase_shifts[nn])] for nn in range(nNets)] 
                       for ass in range(nAssets)]
    listCountPoss = [[[0 for ps in range(phase_shifts[nn])] for nn in range(nNets)] 
                       for ass in range(nAssets)]
    countOutss = [[[np.zeros((int(nChans[nn]*phase_shifts[nn]))).astype(int) 
                    for t in list_t_indexs[nn]] 
                    for nn in range(nNets)] 
                    for ass in range(nAssets)]
    EOFs = [[pd.DataFrame(columns=['DateTime','SymbolBid','SymbolAsk'], 
                          index=range(int(nChans[nn]*phase_shifts[nn]))) 
             for nn in range(nNets)] 
             for ass in range(nAssets)]
    # network inputs
    list_list_X_i = [[[np.zeros((1, int((lBs[nn]-nExSs[nn])/mWs[nn]+1), 
                                 list_n_feats[nn])) 
                       for ps in range(phase_shifts[nn])] 
                       for nn in range(nNets)] 
                       for ass in range(nAssets)]
    list_listAllFeatsLive = [[[np.zeros((list_n_feats[nn],0)) 
                               for ps in range(phase_shifts[nn])] 
                               for nn in range(nNets)] 
                               for ass in range(nAssets)]
    list_listFeaturesLive = [[[None for ps in range(phase_shifts[nn])] 
                               for nn in range(nNets)] 
                               for ass in range(nAssets)]
    list_listParSarStruct = [[[None for ps in range(phase_shifts[nn])] \
        for nn in range(nNets)] for ass in range(nAssets)]
    list_listEM = [[[None for ps in range(phase_shifts[nn])] \
        for nn in range(nNets)] for ass in range(nAssets)]
    # network outputs
    list_list_Ylive = [[[[np.zeros((0,)) for t in list_t_indexs[nn]] 
        for ps in range(phase_shifts[nn])] for nn in range(nNets)] 
        for ass in range(nAssets)]
    list_list_Pmc_live = [[[[np.zeros((0,)) for t in list_t_indexs[nn]] 
        for ps in range(phase_shifts[nn])] for nn in range(nNets)] 
        for ass in range(nAssets)]
    list_list_Pmd_live = [[[[np.zeros((0,)) for t in list_t_indexs[nn]] 
        for ps in range(phase_shifts[nn])] for nn in range(nNets)] 
        for ass in range(nAssets)]
    list_list_Pmg_live = [[[[np.zeros((0,)) for t in list_t_indexs[nn]] 
        for ps in range(phase_shifts[nn])] for nn in range(nNets)] 
        for ass in range(nAssets)]
    # init condition vector to open market
    #condition = np.zeros((model.seq_len))
    list_list_time_to_entry = [[[[[]  for t in list_t_indexs[nn]] 
        for ps in range(phase_shifts[nn])] for nn in range(nNets)] 
        for ass in range(nAssets)] # list tracking times to entry the market
    list_list_list_soft_tildes = [[[[[]  for t in list_t_indexs[nn]] 
        for ps in range(phase_shifts[nn])] for nn in range(nNets)] 
        for ass in range(nAssets)]
    # upper diagonal matrix containing latest weight values
    list_list_weights_matrix = [[[np.zeros((int((lBs[nn]-nExSs[nn])/mWs[nn]+1),\
        int((lBs[nn]-nExSs[nn])/mWs[nn]+1))) for ps in range(phase_shifts[nn])] \
        for nn in range(nNets)] for ass in range(nAssets)]
    
    return (inits,listFillingXs,listCountPoss,countOutss,EOFs,list_list_X_i,
            list_listAllFeatsLive,list_listFeaturesLive,list_listParSarStruct,
            list_listEM,list_list_Ylive,list_list_Pmc_live,list_list_Pmd_live,
            list_list_Pmg_live,list_list_time_to_entry,list_list_list_soft_tildes,
            list_list_weights_matrix)

#if __name__ == '__main__':
    
def run_carefully(config_trader, running_assets, start_time, test, queue, queue_prior, session_json, token_header):
    """  """
    try:
        run(config_trader, running_assets, start_time, test, queue, queue_prior, session_json, token_header)
    except KeyboardInterrupt:
        print("KeyboardInterrupt: Exit program organizedly")
    
def run(config_traders_list, running_assets, start_time, test, queue, queue_prior, session_json, token_header):
    """  """    
    
    if len(config_traders_list)>1 and not run_back_test:
        raise ValueError("Live execution not compatible with more than one trader")
    if len(running_assets)>1:
        raise ValueError("Error! Only one running asset per trader is supported")
    
    # init futures session of API
#    if send_info_api:
#        api.post_token()
#        api.intit_all(list_config_traders[0], running_assets, sessiontype, sessiontest=test)
    # directories
    if run_back_test:
        dir_results = LC.live_results_dict+"back_test/"    
    else:
        dir_results = LC.live_results_dict+"live/"
    #dir_results_trader = dir_results+"trader/"
    dir_results_netorks = dir_results+'networks/'
    if not os.path.exists(dir_results_netorks):
        os.makedirs(dir_results_netorks)
    dir_log = dir_results+'log/'
    if not os.path.exists(dir_log):
        os.makedirs(dir_log)
    
    AllAssets = C.AllAssets
    thisAsset = AllAssets[str(running_assets[0])]
#    for ass in AllAssets:
#        asset = AllAssets[ass]
    snapshot_dir_ass = LC.snapshot_live_dir+thisAsset+'/'
    if not os.path.exists(snapshot_dir_ass):
        os.makedirs(snapshot_dir_ass)
    # unique network list
    unique_nets = []
    list_models = []
    #list_data = []
    list_nExS = []
    list_mW = []
    list_lBs = []
    list_n_feats = []
    unique_IDresults = []
    unique_IDepoch = []
    unique_t_indexs = []
    unique_phase_shifts = []
    list_strategies = []
    list_results = []
    #unique_feats_from = []
    unique_delays = []
    unique_netNames = []
    unique_inv_out = []
    list_net2strategy = [[] for _ in range(len(config_traders_list))]
    list_unique_configs = []
    list_tags = []
    list_tags_modular = []
    #log_files = []
    # init tensorflow graph
    tf.reset_default_graph()
    log_file = dir_log+start_time+'_log.log'
    # loop over list with trader configuration files
    for idx_tr, config_trader in enumerate(config_traders_list):
        config_list = config_trader['config_list']
#        log_files.append(log_file)
        config_name = config_trader['config_name']
#        print("config_name")
#        print(config_name)
        #dateTest = config_trader['dateTest']
        #dateTest = ['2018.12.31','2019.01.01','2019.01.02','2019.01.03','2019.01.04']
        numberNetworks = config_trader['numberNetworks']
        IDweights = config_trader['IDweights']#['000318INVO','000318INVO','000318INVO']#['000289STRO']
        IDresults = config_trader['IDresults']#['100318INVO','000318INVO','000318INVO']
        #lIDs = config_trader['lIDs']#[len(IDweights[i]) for i in range(numberNetworks)]
        
        list_name = config_trader['list_name']#['15e_1t_77m_2p','8e_3t_77m_3p','22e_0t_57m_1p']#['89_4']
        IDepoch = config_trader['IDepoch']#['15','8','22']
        netNames = config_trader['netNames']#['31815','31808','31822']
        list_t_indexs = config_trader['list_t_indexs']#[[1],[3],[3]]
        list_inv_out = config_trader['list_inv_out']#[True,True,True]
        #list_feats_from = config_trader['list_feats_from']
        list_entry_strategy = config_trader['list_entry_strategy']#['spread_ranges' for i in range(numberNetworks)] #'fixed_thr','gre' or 'spread_ranges'
        list_spread_ranges = config_trader['list_spread_ranges']#[{'sp':[2],'th':[(.7,.7)]},{'sp':[3],'th':[(.7,.7)]},{'sp':[1],'th':[(.5,.7)]}]#[2]# in pips
        #[{'sp':[2],'th':[(.5,.7)]},{'sp':[3],'th':[(.6,.8)]},{'sp':[1],'th':[(.5,.7)]}]
        list_priorities = config_trader['list_priorities']#[[1],[2],[0]]
        phase_shifts = config_trader['phase_shifts']#[5,5,5]
        list_thr_sl = config_trader['list_thr_sl']#[20 for i in range(numberNetworks)]
        list_thr_tp = config_trader['list_thr_tp']#[1000 for i in range(numberNetworks)]
        delays = config_trader['delays']#[0,0,0]
        mWs = config_trader['mWs']#[100,100,100]
        nExSs = config_trader['nExSs']#[1000,1000,1000]
        #lBs = config_trader['lBs']#[1300,1300,1300]#[1300]
        list_lim_groi_ext = config_trader['list_lim_groi_ext']#[-.02 for i in range(numberNetworks)]
        list_w_str = config_trader['list_w_str']#['55','55','55']
        list_weights = config_trader['list_weights']#[np.array([.5,.5]) for i in range(numberNetworks)]
        list_lb_mc_op = config_trader['list_lb_mc_op']#[.5 for i in range(numberNetworks)]
        list_lb_md_op = config_trader['list_lb_md_op']#[.8 for i in range(numberNetworks)]
        list_lb_mc_ext = config_trader['list_lb_mc_ext']#[.5 for i in range(numberNetworks)]
        list_lb_md_ext = config_trader['list_lb_md_ext']#[.6 for i in range(numberNetworks)]
        list_ub_mc_op = config_trader['list_ub_mc_op']#[1 for i in range(numberNetworks)]
        list_ub_md_op = config_trader['list_ub_md_op']#[1 for i in range(numberNetworks)]
        list_ub_mc_ext = config_trader['list_ub_mc_ext']#[1 for i in range(numberNetworks)]
        list_ub_md_ext = config_trader['list_ub_md_ext']#[1 for i in range(numberNetworks)]
        list_fix_spread = config_trader['list_fix_spread']#[False for i in range(numberNetworks)]
        list_fixed_spread_pips = config_trader['list_fixed_spread_pips']#[4 for i in range(numberNetworks)]
        list_max_lots_per_pos = config_trader['list_max_lots_per_pos']#[.1 for i in range(numberNetworks)]
        list_flexible_lot_ratio = config_trader['list_flexible_lot_ratio']#[False for i in range(numberNetworks)]
        list_if_dir_change_close = config_trader['list_if_dir_change_close']#[False for i in range(numberNetworks)]
        list_if_dir_change_extend = config_trader['list_if_dir_change_extend']#[False for i in range(numberNetworks)]
        if 'max_opened_positions' in config_trader:
            max_opened_positions = config_trader['max_opened_positions']
        else:
            # an infinite value
            max_opened_positions = 99999
        # add unique networks
        for nn in range(numberNetworks):
            # TODO! Take unique networks!
            unique_nets.append(netNames[nn])
            configs = config_list[nn]
            #list_data.append(data)
            list_nExS.append(nExSs[nn])
            list_mW.append(mWs[nn])
            list_lBs.append(configs[0]['lB'])
            feature_keys_manual = configs[0]['feature_keys_manual']
            n_feats_manual = len(feature_keys_manual)
            list_n_feats.append(n_feats_manual)
            unique_IDresults.append(IDresults[nn])
            unique_IDepoch.append(IDepoch[nn])
            unique_phase_shifts.append(phase_shifts[nn])
            unique_t_indexs.append(list_t_indexs[nn])
            #unique_feats_from.append(list_feats_from[nn])
            unique_delays.append(delays[nn])
            unique_netNames.append(netNames[nn])
            unique_inv_out.append(list_inv_out[nn])
            list_models.append(StackedModel(configs,IDweights[nn]).init_interactive_session(epochs=IDepoch[nn]))
            feats_from_bids = configs[0]['feats_from_bids']
#                movingWindow = configs[0]['movingWindow']
#                nEventsPerStat = configs[0]['nEventsPerStat']
            feature_keys_manual = configs[0]['feature_keys_manual']
            n_feats_manual = len(feature_keys_manual)
            noVarFeats = configs[0]['noVarFeatsManual']
            # load stats
            if feats_from_bids:
                # only get short bets (negative directions)
                #tag = 'IO_mW'
                tag_stats = 'IOB'
                tag_stats_modular = 'bid'
            else:
                # only get long bets (positive directions)
                #tag = 'IOA_mW'
                tag_stats = 'IOA'
                tag_stats_modular = 'ask'
            print(tag_stats)
            list_tags.append(tag_stats)
            list_tags_modular.append(tag_stats_modular)
#                filename_prep_IO = (hdf5_directory+tag+str(movingWindow)+'_nE'+
#                                    str(nEventsPerStat)+'_nF'+str(n_feats_manual)+'.hdf5')
#                list_filenames.append(filename_prep_IO)
            list_unique_configs.append(configs[0])
            #else:
            list_net2strategy[idx_tr].append(netNames[nn])
        ################# Strategies #############################
#        print(list_thr_sl)
#        print("len(list_thr_tp)")
#        print(len(list_thr_tp))
#        print("len(list_fix_spread)")
#        print(len(list_fix_spread))
#        print("len(list_fixed_spread_pips)")
#        print(len(list_fixed_spread_pips))
#        print("len(list_max_lots_per_pos)")
#        print(len(list_max_lots_per_pos))
#        print("len(list_flexible_lot_ratio)")
#        print(len(list_flexible_lot_ratio))
#        print("len(list_lb_mc_op)")
#        print(len(list_lb_mc_op))
#        print("len(list_lb_mc_ext)")
#        print(len(list_lb_mc_ext))
#        print("len(list_lb_md_ext)")
#        print(len(list_lb_md_ext))
#        print("len(list_ub_mc_op)")
#        print(len(list_ub_mc_op))
#        print("len(list_ub_md_op)")
#        print(len(list_ub_md_op))
#        print("len(list_ub_mc_ext)")
#        print(len(list_ub_mc_ext))
#        print("len(list_ub_md_ext)")
#        print(len(list_ub_md_ext))
#        print("len(list_if_dir_change_close)")
#        print(len(list_if_dir_change_close))
#        print("len(list_if_dir_change_extend)")
#        print(len(list_if_dir_change_extend))
#        print("len(list_name)")
#        print(len(list_name))
#        print("len(list_t_indexs)")
#        print(len(list_t_indexs))
#        print("len(list_entry_strategy)")
#        print(len(list_entry_strategy))
#        print("len(IDresults)")
#        print(len(IDresults))
#        print(len(IDepoch))
#        print(len(list_weights))
#        print(len(list_spread_ranges))
#        print(len(list_priorities))
#        print(len(list_lim_groi_ext))
        strategies = [Strategy(direct=LC.results_directory,thr_sl=list_thr_sl[i], 
                              thr_tp=list_thr_tp[i], fix_spread=list_fix_spread[i], 
                              fixed_spread_pips=list_fixed_spread_pips[i], 
                              max_lots_per_pos=list_max_lots_per_pos[i], 
                              flexible_lot_ratio=list_flexible_lot_ratio[i], 
                              lb_mc_op=list_lb_mc_op[i], lb_md_op=list_lb_md_op[i], 
                              lb_mc_ext=list_lb_mc_ext[i], lb_md_ext=list_lb_md_ext[i],
                              ub_mc_op=list_ub_mc_op[i], ub_md_op=list_ub_md_op[i], 
                              ub_mc_ext=list_ub_mc_ext[i], ub_md_ext=list_ub_md_ext[i],
                              if_dir_change_close=list_if_dir_change_close[i], 
                              if_dir_change_extend=list_if_dir_change_extend[i], 
                              name=list_name[i],t_indexs=list_t_indexs[i],
                              entry_strategy=list_entry_strategy[i],IDr=IDresults[i],
                              epoch=IDepoch[i],weights=list_weights[i],
                              info_spread_ranges=list_spread_ranges[i],
                              priorities=list_priorities[i],
                              lim_groi_ext=list_lim_groi_ext[i]) for i in range(numberNetworks)]
        list_strategies.append(strategies)
        
        results = Results(IDresults, IDepoch, list_t_indexs, numberNetworks,
                          list_w_str, start_time, dir_results, config_name=config_name)
        list_results.append(results)
    # end of for idx_tr, config_trader in enumerate(config_list):
    ### MAP CONFIGS TO NETWORKS ###
    nUniqueNetworks = len(unique_nets)
#    print("nUniqueNetworks")
#    print(nUniqueNetworks)
#    print(unique_nets)
#    print("list_net2strategy")
#    print(list_net2strategy)
    ADs = [np.array([]) for i in range(nUniqueNetworks)]
    
    nChans = (np.array(list_nExS)/np.array(list_mW)).astype(int).tolist()
    #print(unique_IDresults)
    #print(unique_IDepoch)
    resultsDir = [[dir_results_netorks+start_time+'/'
                   for t in unique_t_indexs[nn]] for nn in range(nUniqueNetworks)]
#    dir_results_netorks+unique_IDresults[nn]+"T"+
#                   str(t)+"E"+''.join(str(e) for e in unique_IDepoch[nn])+"/" 
    print(resultsDir)
    results_files = [[unique_IDresults[nn]+"T"+str(t)+"E"+
                      ''.join(str(e) for e in unique_IDepoch[nn])+".txt"
                      for t in unique_t_indexs[nn]] for nn in range(nUniqueNetworks)]
     
    print(results_files)
    nCxAxN = np.zeros((len(running_assets),nUniqueNetworks))
#    columnsResultInfo = ["Asset","Entry Time","Exit Time","Bet","Outcome","Diff",
#                         "Bi","Ai","Bo","Ao","GROI","Spread","ROI","P_mc","P_md",
#                         "P_mg"]
        
    for nn in range(nUniqueNetworks):
        
        if unique_phase_shifts[nn] != 0:
            nCxAxN[:,nn] = int(nChans[nn]*unique_phase_shifts[nn])
        else:
            nCxAxN[:,nn] = int(nChans[nn])
        for t in range(len(unique_t_indexs[nn])):
            if not os.path.exists(resultsDir[nn][t]):
                try:
                    os.makedirs(resultsDir[nn][t])
                        
                except:
                    print("Warning. Error when creating directory")
            # check if path exists
            filedirname = resultsDir[nn][t]+results_files[nn][t]
            if not os.path.exists(filedirname):
                pd.DataFrame(columns = columnsResultInfo).to_csv(filedirname, 
                            mode="w",index=False,sep='\t')
    
    buffSizes = list_nExS+np.zeros((len(running_assets),nUniqueNetworks)).astype(int)
    
    ############################################
    # init bufers and file extensions (n buffers = n channels per network and asset)
    nNets = nCxAxN.shape[1]
    nAssets = nCxAxN.shape[0]
    
    #########################################
    
    if not test:
        if modular:
            #raise NotImplementedError("Stats from modular features not implemented yet for live session")
            
            list_stats = [[load_stats_modular_live(AllAssets[str(running_assets[ass])], 
                                                   mWs[nn], nExSs[nn], 
                                                   list_tags_modular[nn], 
                                                   feature_keys=[i for i in range(37)], 
                                                   ass_rel='direct') 
                                                    for nn in range(nNets)] 
                                                    for ass in range(nAssets)]
            
            list_stats_feats = [[list_stats[ass][nn][0] for nn in range(nNets)] for ass in range(nAssets)]
            list_stats_rets = [[list_stats[ass][nn][1] for nn in range(nNets)] for ass in range(nAssets)]
        else:
            list_stats_feats = [[load_stats_input_live(feature_keys_manual, mWs[nn], nExSs[nn], AllAssets[str(running_assets[ass])], 
                            None, 
                            from_stats_file=True, hdf5_directory=LC.stats_live_dir,tag=list_tags[nn]) 
                            for nn in range(nNets)] for ass in range(nAssets)]
            list_stats_rets = [[load_stats_output_live(list_unique_configs[nn], LC.stats_live_dir, AllAssets[str(running_assets[ass])], 
                            tag=tag_stats) for nn in range(nNets)] for ass in range(nAssets)]
        gain = 1
    else:
        list_stats_feats = [[load_stats_input_live(feature_keys_manual, 500, 5000, AllAssets[str(running_assets[ass])], 
                        None, 
                        from_stats_file=True, hdf5_directory=LC.stats_live_dir) 
                        for nn in range(nNets)] for ass in range(nAssets)]
        list_stats_rets = [[load_stats_output_live({}, LC.stats_live_dir, 
                            AllAssets[str(running_assets[ass])]) 
                        for nn in range(nNets)] for ass in range(nAssets)]
        gain = .000000001
    
    
    
        
    list_means_in =  [[list_stats_feats[ass][nn]['means_t_in'] for nn in range(nNets)] 
                                                                 for ass in range(nAssets)]
#    print(list_means_in[0][0].shape)
#    print(list_means_in[0][0][1,:])
    list_stds_in =  [[gain*list_stats_feats[ass][nn]['stds_t_in'] for nn in range(nNets)] 
                                                                    for ass in range(nAssets)]
#    print(list_stds_in[0][0].shape)
#    print(list_stds_in[0][0][1,:])    
    list_stds_out =  [[gain*list_stats_rets[ass][nn]['stds_t_out'] for nn in range(nNets)] 
                                                                  for ass in range(nAssets)]
#    print(list_stds_out[0][0])
#    a=p
    # pre allocate memory size
    
    # init non-variation features
    list_nonVarFeats = [np.intersect1d(noVarFeats,feature_keys_manual) for nn in range(nNets)]
    list_nonVarIdx = [np.zeros((len(list_nonVarFeats[nn]))).astype(int) for nn in range(nNets)]
    
    
    for nn in range(nNets):
        nv = 0
        for allnv in range(n_feats_manual):
            if feature_keys_manual[allnv] in list_nonVarFeats[nn]:
                list_nonVarIdx[nn][nv] = int(allnv)
                nv += 1
        
    ass2index_mapping = {}
    #ass_index = 0
    
    for ass_index, ass in enumerate(running_assets):
        ass2index_mapping[AllAssets[str(ass)]] = ass_index
        #ass_index += 1
    
    tic = time.time()
    
    ################### RNN ###############################
    
    shutdown = False
    if run_back_test:
        first_day = '2020.01.06'#'2018.11.12'
        last_day = '2020.01.10'#'2019.08.22'
        init_day = dt.datetime.strptime(first_day,'%Y.%m.%d').date()
        end_day = dt.datetime.strptime(last_day,'%Y.%m.%d').date()
        delta_dates = end_day-init_day
        dateTestDt = [init_day + dt.timedelta(i) for i in range(delta_dates.days + 1)]
        dateTest = []
        for d in dateTestDt:
            if d.weekday()<5:
                dateTest.append(dt.date.strftime(d,'%Y.%m.%d'))
        ##### TEMP! #####
#            dateTest = dateTest+['2019.03.11','2019.03.12','2019.03.13','2019.03.14','2019.03.15']
        day_index = 0
        #t_journal_entries = 0
        week_counter = 0
        while day_index<len(dateTest) and not shutdown:
            counter_back = 0
            init_list_index = day_index#data.dateTest[day_index]
            
            # find sequence of consecutive days in test days
            while (day_index<len(dateTest)-1 and dt.datetime.strptime(
                    dateTest[day_index],'%Y.%m.%d')+dt.timedelta(1)==
                    dt.datetime.strptime(dateTest[day_index+1],'%Y.%m.%d')):
                day_index += 1
            
            end_list_index = day_index+counter_back
#                out = "Week from "+dateTest[init_list_index]+" to "+dateTest[end_list_index]
#                print(out)
 
            day_index += counter_back+1
            
            buffers = [[[pd.DataFrame() for k in range(int(nCxAxN[ass,nn]))] 
                            for nn in range(nNets)] 
                            for ass in range(nAssets)]
            buffersCounter = [[[-k*buffSizes[ass,nn]/nCxAxN[ass,nn]-unique_delays[nn] 
                                for k in range(int(nCxAxN[ass,nn]))] 
                                for nn in range(nNets)] 
                                for ass in range(nAssets)]
            bufferExt = [[[0 for k in range(int(nCxAxN[ass,nn]))] 
                          for nn in range(nNets)] 
                          for ass in range(nAssets)]
            #timers_till_open = [0 for ass in range(nAssets)]
            
            lists = {}
            lists['phase_shifts'] = unique_phase_shifts
            lists['nChans'] = nChans
            lists['list_t_indexs'] = unique_t_indexs
            lists['lBs'] = list_lBs
            lists['nExSs'] = list_nExS
            lists['mWs'] = list_mW
            #lists['list_data'] = list_data
            lists['nCxAxN'] = nCxAxN
            lists['buffSizes'] = buffSizes
            lists['list_unique_configs'] = list_unique_configs
            lists['list_n_feats'] = list_n_feats
            
            (inits,listFillingXs,listCountPoss,countOutss,EOFs,list_list_X_i,
             list_listAllFeatsLive,list_listFeaturesLive,list_listParSarStruct,
             list_listEM,list_list_Ylive,list_list_Pmc_live,list_list_Pmd_live,
             list_list_Pmg_live,list_list_time_to_entry,list_list_list_soft_tildes,
             list_list_weights_matrix) = init_network_structures(lists, nNets, nAssets)
            
            lists['buffers'] = buffers
            lists['buffersCounter'] = buffersCounter
            lists['bufferExt'] = bufferExt
            lists['listFillingXs'] = listFillingXs
            lists['inits'] = inits
            lists['list_listFeaturesLive'] = list_listFeaturesLive
            lists['list_listParSarStruct']= list_listParSarStruct
            lists['list_listEM'] = list_listEM
            lists['list_listAllFeatsLive'] = list_listAllFeatsLive
            lists['list_list_X_i'] = list_list_X_i
            lists['list_means_in'] = list_means_in
            lists['phase_shifts'] = unique_phase_shifts
            lists['list_stds_in'] = list_stds_in
            lists['list_stds_out'] = list_stds_out
            lists['ADs'] = ADs
            lists['netNames'] = netNames
            lists['listCountPoss'] = listCountPoss
            lists['list_list_weights_matrix'] = list_list_weights_matrix
            lists['list_list_time_to_entry'] = list_list_time_to_entry
            lists['list_list_list_soft_tildes'] = list_list_list_soft_tildes
            lists['list_list_Ylive'] = list_list_Ylive
            lists['list_list_Pmc_live'] = list_list_Pmc_live
            lists['list_list_Pmd_live'] = list_list_Pmd_live
            lists['list_list_Pmg_live'] = list_list_Pmg_live
            lists['EOFs'] = EOFs
            lists['countOutss'] = countOutss
            lists['resultsDir'] = resultsDir
            lists['results_files'] = results_files
#                lists['list_models'] = list_models
            #lists['list_data'] = list_data
            lists['list_nonVarIdx'] = list_nonVarIdx
            lists['list_inv_out'] = unique_inv_out
            #lists['list_feats_from'] = unique_feats_from
            
            traders = []
            for idx_tr, config_trader in enumerate(config_traders_list):
                trader = Trader(running_assets,
                                ass2index_mapping, list_strategies[idx_tr], 
                                log_file, results_dir=dir_results, 
                                start_time=start_time, config_name=config_name,
                                net2strategy=list_net2strategy[idx_tr], 
                                queue=queue, queue_prior=queue_prior, 
                                max_opened_positions=max_opened_positions, 
                                session_json=session_json, token_header=token_header)#, api=api
                # pass trader to api
#                if send_info_api:
#                    api.init_trader(trader)
#                    if not os.path.exists(trader.log_file):
#                        write_log(out, trader.log_file)
#                        write_log(out, trader.log_summary)
                
                traders.append(trader)
            out = ("Week counter "+str(week_counter)+". From "+dateTest[init_list_index]+
                   " to "+dateTest[end_list_index])
            week_counter += 1
            print(out)
            #if not os.path.exists(trader.log_file):
            trader.write_log(out)
            send_log_info(queue, "ALL", {"FUNC":"LOG","ORIGIN":"TRADE","ASS":"ALL","MSG":out})
#            if log_thu_control:
#                queue.put({"FUNC":"LOG","ORIGIN":"MONITORING","ASS":"ALL","MSG":out})
            write_log(out, trader.log_summary)
            DateTimes, SymbolBids, SymbolAsks, Assets, nEvents = \
                load_in_memory(running_assets, AllAssets, dateTest, init_list_index, 
                               end_list_index, root_dir=LC.data_test_dir)
            shutdown = back_test(DateTimes, SymbolBids, SymbolAsks, 
                                    Assets, nEvents ,
                                    traders, list_results, running_assets, 
                                    ass2index_mapping, lists, list_models, AllAssets, 
                                    log_file, queue)
    else:
        
        buffers = [[[pd.DataFrame() for k in range(int(nCxAxN[ass,nn]))] 
                            for nn in range(nNets)] 
                            for ass in range(nAssets)]
        buffersCounter = [[[-k*buffSizes[ass,nn]/nCxAxN[ass,nn]-delays[nn] 
                                for k in range(int(nCxAxN[ass,nn]))] 
                                for nn in range(nNets)] 
                                for ass in range(nAssets)]
        bufferExt = [[[0 for k in range(int(nCxAxN[ass,nn]))] 
                          for nn in range(nNets)] 
                          for ass in range(nAssets)]
        #timers_till_open = [0 for ass in range(nAssets)]
            
        lists = {}
        lists['phase_shifts'] = phase_shifts
        lists['nChans'] = nChans
        lists['list_t_indexs'] = list_t_indexs
        lists['lBs'] = list_lBs
        lists['nExSs'] = nExSs
        lists['mWs'] = mWs
        lists['nCxAxN'] = nCxAxN
        lists['buffSizes'] = buffSizes
        lists['list_unique_configs'] = list_unique_configs
        lists['list_n_feats'] = list_n_feats
            
        (inits,listFillingXs,listCountPoss,countOutss,EOFs,list_list_X_i,
         list_listAllFeatsLive,list_listFeaturesLive,list_listParSarStruct,
         list_listEM,list_list_Ylive,list_list_Pmc_live,list_list_Pmd_live,
         list_list_Pmg_live,list_list_time_to_entry,list_list_list_soft_tildes,
         list_list_weights_matrix) = init_network_structures(lists, nNets, nAssets)
        lists['buffers'] = buffers
        lists['buffersCounter'] = buffersCounter
        lists['bufferExt'] = bufferExt
        lists['listFillingXs'] = listFillingXs
        lists['inits'] = inits
        lists['list_listFeaturesLive'] = list_listFeaturesLive
        lists['list_listParSarStruct']= list_listParSarStruct
        lists['list_listEM'] = list_listEM
        lists['list_listAllFeatsLive'] = list_listAllFeatsLive
        lists['list_list_X_i'] = list_list_X_i
        lists['list_means_in'] = list_means_in
        lists['phase_shifts'] = phase_shifts
        lists['list_stds_in'] = list_stds_in
        lists['list_stds_out'] = list_stds_out
        lists['ADs'] = ADs
        lists['netNames'] = netNames
        lists['listCountPoss'] = listCountPoss
        lists['list_list_weights_matrix'] = list_list_weights_matrix
        lists['list_list_time_to_entry'] = list_list_time_to_entry
        lists['list_list_list_soft_tildes'] = list_list_list_soft_tildes
        lists['list_list_Ylive'] = list_list_Ylive
        lists['list_list_Pmc_live'] = list_list_Pmc_live
        lists['list_list_Pmd_live'] = list_list_Pmd_live
        lists['list_list_Pmg_live'] = list_list_Pmg_live
        lists['EOFs'] = EOFs
        lists['countOutss'] = countOutss
        lists['resultsDir'] = resultsDir
        lists['results_files'] = results_files
#            lists['list_models'] = list_models
        lists['list_nonVarIdx'] = list_nonVarIdx
        lists['list_inv_out'] = list_inv_out
#            pickle.dump( lists, open( LC.io_live_dir+AllAssets[str(running_assets[0])]+"/lists.p", "wb" ))
        #lists['list_feats_from'] = list_feats_from
        # init traders
#        session_json = open_session(config_name, sessiontype, test)
        #traders = []
        #for idx_tr, config_trader in enumerate(config_traders_list):
        trader = Trader(running_assets,
                        ass2index_mapping, list_strategies[idx_tr], 
                        log_file, results_dir=dir_results, session_json=session_json,
                        token_header=token_header, 
                        start_time=start_time, config_name=config_name,
                        net2strategy=list_net2strategy[idx_tr], queue=queue, 
                        queue_prior=queue_prior, max_opened_positions=max_opened_positions)
        if send_info_api:
            # send config confirmation
            if log_thu_control:
                trader.queue_prior.put({"FUNC":"CONFIG", 
                                      "CONFIG":config_trader, 
                                      "ASSET":thisAsset, 
                                      "ORIGIN":"PARAM_UPDATE"})
            else:
                ct.confirm_config_info(config_trader, thisAsset, "PARAM_UPDATE", trader.token_header)
        # Resume after hibernation
        snapshot_filename = LC.snapshot_live_dir+AllAssets[str(running_assets[0])]+"/snapshot.p"
        if resume:
            if os.path.exists(snapshot_filename):
                # Load
                # WARNING! Not compatible with more than one asset per trader
                print("Loading "+snapshot_filename)
                snapshot = pickle.load( open( snapshot_filename, "rb" ))
                lists = snapshot['network']
                for attr in snapshot['trader']:
                    setattr(trader, attr, snapshot['trader'][attr])
            else:
                print("\n\n"+AllAssets[str(running_assets[0])]+" WARNING! Snapshot does not exist. Skipped.\n\n")
        
        # launch fetcher
        fetch(lists, list_models, trader, LC.directory_MT5_IO, AllAssets, 
              running_assets, log_file, results, queue, queue_prior)
    
    for idx in range(1):
        # gather results
        total_entries = int(np.sum(list_results[idx].number_entries))
        total_successes = int(np.sum(list_results[idx].net_successes))
        total_failures = total_entries-total_successes
        per_gross_success = 100*np.sum(list_results[idx].gross_successes)/total_entries
        per_net_succsess = 100*np.sum(list_results[idx].net_successes)/total_entries
        average_loss = np.sum(list_results[idx].total_losses)/\
            ((total_entries-np.sum(list_results[idx].net_successes))*trader.pip)
        average_win = np.sum(list_results[idx].total_wins)/\
            (np.sum(list_results[idx].net_successes)*trader.pip)
        RR = total_successes*average_win/(average_loss*total_failures)
        
        out = ("\nTotal GROI = {0:.3f}% ".format(list_results[idx].total_GROI)+
               "Total ROI = {0:.3f}% ".format(list_results[idx].total_ROI)+
               "Sum GROI = {0:.3f}% ".format(list_results[idx].sum_GROI)+
               "Sum ROI = {0:.3f}%".format(list_results[idx].sum_ROI)+
               " Accumulated earnings {0:.2f}E".format(list_results[idx].total_earnings))
        print(out)
        write_log(out, trader.log_file)
        write_log(out, trader.log_summary)
        out = ("Total entries "+str(total_entries)+
               " percent gross success {0:.2f}%".format(per_gross_success)+
              " percent nett success {0:.2f}%".format(per_net_succsess)+
              " average loss {0:.2f}p".format(average_loss)+
              " average win {0:.2f}p".format(average_win)+
              " RR 1 to {0:.2f}".format(RR))
        print(out)
        write_log(out, trader.log_file)
        write_log(out, trader.log_summary)
        out = ("DONE. Total time: "+"{0:.2f}".format((time.time()-tic)/60)+" mins\n")
        print(out)
        write_log(out, trader.log_file)
        write_log(out, trader.log_summary)
        list_results[idx].save_results()
#[1,2,3,4,7,8,10,11,12,13,14,16,17,19,27,28,29,30,31,32]

def launch(synchroned_run=False, test=False, session_json=None, token_header=None):
    
    #print("\n\n\nLaunching\n\n\n")
    if synchroned_run:
        run(list_config_traders, running_assets, start_time, test)
        return []
#        disp = Process(target=run, args=[running_assets,start_time])
#        disp.start()
    else:
        queues = []
        queues_prior = []
        processes = []
        for ass_idx in range(len(running_assets)):
            queue = Queue()
            queue_prior = Queue()
            disp = Process(target=run_carefully, args=[list_config_traders, 
                           running_assets[ass_idx:ass_idx+1], start_time, test, 
                           queue, queue_prior, session_json, token_header])
            disp.start()
            processes.append(disp)
            queues.append(queue)
            queues_prior.append(queue_prior)
            time.sleep(10)
        time.sleep(30)
        print("All RNNs launched")
        return processes, queues, queues_prior

if __name__=='__main__':
    import sys
    this_path = os.getcwd()
    path = '\\'.join(this_path.split('\\')[:-2])+'\\'
    print(path)
    if path not in sys.path:
        sys.path.insert(0, path)
        print(path+" added to python path")
    else:
        print(path+" already added to python path")
    
#    synchroned_run = False
#    test = False
#    config_names = ['TN01010FS2NYREDOK2K52145314SRv3']#['TTEST10']#'TPRODN01010N01011'
#    
#    for arg in sys.argv:
#        if re.search('^synchroned_run=False',arg)!=None:
#            synchroned_run = False
#        if re.search('^test',arg)!=None:
#            print(arg)
#            if (arg.split('=')[-1])=='True' or (arg.split('=')[-1])=='true':
#                test = True
#        if re.search('^config_names',arg)!=None:
#            config_names = (arg.split('=')[-1]).split(',')
#            print(config_names)
#    if ('TTEST10' not in config_names or len(config_names)>1) and test:
#        raise ValueError("test cannot be False if config_names is TTEST")
    #
from kaissandra.updateRaw import load_in_memory
from kaissandra.prod.preprocessing import load_stats_input_live, \
                                          load_stats_output_live, \
                                          init_features_live,\
                                          get_features_live, \
                                          load_stats_modular_live
from kaissandra.models import StackedModel
import shutil
from kaissandra.local_config import local_vars as LC
#if send_info_api:
#from kaissandra.prod.api import API
#from kaissandra.prod.communication import open_session, \
#                                          close_session, \
#                                          shutdown_control
from kaissandra.config import Config as C
# runLive in multiple processes
from multiprocessing import Process, Queue
from kaissandra.config import retrieve_config
import kaissandra.prod.communication as ct

verbose_RNN = LC.VERBOSE
verbose_trader = LC.VERBOSE
test = LC.TEST
synchroned_run = LC.SYNCHED
run_back_test = LC.BACK_TEST
send_info_api = LC.API
running_assets= LC.ASSETS#
config_names = [LC.CONFIG_FILE]
if hasattr(LC,'CRISIS_MODE'):
    crisis_mode = LC.CRISIS_MODE
else:
    print("\n\nWARNING! Crisis mode not in LC\n\n")
    crisis_mode = False
if hasattr(LC,'DOUBLE_DOWN'):
    double_down = LC.DOUBLE_DOWN
    if double_down['on']:
        print("\nDOUBLE DOWN ON!")
else:
    double_down = {'on':False}

if hasattr(LC,'MARGIN_ADAPT'):    
    margin_adapt = LC.MARGIN_ADAPT
    if margin_adapt:
        print("\nMARGIN ADAPT TRUE!")
else:
    margin_adapt = False

resume = LC.RESUME
if resume:
    print("\n\nRESUME ON\n\n")

# depricated
spread_ban = False
ban_only_if_open = False # not in use
force_no_extesion = False

modular = True
if hasattr(LC,'LOG_CONTROL'):
    log_thu_control = LC.LOG_CONTROL
else:
    print("\n\nWARNING! Crisis mode not in LC\n\n")
    log_thu_control = False

if not test:
    if not crisis_mode:
        n_samps_buffer = 250
    else:
        n_samps_buffer = 100
else:
    n_samps_buffer = 10

if len(config_names)>1:
    raise ValueError("ERROR! config_names len must be 1")
print("config_names")
print(config_names)
list_config_traders = [retrieve_config(LC.CONFIG_FILE)]
if test:
    print("WARNING! TEST ON")
#if not test:
#    if len(config_names)>0: 
#        list_config_traders = [retrieve_config(config_name) for config_name in config_names]
#        print("config_names")
#        print(config_names)
#    #        for config_name in ins:
#    #            #config_trader = retrieve_config(ins[0])
#    #            list_config_traders.append(retrieve_config(config_name))
#    else:
#        list_config_traders = [retrieve_config(LC.CONFIG_FILE)]#'TPRODN01010GREV2', 'TPRODN01010N01011'
## override list configs if test is True
#else:
#    list_config_traders = [retrieve_config('T6N504060TESTv1')]#TTESTv3#'TTEST10'#'TPRODN01010N01011'
#    print("WARNING! TEST ON")
#print("synchroned_run: "+str(synchroned_run))
#print("Test "+str(test))
#running_assets = [10]#assets#[7,10,12,14]#assets#[12,7,14]#
if run_back_test:
    sessiontype = 'backtest'
else:
    sessiontype = 'live'



start_time = dt.datetime.strftime(dt.datetime.now(),'%y_%m_%d_%H_%M_%S')

#if send_info_api:
#print("API")
#api = API()
#else:
#     api = None
#     print("send_info_api")
#     print(send_info_api)

if __name__=='__main__':
    # lauch
    if send_info_api:
        session_json = ct.open_session(list_config_traders[0]['config_name'], sessiontype, test)
        token_header = ct.build_token_header(ct.post_token())
    else:
        session_json = None
        token_header = None
    
    clean_mt5 = False
    for arg in sys.argv:
        if arg=='clean':
            # clean directory before running
            clean_mt5 = True
    renew_directories(C.AllAssets, running_assets, clean_mt5)
#    if synchroned_run and send_info_api:
#        api.intit_all(list_config_traders[0], running_assets, sessiontype, sessiontest=test)
    processes, queues, queues_prior = launch(synchroned_run=synchroned_run, 
                                             test=test, session_json=session_json, token_header=token_header)#
    if not synchroned_run:
        # Controlling and message passing to releave traders of these tasks
        if log_thu_control:
            from kaissandra.prod.control import control
            kwargs = {'queues':queues, 'queues_prior':queues_prior, 'send_info_api':send_info_api}
            #Process(target=control, args=[running_assets], kwargs=kwargs).start()
            
            control(running_assets, queues=queues, queues_prior=queues_prior, send_info_api=send_info_api, test=test)
        # wait for last trader to finish
        print("WAITING FOR PROCESSES TO FINISH")
        for p in processes:
            p.join()
        # close session
        if send_info_api:
            ct.close_session(session_json)
        # shutdown control
        print("TRADER PROCESSES FINISHED")
        ct.shutdown_control()
#        control(running_assets, queues=queues, queues_prior=queues_prior, send_info_api=send_info_api)
        