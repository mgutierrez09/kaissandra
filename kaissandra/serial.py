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
import h5py
import re
import tensorflow as tf
#from multiprocessing import Process
from kaissandra.simulateTrader import load_in_memory
from kaissandra.inputs import (Data, 
                               initFeaturesLive, 
                               extractFeaturesLive, 
                               load_stats_manual, 
                               load_stats_output,
                               initFeaturesLive_v2,
                               extractFeaturesLive_v2)
from kaissandra.preprocessing import (load_stats_manual_v2,
                                      load_stats_output_v2,)
from kaissandra.RNN import modelRNN
from kaissandra.models import RNN, StackedModel
import shutil
from kaissandra.local_config import local_vars

entry_time_column = 'Entry Time'#'Entry Time
exit_time_column = 'Exit Time'#'Exit Time
entry_bid_column = 'Bi'
entry_ask_column = 'Ai'
exit_ask_column = 'Ao'
exit_bid_column = 'Bo'

verbose_RNN = True
verbose_trader = True
test = False
run_back_test = True
spread_ban = False
ban_only_if_open = False # not in use
force_no_extesion = False

# TODO: add it in parameters
n_samps_buffer = 500
columnsResultInfo = ["Asset","Entry Time","Exit Time","Bet",
                     "Outcome","Diff","Bi","Ai","Bo","Ao",
                     "GROI","Spread","ROI","P_mc","P_md",
                     "P_mg"]

data_dir = local_vars.data_test_dir
directory_MT5 = local_vars.directory_MT5#("C:/Users/mgutierrez/AppData/Roaming/MetaQuotes/Terminal/"+
                #     "D0E8209F77C8CF37AD8BF550E51FF075/MQL5/Files/IOlive/")
io_dir = local_vars.io_dir
ADsDir = local_vars.ADsDir
hdf5_directory = local_vars.hdf5_directory

init_budget = 10000.0
#start_time = dt.datetime.strftime(dt.datetime.now(),'%y_%m_%d_%H_%M_%S')


class Results:
        
    def __init__(self, IDresults, IDepoch, list_t_indexs, 
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
        numberNetworks = len(IDresults)
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
        results_dict = {'dts_close':self.dts_close,
                    'GROIs':self.GROIs,
                    'ROIs':self.ROIs,
                    'earnings':self.earnings}
        pickle.dump( results_dict, open( self.results_dir_and_file, "wb" ))
    
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
    
    def save_pos_evolution(self, filename, dts, bids, asks, ems):
        """ Save evolution of the position from opening till close """
        # format datetime for filename
        
        df = pd.DataFrame({'DateTime':dts,
                           'SymbolBid':bids,
                           'SymbolAsk':asks,
                           'EmBid':ems})
        df.to_csv(self.dir_positions+filename+'.txt', index=False)
    
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
        self.idx_mc = strategy._get_idx(self.p_mc)
        self.idx_md = strategy._get_idx(self.p_md)

class Strategy():
    
    thresholds_mc = [.5,.6,.7,.8,.9]
    thresholds_md = [.5,.6,.7,.8,.9]
    
    def __init__(self, direct='',thr_sl=1000, thr_tp=1000, fix_spread=False, 
                 fixed_spread_pips=2, max_lots_per_pos=.1, 
                 flexible_lot_ratio=False, lb_mc_op=0.6, lb_md_op=0.6, 
                 lb_mc_ext=0.6, lb_md_ext=0.6, ub_mc_op=1, ub_md_op=1, 
                 ub_mc_ext=1, ub_md_ext=1,if_dir_change_close=False, 
                 if_dir_change_extend=False, name='',t_indexs=[3],entry_strategy='gre',
                 IDr=None,epoch='11',weights=np.array([0,1]),info_spread_ranges={},
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
        self.epoch = epoch
        self.t_indexs = t_indexs
        self.weights = weights
        self.priorities = priorities
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
            
            if test:
                self.GRE = self.GRE+20
#            print("GRE combined level 1:")
#            print(self.GRE[:,:,0])
#            print("GRE combined level 2:")
#            print(self.GRE[:,:,1])
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
        else:
            # here it represents a priority
            return self.priorities[t]
        
 
class Trader:
    
    def __init__(self, running_assets, ass2index_mapping, strategies,
                 AllAssets, log_file, results_dir="", 
                 start_time='', config_name='',net2strategy=[]):
        
        self.list_opened_positions = []
        self.AllAssets = AllAssets
        self.map_ass_idx2pos_idx = np.array([-1 for i in range(len(AllAssets))])
        self.list_count_events = []
        self.list_count_all_events = []
        self.list_stop_losses = []
        self.list_take_profits = []
        self.list_lots_per_pos = []
        self.list_lots_entry = []
        self.list_last_bid = []
        self.list_EM = []
        self.list_last_ask = []
        self.list_last_dt = []
        self.list_sl_thr_vector = []
        self.list_deadlines = []
        self.positions_tracker = []
        
        self.journal_idx = 0
        self.sl_thr_vector = np.array([5, 10, 15, 20, 25, 30])
        
        self.budget = init_budget
        self.init_budget = init_budget
        self.LOT = 100000.0
        
        self.pip = 0.0001
        self.leverage = 30
        #self.budget_in_lots = self.leverage*self.budget/self.LOT
        self.available_budget = self.budget*self.leverage
        self.available_bugdet_in_lots = self.available_budget/self.LOT
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
        self.n_pos_extended = 0
        
        # log
        self.save_log = 1
        self.results_dir_trader = results_dir+'trader/'
        
        self.running_assets = running_assets
        self.ass2index_mapping = ass2index_mapping
        
        self.strategies = strategies
        if len(net2strategy)==0:
            self.net2strategy = range(len(strategies))
        else:
            self.net2strategy = net2strategy
        
        if start_time=='':
            
            raise ValueError("Depricated. String start_time cannot be empty")
            
        else:
            if run_back_test:
                tag = '_BT_'
            else:
                tag = '_LI_'
            self.log_file_trader = self.results_dir_trader+start_time+tag+config_name+"trader.log"
            self.log_file = log_file
            self.log_positions_soll = self.results_dir_trader+start_time+tag+config_name+"positions_soll.log"
            self.log_positions_ist = self.results_dir_trader+start_time+tag+config_name+"positions_ist.log"
            self.log_summary = self.results_dir_trader+start_time+tag+config_name+"summary.log"
            self.dir_positions = results_dir+'/positions/'+start_time+config_name+'/'
            self.budget_file = self.results_dir_trader+start_time+tag+config_name+"budget.log"
            self.ban_currencies_dir = io_dir+'/ban'+config_name+'/'
            
        
        self.start_time = start_time
        
        if not os.path.exists(self.results_dir_trader):
            os.makedirs(self.results_dir_trader)
        if not os.path.exists(self.dir_positions):
                os.makedirs(self.dir_positions)
        if not os.path.exists(self.ban_currencies_dir):
                os.makedirs(self.ban_currencies_dir)
        # little pause to garantee no 2 processes access at the same time
        time.sleep(np.random.rand())
        # results tracking
        if not os.path.exists(self.log_positions_soll):
            resultsInfoHeader = "Asset,Entry Time,Exit Time,Position,"+\
                "Bi,Ai,Bo,Ao,ticks_d,GROI,Spread,ROI,strategy,Profit,E_spread,stoploss,stGROI,stROI"
            write_log(resultsInfoHeader, self.log_positions_soll)
            if not run_back_test:
                write_log(resultsInfoHeader, self.log_positions_ist)
            write_log(str(self.available_bugdet_in_lots), self.budget_file)
        # flow control
        self.EXIT = 0
        self.rewind = 0
        self.approached = 0
        self.swap_pending = 0
    
    def _get_thr_sl_vector(self):
        '''
        '''
        return self.next_candidate.entry_bid*(1-self.next_candidate.direction*
                                              self.sl_thr_vector)
    
    def add_new_candidate(self, position):
        '''
        '''
        self.next_candidate = position
    
    def add_position(self, idx, lots, datetime, bid, ask, deadline):
        '''
        '''
        self.list_opened_positions.append(self.next_candidate)
        self.list_count_events.append(0)
        self.list_count_all_events.append(0)
        self.list_lots_per_pos.append(lots)
        self.list_lots_entry.append(lots)
        self.list_last_bid.append([bid])
        self.list_EM.append([bid])
        self.list_last_ask.append([ask])
        self.list_last_dt.append([datetime])
        self.map_ass_idx2pos_idx[idx] = len(self.list_count_events)-1
        self.list_stop_losses.append(self.next_candidate.entry_bid*\
                                (1-self.next_candidate.direction*\
                                self.next_candidate.strategy.thr_sl*self.pip))
        self.list_take_profits.append(self.next_candidate.entry_bid*\
                                (1+self.next_candidate.direction*\
                                self.next_candidate.strategy.thr_tp*self.pip))
        self.list_deadlines.append(deadline)
        
        
    def remove_position(self, idx):
        '''
        '''
        self.list_opened_positions = self.list_opened_positions\
            [:self.map_ass_idx2pos_idx[idx]]+self.list_opened_positions\
            [self.map_ass_idx2pos_idx[idx]+1:]
        self.list_count_events = self.list_count_events\
            [:self.map_ass_idx2pos_idx[idx]]+self.list_count_events\
            [self.map_ass_idx2pos_idx[idx]+1:]
        self.list_count_all_events = self.list_count_all_events\
            [:self.map_ass_idx2pos_idx[idx]]+self.list_count_all_events\
            [self.map_ass_idx2pos_idx[idx]+1:]
        self.list_stop_losses = self.list_stop_losses\
            [:self.map_ass_idx2pos_idx[idx]]+self.list_stop_losses\
            [self.map_ass_idx2pos_idx[idx]+1:]
        self.list_take_profits = self.list_take_profits\
            [:self.map_ass_idx2pos_idx[idx]]+self.list_take_profits\
            [self.map_ass_idx2pos_idx[idx]+1:]
        self.list_lots_per_pos = self.list_lots_per_pos\
            [:self.map_ass_idx2pos_idx[idx]]+self.list_lots_per_pos\
            [self.map_ass_idx2pos_idx[idx]+1:]
        self.list_lots_entry = self.list_lots_entry\
            [:self.map_ass_idx2pos_idx[idx]]+self.list_lots_entry\
            [self.map_ass_idx2pos_idx[idx]+1:]
        self.list_last_bid = self.list_last_bid\
            [:self.map_ass_idx2pos_idx[idx]]+self.list_last_bid\
            [self.map_ass_idx2pos_idx[idx]+1:]
        self.list_EM = self.list_EM\
            [:self.map_ass_idx2pos_idx[idx]]+self.list_EM\
            [self.map_ass_idx2pos_idx[idx]+1:]
        self.list_last_ask = self.list_last_ask\
            [:self.map_ass_idx2pos_idx[idx]]+self.list_last_ask\
            [self.map_ass_idx2pos_idx[idx]+1:]
        self.list_last_dt = self.list_last_dt\
            [:self.map_ass_idx2pos_idx[idx]]+self.list_last_dt\
            [self.map_ass_idx2pos_idx[idx]+1:]
        self.list_deadlines = self.list_deadlines\
            [:self.map_ass_idx2pos_idx[idx]]+self.list_deadlines\
            [self.map_ass_idx2pos_idx[idx]+1:]
        self.positions_tracker = self.positions_tracker\
            [:self.map_ass_idx2pos_idx[idx]]+self.positions_tracker\
            [self.map_ass_idx2pos_idx[idx]+1:]
            
        
        mask = self.map_ass_idx2pos_idx>self.map_ass_idx2pos_idx[idx]
        self.map_ass_idx2pos_idx[idx] = -1
        self.map_ass_idx2pos_idx = self.map_ass_idx2pos_idx-mask*1#np.maximum(,-1)
        
    def update_position(self, idx):
        '''
        '''
        # reset counter
        self.list_count_events[self.map_ass_idx2pos_idx[idx]] = 0
        self.list_deadlines[self.map_ass_idx2pos_idx[idx]] = self.next_candidate.deadline
        
        entry_bid = self.list_opened_positions[self.map_ass_idx2pos_idx[idx]].entry_bid
        entry_ask = self.list_opened_positions[self.map_ass_idx2pos_idx[idx]].entry_ask
        direction = self.list_opened_positions[self.map_ass_idx2pos_idx[idx]].direction
        bet = self.list_opened_positions[self.map_ass_idx2pos_idx[idx]].bet
        entry_time = self.list_opened_positions[self.map_ass_idx2pos_idx[idx]].entry_time
        p_mc = self.list_opened_positions[self.map_ass_idx2pos_idx[idx]].p_mc
        p_md = self.list_opened_positions[self.map_ass_idx2pos_idx[idx]].p_md
        strategy = self.list_opened_positions[self.map_ass_idx2pos_idx[idx]].strategy
        
        self.list_opened_positions[self.map_ass_idx2pos_idx[idx]] = self.next_candidate
        
        self.list_opened_positions[self.map_ass_idx2pos_idx[idx]].entry_bid = entry_bid
        self.list_opened_positions[self.map_ass_idx2pos_idx[idx]].entry_ask = entry_ask
        self.list_opened_positions[self.map_ass_idx2pos_idx[idx]].direction = direction
        self.list_opened_positions[self.map_ass_idx2pos_idx[idx]].bet = bet
        self.list_opened_positions[self.map_ass_idx2pos_idx[idx]].entry_time = entry_time
        self.list_opened_positions[self.map_ass_idx2pos_idx[idx]].p_mc = p_mc
        self.list_opened_positions[self.map_ass_idx2pos_idx[idx]].p_md = p_md
        self.list_opened_positions[self.map_ass_idx2pos_idx[idx]].strategy = strategy
    
    def is_opened(self, idx):
        '''
        '''
        if self.map_ass_idx2pos_idx[idx]>=0:
            return True
        else:
            return False
    
    def count_events(self, idx, n_events):
        self.list_count_events[self.map_ass_idx2pos_idx[idx]] += n_events
        self.list_count_all_events[self.map_ass_idx2pos_idx[idx]] += n_events
    
    def direction_map(self, candidate_direction, strategy_direction):
        """  """
        if strategy_direction=='COMB':
            condition = True
        elif strategy_direction=='BIDS' and candidate_direction<0:
            condition =True
        elif strategy_direction=='ASKS' and candidate_direction>0:
            condition = True
        else:
            condition = False
        return condition
    
    def check_contition_for_opening(self, t):
        '''
        '''
        reason = ''
        this_strategy = self.next_candidate.strategy
        e_spread = self.next_candidate.e_spread
        margin = 0.0
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
        elif not this_strategy.fix_spread and this_strategy.entry_strategy=='gre':
            condition_open = self.next_candidate.profitability>e_spread+margin
        elif this_strategy.entry_strategy=='spread_ranges':
            cond_pmc = self.next_candidate.p_mc>=this_strategy.info_spread_ranges['th'][t][0]
            cond_pmd = self.next_candidate.p_md>=this_strategy.info_spread_ranges['th'][t][1]
            cond_spread = e_spread<=this_strategy.info_spread_ranges['sp'][t]
            cond_bet = self.direction_map(self.next_candidate.direction, 
                                   self.next_candidate.strategy.info_spread_ranges['dir'])
            condition_open= cond_pmc and\
                cond_pmd and\
                cond_spread and\
                cond_bet
            if not cond_pmc:
                reason += 'pmc'
            if not cond_pmd:
                reason += 'pmd'
            if not cond_spread:
                reason += 'spread'
            if not cond_bet:
                reason += 'bet'
        else:
            #print("ERROR: fix_spread cannot be fixed if GRE is in use")
            raise ValueError("fix_spread cannot be fixed if GRE is in use")
            
        return condition_open, reason
    
    def check_same_direction(self, ass_id):
        """ Check that extension candidate is in the same direction as current
        position. """
        return self.list_opened_positions[self.map_ass_idx2pos_idx[ass_id]]\
                                .direction==self.next_candidate.direction
                                
    def check_same_strategy(self, ass_id):
        """ Check if candidate and current position share same strategy """
        return self.list_opened_positions[self.map_ass_idx2pos_idx[ass_id]]\
                                .strategy.name==self.next_candidate.strategy.name
    
    def check_remain_samps(self, ass_id):
        """ Check that the number of samples for extantion is larger than the 
        remaining ones. This situation can happen in multi-network environments
        if the number of events of one network is smaller than others. """
        samps_extension = self.next_candidate.deadline
        samps_remaining = self.list_deadlines[self.map_ass_idx2pos_idx[ass_id]]-\
            self.list_count_events[self.map_ass_idx2pos_idx[ass_id]]
        return samps_remaining<samps_extension
    
    def check_primary_condition_for_extention(self, ass_id):
        """  """
        return self.check_same_direction(ass_id) and self.check_remain_samps(ass_id)
        
        
    def check_secondary_condition_for_extention(self, ass_id, ass_idx, curr_GROI, t):
        '''
        '''
        this_strategy = self.next_candidate.strategy
        margin = 0.5
        if this_strategy.entry_strategy=='fixed_thr':
            condition_extension = (self.next_candidate.p_mc>=this_strategy.lb_mc_ext and 
                              self.next_candidate.p_md>=this_strategy.lb_md_ext and
                              self.next_candidate.p_mc<this_strategy.ub_mc_ext and 
                              self.next_candidate.p_md<this_strategy.ub_md_ext)
        elif this_strategy.entry_strategy=='gre':
            condition_extension = (self.next_candidate.profitability>margin and 
                                  100*curr_GROI>=this_strategy.lim_groi_ext)
        elif this_strategy.entry_strategy=='spread_ranges':
            cond_pmc = self.next_candidate.p_mc>=this_strategy.info_spread_ranges['th'][t][0]
            cond_pmd = self.next_candidate.p_md>=this_strategy.info_spread_ranges['th'][t][1]
            cond_groi = 100*curr_GROI>=this_strategy.lim_groi_ext
            cond_bet = self.direction_map(self.next_candidate.direction, 
                                   this_strategy.info_spread_ranges['dir'])
            condition_extension = cond_pmc and\
                cond_pmd and \
                cond_groi and cond_bet \
                and not force_no_extesion
        else:
            raise ValueError("Wrong entry strategy")            
        reason = ''
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
        return condition_extension, reason

    def update_stoploss(self, idx, bid):
        # update stoploss
        this_strategy = self.next_candidate.strategy
        if self.list_opened_positions[self.map_ass_idx2pos_idx[idx]].direction == 1:
            self.list_stop_losses[self.map_ass_idx2pos_idx[idx]] = (max(
                    self.list_stop_losses[self.map_ass_idx2pos_idx[idx]],
                    bid*(1-self.list_opened_positions[
                    self.map_ass_idx2pos_idx[idx]].direction*this_strategy.thr_sl*
                    this_strategy.fixed_spread_ratio)))
        else:
            self.list_stop_losses[self.map_ass_idx2pos_idx[idx]] = (min(
                    self.list_stop_losses[self.map_ass_idx2pos_idx[idx]],
                    bid*(1-self.list_opened_positions[self.map_ass_idx2pos_idx[
                    idx]].direction*this_strategy.thr_sl*
                    this_strategy.fixed_spread_ratio)))
    
    def is_stoploss_reached(self, lists, datetime, ass_id, bid, em, event_idx, results):
        """ check stop loss reachead """
        direction = self.list_opened_positions[self.map_ass_idx2pos_idx[ass_id]].direction
        if (direction*
            (bid-self.list_stop_losses[self.map_ass_idx2pos_idx[ass_id]])<=0):
            # exit position due to stop loss
            asset = self.list_opened_positions[self.\
                                               map_ass_idx2pos_idx[ass_id]].asset
            self.stoplosses += 1
            out = (asset+" Exit position due to stop loss @event idx "+
                   str(event_idx)+" bid="+str(bid)+" sl="+
                   str(self.list_stop_losses[self.map_ass_idx2pos_idx[ass_id]]))
            print("\r"+out)
            self.write_log(out)
            stoploss_flag = True
            self.ban_currencies(lists, asset, datetime, results, direction)
        else:
            stoploss_flag = False
        return stoploss_flag
    
    def is_takeprofit_reached(self, this_pos, take_profit, takeprofits, bid, event_idx):
        """ check if take-profit threshold has been reached """
        # check take profit reachead
        if this_pos.direction*(bid-take_profit)>=0:
            # exit position due to stop loss
            exit_pos = 1
            takeprofits += 1
            out = "Exit position due to take profit @event idx "+str(event_idx)+\
                ". tp="+str(take_profit)
            print("\r"+out)
            self.write_log(out)
        else:
            exit_pos = 0
        return exit_pos, takeprofits
    
    def get_rois(self, idx, date_time='', roi_ratio=1, ass=''):
        """ Get current GROI and ROI of a given asset idx """
        strategy_name = self.list_opened_positions[self.map_ass_idx2pos_idx[idx]].strategy.name
        direction = self.list_opened_positions[self.map_ass_idx2pos_idx[idx]].direction
        Ti = self.list_opened_positions[self.map_ass_idx2pos_idx[idx]].entry_time
        bet = self.list_opened_positions[self.map_ass_idx2pos_idx[idx]].bet
        Bi = self.list_opened_positions[self.map_ass_idx2pos_idx[idx]].entry_bid
        Ai = self.list_opened_positions[self.map_ass_idx2pos_idx[idx]].entry_ask
        Ao = self.list_last_ask[self.map_ass_idx2pos_idx[idx]][-1]
        Bo = self.list_last_bid[self.map_ass_idx2pos_idx[idx]][-1]
        
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
        
        return GROI_live, ROI_live, spread, info
        
    
    def close_position(self, date_time, ass, idx, results,
                       lot_ratio=None, partial_close=False, from_sl=0):
        """ Close position """
        list_idx = self.map_ass_idx2pos_idx[idx]
        # if it's full close, get the raminings of lots as lots ratio
        if not partial_close:
            lot_ratio = 1.0
        
        roi_ratio = lot_ratio*self.list_lots_per_pos\
            [list_idx]/self.list_lots_entry\
            [list_idx]
        if np.isnan(roi_ratio):
            raise AssertionError("np.isnan(roi_ratio)")
        # get returns
        GROI_live, ROI_live, spread, info = self.get_rois(idx, 
                                                          date_time=date_time,
                                                          roi_ratio=roi_ratio,
                                                          ass=ass)
        
        lots2add = self.list_lots_per_pos[list_idx]*(lot_ratio+ROI_live)
        self.available_bugdet_in_lots = self.get_current_available_budget()+lots2add
        self.available_budget = self.available_bugdet_in_lots*self.LOT 
        # update available budget file
        self.update_current_available_budget()
        
        self.budget_in_lots += self.list_lots_per_pos[list_idx]*ROI_live
                                                    
        nett_win = self.list_lots_entry[list_idx]*ROI_live*self.LOT
        gross_win = self.list_lots_entry[list_idx]*GROI_live*self.LOT
        self.budget += nett_win
        earnings = self.budget-self.init_budget
        self.gross_earnings += gross_win
        self.nett_earnigs += nett_win
        
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
        
        e_spread = self.list_opened_positions[self.map_ass_idx2pos_idx[idx]].e_spread
        # check if close comes from external ban
        if not from_sl and self.list_count_events[self.map_ass_idx2pos_idx[idx]]!=\
            self.list_deadlines[self.map_ass_idx2pos_idx[idx]]:
                from_sl = 2
        # write output to trader summary
        info_close = info+","+str(nett_win)+","+str(e_spread*100*self.pip)+","+\
            str(from_sl)+","+str(100*self.tGROI_live)+","+str(100*self.tROI_live)
        write_log(info_close, self.log_positions_soll)
        pos_filename = get_positions_filename(ass, self.list_last_dt[list_idx][0], 
                                              self.list_last_dt[list_idx][-1])
        # save position evolution
        self.track_position('close', date_time, idx=idx, groi=GROI_live, filename=pos_filename)
        results.save_pos_evolution(pos_filename, self.list_last_dt[list_idx],
                                   self.list_last_bid[list_idx], 
                                   self.list_last_ask[list_idx], 
                                   self.list_EM[list_idx])
        # update output lists
        results.update_outputs(date_time, 100*GROI_live, 100*ROI_live, nett_win)
        
        if not partial_close:
            self.remove_position(idx)
        else:
            # decrease the lot ratio in case the position is not fully closed
            self.list_lots_per_pos[list_idx] = \
                self.list_lots_per_pos[list_idx]*(1-lot_ratio)

        if partial_close:
            partial_string = ' Partial'
        else:
            partial_string = ' Full'
        
        out =( date_time+partial_string+" close "+ass+" Ratio {0:.2f}"\
              .format(lot_ratio)+
              " GROI {2:.3f}% Spread {1:.3f}% ROI = {0:.3f}%".format(
                      100*ROI_live,100*spread,100*GROI_live)+
                      " TGROI {1:.3f}% TROI = {0:.3f}%".format(
                      100*self.tROI_live,100*self.tGROI_live)+
                      " Earnings {0:.2f}".format(earnings)+
                      ". Remeining open "+str(len(self.list_opened_positions)))
        self.write_log(out)
        print("\r"+out)
        
        
        assert(lot_ratio<=1.00 and lot_ratio>0)
    
    def get_current_available_budget(self):
        """ get available budget from shared file among traders """
        fh = open(self.budget_file,"r")
        # read output
        av_bugdet_in_lots = float(fh.read())
        fh.close()
        print("get_current_available_budget: "+str(av_bugdet_in_lots))
        return av_bugdet_in_lots
    
    def update_current_available_budget(self):
        """ update available budget from shared file among traders """
        fh = open(self.budget_file,"w")
        fh.write(str(self.available_bugdet_in_lots))
        fh.close()
        print("update_current_available_budget: "+str(self.available_bugdet_in_lots))
        return None
    
    def open_position(self, idx, lots, DateTime, e_spread, bid, ask, deadline):
        """ Open position """
        # update available budget
        self.available_bugdet_in_lots -= lots
        self.available_budget = self.available_bugdet_in_lots*self.LOT
        # update available budget file
        self.update_current_available_budget()
        #self.available_bugdet_in_lots -= lots
        self.n_entries += 1
        self.n_pos_opened += 1
        
        # update vector of opened positions
        self.add_position(idx, lots, DateTime, bid, ask, deadline)
        # track position
        self.track_position('open', DateTime)
            
        out = (DateTime+" Open "+self.list_opened_positions[-1].asset+
              " Lots {0:.1f}".format(lots)+" "+str(self.list_opened_positions[-1].bet)+
              " p_mc={0:.2f}".format(self.list_opened_positions[-1].p_mc)+
              " p_md={0:.2f}".format(self.list_opened_positions[-1].p_md)+
              " spread={0:.3f} ".format(e_spread)+" strategy "+
              self.list_opened_positions[-1].strategy.name)
        if verbose_trader:
            print("\r"+out)
        self.write_log(out)
        
        
        return None
    
    def track_position(self, event, DateTime, idx=None, groi=0.0, 
                       filename=''):
        """ track position.
        Args:
            - event (str): {open, extend, close} """
        if event=='open':
            pos_info = {'id':0,
                        'n_ext':0,
                        'dts':[DateTime],
                        '@tick#':[0],
                        'grois':[groi],
                        'p_mcs':[self.list_opened_positions[-1].p_mc],
                        'p_mds':[self.list_opened_positions[-1].p_md],
                        'levels':[self.list_opened_positions[-1].bet],
                        'strategy':[self.list_opened_positions[-1].strategy.name]}
            #print(pos_info)
            self.positions_tracker.append(pos_info)
        elif event=='extend':
            #print(self.map_ass_idx2pos_idx[idx])
            #print(self.positions_tracker)
            pos_info = self.positions_tracker[self.map_ass_idx2pos_idx[idx]]
            p_mc = self.list_opened_positions[self.map_ass_idx2pos_idx[idx]].p_mc
            p_md = self.list_opened_positions[self.map_ass_idx2pos_idx[idx]].p_md
            bet = self.list_opened_positions[self.map_ass_idx2pos_idx[idx]].bet
            strategy_name = self.list_opened_positions[self.map_ass_idx2pos_idx[idx]].strategy.name
            tick_counts = self.list_count_all_events[self.map_ass_idx2pos_idx[idx]]
            pos_info['n_ext'] += 1
            pos_info['dts'].append(DateTime)
            pos_info['p_mcs'].append(p_mc)
            pos_info['p_mds'].append(p_md)
            pos_info['levels'].append(bet)
            pos_info['grois'].append(groi)
            pos_info['@tick#'].append(tick_counts)
            pos_info['strategy'].append(strategy_name)
        elif event=='close':
            pos_info = self.positions_tracker[self.map_ass_idx2pos_idx[idx]]
            n_ticks = len(self.list_last_bid[self.map_ass_idx2pos_idx[idx]])
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
            self.available_bugdet_in_lots = self.get_current_available_budget()
            # check if there's enough bugdet available
            if self.available_bugdet_in_lots>0:
                open_lots = min(this_strategy.max_lots_per_pos, 
                                self.available_bugdet_in_lots)
            else:
                open_lots = this_strategy.max_lots_per_pos
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
#            
            # make sure the available resources are smaller or equal than slots to open
            open_lots = min(open_lots,self.available_bugdet_in_lots)
        #print("open_lots corrected "+str(open_lots))
            
        return open_lots
    
    def select_new_entry(self, inputs, thisAsset):
        """ get new entry from inputs comming from network
        Arg:
            - inputs (list): [n_inputs][list, network_index][t_index][input] """
        #print(inputs)
        e_spread = inputs[0][0][0][1]
        DateTime = inputs[0][0][0][2]
        Bi = inputs[0][0][0][3]
        Ai = inputs[0][0][0][4]
        e_spread_pip = e_spread/self.pip
        s_prof = -10000 # -infinite
        for nn in range(len(inputs)):
            network_index = inputs[nn][-1]
            for i in range(len(inputs[nn])-1):
                
                deadline = inputs[nn][i][0][5]
                
                for t in range(len(inputs[nn][i])):
                    soft_tilde = inputs[nn][i][t][0]
                    #t_index = inputs[nn][i][t][6]
                    #print("nn "+str(network_index)+" i "+str(i)+" t "+str(t_index))
                    # get probabilities
                    max_bit_md = int(np.argmax(soft_tilde[1:3]))
                    if not max_bit_md:
                        #Y_tilde = -1
                        Y_tilde = np.argmax(soft_tilde[3:5])-2
                    else:
                        #Y_tilde = 1
                        Y_tilde = np.argmax(soft_tilde[6:])+1
                        
                    p_mc = soft_tilde[0]
                    p_md = np.max([soft_tilde[1],soft_tilde[2]])
                    profitability = self.strategies[network_index].get_profitability(
                            t, p_mc, p_md, int(np.abs(Y_tilde)-1))
                    #print("profitability: "+str(profitability))
                    if profitability>s_prof:
                        s_prof = profitability
                        s_deadline = deadline
                        s_p_mc = p_mc
                        s_p_md = p_md
                        s_Y_tilde = Y_tilde
                        s_network_index = network_index
                        s_t = t
                    # end of for t in range(len(inputs[nn][i])):
            # end of for i in range(len(inputs[nn])-1):
        # end of for nn in range(len(inputs)):
        # add profitabilities
        #print("s_prof: "+str(s_prof))
        new_entry = {}
        new_entry[entry_time_column] = DateTime
        new_entry['Asset'] = thisAsset
        new_entry['Bet'] = s_Y_tilde
        new_entry['P_mc'] = s_p_mc
        new_entry['P_md'] = s_p_md
        new_entry[entry_bid_column] = Bi
        new_entry[entry_ask_column] = Ai
        new_entry['E_spread'] = e_spread_pip
        new_entry['Deadline'] = s_deadline
        new_entry['network_index'] = s_network_index
        new_entry['profitability'] = s_prof
        new_entry['t'] = s_t
        
        return new_entry
    
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
                    if not max_bit_md:
                        #Y_tilde = -1
                        Y_tilde = np.argmax(soft_tilde[3:5])-2
                    else:
                        #Y_tilde = 1
                        Y_tilde = np.argmax(soft_tilde[6:])+1
                        
                    p_mc = soft_tilde[0]
                    p_md = np.max([soft_tilde[1],soft_tilde[2]])
                    if network_name in self.net2strategy:
                        strategy_index = self.net2strategy.index(network_name)
                        #print("str_index in trader")
                        #print(strategy_index)
                        profitability = self.strategies[strategy_index].get_profitability(
                                t, p_mc, p_md, int(np.abs(Y_tilde)-1))
                        if network_name=='327T21E0S':
                            out = "327T21E0S network_name in self.net2strategy"
                            print(out)
                            self.write_log(out)
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
                            
    
    def check_new_inputs(self, inputs, thisAsset, results, directory_MT5_ass=''):
        '''
        <DocString>
        '''
        #new_entry = self.select_new_entry(inputs, thisAsset)
        for new_entry in self.select_next_entry(inputs, thisAsset):
            if not type(new_entry)==list:
                t = new_entry['t']
                strategy_name = self.strategies[new_entry['strategy_index']].name
                # check for opening/extension in order of expected returns
                out = ("New entry @ "+new_entry[entry_time_column]+" "+
                       new_entry['Asset']+
                       " P_mc {0:.3f} ".format(new_entry['P_mc'])+
                       "P_md {0:.3f} ".format(new_entry['P_md'])+
                       "profitability {0:.2f} ".format(new_entry['profitability'])+
                       "Bet {0:d} ".format(new_entry['Bet'])+
                       "E_spread {0:.3f} ".format(new_entry['E_spread'])+
                       "Strategy "+strategy_name)
                if verbose_trader:
                    print("\r"+out)
                    self.write_log(out)
                position = Position(new_entry, self.strategies[new_entry['strategy_index']])
                
                self.add_new_candidate(position)
                ass_idx = self.ass2index_mapping[thisAsset]
                ass_id = self.running_assets[ass_idx]
                
                # open market
                if not self.is_opened(ass_id):
                    # check if condition for opening is met
                    condition_open, reason = self.check_contition_for_opening(t)
                    if condition_open:
                        # assign budget
                        lots = self.assign_lots(new_entry[entry_time_column])
                        # check if there is enough budget
                        if self.available_bugdet_in_lots>=lots:
                            if not run_back_test:
                                self.send_open_command(directory_MT5_ass, ass_idx)
                            self.open_position(ass_id, lots, 
                                               new_entry[entry_time_column], 
                                               self.next_candidate.e_spread, 
                                               self.next_candidate.entry_bid, 
                                               self.next_candidate.entry_ask, 
                                               self.next_candidate.deadline)
                        else: # no opening due to budget lack
                            out = "Not enough budget"
                            print("\r"+out)
                            self.write_log(out)
                            # check swap of resources
                            if self.next_candidate.strategy.entry_strategy=='gre':
    #                             and self.check_resources_swap()
                                # TODO: Check propertly function of swapinng
                                # lauch swap of resourves
                                pass
                                ### WARNING! With asynched traders no swap
                                ### possible
                                #self.initialize_resources_swap(directory_MT5_ass)
                    else:
                        out = new_entry[entry_time_column]+" not opened "+\
                                  thisAsset+" due to "+reason
                        if verbose_trader:
                            print("\r"+out)
                        self.write_log(out)
                else: # position is opened
                    # check for extension
                    if self.check_primary_condition_for_extention(ass_id):
                        curr_GROI, _, _, _ = self.get_rois(ass_id, date_time='', roi_ratio=1)
                        extention, reason = self.check_secondary_condition_for_extention(ass_id, ass_idx, curr_GROI, t)
                        if extention:    
                            # include third condition for thresholds
                            # extend deadline
                            if not run_back_test:
                                self.send_open_command(directory_MT5_ass, ass_idx)
                            self.update_position(ass_id)
                            self.n_pos_extended += 1
                            # track position
                            self.track_position('extend', new_entry[entry_time_column], idx=ass_id, groi=curr_GROI)
                            # print out
                            
                            out = (new_entry[entry_time_column]+" "+
                                   thisAsset+" Extended "+str(self.list_deadlines[
                                       self.map_ass_idx2pos_idx[ass_id]])+" samps "+
#                               " Lots {0:.1f} ".format(self.list_lots_per_pos[
#                                       self.map_ass_idx2pos_idx[ass_id]])+
                               "bet "+str(new_entry['Bet'])+
                               " p_mc={0:.2f}".format(new_entry['P_mc'])+
                               " p_md={0:.2f}".format(new_entry['P_md'])+ 
                               " spread={0:.3f}".format(new_entry['E_spread'])+
                               " strategy "+strategy_name+
                               " current GROI = {0:.2f}%".format(100*curr_GROI))
                            #out = new_entry[entry_time_column]+" Extended "+thisAsset
                            if verbose_trader:
                                print("\r"+out)
                            self.write_log(out)
                        else: # if candidate for extension does not meet requirements
                            out = new_entry[entry_time_column]+" not extended "+\
                                  thisAsset+" due to "+reason
                            if verbose_trader:
                                print("\r"+out)
                            self.write_log(out)
                    else: # if direction is different
                        this_strategy = self.next_candidate.strategy
                        #if this_strategy.if_dir_change_close:
                        if this_strategy.if_dir_change_close and not self.check_same_direction(ass_id) and \
                        this_strategy.entry_strategy=='spread_ranges' and \
                        self.check_same_strategy(ass_id):
#                        self.next_candidate.p_mc>=this_strategy.info_spread_ranges['th'][t][0] and \
#                        self.next_candidate.p_md>=this_strategy.info_spread_ranges['th'][t][1]:
                            # close position due to direction change
                            out = "WARNING! "+new_entry[entry_time_column]+" "+thisAsset\
                            +" closing due to direction change!"
                            if verbose_trader:
                                print("\r"+out)
                            self.write_log(out)
                            if run_back_test:
                                self.close_position(new_entry[entry_time_column], 
                                                    thisAsset, ass_id, results)
                            else:
                                send_close_command(thisAsset)
#                        # if new position has higher GRE, close
#                        if self.next_candidate.profitability>=self.list_opened_positions[
#                                self.map_ass_idx2pos_idx[ass_id]].profitability:
#                            if not run_back_test:
#                                pass
#                            # TODO: check proper function of  close_command
#                                #self.send_close_command(directory_MT5_ass)
#                                # TODO: study the option of not only closing 
#                                #postiion but also changing direction
#                            else:
#                                # TODO: implement it for back test
#                                pass
                    # end of extention options
                # end of if not self.is_opened(ass_id):
            # end of for io in indexes_ordered:    
            else:
                pass
                #print("Network not in Trader. Skipped")
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
    
    def send_open_command(self, directory_MT5_ass, ass_idx):
        """
        Send command for opening position to MT5 software   
        """
        success = 0
        # load network output
        while not success:
            try:
                fh = open(directory_MT5_ass+"TT","w", encoding='utf_16_le')
                fh.write(str(self.next_candidate.direction)+","+
                         str(self.next_candidate.strategy.max_lots_per_pos)+
                         ","+str(self.next_candidate.deadline))
                fh.close()
                success = 1
                #stop_timer(ass_idx)
            except PermissionError:
                print("Error writing TT")
                
    def check_resources_swap(self):
        """
        <DocString>
        """
        min_op = 0
        min_profitability = self.list_opened_positions[min_op].profitability
        for op in range(1,len(self.list_opened_positions)):
            if self.list_opened_positions[op].profitability<min_profitability:
                min_profitability = self.list_opened_positions[op].profitability
                min_op = op
        if self.next_candidate.profitability>min_profitability:    
            self.swap_pos = min_op
            out = "Swap "+self.next_candidate.asset+" with "\
                +self.list_opened_positions[op].asset
            self.write_log(out)
            print(out)
            return True
        else:
            return False
    
    def initialize_resources_swap(self, directory_MT5_ass):
        """
        <DocString>
        """
        out = self.next_candidate.entry_time+" "+\
            self.next_candidate.asset+" initialize swap"
        self.write_log(out)
        print(out)
        #send close command if running live
        if not run_back_test:
            directory_MT5_ass2close = directory_MT5+\
                self.list_opened_positions[self.swap_pos].asset+"/"
            self.send_close_command(directory_MT5_ass2close)
        # activate swap pending flag
        self.swap_pending = 1
        self.new_swap_candidate = self.next_candidate
        self.swap_directory_MT5_ass = directory_MT5_ass
        # if back test, finlaize swap
        if run_back_test:
            self.finalize_resources_swap()
        
    
    def finalize_resources_swap(self):
        """
        <DocString>
        """
        out = self.new_swap_candidate.entry_time+" "+\
            self.new_swap_candidate.asset+" finlaize swap"
        self.write_log(out)
        print(out)
        # open new position
        lots = self.assign_lots(self.new_swap_candidate.entry_time)
        ass_idx = self.running_assets[self.ass2index_mapping\
                                      [self.new_swap_candidate.asset]]
        # check if there is enough budget
        if self.available_bugdet_in_lots>=lots:
            # send open command to MT5 if running live
            if not run_back_test:
                self.send_open_command(self.swap_directory_MT5_ass)
            # open position
            self.open_position(ass_idx, lots, self.new_swap_candidate.entry_time, 
                               self.new_swap_candidate.e_spread, 
                               self.new_swap_candidate.entry_bid, 
                               self.new_swap_candidate.entry_ask, 
                               self.new_swap_candidate.deadline)
            self.swap_pending = 0
    
    def update_list_last(self, list_idx, datetime, bid, ask):
        """ Update bids and asks list with new value """
        # update bid and ask lists if exist
        if list_idx>-1:
            self.list_last_bid[list_idx].append(bid)
            self.list_last_ask[list_idx].append(ask)
            self.list_last_dt[list_idx].append(datetime)
            w = 1-1/20
            em = self.list_EM[list_idx][-1]
            self.list_EM[list_idx].append(w*em+(1-w)*bid)
            
    def save_pos_evolution(self, asset, list_idx):
        """ Save evolution of the position from opening till close """
        # format datetime for filename
        dt_open = dt.datetime.strftime(dt.datetime.strptime(
                        self.list_last_dt[list_idx][0],'%Y.%m.%d %H:%M:%S'),
                '%y%m%d%H%M%S')
        dt_close = dt.datetime.strftime(dt.datetime.strptime(
                        self.list_last_dt[list_idx][-1],'%Y.%m.%d %H:%M:%S'),
                '%y%m%d%H%M%S')
        filename = 'O'+dt_open+'C'+dt_close+asset+'.txt'
        direct = self.dir_positions
        df = pd.DataFrame({'DateTime':self.list_last_dt[list_idx],
                           'SymbolBid':self.list_last_bid[list_idx],
                           'SymbolAsk':self.list_last_ask[list_idx],
                           'EmBid':self.list_EM[list_idx]})
        df.to_csv(direct+filename, index=False)
    
    def ban_currencies(self, lists, thisAsset, DateTime, results, direction):
        """ Ban currency pairs related to ass_idx asset. WARNING! Assets 
        involving GOLD are not supported """
        # WARNING! Ban of only the asset closing stoploss. Change and for or
        # for ban on all assets sharing one currency
        ass_idx = 0
        message = thisAsset+','+str(direction)
        for ass_key in self.AllAssets:
            asset = self.AllAssets[ass_key]
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
                    if self.is_opened(ass_id):
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
                            
                            list_idx = self.map_ass_idx2pos_idx[ass_id]
                            bid = self.list_last_bid[list_idx][-1]
                            self.close_position(DateTime, asset, ass_id, results, from_sl=1)
                            lists = flush_asset(lists, ass_idx, bid)
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
    
    def send_ban_command(self, ban_asset, message):
        """ Send ban command to an asset """
        file = open(self.ban_currencies_dir+self.start_time+ban_asset,"w")
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
                  results_file, model, config, log_file, nonVarIdx, list_inv_out):
    """
    <DocString>
    """
    thr_mc = 0.5
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
    if listFillingX[sc] and verbose_RNN:# and not simulate
        out = tradeInfoLive.DateTime.iloc[-1]+" "+thisAsset+netName+"C"+\
            str(c)+"F"+str(file)
        print("\r"+out)
        write_log(out, log_file)
                
    # launch features extraction
    if init[sc]==False:
        if verbose_RNN:
            out = tradeInfoLive.DateTime.iloc[-1]+" "+thisAsset+netName+\
                " Features inited"
            print("\r"+out)
            write_log(out, log_file)
        listFeaturesLive[sc],listParSarStruct[sc],listEM[sc] = \
            initFeaturesLive_v2(config, tradeInfoLive)
        init[sc] = True

    listFeaturesLive[sc],listParSarStruct[sc],listEM[sc] = extractFeaturesLive_v2\
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
            if verbose_RNN:
                out = tradeInfoLive.DateTime.iloc[-1]+" "+\
                    thisAsset+netName+" Filling X sc "+str(sc)
                print("\r"+out)
                write_log(out, log_file)
                
            if listCountPos[sc]>=seq_len-1:
                if verbose_RNN:
                    out = tradeInfoLive.DateTime.iloc[-1]+" "+\
                        thisAsset+netName+" Filling X sc "+str(sc)+\
                        " done. Waiting for output..."
                    print("\r"+out)
                    write_log(out, log_file)
                listFillingX[sc] = False
        else:
########################################### Predict ###########################            
            if verbose_RNN:
                out = tradeInfoLive.DateTime.iloc[-1]+" "+thisAsset+netName
                print("\r"+out, sep=' ', end='', flush=True)
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
                            
    ################################# Send prediciton to trader ###############
                # get condition to 
                condition = soft_tilde_t[0,0]>thr_mc
          
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
                elif verbose_RNN:
                    Bi = tradeInfoLive.SymbolBid.iloc[-1]
                    Ai = tradeInfoLive.SymbolAsk.iloc[-1]
                    out = thisAsset+netName+\
                        tradeInfoLive.DateTime.iloc[-1]+\
                        " Bi "+str(Bi)+" Ai "+str(Ai)+\
                        " P_mc "+str(soft_tilde_t[0,0])+\
                        " P_md "+str(np.max(soft_tilde_t[0,1:3]))
                    print("\r"+out)
                    write_log(out, log_file)
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
                    
                    if verbose_RNN:
                        out = thisAsset+netName+" Sent DateTime "+\
                              tradeInfoLive.DateTime.iloc[-1]+\
                              " P_mc "+str(list_list_soft_tildes\
                                           [sc][t_index][0][0])+\
                              " P_md "+str(np.max(list_list_soft_tildes\
                                                  [sc][t_index][0][1:3]))
                        print(out)
                        write_log(out, log_file)
                    
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
                            
                if prob_mc>thr_mc:
                    Y_tilde_idx = np.argmax(soft_tilde_t[0,3:])#
                    Y_tilde_check = np.array([(-1)**(1-np.argmax(soft_tilde_t[0,1:3]))]).astype(int)
                else:
                    Y_tilde_idx = int((n_bits_mg-1)/2) # zero index
                    Y_tilde_check = np.array([0])
                Y_tilde = np.array([Y_tilde_idx-(n_bits_mg-1)/2]).astype(int)
                
                if np.sign(Y_tilde)!=np.sign(Y_tilde_check):
                    out = "WARNING! Sign Y_tilde!=sign Y_tilde_check"
                    print("\r"+out)
                    write_log(out, log_file)
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
                    Output_i=(tradeInfoLive.SymbolBid.iloc[-2]-EOF.SymbolBid.iloc[c]
                             )/stds_out[0,lookAheadIndex]
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
                    if p_mc>.5:#pred!=0:
                        
                        # entry ask and bid
                        Ai = EOF.SymbolAsk.iloc[c]
                        Bi = EOF.SymbolBid.iloc[c]
                        # exit ask and bid
                        Ao = tradeInfoLive.SymbolAsk.iloc[-2]
                        Bo = tradeInfoLive.SymbolBid.iloc[-2]
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
                        newEntry['Bi'] = EOF.SymbolBid.iloc[c]
                        newEntry['Ai'] = EOF.SymbolAsk.iloc[c]
                        newEntry['Bo'] = tradeInfoLive.SymbolBid.iloc[-2]
                        newEntry['Ao'] = tradeInfoLive.SymbolAsk.iloc[-2]
                        
                        
                        #resultInfo = pd.DataFrame(columns = columnsResultInfo)
                        resultInfo = pd.DataFrame(newEntry,index=[0])[pd.DataFrame(
                                     columns = columnsResultInfo).columns.tolist()]
                        #if p_mc>.5:
                        resultInfo.to_csv(results_network[t_index]+results_file[t_index],mode="a",
                                              header=False,index=False,sep='\t',
                                              float_format='%.5f')
                        # print entry
                        if verbose_RNN:
                            out = netName+resultInfo.to_string(index=False,\
                                                               header=False)
                            print("\r"+out)
                            write_log(out, log_file)
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

def dispatch(lists, tradeInfo, AllAssets, ass_id, ass_idx, log_file):
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
                                       lists['list_models'][nn],
                                       lists['list_unique_configs'][nn], 
                                       log_file, 
                                       lists['list_nonVarIdx'][nn],
                                       lists['list_inv_out'][nn])
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
                
#    if len(outputs)>1:
#        print("WARNING! Outputs length="+str(len(outputs))+
#              ". No support for multiple outputs at the same time yet.") 
    return outputs, new_outputs

def renew_directories(AllAssets, running_assets, directory_MT5):
    """ Renew MT5 directories """
    for ass_id in running_assets:
        thisAsset = AllAssets[str(ass_id)]
        
        directory_MT5_ass = directory_MT5+thisAsset+"/"
        io_ass_dir = io_dir+thisAsset+"/"
        
        if os.path.exists(io_ass_dir):
            try:
                shutil.rmtree(directory_MT5_ass)
                time.sleep(1)
            except:
                print(thisAsset+" Warning. Error when renewing MT5 directory")
                # TODO: Synch fetcher with current file if MT5 is recording
            
        #try:
        if os.path.exists(io_ass_dir):
            try:
                shutil.rmtree(io_ass_dir)
                print(io_ass_dir+" Directiory renewed")
                time.sleep(1)
            except:
                print(thisAsset+" Warning. Error when renewing IO directory") 
            
        if not os.path.exists(directory_MT5_ass):
            #try:
            os.makedirs(directory_MT5_ass)
            print(directory_MT5_ass+" Directiory created")
            #except:
                #print(directory_MT5_ass+" Warning. Error when creating MT5 directory")
        
        if not os.path.exists(io_ass_dir):
            os.makedirs(io_ass_dir)
            print(io_ass_dir+" Directiory created")

#def start_timer(ass_idx):
#    """  """
#    timers_till_open[ass_idx] = time.time()
#
#def stop_timer(ass_idx):
#    """  """
#    out = "Timer stoped for asset "+\
#        data.AllAssets[str(running_assets[ass_idx])]+" @ "+\
#        str(time.time()-timers_till_open[ass_idx])+" secs"
#    print(out)
#    write_log(out, log_file)
    
def send_close_command(asset):
        """ Send command for closeing position to MT5 software """
        directory_MT5_ass2close = directory_MT5+asset+"/"
        success = 0
        # load network output
        while not success:
            try:
                fh = open(directory_MT5_ass2close+"LC","w", encoding='utf_16_le')
                fh.close()
                success = 1
            except PermissionError:
                print("Error writing LC")
    
def fetch(lists, trader, directory_MT5, 
          AllAssets, running_assets, 
          log_file, results):
    """ Fetch info coming from MT5 """
    
    nAssets = len(running_assets)
    
    print("Fetcher lauched")

    extension = ".txt"
    deli = "_"
    fileExt = [0 for ass in range(nAssets)]
    nFiles = 100
    
    first_info_fetched = False
    
    flag_cl_name = "CL"
    flag_sl_name = "SL"
    nMaxFilesInDir = 0
    tic = time.time()
    run = True
    while run:
#        tic = time.time()
        for ass_idx, ass_id in enumerate(running_assets):
            
            thisAsset = AllAssets[str(ass_id)]
            
            #test_multiprocessing(ass_idx)
            
#            disp = Process(target=test_multiprocessing, args=[ass_idx])
#            disp.start()
            
            directory_MT5_ass = directory_MT5+thisAsset+"/"
            # Fetching buffers
            fileID = thisAsset+deli+str(fileExt[ass_idx])+extension
            success = 0
            
            try:
                buffer = pd.read_csv(directory_MT5_ass+fileID, encoding='utf_16_le')#
                #print(thisAsset+" new buffer received")
                os.remove(directory_MT5_ass+fileID)
                success = 1
                nFilesDir = len(os.listdir(directory_MT5_ass))
                #start_timer(ass_idx)
                if not first_info_fetched:
                    print(thisAsset+" First info fetched")
                    first_info_fetched = True
                elif nMaxFilesInDir<nFilesDir:
                    nMaxFilesInDir = nFilesDir
                    out = "new max number files in dir "+thisAsset+": "+str(nMaxFilesInDir)
                    print(out)
                    write_log(out, log_file)
                    
            except (FileNotFoundError,PermissionError):
                io_ass_dir = io_dir+thisAsset+"/"
                # check shut down command
                if os.path.exists(io_ass_dir+'SD'):
                    print(thisAsset+" Shutting down")
                    os.remove(io_ass_dir+'SD')
                    run = False
                    time.sleep(5*np.random.rand(1))
                time.sleep(.01)
            # update file extension
            if success:
                fileExt[ass_idx] = (fileExt[ass_idx]+1)%nFiles
                # update list
                list_idx = trader.map_ass_idx2pos_idx[ass_id]
                bid = buffer.SymbolBid.iloc[-1]
                ask = buffer.SymbolAsk.iloc[-1]
                DateTime = buffer.DateTime.iloc[-1]
                trader.update_list_last(list_idx, DateTime, bid, ask)
                if list_idx>-1:
                    trader.count_events(ass_id, buffer.shape[0])
                # dispatch
                outputs, new_outputs = dispatch(lists, buffer, AllAssets, 
                                                ass_id, ass_idx, 
                                                log_file)
                ################# Trader ##################
                if new_outputs and not trader.swap_pending:
                    #print(outputs)
                    trader.check_new_inputs(outputs, thisAsset, results,
                                            directory_MT5_ass=directory_MT5_ass)
            
            # check for closing
            flag_cl = 0
            # check if position closed
            if trader.is_opened(ass_id) and os.path.exists(directory_MT5_ass+flag_cl_name):
                
                success = 0
                while not success:
                    try:
                        fh = open(directory_MT5_ass+flag_cl_name,"r",encoding='utf_16_le')
                        # read output
                        info_close = fh.read()[1:-1]
                        #print(info_close)
                        # close file
                        fh.close()
                        
                        flag_cl = 1
                        if len(info_close)>1:
                            os.remove(directory_MT5_ass+flag_cl_name)
                            success = 1
                        else:
                            pass
                            #print("Error in reading file. Length "+str(len(info_close)))
                    except (FileNotFoundError,PermissionError):
                        pass
                
                
            # check for stoploss closing
            flag_sl = 0
            if trader.is_opened(ass_id) and os.path.exists(directory_MT5_ass+flag_sl_name):
                success = 0
                while not success:
                    try:
                        #print(dirOr+flag_name)
                        fh = open(directory_MT5_ass+flag_sl_name,"r",encoding='utf_16_le')
                        # read output
                        info_close = fh.read()[1:-1]
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
                    except (FileNotFoundError,PermissionError):
                        pass

            # check if asset has been banned from outside
            if os.path.exists(trader.ban_currencies_dir+trader.start_time+thisAsset):
                fh = open(trader.ban_currencies_dir+trader.start_time+thisAsset,"r")
                message = fh.read()
                fh.close()
                info = message.split(",")
                otherAsset = info[0]
                otherDirection = int(info[1])
                out = thisAsset+" flag ban from found: "+message[:-1]
                print(out)
                trader.write_log(out)
                # for now, ban only if asset is opened AND they go in same direction
                if trader.is_opened(ass_id):
                    thisDirection = trader.list_opened_positions\
                        [trader.map_ass_idx2pos_idx[ass_id]].direction
                    if trader.assets_same_direction(ass_id, thisAsset, thisDirection, 
                                  otherAsset, otherDirection):
                        lists = flush_asset(lists, ass_idx, 0.0)
                        out = thisAsset+" flushed"
                        print(out)
                        trader.write_log(out)
                        send_close_command(thisAsset)
                    else:
                        out = thisAsset+" NOT flushed due to different directions"
                        print(out)
                        trader.write_log(out)
                else:
                    out = thisAsset+" NOT flushed due to not opened"
                    print(out)
                    trader.write_log(out)
                os.remove(trader.ban_currencies_dir+trader.start_time+thisAsset)
                
                
            # i stoploss
            if flag_cl:
                # update postiions vector
                info_split = info_close.split(",")
                list_idx = trader.map_ass_idx2pos_idx[ass_id]
                bid = float(info_split[6])
                ask = float(info_split[7])
                DateTime = info_split[2]
                # update bid and ask lists if exist
                trader.update_list_last(list_idx, DateTime, bid, ask)
                    
                trader.close_position(DateTime, thisAsset, ass_id, results)
                
                write_log(info_close, trader.log_positions_ist)
                # open position if swap process is on
                if trader.swap_pending:
                    trader.finalize_resources_swap()
                
                
            elif flag_sl:
                # update positions vector
                info_split = info_close.split(",")
                list_idx = trader.map_ass_idx2pos_idx[ass_id]
                bid = float(info_split[6])
                ask = float(info_split[7])
                DateTime = info_split[2]
                direction = int(info_split[3])
                # update bid and ask lists if exist
                trader.update_list_last(list_idx, DateTime, bid, ask)
                    
                #trader.close_position(DateTime, thisAsset, ass_id, results)
                
                trader.stoplosses += 1
                out = (thisAsset+" Exit position due to stop loss "+" sl="+
                       str(trader.list_stop_losses[trader.map_ass_idx2pos_idx[ass_id]]))
                print("\r"+out)
                write_log(out, trader.log_file)
                
                #if not simulate:
                write_log(info_close, trader.log_positions_ist)
                
                lists = trader.ban_currencies(lists, thisAsset, DateTime, 
                                              results, direction)
            
    # end of while run
    budget = get_intermediate_results(trader, AllAssets, running_assets, tic, results)
    return budget

def back_test(DateTimes, SymbolBids, SymbolAsks, Assets, nEvents,
              traders, list_results, running_assets, ass2index_mapping, lists,
              AllAssets, log_file):
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
        
        for idx, trader in enumerate(traders):
            list_idx = trader.map_ass_idx2pos_idx[ass_id]
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
                
                out = thisAsset+" flag ban from found: "+message[:-1]
                print(out)
                trader.write_log(out)
                # for now, ban only if asset is opened AND they go in same direction
                if trader.is_opened(ass_id):
                    thisDirection = trader.list_opened_positions\
                        [trader.map_ass_idx2pos_idx[ass_id]].direction
                    if trader.assets_same_direction(ass_id, thisAsset, thisDirection, 
                                  otherAsset, otherDirection):
                        lists = flush_asset(lists, ass_idx, 0.0)
                        out = thisAsset+" flushed"
                        print(out)
                        trader.write_log(out)
                        trader.close_position(DateTime, thisAsset, ass_id, list_results[idx])
                    else:
                        out = thisAsset+" NOT flushed due to different directions"
                        print(out)
                        trader.write_log(out)
                else:
                    out = thisAsset+" NOT flushed due to not opened"
                    print(out)
                    trader.write_log(out)
                os.remove(trader.ban_currencies_dir+trader.start_time+thisAsset)

        if sampsBuffersCounter[ass_idx]==0:
            outputs, new_outputs = dispatch(lists, buffers[ass_idx], AllAssets, 
                                            ass_id, ass_idx, 
                                            log_file)

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
        io_ass_dir = io_dir+thisAsset+"/"
        # check shut down command
        if os.path.exists(io_ass_dir+'SD'):
            print(thisAsset+" Shutting down")
            os.remove(io_ass_dir+'SD')
            shutdown = True
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
        ###################### End of Trader ###########################
        event_idx += 1
    # end of while events
    for idx, trader in enumerate(traders):
        # get intermediate results
        get_intermediate_results(trader, AllAssets, running_assets, tic, list_results[idx])
    
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
    
def run_carefully(config_trader, running_assets, start_time):
    """  """
    try:
        run(config_trader, running_assets, start_time)
    except KeyboardInterrupt:
        print("KeyboardInterrupt: Exit program organizedly")
    
def run(config_traders_list, running_assets, start_time):
    """  """    
    
    if len(config_traders_list)>1 and not run_back_test:
        raise ValueError("Live execution not compatible with more than one trader")
    # directories
    if run_back_test:
        dir_results = local_vars.RNN_dir+"resultsLive/back_test/"    
    else:
        dir_results = local_vars.RNN_dir+"resultsLive/live/"
    #dir_results_trader = dir_results+"trader/"
    dir_results_netorks = dir_results+'networks/'
    if not os.path.exists(dir_results_netorks):
        os.makedirs(dir_results_netorks)
    dir_log = dir_results+'log/'
    if not os.path.exists(dir_log):
        os.makedirs(dir_log)
    AllAssets = Data().AllAssets
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
    list_filenames = []
    list_unique_configs = []
    list_tags = []
    #log_files = []
    # init tensorflow graph
    tf.reset_default_graph()
    log_file = dir_log+start_time+'_log.log'
    # loop over list with trader configuration files
    for idx_tr, config_trader in enumerate(config_traders_list):
        config_list = config_trader['config_list']
#        log_files.append(log_file)
        config_name = config_trader['config_name']
        print("config_name")
        print(config_name)
        dateTest = config_trader['dateTest']
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
        #model_dict = config_trader['model_dict']
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
#        list_data = [Data(movingWindow=mWs[i],nEventsPerStat=nExSs[i],lB=lBs[i],
#                          dateTest = dateTest,feature_keys_tsfresh=[]) for i in range(numberNetworks)]        
        # add unique networks
        for nn in range(numberNetworks):
            if netNames[nn] not in unique_nets:
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
                movingWindow = configs[0]['movingWindow']
                nEventsPerStat = configs[0]['nEventsPerStat']
                feature_keys_manual = configs[0]['feature_keys_manual']
                n_feats_manual = len(feature_keys_manual)
                noVarFeats = configs[0]['noVarFeatsManual']
                # load stats
                if feats_from_bids:
                    # only get short bets (negative directions)
                    tag = 'IO_mW'
                    tag_stats = 'IOB'
                else:
                    # only get long bets (positive directions)
                    tag = 'IOA_mW'
                    tag_stats = 'IOA'
                print(tag_stats)
                list_tags.append(tag_stats)
                filename_prep_IO = (hdf5_directory+tag+str(movingWindow)+'_nE'+
                                    str(nEventsPerStat)+'_nF'+str(n_feats_manual)+'.hdf5')
                list_filenames.append(filename_prep_IO)
                list_unique_configs.append(configs[0])
            #else:
            list_net2strategy[idx_tr].append(netNames[nn])
        ################# Strategies #############################
#        print(list_thr_sl)
#        print(len(list_thr_tp))
#        print(len(list_fix_spread))
#        print(len(list_fixed_spread_pips))
#        print(len(list_max_lots_per_pos))
#        print(len(list_flexible_lot_ratio))
#        print(len(list_lb_mc_op))
#        print(len(list_lb_mc_ext))
#        print(len(list_lb_md_ext))
#        print(len(list_ub_mc_op))
#        print(len(list_ub_md_op))
#        print(len(list_ub_mc_ext))
#        print(len(list_ub_md_ext))
#        print(len(list_if_dir_change_close))
#        print(len(list_if_dir_change_extend))
#        print(len(list_name))
#        print(len(list_t_indexs))
#        print(len(list_entry_strategy))
#        print(len(IDresults))
#        print(len(IDepoch))
#        print(len(list_weights))
#        print(len(list_spread_ranges))
#        print(len(list_priorities))
#        print(len(list_lim_groi_ext))
        strategies = [Strategy(direct=local_vars.ADsDir,thr_sl=list_thr_sl[i], 
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
        
        results = Results(IDresults, IDepoch, list_t_indexs, list_w_str, start_time,
                          dir_results, config_name=config_name)
        list_results.append(results)
    # end of for idx_tr, config_trader in enumerate(config_list):
    ### MAP CONFIGS TO NETWORKS ###
    nUniqueNetworks = len(unique_nets)
    print("nUniqueNetworks")
    print(nUniqueNetworks)
    print(unique_nets)
    print("list_net2strategy")
    print(list_net2strategy)
    ADs = [np.array([]) for i in range(nUniqueNetworks)]
    
    nChans = (np.array(list_nExS)/np.array(list_mW)).astype(int).tolist()
    #print(unique_IDresults)
    #print(unique_IDepoch)
    resultsDir = [[dir_results_netorks+unique_IDresults[nn]+"T"+
                   str(t)+"E"+''.join(str(e) for e in unique_IDepoch[nn])+"/" 
                   for t in unique_t_indexs[nn]] for nn in range(nUniqueNetworks)]
    #print(resultsDir)
    results_files = [[start_time+'_'+unique_IDresults[nn]+"T"+str(t)+"E"+
                      ''.join(str(e) for e in unique_IDepoch[nn])+".txt" 
                      for t in unique_t_indexs[nn]] for nn in range(nUniqueNetworks)]
    #print(results_files)
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
    
    #print(h5py.File(list_filenames[0],'r')[AllAssets[str(running_assets[0])]])
#    f_prep_IO = h5py.File(filename_prep_IO,'r')
    list_stats_feats = [[load_stats_manual_v2(list_unique_configs[nn], AllAssets[str(running_assets[ass])], 
                    None, 
                    from_stats_file=True, hdf5_directory=hdf5_directory+'stats/',tag=list_tags[nn]) 
                    for nn in range(nNets)] for ass in range(nAssets)]
    list_stats_rets = [[load_stats_output_v2(list_unique_configs[nn], hdf5_directory+
                    'stats/', AllAssets[str(running_assets[ass])], 
                    tag=tag_stats) for nn in range(nNets)] for ass in range(nAssets)]
    if test:
        gain = .000000001
        
    else: 
        gain = 1
    
    list_means_in =  [[list_stats_feats[ass][nn]['means_t_in'] for nn in range(nNets)] 
                                                             for ass in range(nAssets)]
    list_stds_in =  [[gain*list_stats_feats[ass][nn]['stds_t_in'] for nn in range(nNets)] 
                                                                for ass in range(nAssets)]
    list_stds_out =  [[gain*list_stats_rets[ass][nn]['stds_t_out'] for nn in range(nNets)] 
                                                                  for ass in range(nAssets)]
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
    ass_index = 0
    
    for ass in running_assets:
        ass2index_mapping[AllAssets[str(ass)]] = ass_index
        ass_index += 1
    
    tic = time.time()
    
    ################### RNN ###############################
    
    
    
    if 1:
        shutdown = False
        if run_back_test:
            day_index = 0
            #t_journal_entries = 0
            while day_index<len(dateTest) and not shutdown:
                counter_back = 0
                init_list_index = day_index#data.dateTest[day_index]
                
                # find sequence of consecutive days in test days
                while (day_index<len(dateTest)-1 and dt.datetime.strptime(
                        dateTest[day_index],'%Y.%m.%d')+dt.timedelta(1)==
                        dt.datetime.strptime(dateTest[day_index+1],'%Y.%m.%d')):
                    day_index += 1
                
                end_list_index = day_index+counter_back
                out = "Week from "+dateTest[init_list_index]+" to "+dateTest[end_list_index]
                print(out)
                
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
                lists['netNames'] = unique_netNames
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
                lists['list_models'] = list_models
                #lists['list_data'] = list_data
                lists['list_nonVarIdx'] = list_nonVarIdx
                lists['list_inv_out'] = unique_inv_out
                #lists['list_feats_from'] = unique_feats_from
                
                traders = []
                for idx_tr, config_trader in enumerate(config_traders_list):
                    trader = Trader(running_assets,
                                    ass2index_mapping, list_strategies[idx_tr], AllAssets, 
                                    log_file, results_dir=dir_results, 
                                    start_time=start_time, config_name=config_trader['config_name'],
                                    net2strategy=list_net2strategy[idx_tr])
                    
                    if not os.path.exists(trader.log_file):
                        write_log(out, trader.log_file)
                        write_log(out, trader.log_summary)
                    
                    traders.append(trader)
                    
                DateTimes, SymbolBids, SymbolAsks, Assets, nEvents = \
                    load_in_memory(running_assets, AllAssets, dateTest, init_list_index, 
                                   end_list_index, root_dir=data_dir)
                shutdown = back_test(DateTimes, SymbolBids, SymbolAsks, 
                                        Assets, nEvents ,
                                        traders, list_results, running_assets, 
                                        ass2index_mapping, lists, AllAssets, 
                                        log_file)
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
            lists['list_models'] = list_models
            #lists['list_data'] = list_data
            lists['list_nonVarIdx'] = list_nonVarIdx
            lists['list_inv_out'] = list_inv_out
            #lists['list_feats_from'] = list_feats_from
            # init traders
            traders = []
            for idx_tr, config_trader in enumerate(config_list):
                trader = Trader(running_assets,
                                ass2index_mapping, list_strategies[idx_tr], AllAssets, 
                                log_file, results_dir=dir_results, 
                                start_time=start_time, config_name=config_trader['config_name'],
                                net2strategy=list_net2strategy[idx_tr])
                    
#                if not os.path.exists(trader.log_file):
#                    write_log(out, trader.log_file)
#                    write_log(out, trader.log_summary)
                    
                traders.append(trader)
            
#            trader = Trader(running_assets,
#                                ass2index_mapping, strategies, AllAssets, 
#                                log_file, results_dir=dir_results, 
#                                start_time=start_time)
            # launch fetcher
            fetch(lists, trader, directory_MT5, AllAssets, 
                  running_assets, log_file, results)
        
        for idx, trader in enumerate(traders):
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
        
def launch(*ins):
    # runLive in multiple processes
    from multiprocessing import Process
    import datetime as dt
    import time
    from kaissandra.config import retrieve_config
    
    if len(ins)>0:
        list_config_traders = [retrieve_config(config_name) for config_name in ins]
#        for config_name in ins:
#            #config_trader = retrieve_config(ins[0])
#            list_config_traders.append(retrieve_config(config_name))
    else:
        list_config_traders = [retrieve_config('T01010-1k1k2')]
    synchroned_run = False
    assets = [1,2,3,4,7,8,10,11,12,13,14,15,16,17,19,27,28,29,30,31,32]#[1]
    running_assets = assets#[12,7,14]#
    renew_directories(Data().AllAssets, running_assets, directory_MT5)
    start_time = dt.datetime.strftime(dt.datetime.now(),'%y_%m_%d_%H_%M_%S')
    if synchroned_run:
        run(list_config_traders, running_assets, start_time)
#        disp = Process(target=run, args=[running_assets,start_time])
#        disp.start()
    else:
        for ass_idx in range(len(running_assets)):
            disp = Process(target=run_carefully, args=[list_config_traders, running_assets[ass_idx:ass_idx+1],start_time])
            disp.start()
            time.sleep(2)
        time.sleep(30)
    print("All RNNs launched")
if __name__=='__main__':
    
        launch()
        
#
#GROI = -0.668% ROI = -1.028% Sum GROI = -0.668% Sum ROI = -1.028% Final budget 9897.22E Earnings -102.78E per earnings -1.028% ROI per position -0.029%
#Number entries 36 per entries 0.00% per net success 36.111% per gross success 44.444% av loss 0.071% per sl 0.000%
#DONE. Time: 98.29 mins

# .6/.6 real spread<.04 approx 03.09-06.09
#GROI = -0.315% ROI = -1.155% Sum GROI = -0.315% Sum ROI = -1.155% Final budget 9884.52E Earnings -115.48E per earnings -1.155% ROI per position -0.030%
#Number entries 39 per entries 0.00% per net success 33.333% per gross success 51.282% av loss 0.069% per sl 0.000%
#DONE. Time: 14.07 mins

# .6/.6 real spread<.04 exact 03.09-06.09
#GROI = -0.673% ROI = -1.472% Sum GROI = -0.673% Sum ROI = -1.472% Final budget 9852.80E Earnings -147.20E per earnings -1.472% ROI per position -0.041%
#Number entries 36 per entries 0.00% per net success 25.000% per gross success 41.667% av loss 0.066% per sl 0.000%
#DONE. Time: 106.78 mins
        
# GRE real spread<.04 approx 03.09-06.09
#GROI = 0.272% ROI = 0.112% Sum GROI = 0.272% Sum ROI = 0.112% Final budget 10011.18E Earnings 11.18E per earnings 0.112% ROI per position 0.012%
#Number entries 9 per entries 0.00% per net success 77.778% per gross success 77.778% av loss 0.126% per sl 0.000%
#DONE. Time: 7.07 mins

# .6/.7 real spread<.04 approx 03.09-06.09
#GROI = 0.209% ROI = -0.199% Sum GROI = 0.209% Sum ROI = -0.199% Final budget 9980.06E Earnings -19.94E per earnings -0.199% ROI per position -0.012%
#Number entries 17 per entries 0.00% per net success 52.941% per gross success 64.706% av loss 0.081% per sl 0.000%
#DONE. Time: 7.13 mins

# .6/.7 real spread<.04 approx 27.11.17-10.08.18
#Total GROI = 7.220% Total ROI = 3.457% Sum GROI = 7.220% Sum ROI = 3.457% Accumulated earnings 345.70E

# GRE exact 03.09.18-10.09.18
#DONE. Time: 175.96 mins
#Total GROI = 0.235% Total ROI = -0.051% Sum GROI = 0.235% Sum ROI = -0.051% Accumulated earnings -5.14E