# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 13:52:57 2018

@author: mgutierrez
Script to simulate trader

Advice from Brian Chesky: Get 100 people to love your product! Originaly from Paul Graham :)
Word of mouth
"""

import os
import re
import time
import pandas as pd
import datetime as dt
import numpy as np
from kaissandra.inputs import Data
from kaissandra.local_config import local_vars
import pickle

#from TradingManager_v10 import write_log

class Results:
    
    def __init__(self):
        
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
        
        self.datetimes = []
        self.number_entries = np.array([])
        
        self.net_successes = np.array([])
        self.total_losses = np.array([])
        self.total_wins = np.array([])
        self.gross_successes = np.array([])
        self.number_stoplosses = np.array([])
        
        self.sl_levels = np.array([5, 10, 15, 20])
        self.n_slots = 10
        self.expectations_matrix = np.zeros((self.sl_levels.shape[0], self.n_slots))
        
        self.n_entries = 0
        
    def update_results(self):
        
        self.n_entries += trader.n_entries
        
        self.sum_GROI += 100*trader.tGROI_live
        self.sum_GROIs_week = np.append(self.sum_GROIs_week, 100*trader.tGROI_live)
        
        self.sum_ROI += 100*trader.tROI_live
        self.sum_ROIs_week = np.append(self.sum_ROIs_week, 100*trader.tROI_live)
        
        self.total_GROI += 100*GROI
        self.GROIs_week = np.append(self.GROIs_week, 100*GROI)
        
        self.total_ROI += 100*ROI
        self.ROIs_week = np.append(self.ROIs_week, 100*ROI)
        
        self.total_earnings += earnings
        self.earnings_week = np.append(self.earnings_week, earnings)
        
        self.number_entries = np.append(self.number_entries, trader.n_entries)
        self.net_successes = np.append(self.net_successes, trader.net_successes)
        self.total_losses = np.append(self.total_losses, np.abs(trader.average_loss))
        self.total_wins = np.append(self.total_wins, np.abs(trader.average_win))
        self.gross_successes = np.append(self.gross_successes, trader.gross_successes)
        self.number_stoplosses = np.append(self.number_stoplosses, stoplosses)
        

class Position:
    """
    Class containing market position info.
    """
    pos_str = {'-1':'short','1':'long'}
    #stop_loss = 0.0 # never reacheable
    #take_profit = 1000000 # never reachable
    ROI = 0.0
    
    def __init__(self, journal_entry, AD_resume, eROIpb):
        
        self.asset = journal_entry['Asset'].encode("utf-8")
        self.entry_time = dt.datetime.strptime(journal_entry[entry_time_column],
                                               '%Y.%m.%d %H:%M:%S')
        self.exit_time = dt.datetime.strptime(journal_entry[exit_time_column],
                                              '%Y.%m.%d %H:%M:%S')
        self.entry_bid = int(np.round(journal_entry[entry_bid_column]*100000))/100000
        self.exit_bid = int(np.round(journal_entry[exit_bid_column]*100000))/100000
        self.entry_ask = int(np.round(journal_entry[entry_ask_column]*100000))/100000
        self.exit_ask = int(np.round(journal_entry[exit_ask_column]*100000))/100000
        self.bet = int(journal_entry['Bet'])
        self.direction = np.sign(self.bet)
        self.p_mc = journal_entry['P_mc']
        self.p_md = journal_entry['P_md']
        self.ADm = self._get_ADm(AD_resume)
        self.strategy = journal_entry.strategy
#        self.idx_mc = strategys[name2str_map[self.strategy]]._get_idx(self.p_mc)
#        self.idx_md = strategys[name2str_map[self.strategy]]._get_idx(self.p_md)
                
        
    def _get_ADm(self, AD_resume):
        if AD_resume!=None:
            return AD_resume[str(int(np.floor(self.p_mc*10)))+
                             str(int(np.floor(self.p_md*10)))]
        else:
            return None
    
    def _get_eROI(self, journal_entry, eROIpb):
        # find threshold it belongs to
        if journal_entry.P_mc>0.5 and journal_entry.P_mc<=0.6:
            idx_mc = 0
        elif journal_entry.P_mc>0.6 and journal_entry.P_mc<=0.7:
            idx_mc = 1
        elif journal_entry.P_mc>0.7 and journal_entry.P_mc<=0.8:
            idx_mc = 2
        elif journal_entry.P_mc>0.8 and journal_entry.P_mc<=0.9:
            idx_mc = 3
        elif journal_entry.P_mc>0.9:
            idx_mc = 4
        
            
        # find threshold md it belongs to
        if journal_entry.P_md>0.5 and journal_entry.P_md<=0.6:
            idx_md = 0
        elif journal_entry.P_md>0.6 and journal_entry.P_md<=0.7:
            idx_md = 1
        elif journal_entry.P_md>0.7 and journal_entry.P_md<=0.8:
            idx_md = 2
        elif journal_entry.P_md>0.8 and journal_entry.P_md<=0.9:
            idx_md = 3
        elif journal_entry.P_md>0.9:
            idx_md = 4
        if eROIpb.shape[0]>idx_mc and eROIpb.shape[1]>idx_mc:
            print(eROIpb)
            print(idx_mc)
            print(idx_md)
            return eROIpb[idx_mc, idx_md, int(np.abs(journal_entry.Bet))+1]
        else:
            return eROIpb[0, 0, int(np.abs(journal_entry.Bet))+1]
    
#    def _get_lots(self, journal_entry):
#        # find threshold it belongs to
#        lots = 3
#        #print(lots)
#        return lots
        
class Strategy():
    
    def __init__(self, direct='', thr_sl=1000, lim_groi_ext=-.1, thr_tp=1000, fix_spread=False, 
                 fixed_spread_pips=2, max_lots_per_pos=.1, flexible_lot_ratio=False, 
                 lb_mc_op=0.6, lb_md_op=0.6, lb_mc_ext=0.6, lb_md_ext=0.6, 
                 ub_mc_op=1, ub_md_op=1, ub_mc_ext=1, ub_md_ext=1,
                 if_dir_change_close=False, if_dir_change_extend=False, 
                 name='',t_index=3,IDr=None,IDgre=None,epoch='11',p_md_margin=0.02,
                 weights=np.array([0,1]),info_spread_ranges=[],entry_strategy='fixed_thr',
                 extend_for_any_thr=True):
        
        self.name = name
        self.dir_origin = direct
        
        self.thr_sl = thr_sl
        self.thr_tp = thr_tp
        self.lim_groi_ext = lim_groi_ext
        
        self.fix_spread = fix_spread
        self.pip = 0.0001
        self.fixed_spread_ratio = fixed_spread_pips*self.pip
        
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
        self.extend_for_any_thr = extend_for_any_thr
        
        self.if_dir_change_close = if_dir_change_close
        self.if_dir_change_extend = if_dir_change_extend
        # strategies
        self.entry_strategy = entry_strategy
        # load GRE
        self.IDr = IDr
        self.IDgre = IDgre
        self.epoch = epoch
        self.t_index = t_index
        self.weights = weights
        self._load_GRE()
        # spread range strategy
        
        if 'mar' not in info_spread_ranges:
            info_spread_ranges['mar'] = [(0.0,0.0) for _ in range(len(info_spread_ranges['th']))]
        self.info_spread_ranges = info_spread_ranges
        self.p_md_margin = p_md_margin
        
    def _load_GRE(self):
        """ Load strategy efficiency matrix GRE """
        # shape GRE: (model.seq_len+1, len(thresholds_mc), len(thresholds_md), 
        #int((model.size_output_layer-1)/2))
        if self.entry_strategy=='gre':
            assert(np.sum(self.weights)==1)
            allGREs = pickle.load( open( self.dir_origin+self.IDgre+
                                        "/GRE_e"+self.epoch+".p", "rb" ))
            GRE = allGREs[self.t_index, :, :, :]/self.pip
            print("GRE level 1:")
            print(GRE[:,:,0])
            print("GRE level 2:")
            print(GRE[:,:,1])
            if os.path.exists(self.dir_origin+self.IDgre+"/GREex_e"+self.epoch+".p"):
                allGREs = pickle.load( open( self.dir_origin+self.IDgre+
                                            "/GREex_e"+self.epoch+".p", "rb" ))
                GREex = allGREs[self.t_index, :, :, :]/self.pip
                print("GREex level 1:")
                print(GREex[:,:,0])
                print("GREex level 2:")
                print(GREex[:,:,1])
            else: 
                GREex = 1-self.weights[0]*GRE
            
            self.GRE = self.weights[0]*GRE+self.weights[1]*GREex
            print("GRE combined level 1:")
            print(self.GRE[:,:,0])
            print("GRE combined level 2:")
            print(self.GRE[:,:,1])
        elif self.entry_strategy=='gre_v2':
            # New GRE implementation
            gre_list = pickle.load( open( local_vars.gre_directory+self.IDgre+".p", "rb" ))
            self.GRE = gre_list[0]
            
            self.gre_model = gre_list[1]
            if len(gre_list)>2:
                # weighted LLS
                print("weighted LLS model")
                self.gre_model = gre_list[2]
            #print(self.gre_model)
            self.resolution_mc = self.GRE.shape[0]
            self.resolution_md = int(2*self.GRE.shape[1])
        else:
            self.GRE = None
    
    def _get_idx(self, p):
        """ Get probability index """
#        idx = max(int(np.floor(p*2*self.resolution)-self.resolution),0)
        if p>=0.5 and p<=0.6:
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
            print(p)
            print("WARNING: p<0.5")
            idx = 0
            
        return idx
    
    def _get_idx_pc(self, p):
        """ Get probability index """
        idx = max(int(np.floor(p*self.resolution_mc)),0)
        
        return idx
    
    def _get_idx_pm(self, p):
        """ Get probability index """
        idx = max(int(np.floor(p*self.resolution_md)-self.resolution_md/2),0)
            
        return idx
    
    def get_profitability(self, p_mc, p_md, level):
        """ """
        if self.entry_strategy=='gre':
            
            return self.GRE[self._get_idx(p_mc), self._get_idx(p_md), level]
        elif self.entry_strategy=='gre_v2':
            return self.GRE[self._get_idx_pc(p_mc), self._get_idx_pm(p_md), level]
#            return self.gre_model.predict(np.array([[p_mc, p_md-self.p_md_margin, level]]))[0]
        else:
            return 0.0

class Trader:
    
    def __init__(self, next_candidate, init_budget=10000,log_file='',summary_file='',
                 positions_dir='',positions_file='', allow_candidates=False):
        """  """
        self.list_opened_positions = []
        self.map_ass_idx2pos_idx = np.array([-1 for i in range(len(data.AllAssets))])
        self.list_count_events = []
        self.list_stop_losses = []
        self.list_lim_groi = []
        self.list_take_profits = []
        self.list_lots_per_pos = []
        self.list_lots_entry = []
        self.list_sps = []
        self.list_last_bid = []
        self.list_last_ask = []
        self.list_sl_thr_vector = []
        self.list_EM = []
        self.list_banned_counter = np.zeros((len(data.assets)))-1
        self.list_is_asset_banned = [False for _ in data.assets]
        
        self.next_candidate = next_candidate
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
        self.flexible_lot_ratio = False
        self.margin_open = 2 # gre product margin
#        self.margin_p_mc = .05 # spread_ranges p_mc opening margin
#        self.margin_p_md = .05 # spread_ranges p_mc opening margin
        
        self.tROI_live = 0.0
        self.tGROI_live = 0.0
        self.GROIs = np.array([])
        self.ROIs = np.array([])
        
        self.net_successes = 0 
        self.average_loss = 0.0
        self.average_win = 0.0
        self.gross_successes = 0
        self.n_pos_extended = 0
        self.n_entries = 0
        self.allow_candidates = allow_candidates
        
        self.save_log = 1
        if log_file=='':
            start_time = dt.datetime.strftime(dt.datetime.now(),'%y_%m_%d_%H_%M_%S')
            self.log_file = (local_vars.live_results_dict+"simulate/trader/"+
                             start_time+"trader_v30.log")
            self.positions_dir = local_vars.live_results_dict+"simulate/trader/positions/"
        else:
            self.log_file = log_file
        if summary_file=='':
            self.summary_file = (local_vars.live_results_dict+"simulate/trader/"+
                             start_time+"summary.log")
        else:
            self.summary_file = summary_file
            self.positions_dir = positions_dir
        self.positions_file = positions_file
    
    def _get_thr_sl_vector(self):
        """  """
        return self.next_candidate.entry_bid*(1-self.next_candidate.direction*
                                              self.sl_thr_vector*self.pip)
    
    def add_position(self, idx, lots, sp):
        """  """
        self.list_opened_positions.append(self.next_candidate)
        self.list_count_events.append(0)
        self.list_lots_per_pos.append(lots)
        self.list_lots_entry.append(lots)
        self.list_sps.append(sp)
        self.list_last_bid.append(bid)
        self.list_last_ask.append(ask)
        self.list_EM.append(bid)
        self.map_ass_idx2pos_idx[idx] = len(self.list_count_events)-1
        self.list_stop_losses.append(self.next_candidate.entry_bid*
                                     (1-self.next_candidate.direction*strategys
                                      [name2str_map[
                                              self.next_candidate.strategy]].thr_sl*
                                              self.pip))
        self.list_take_profits.append(self.next_candidate.entry_bid*
                                      (1+self.next_candidate.direction*
                                       strategys[name2str_map[
                                               self.next_candidate.strategy]].thr_tp
                                               *self.pip))
        self.list_lim_groi.append(strategys[name2str_map[
                                               self.next_candidate.strategy]].lim_groi_ext)
        
    def add_candidate(self):
        """  """
        pass
    
    def remove_position(self, idx):
        """  """
        self.list_opened_positions = self.list_opened_positions[
                :self.map_ass_idx2pos_idx[idx]]+self.list_opened_positions[
                        self.map_ass_idx2pos_idx[idx]+1:]
        self.list_count_events = self.list_count_events[
                :self.map_ass_idx2pos_idx[idx]]+self.list_count_events[
                        self.map_ass_idx2pos_idx[idx]+1:]
        self.list_stop_losses = self.list_stop_losses[
                :self.map_ass_idx2pos_idx[idx]]+self.list_stop_losses[
                        self.map_ass_idx2pos_idx[idx]+1:]
        self.list_take_profits = self.list_take_profits[
                :self.map_ass_idx2pos_idx[idx]]+self.list_take_profits[
                        self.map_ass_idx2pos_idx[idx]+1:]
        self.list_lim_groi = self.list_lim_groi[
                :self.map_ass_idx2pos_idx[idx]]+self.list_lim_groi[
                        self.map_ass_idx2pos_idx[idx]+1:]
        self.list_lots_per_pos = self.list_lots_per_pos[
                :self.map_ass_idx2pos_idx[idx]]+self.list_lots_per_pos[
                        self.map_ass_idx2pos_idx[idx]+1:]
        self.list_lots_entry = self.list_lots_entry[
                :self.map_ass_idx2pos_idx[idx]]+self.list_lots_entry[
                        self.map_ass_idx2pos_idx[idx]+1:]
        self.list_sps = self.list_sps[
                :self.map_ass_idx2pos_idx[idx]]+self.list_sps[
                        self.map_ass_idx2pos_idx[idx]+1:]
        self.list_last_bid = self.list_last_bid[
                :self.map_ass_idx2pos_idx[idx]]+self.list_last_bid[
                        self.map_ass_idx2pos_idx[idx]+1:]
        self.list_last_ask = (self.list_last_ask[:self.map_ass_idx2pos_idx[idx]]+
                              self.list_last_ask[self.map_ass_idx2pos_idx[idx]+1:])
        self.list_EM = self.list_EM\
            [:self.map_ass_idx2pos_idx[idx]]+self.list_EM\
            [self.map_ass_idx2pos_idx[idx]+1:]
        mask = self.map_ass_idx2pos_idx>self.map_ass_idx2pos_idx[idx]
        self.map_ass_idx2pos_idx[idx] = -1
        self.map_ass_idx2pos_idx = self.map_ass_idx2pos_idx-mask*1#np.maximum(,-1)
        
    def update_position(self, idx):
        # reset counter
        self.list_count_events[self.map_ass_idx2pos_idx[idx]] = 0
        
        entry_bid = self.list_opened_positions[self.map_ass_idx2pos_idx[idx]].entry_bid
        entry_ask = self.list_opened_positions[self.map_ass_idx2pos_idx[idx]].entry_ask
        direction = self.list_opened_positions[self.map_ass_idx2pos_idx[idx]].direction
        bet = self.list_opened_positions[self.map_ass_idx2pos_idx[idx]].bet
        p_mc = self.list_opened_positions[self.map_ass_idx2pos_idx[idx]].p_mc
        p_md = self.list_opened_positions[self.map_ass_idx2pos_idx[idx]].p_md
        entry_time = self.list_opened_positions[self.map_ass_idx2pos_idx[idx]].entry_time
        #entry_time = self.list_opened_positions[self.map_ass_idx2pos_idx[idx]].entry_time
        
        self.list_opened_positions[self.map_ass_idx2pos_idx[idx]] = self.next_candidate
        
        self.list_opened_positions[self.map_ass_idx2pos_idx[idx]].entry_bid = entry_bid
        self.list_opened_positions[self.map_ass_idx2pos_idx[idx]].entry_ask = entry_ask
        self.list_opened_positions[self.map_ass_idx2pos_idx[idx]].direction = direction
        self.list_opened_positions[self.map_ass_idx2pos_idx[idx]].bet = bet
        self.list_opened_positions[self.map_ass_idx2pos_idx[idx]].p_mc = p_mc
        self.list_opened_positions[self.map_ass_idx2pos_idx[idx]].p_md = p_md
        self.list_opened_positions[self.map_ass_idx2pos_idx[idx]].entry_time = entry_time
        #self.list_opened_positions[self.map_ass_idx2pos_idx[idx]].entry_time = entry_time
    
    def is_opened(self, idx):
        if self.map_ass_idx2pos_idx[idx]>=0:
            return True
        else:
            return False
    
    def count_one_event(self, idx):
        self.list_count_events[self.map_ass_idx2pos_idx[idx]] += 1
                
    def update_candidates(self):
        rewind = 0
        if self.journal_idx<journal.shape[0]-1:
            self.journal_idx += 1
            new_pos = Position(journal.iloc[self.journal_idx], AD_resume, eROIpb)
            EXIT = 0
            if (new_pos.entry_time==self.next_candidate.entry_time and 
                self.journal_idx<journal.shape[0]-1):
                rewind = 1
            self.next_candidate = new_pos
            if self.journal_idx==journal.shape[0]-1:
                # no more journal entries. Exit
                print("EXIT NEXT")
    #            EXIT = 1
            else:# otherwise, update next position
                pass
        elif self.next_candidate!=None:
            self.last_fix_spread = strategys[name2str_map[
                    self.next_candidate.strategy]].fix_spread
            self.last_fixed_spread_ratio = strategys[
                    name2str_map[self.next_candidate.strategy]].fixed_spread_ratio
            self.next_candidate = None
            EXIT = 1
        else:
            EXIT = 1
            print("EXIT")
        return EXIT, rewind
    
    def chech_ground_condition_for_opening(self):
        
        ground_condition = (self.next_candidate!= None and 
                            time_stamp==self.next_candidate.entry_time and 
                            Assets[event_idx]==self.next_candidate.asset)
        return ground_condition
    
    def check_primary_condition_for_opening(self):
        
        first_condition_open = (self.next_candidate!= None and 
                                bid==self.next_candidate.entry_bid and 
                                ask==self.next_candidate.entry_ask)
        return first_condition_open
    
    def check_secondary_contition_for_opening(self):
        
        margin = 0.5
        this_strategy = strategys[name2str_map[self.next_candidate.strategy]]
        if this_strategy.entry_strategy=='fixed_thr':
            second_condition_open = (self.next_candidate!= None and 
                                     self.next_candidate.p_mc>=this_strategy.lb_mc_op and 
                                    self.next_candidate.p_md>=this_strategy.lb_md_op and
                                     self.next_candidate.p_mc<this_strategy.ub_mc_op and 
                                    self.next_candidate.p_md<this_strategy.ub_md_op)
            
        elif this_strategy.entry_strategy=='gre':
            second_condition_open = (self.next_candidate!= None and 
                                     this_strategy.get_profitability(
                                             self.next_candidate.p_mc, 
                                    self.next_candidate.p_md, 
                                    int(np.abs(self.next_candidate.bet
                                    )-1))>e_spread/self.pip+margin)
        elif this_strategy.entry_strategy=='gre_v2':
            second_condition_open = (self.next_candidate!= None and 
                                     this_strategy.get_profitability(
                                             self.next_candidate.p_mc, 
                                    self.next_candidate.p_md, 
                                    int(np.abs(self.next_candidate.bet
                                    )-1))>self.margin_open*e_spread/self.pip)
        elif this_strategy.entry_strategy=='spread_ranges':
#            print("\n\n\n")
#            print(e_spread/self.pip)
#            print("\n\n\n")
            for t in range(len(this_strategy.info_spread_ranges['th'])):
                second_condition_open = self.next_candidate.p_mc>=\
                this_strategy.info_spread_ranges['th'][t][0]+\
                this_strategy.info_spread_ranges['mar'][t][0] and\
                self.next_candidate.p_md>=this_strategy.info_spread_ranges['th'][t][1]+\
                this_strategy.info_spread_ranges['mar'][t][1] and\
                e_spread/self.pip<=this_strategy.info_spread_ranges['sp'][t]
                if second_condition_open:
                    sp = this_strategy.info_spread_ranges['sp'][t]
                    return second_condition_open, sp
                    
        else:
            raise ValueError("Unknown entry strategy")
            
        return second_condition_open, -1
    
#    def check_asset_not_banned(self):
#        """  """
#        condition = self.list_banned_counter[ass_idx]==-1
##        if not condition:
##            out = data.AllAssets[str(data.assets[ass_idx])]+" is banned"
##            self.write_log(out)
##            print(out)
#        return condition
        
    def check_primary_condition_for_extention(self, time_stamp, idx):
        """
        
        """
        return (self.next_candidate.asset==self.list_opened_positions[
                self.map_ass_idx2pos_idx[idx]].asset and 
            time_stamp==self.next_candidate.entry_time)

    def check_secondary_condition_for_extention(self, idx, curr_GROI):
        """
            
        """
        margin = 0.5
        this_strategy = strategys[name2str_map[self.next_candidate.strategy]]
        
        dir_condition = self.list_opened_positions[
                        self.map_ass_idx2pos_idx[idx]
                        ].direction==self.next_candidate.direction
        # if not use GRE matrix
        if this_strategy.entry_strategy=='fixed_thr':
            condition =  (self.list_opened_positions[
                        self.map_ass_idx2pos_idx[idx]
                        ].direction==self.next_candidate.direction and 
                        self.next_candidate.p_mc>=this_strategy.lb_mc_ext and 
                        self.next_candidate.p_md>=this_strategy.lb_md_ext and
                        self.next_candidate.p_mc<this_strategy.ub_mc_ext and 
                        self.next_candidate.p_md<this_strategy.ub_md_ext)
            prods_condition = condition
        elif this_strategy.entry_strategy=='gre' or this_strategy.entry_strategy=='gre_v2':
#            previous_p_mc = self.list_opened_positions[self.map_ass_idx2pos_idx[idx]].p_mc
#            previous_p_md = self.list_opened_positions[self.map_ass_idx2pos_idx[idx]].p_md
#            sum_previous_p = previous_p_mc+previous_p_md
            #print("sum_previous_p: "+str(sum_previous_p))
            #self.next_candidate.p_mc+self.next_candidate.p_md>=sum_previous_p
#            if self.next_candidate!= None:
#                print("sum_p: "+str(self.next_candidate.p_mc+self.next_candidate.p_md))
            
            condition = (self.next_candidate!= None and 
                         this_strategy.get_profitability(
                         self.next_candidate.p_mc, self.next_candidate.p_md, 
                         int(np.abs(self.next_candidate.bet)-1))>margin and 
                         100*curr_GROI>=self.list_lim_groi[self.map_ass_idx2pos_idx[idx]])
            prods_condition = condition
            #out = "lim GROI: "+str(self.list_lim_groi[self.map_ass_idx2pos_idx[idx]])
            #print(out)
            #self.write_log(out)
#             and
#                              self.next_candidate.p_mc>=previous_p_mc-.05 and 
#                              self.next_candidate.p_md>=previous_p_md-.05
        elif this_strategy.entry_strategy=='spread_ranges':
            
            if this_strategy.extend_for_any_thr:
                t = 0
            else:
                sp = self.list_sps[self.map_ass_idx2pos_idx[idx]]
                t = next((i for i,x in enumerate(this_strategy.info_spread_ranges['sp']) if x>=sp), 
                         len(this_strategy.info_spread_ranges['sp'])-1)
#            print(t)
#            print(this_strategy.info_spread_ranges)
            prods_condition = self.next_candidate.p_mc>=\
                this_strategy.info_spread_ranges['th'][t][0]+\
                this_strategy.info_spread_ranges['mar'][t][0] and\
                self.next_candidate.p_md>=this_strategy.info_spread_ranges['th'][t][1]+\
                this_strategy.info_spread_ranges['mar'][t][1]
            condition = dir_condition and prods_condition and \
                100*curr_GROI>=self.list_lim_groi[self.map_ass_idx2pos_idx[idx]]
        return condition, dir_condition, prods_condition
    
#    def check_close_dueto_dirchange(self):
#        """  """
#        this_strategy = strategys[name2str_map[self.next_candidate.strategy]]
#        if this_strategy.entry_strategy=='spread_ranges':
#            if self.next_candidate.p_mc>=\
#                this_strategy.info_spread_ranges['th'][0][0]+\
#                this_strategy.info_spread_ranges['mar'][0][0] and\
#                self.next_candidate.p_md>=this_strategy.info_spread_ranges['th'][0][1]+\
#                this_strategy.info_spread_ranges['mar'][0][1]:
#                    return True
        return False

    def update_stoploss(self, idx):
        # update stoploss
        if self.list_opened_positions[self.map_ass_idx2pos_idx[idx]].direction == 1:
            self.list_stop_losses[self.map_ass_idx2pos_idx[idx]] = max(
                    self.list_stop_losses[self.map_ass_idx2pos_idx[idx]],
                    bid*(1-self.list_opened_positions[
                    self.map_ass_idx2pos_idx[idx]].direction*strategys[
                    name2str_map[self.next_candidate.strategy]].thr_sl*strategys[
                    name2str_map[self.next_candidate.strategy]].fixed_spread_ratio))
        else:
            self.list_stop_losses[self.map_ass_idx2pos_idx[idx]] = min(
                    self.list_stop_losses[self.map_ass_idx2pos_idx[idx]],
                    bid*(1-self.list_opened_positions[
                    self.map_ass_idx2pos_idx[idx]].direction*strategys[
                    name2str_map[self.next_candidate.strategy]
                    ].thr_sl*strategys[name2str_map[
                    self.next_candidate.strategy]].fixed_spread_ratio))
    
    def is_stoploss_reached(self, idx, stoplosses):
        # check stop loss reachead
        #print(self.list_stop_losses)
        list_idx = self.map_ass_idx2pos_idx[idx]
        if (self.list_opened_positions[self.map_ass_idx2pos_idx[idx]].direction*(
                self.list_last_bid[list_idx]-self.list_stop_losses[self.map_ass_idx2pos_idx[idx]])<=0):
            # exit position due to stop loss
            exit_pos = 1
            stoplosses += 1
            out = ("Exit position due to stop loss @event idx "+str(event_idx)+
                   " bid="+str(bid)+" sl="+
                   str(self.list_stop_losses[self.map_ass_idx2pos_idx[idx]]))
            self.write_log(out)
            print(out)
        else:
            exit_pos = 0
        return exit_pos, stoplosses
    
#    def ban_asset(self, ass_idx):
#        """  """
#        self.list_banned_counter[ass_idx] = data.lB
#        out = data.AllAssets[str(ass_idx)]+" banned"
#        print(out)
#        self.write_log(out)
#        return None
    
    def decrease_ban_counter(self, ass_idx, n_counts):
        """  """
        self.list_banned_counter[ass_idx] = max(self.list_banned_counter[ass_idx]-n_counts,-1)
        return None
    
    def is_takeprofit_reached(self, this_pos, take_profit, takeprofits):
        
        # check take profit reachead
        if this_pos.direction*(bid-take_profit)>=0:
            # exit position due to stop loss
            exit_pos = 1
            takeprofits += 1
            out = ("Exit position due to take profit @event idx "+str(event_idx)+
                   ". tp="+str(take_profit))
            print(out)
            self.write_log(out)
        else:
            exit_pos = 0
        return exit_pos, takeprofits
    
    def get_rois(self, idx, date_time='', roi_ratio=1, ass=''):
        """ Get current GROI and ROI of a given asset idx """
        
        direction = self.list_opened_positions[self.map_ass_idx2pos_idx[idx]].direction
        Ti = \
            dt.datetime.strftime(self.list_opened_positions
                                 [self.map_ass_idx2pos_idx[idx]].entry_time,
                                 '%Y.%m.%d %H:%M:%S')
        Dir = np.sign(self.list_opened_positions[self.map_ass_idx2pos_idx[idx]].bet)
        strategy_name = self.list_opened_positions[self.map_ass_idx2pos_idx[idx]].strategy
############################ WARNING!!!! ##############################################        
        Bi = self.list_opened_positions[self.map_ass_idx2pos_idx[idx]].entry_bid
        Ai = self.list_opened_positions[self.map_ass_idx2pos_idx[idx]].entry_ask
        Bo = self.list_last_bid[self.map_ass_idx2pos_idx[idx]]
        Ao = self.list_last_ask[self.map_ass_idx2pos_idx[idx]]
        espread = (Ai-Bi)/Ai
        if direction>0:
            GROI_live = roi_ratio*(Ao-Ai)/Ai
            spread = (Ao-Bo)/Ai
            
        else:
            GROI_live = roi_ratio*(Bi-Bo)/Ao
            spread = (Ao-Bo)/Ao
        
        if type(self.next_candidate)!=type(None):
            if strategys[name2str_map[self.next_candidate.strategy]].fix_spread:
                ROI_live = GROI_live-roi_ratio*strategys[name2str_map[
                        self.next_candidate.strategy]].fixed_spread_ratio
            else:
                ROI_live = GROI_live-roi_ratio*spread
        else:
            if self.last_fix_spread:
                ROI_live = GROI_live-roi_ratio*self.last_fixed_spread_ratio
            else:
                ROI_live = GROI_live-roi_ratio*spread
        
        info = ass+"\t"+Ti[:10]+"\t"+Ti[11:]+"\t"+date_time[:10]+"\t"+date_time[11:]+"\t"+\
                str(100*GROI_live)+"\t"+str(100*ROI_live)+"\t"+str(100*spread)+"\t"+str(100*espread)+"\t"+\
                "0"+"\t"+str(Dir)+"\t"+str(Bi)+"\t"+str(Bo)+"\t"+str(Ai)+"\t"+\
                str(Ao)+"\t"+strategy_name
        return GROI_live, ROI_live, spread, info
    
    def update_groi_limit(self, idx, curr_GROI):
        self.list_lim_groi[self.map_ass_idx2pos_idx[idx]] = \
            max(self.list_lim_groi[self.map_ass_idx2pos_idx[idx]], 
                100*curr_GROI+strategys[name2str_map[
                self.next_candidate.strategy]].lim_groi)
        return None
    
    def close_position(self, date_time, ass, idx, lot_ratio=None, partial_close=False):
        # update results and exit market
        self.n_entries += 1
        
        # if it's full close, get the raminings of lots as lots ratio
        if not partial_close:
            lot_ratio = 1.0
        roi_ratio = lot_ratio*self.list_lots_per_pos[
                self.map_ass_idx2pos_idx[idx]]/self.list_lots_entry[
                        self.map_ass_idx2pos_idx[idx]]
        
        if np.isnan(roi_ratio):
            raise ValueError("roi_ratio NaN")
        
        direction = self.list_opened_positions[self.map_ass_idx2pos_idx[idx]].direction
        Bi = self.list_opened_positions[self.map_ass_idx2pos_idx[idx]].entry_bid
        Ai = self.list_opened_positions[self.map_ass_idx2pos_idx[idx]].entry_ask
        Ao = self.list_last_ask[self.map_ass_idx2pos_idx[idx]]
        Bo = self.list_last_bid[self.map_ass_idx2pos_idx[idx]]
        p_mc = self.list_opened_positions[self.map_ass_idx2pos_idx[idx]].p_mc
        p_md = self.list_opened_positions[self.map_ass_idx2pos_idx[idx]].p_md
#        
#        if direction>0:
#            GROI_live = roi_ratio*(Ao-Ai)/Ai
#            spread = (Ao-Bo)/Ai
#            
#        else:
#            GROI_live = roi_ratio*(Bi-Bo)/Ao
#            spread = (Ao-Bo)/Ao
        
#        if type(self.next_candidate)!=type(None):
#            if strategys[name2str_map[self.next_candidate.strategy]].fix_spread:
#                ROI_live = GROI_live-roi_ratio*strategys[name2str_map[
#                        self.next_candidate.strategy]].fixed_spread_ratio
#            else:
#                ROI_live = GROI_live-roi_ratio*spread
#        else:
#            if self.last_fix_spread:
#                ROI_live = GROI_live-roi_ratio*self.last_fixed_spread_ratio
#            else:
#                ROI_live = GROI_live-roi_ratio*spread
        GROI_live, ROI_live, spread, pos_info = self.get_rois(idx, date_time=date_time,
                                                          roi_ratio=roi_ratio,
                                                          ass=ass)
        self.write_pos_info(pos_info)
        
        self.available_budget += self.list_lots_per_pos[
                self.map_ass_idx2pos_idx[idx]]*self.LOT*(lot_ratio+ROI_live)
        self.available_bugdet_in_lots = self.available_budget/self.LOT
        self.budget_in_lots += self.list_lots_per_pos[
                self.map_ass_idx2pos_idx[idx]]*ROI_live
        nett_win = self.list_lots_entry[
                self.map_ass_idx2pos_idx[idx]]*ROI_live*self.LOT
        gross_win = self.list_lots_entry[
                self.map_ass_idx2pos_idx[idx]]*GROI_live*self.LOT
        self.budget += nett_win
        earnings = self.budget-self.init_budget
        self.gross_earnings += gross_win
        if ROI_live>0:
            self.net_successes += 1
            self.average_win += ROI_live
        else:
            self.average_loss += ROI_live
        if GROI_live>0:
            self.gross_successes += 1
        self.tROI_live += ROI_live
        self.tGROI_live += GROI_live
        
        if not partial_close:
            self.remove_position(idx)
        else:
            # decrease the lot ratio in case the position is not fully closed
            self.list_lots_per_pos[self.map_ass_idx2pos_idx[idx]
            ] = self.list_lots_per_pos[self.map_ass_idx2pos_idx[idx]]*(1-lot_ratio)
#        if partial_close:
#            partial_string = ' Partial'
#        else:
#            partial_string = ' Full'
#        " Bi {0:.5f} ".format(Bi)+"Bo {0:.5f} ".format(Bo)+
#              "Ai {0:.5f} ".format(Ai)+"Ao {0:.5f} ".format(Ao)+
#        profitability = strategys[name2str_map[
#                self.next_candidate.strategy]
#            ].get_profitability(self.next_candidate.p_mc, 
#            self.next_candidate.p_md, 
#            int(np.abs(self.next_candidate.bet)-1))
        out =( date_time+" "+str(direction)+" close "+ass+
              " p_mc={0:.2f}".format(p_mc)+
              " p_md={0:.2f}".format(p_md)+
              " GROI {2:.3f}% Spread {1:.3f}% ROI = {0:.3f}%".format(
                      100*ROI_live,100*spread,100*GROI_live)+
                      " TGROI {1:.3f}% TROI = {0:.3f}%".format(
                      100*self.tROI_live,100*self.tGROI_live)+
                              " Earnings {0:.2f}".format(earnings)
              +". Remeining open "+str(len(self.list_opened_positions)))
        # update results
        results.datetimes.append(date_time)
        results.GROIs = np.append(results.GROIs,100*GROI_live)
        results.ROIs = np.append(results.ROIs,100*ROI_live)
        results.earnings = np.append(results.GROIs,nett_win)
        self.write_log(out)
        print(out)
        assert(lot_ratio<=1.00 and lot_ratio>0)
        
    def create_candidate(self, idx, approached, n_pos_opened, lots):
        """ """
        if self.allow_candidates:
            pass
        else:
            self.open_position(idx, approached, n_pos_opened, lots)
    
    def open_position(self, idx, approached, n_pos_opened, lots, sp):
        """
        
        """
        self.available_budget -= lots*self.LOT
        self.available_bugdet_in_lots -= lots
        approached = 1
        n_pos_opened += 1
        
        # update vector of opened positions
        self.add_position(idx, lots, sp)
        # update candidate positions
        EXIT, rewind = self.update_candidates()
        out = (time_stamp.strftime('%Y.%m.%d %H:%M:%S')+" Open "+
               self.list_opened_positions[-1].asset.decode("utf-8")+
              " Lots {0:.1f}".format(lots)+" "+
              str(self.list_opened_positions[-1].bet)+
              " p_mc={0:.2f}".format(self.list_opened_positions[-1].p_mc)+
              " p_md={0:.2f}".format(self.list_opened_positions[-1].p_md)+
              " spread={0:.3f}".format(100*e_spread))
        print(out)
        self.write_log(out)
        
        return approached, n_pos_opened, EXIT, rewind
    
    def extend_position(self, idx, curr_GROI):
        """
        <DocString>
        """
        self.list_opened_positions[self.map_ass_idx2pos_idx[idx]].direction
        out = (time_stamp.strftime('%Y.%m.%d %H:%M:%S')+" Extended "+
               self.list_opened_positions[self.map_ass_idx2pos_idx[idx]
               ].asset.decode("utf-8")+
               " Lots {0:.1f}".format(lots)+" "+
               str(self.list_opened_positions[self.map_ass_idx2pos_idx[idx]].bet)+
               " p_mc={0:.2f}".format(self.list_opened_positions[
                       self.map_ass_idx2pos_idx[idx]].p_mc)+
               " p_md={0:.2f}".format(self.list_opened_positions[
                       self.map_ass_idx2pos_idx[idx]].p_md)+
               " spread={0:.3f}".format(100*e_spread)+ " cGROI {0:.2f}".format(100*curr_GROI))
        print(out)
        self.write_log(out)
        self.n_pos_extended += 1
        
        self.update_position(ass_idx)
        EXIT, rewind = self.update_candidates()
        
        return EXIT, rewind
        
    
    def skip_positions(self, original_exit_time):
        """
        
        """
        EXIT = 0
        print("Original exit time "+str(original_exit_time))
        while (self.next_candidate!= None and 
               self.next_candidate.entry_time<=original_exit_time and not EXIT):
            out = "Skipping candidate entry time "+str(self.next_candidate.entry_time)
            print(out)
            self.write_log(out)
            EXIT, rewind = self.update_candidates()
            
        
        return EXIT
    
    def assign_lots(self, date_time):#, date_time, ass, idx
        
        if not strategys[name2str_map[self.next_candidate.strategy]].flexible_lot_ratio:
            if self.available_bugdet_in_lots>0:
                open_lots = min(strategys[name2str_map[
                            self.next_candidate.strategy]].max_lots_per_pos, 
                            self.available_bugdet_in_lots)
            else:
                open_lots = strategys[name2str_map[self.next_candidate.strategy
                                                   ]].max_lots_per_pos
        else:
            # lots ratio to asssign to new asset
            open_lots = min(self.budget_in_lots/(len(
                    self.list_opened_positions)+1),
                    strategys[name2str_map[
                            self.next_candidate.strategy]].max_lots_per_pos)
            margin = 0.0001
            # check if necessary lots for opening are available
            if open_lots>self.available_bugdet_in_lots+margin:
                close_lots = (open_lots-self.available_bugdet_in_lots)/len(
                        self.list_opened_positions)
                out = "close_lots "+str(close_lots)
                print(out)
                self.write_log(out)
                if close_lots==np.inf:
                    raise ValueError("close_lots==np.inf")
                # loop over open positions to close th close_lot_ratio 
                #ratio to allocate the new position
                for pos in range(len(self.list_opened_positions)):
                    # lots ratio to be closed for this asset
                    ass = self.list_opened_positions[pos].asset.decode("utf-8")
#                    print(ass)
#                    print("self.list_lots_per_pos[pos] "+str(self.list_lots_per_pos[pos]))
                    close_lot_ratio = close_lots/self.list_lots_per_pos[pos]
#                    print("close_lot_ratio "+str(close_lot_ratio))
                    idx = ass2index_mapping[ass]
                    self.close_position(date_time, ass, idx, lot_ratio = close_lot_ratio, 
                                        partial_close = True)
            
            # make sure the available resources are smaller or equal than slots to open
            open_lots = min(open_lots,self.available_bugdet_in_lots)
        #print("open_lots corrected "+str(open_lots))
            
        return open_lots
    
    def write_log(self, log):
        """
        <DocString>
        """
        if self.save_log:
            file = open(self.log_file,"a")
            file.write(log+"\n")
            file.close()
        return None
    
    def write_pos_info(self, pos_info):
        """  """
        if self.save_log:
            file = open(self.positions_dir+self.positions_file,"a")
            file.write(pos_info+"\n")
            file.close()
    
    def write_summary(self, out):
        """
        Write summary into a file
        """
        if self.save_log:
            file = open(self.summary_file,"a")
            file.write(out+"\n")
            file.close()
        return None
    
    def ban_asset(self, position, DateTime, thisAsset, ass_idx, Bo, Ao):
        """  """
        self.list_is_asset_banned[ass_idx] = True
        if not hasattr(self, 'list_dict_banned_assets'):
            self.list_dict_banned_assets = [None for _ in self.list_is_asset_banned]
        GROI = 100*self.get_groi_banned_asset(position, Bo, Ao)
        tracing_dict = {'lastDateTime':DateTime,
                        'counter':50,
                        'position':position}
        self.list_dict_banned_assets[ass_idx] = tracing_dict
        out = DateTime+" "+thisAsset+\
                " ban counter set to "\
                +str(self.list_dict_banned_assets[ass_idx]['counter'])+\
                " GROI = "+str(GROI)
        print("\r"+out)
        self.write_log(out)
    
    def get_groi_banned_asset(self, position, Bo, Ao):
        """  """
        direction = position.direction
#        Ti = \
#            dt.datetime.strftime(position.entry_time,
#                                 '%Y.%m.%d %H:%M:%S')
#        Dir = np.sign(position.bet)     
        Bi = position.entry_bid
        Ai = position.entry_ask
#        Bo = self.list_last_bid[self.map_ass_idx2pos_idx[idx]]
#        Ao = self.list_last_ask[self.map_ass_idx2pos_idx[idx]]
#        espread = (Ai-Bi)/Ai
        if direction>0:
            GROI_live = (Ao-Ai)/Ai
#            spread = (Ao-Bo)/Ai
            
        else:
            GROI_live = (Bi-Bo)/Ao
#            spread = (Ao-Bo)/Ao
        

#        ROI_live = GROI_live-spread

        return GROI_live
    
    def track_banned_asset(self, DateTime, thisAsset, ass_idx, Bo, Ao):
        """ """
        
        self.list_dict_banned_assets[ass_idx]['counter'] -= 1
        GROI = 100*self.get_groi_banned_asset(self.list_dict_banned_assets[ass_idx]['position'], Bo, Ao)
        out = DateTime+" "+thisAsset+\
        " ban counter set to "\
        +str(self.list_dict_banned_assets[ass_idx]['counter'])+\
        " GROI = "+str(GROI)
        print("\r"+out)
        self.write_log(out)
        if self.list_dict_banned_assets[ass_idx]['counter'] == 0:
            self.lift_ban_asset(ass_idx)
            out = "Ban lifted due to counter 0"
            print("\r"+out)
            self.write_log(out)
        # check if trend has changed and get in again
        elif GROI>-0.3: # hard coded stoploss
            self.lift_ban_asset(ass_idx)
            out = "Ban lifted due to GROI > stoploss"
            print("\r"+out)
            self.write_log(out)
    
    def lift_ban_asset(self, ass_idx):
        """  """
        self.list_is_asset_banned[ass_idx] = False


def load_in_memory(assets, AllAssets, dateTest, init_list_index, end_list_index,root_dir='D:/SDC/py/Data/'):
    print("Loading info from original files...")
    files_list = []
    DateTimes = np.chararray((0,),itemsize=19)
    Assets = np.chararray((0,),itemsize=19)
    SymbolBids = np.array([])
    SymbolAsks = np.array([])
    alloc = 10000000
    DateTimes = np.chararray((alloc,),itemsize=19)
    Assets = np.chararray((alloc,),itemsize=19)
    
    SymbolBids = np.zeros((alloc,))
    SymbolAsks = np.zeros((alloc,))
    idx_counter = 0
    # read trade info from raw files
    for ass in assets:
        thisAsset = AllAssets[str(ass)]
        #print(thisAsset)
        directory_origin = root_dir+thisAsset+'/'#'../Data/'+thisAsset+'/'
        # get files list, and beginning and end current dates
        if os.path.isdir(directory_origin):
            files_list_all = sorted(os.listdir(directory_origin))
            for file in files_list_all:
                m = re.search('^'+thisAsset+'_\d+'+'.txt$',file)
                if m!=None:
                    last_date_s = re.search('\d+',m.group()).group()
                    day = dt.datetime.strftime(dt.datetime.strptime(last_date_s, 
                                                                    '%Y%m%d%H%M%S'), 
                                                                    '%Y.%m.%d')
                    if day in dateTest[init_list_index:end_list_index+1]:
                        #print(file)
                        files_list.append(file)
                        pd_file = pd.read_csv(directory_origin+file)
                        n_new_samps = pd_file.shape[0]
                        if n_new_samps+idx_counter>DateTimes.shape[0]:
                            #print("Extending size")
                            DateTimes = np.append(DateTimes,np.chararray((alloc,),
                                                                itemsize=19))
                            Assets = np.append(Assets,np.chararray((alloc,),
                                                                itemsize=19))
                            SymbolBids = np.append(SymbolBids,np.zeros((alloc,)))
                            SymbolAsks = np.append(SymbolAsks,np.zeros((alloc,)))
                            
                        DateTimes[idx_counter:idx_counter+n_new_samps] = pd_file['DateTime']
                        
                        Ass = np.chararray((pd_file.shape[0],),itemsize=6)
                        Ass[:] = thisAsset
                        Assets[idx_counter:idx_counter+n_new_samps] = Ass
                        SymbolBids[idx_counter:idx_counter+n_new_samps] = pd_file['SymbolBid']
                        SymbolAsks[idx_counter:idx_counter+n_new_samps] = pd_file['SymbolAsk']
                        idx_counter += n_new_samps
    sorted_idx = np.argsort(DateTimes[:idx_counter],kind='mergesort')
    DateTimes = DateTimes[sorted_idx].astype('S19')
    SymbolBids = SymbolBids[sorted_idx]
    SymbolAsks = SymbolAsks[sorted_idx]
    Assets = Assets[sorted_idx]
    nEvents = DateTimes.shape[0]
    print("DONE")
    return DateTimes, SymbolBids, SymbolAsks, Assets, nEvents

def round_num(number, prec):
    return round(prec*number)/prec

if __name__ == '__main__':
    
    start_time = dt.datetime.strftime(dt.datetime.now(),'%y%m%d%H%M%S')
    numberNetwors = 2
    list_IDresults = ['R21050PS2CMF181112T190822ALk12K2E2215','R21050PS2CMF181112T190822BSk12K2E2215']
    list_name = ['21050PS2ALk12K2E2215SRNSP60','21050PS2BSk12K2E2215SRNSP60']
    list_epoch_journal = [0 for _ in range(numberNetwors)]
    list_t_index = [0 for _ in range(numberNetwors)]
    list_spread_ranges = [{'sp':[.5, .7, .9, 1.2, 1.5, 1.7, 1.8, 2.2, 5],
                           'th':[(.5,.61),(.52,.64),(.52,.65),(.56,.68),(.58,.69),(.59,.79),(.59,.8),(.64,.8),(.93,.59)],
                           'mar':[(0,0) for _ in range(9)]},
                          {'sp':[.6, .8, 1, 2.4, 2.9, 3.9, 4.3, 4.9, 5],
                           'th':[(.8,.61), (.81,.61), (.83,.61), (.88,.58), (.88,.59), (.87,.61), (.89,.58), (.88,.62), (.9,.57)],
                           'mar':[(0,0) for _ in range(9)]}]
    list_lim_groi_ext = [-10 for i in range(numberNetwors)] # in %
    list_lb_mc_ext = [.5, .8]
    list_lb_md_ext = [.65,.61]
    list_max_lots_per_pos = [.1 for i in range(numberNetwors)]
    list_entry_strategy = ['spread_ranges' for i in range(numberNetwors)]#'fixed_thr','gre' or 'spread_ranges', 'gre_v2'
    list_IDgre = ['' for i in range(numberNetwors)]
    list_if_dir_change_close = [False for i in range(numberNetwors)]
    list_extend_for_any_thr = [True for i in range(numberNetwors)]
    list_thr_sl = [1000 for i in range(numberNetwors)]

    # depricated/not supported
    list_IDgre = ['' for i in range(numberNetwors)]
    list_epoch_gre = [None for i in range(numberNetwors)]
    list_weights = [np.array([0,1]) for i in range(numberNetwors)]
    list_w_str = ["" for i in range(numberNetwors)]
    #root_dir = local_vars.data_dir
    root_dir = local_vars.data_test_dir
    
    init_day_str = '20181112'#'20190701'#
    end_day_str = '20190822'
    init_day = dt.datetime.strptime(init_day_str,'%Y%m%d').date()
    end_day = dt.datetime.strptime(end_day_str,'%Y%m%d').date()
    
    positions_file = start_time+'_F'+init_day_str+'T'+end_day_str+'.csv'
#    end_day = dt.datetime.strptime('2019.04.26','%Y.%m.%d').date()
#    delta_dates = dt.datetime.strptime('2018.11.09','%Y.%m.%d').date()-edges[-2]
#    dateTestDt = [edges[-2] + dt.timedelta(i) for i in range(delta_dates.days + 1)]
    delta_dates = end_day-init_day
    dateTestDt = [init_day + dt.timedelta(i) for i in range(delta_dates.days + 1)]
#    dateTestDt = [edges[-2] + dt.timedelta(i) for i in range(delta_dates.days + 1)]
    dateTest = []
    for d in dateTestDt:
        if d.weekday()<5:
            dateTest.append(dt.date.strftime(d,'%Y.%m.%d'))
    #dateTest = ['2018.11.15','2018.11.16']
    load_from_live = False
    # data structure
    data=Data(movingWindow=100,nEventsPerStat=1000,
     dateTest = dateTest)

    tic = time.time()
    # init positions vector
    # build asset to index mapping
    ass2index_mapping = {}
    ass_index = 0
    for ass in data.assets:
        ass2index_mapping[data.AllAssets[str(ass)]] = ass_index
        ass_index += 1

    #list_thr_sl = [1000 for i in range(numberNetwors)]
    list_thr_tp = [1000 for i in range(numberNetwors)]
    list_lb_mc_op = [.5 for i in range(numberNetwors)]
    list_lb_md_op = [.8 for i in range(numberNetwors)]
    list_ub_mc_op = [1 for i in range(numberNetwors)]
    list_ub_md_op = [1 for i in range(numberNetwors)]
    list_ub_mc_ext = [1 for i in range(numberNetwors)]
    list_ub_md_ext = [1 for i in range(numberNetwors)]
    list_fix_spread = [False for i in range(numberNetwors)]
    list_fixed_spread_pips = [4 for i in range(numberNetwors)]
    list_flexible_lot_ratio = [False for i in range(numberNetwors)]
    list_if_dir_change_extend = [False for i in range(numberNetwors)]
    
#    +'_'+'_'.join([list_IDresults[i]+'E'+str(list_epoch_journal[i])+'TI'+
#                         str(list_t_index[i])+'MC'+str(list_spread_ranges[i]['th'][0][0])+'MD'+
#                         str(list_spread_ranges[i]['th'][0][1])
#                         for i in range(numberNetwors)])
    
    
    strategys = [Strategy(direct=local_vars.results_directory,thr_sl=list_thr_sl[i], 
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
                          name=list_name[i],
                          t_index=list_t_index[i],IDr=list_IDresults[i],
                          IDgre=list_IDgre[i],
                          epoch=str(list_epoch_gre[i]),weights=list_weights[i],
                          info_spread_ranges=list_spread_ranges[i],
                          lim_groi_ext=list_lim_groi_ext[i],
                          entry_strategy=list_entry_strategy[i],
                          extend_for_any_thr=list_extend_for_any_thr[i]) 
                          for i in range(numberNetwors)]
    
    name2str_map = {}
    for n in range(len(list_name)):
        name2str_map[strategys[n].name] = n
    
    if not load_from_live:
        new_results = True
        resultsDir = local_vars.results_directory#"../RNN/results/"
        if not new_results:
            list_journal_dir = [resultsDir+list_IDresults[i]+"/t"+
                                str(list_t_index[i])+"/"+list_IDresults[i]+"t"+
                                str(list_t_index[i])+"mc"+str(list_lb_mc_ext[i])+
                                "/"+list_IDresults[i]+"t"+str(list_t_index[i])+
                                "mc"+str(list_lb_mc_ext[i])+"md"+str(list_lb_md_ext[i])+
                                "/" for i in range(len(list_t_index))]
            list_journal_name = ["J"+list_IDresults[i]+"t"+str(list_t_index[i])+"mc"+
                                 str(list_lb_mc_ext[i])+"md"+str(list_lb_md_ext[i])+
                                 "e"+str(list_epoch_journal[i])+".csv" 
                                 for i in range(len(list_t_index))]
            entry_time_column = 'DT1'#'Entry Time
            exit_time_column = 'DT2'#'Exit Time
            entry_bid_column = 'B1'
            entry_ask_column = 'A1'
            exit_ask_column = 'A2'
            exit_bid_column = 'B2'
            root_dir = local_vars.data_dir##'D:/SDC/py/Data/'
            
            list_journal_all_days = [pd.read_csv(list_journal_dir[i]+
                                                 list_journal_name[i], 
                                                 sep='\t').sort_values(
                                                by=[entry_time_column]).reset_index(
                                                        ).drop(labels='level_0',
                                                        axis=1).assign(
                                                        strategy=list_name[i]) 
                                                for i in range(len(list_t_index))]# .drop(labels='level_0',axis=1)
        else:
            list_journal_dir = [resultsDir+list_IDresults[i]+"/journal/"
                                 for i in range(len(list_t_index))]
            list_journal_name = ["J_E"+
                                str(list_epoch_journal[i])+"TI"+str(list_t_index[i])
                                +"MC"+str(list_lb_mc_ext[i])+"MD"+str(list_lb_md_ext[i])+".csv"
                                for i in range(len(list_t_index))]
            entry_time_column = 'DTi'#'Entry Time
            exit_time_column = 'DTo'#'Exit Time
            entry_bid_column = 'Bi'
            entry_ask_column = 'Ai'
            exit_ask_column = 'Ao'
            exit_bid_column = 'Bo'
            
            list_journal_all_days = [pd.read_csv(list_journal_dir[i]+
                                                 list_journal_name[i], 
                                                 sep='\t').sort_values(
                                                by=[entry_time_column]).reset_index(
                                                        ).drop(labels='level_0',
                                                        axis=1).assign(
                                                        strategy=list_name[i]) 
                                                for i in range(len(list_t_index))]
    else:
        
        resultsDir = "../../RNN/resultsLive/back_test/"
        IDresults = ["100287Nov09"]
        list_journal_dir = [resultsDir+IDresults[0]+'T'+
                            str(list_t_index[0])+
                            'E'+str(list_epoch_journal[0])+'/']
        list_journal_name = [IDresults[0]+'T'+str(list_t_index[0])+
                             'E'+str(list_epoch_journal[0])+'_DL3.txt']
        entry_time_column = 'Entry Time'#'Entry Time
        exit_time_column = 'Exit Time'#'Exit Time
        entry_bid_column = 'Bi'
        entry_ask_column = 'Ai'
        exit_ask_column = 'Ao'
        exit_bid_column = 'Bo'
        root_dir = 'D:/SDC/py/Data_DL3/'#'D:/SDC/py/Data_aws_3/'#'D:/SDC/py/Data_DL3/'
        list_journal_all_days = [pd.read_csv(list_journal_dir[i]+
                                             list_journal_name[i], 
                                             sep='\t').sort_values(by=[
                                                     entry_time_column]
                                             ).reset_index().assign(
                                                     strategy=list_name[i]) 
                                            for i in range(len(list_t_index))]
    
    AD_resume = None
    eROIpb = None
    # conver list of journals into a long journal
    journal_all_days = pd.DataFrame()
    for l in range(len(list_journal_all_days)):
        journal_all_days = journal_all_days.append(list_journal_all_days[l]).sort_values(by=[entry_time_column]).reset_index().drop(labels='level_0',axis=1)
    #journal_all_days = journal_all_days.drop(labels='level_0',axis=1)
    
#    print("Purging journal...")
#    journal_purged = pd.DataFrame()
#    # purge journal from unnecessary entries
#    for e in journal_all_days.index:
#        # drop entry if profitability is <=0
#        if strategys[name2str_map[journal_all_days.loc[e].strategy]].get_profitability(journal_all_days.loc[e].P_mc, journal_all_days.loc[e].P_md, int(np.abs(journal_all_days.loc[e].Bet)-1))>0:
#            # keep entry
#            journal_purged = journal_purged.append(journal_all_days.loc[e])
#            
#    journal_all_days = journal_purged
    
    total_journal_entries = journal_all_days.shape[0]
    init_budget = 10000
    results = Results()
    
    directory = local_vars.live_results_dict+"simulate/trader/"
    log_file = directory+start_time+"trader_v30.log"
    summary_file = directory+start_time+"summary.log"
    positions_dir = directory+"positions/"
    if not os.path.exists(positions_dir):
        os.mkdir(positions_dir)
    
    
    columns_positions = 'Asset\tDi\tTi\tDo\tTo\tGROI\tROI\tspread\tespread\text\tDir\tBi\tBo\tAi\tAo\tstrategy'
    file = open(positions_dir+positions_file,"a")
    file.write(columns_positions+"\n")
    file.close()
            #'_E'+str(epoch)+'TI'+str(t_index)+'MC'+str(thr_mc)+'MD'+str(thr_md)+'.csv'
    # save sorted journal
    #journal_all_days.drop('index',axis=1).to_csv(directory+start_time+'journal.log',sep='\t',float_format='%.3f',index_label='index')
    ##### loop over different day groups #####
    day_index = 0
    t_journal_entries = 0
    change_dir = 0
    rewinded = 0
    week_counter = 0
    print(root_dir)
    while day_index<len(data.dateTest):
        counter_back = 0
        init_list_index = day_index#data.dateTest[day_index]
        pd_init_index = journal_all_days[journal_all_days[entry_time_column].str.find(data.dateTest[day_index])>-1]
        while day_index<len(data.dateTest)-1 and pd_init_index.shape[0]==0:
            day_index += 1
            pd_init_index = journal_all_days[journal_all_days[entry_time_column].str.find(data.dateTest[day_index])>-1]
        try:
            init_index = pd_init_index.index[0]
        except:
            pass
            #break    
        # find sequence of consecutive days in test days
        while day_index<len(data.dateTest)-1 and dt.datetime.strptime(data.dateTest[day_index],'%Y.%m.%d')+dt.timedelta(1)==dt.datetime.strptime(data.dateTest[day_index+1],'%Y.%m.%d'):
            day_index += 1
        pd_end_index = journal_all_days[journal_all_days[exit_time_column].str.find(data.dateTest[day_index])>-1]
        while day_index>init_list_index and pd_end_index.shape[0]==0:
            day_index -= 1
            counter_back += 1
            pd_end_index = journal_all_days[journal_all_days[exit_time_column].str.find(data.dateTest[day_index])>-1]
        # only one entry in the week and out is next week
        if pd_end_index.shape[0]==0:
            end_index = pd_init_index.index[-1]
        else:
            end_index = pd_end_index.index[-1]
        end_list_index = day_index+counter_back
        journal = journal_all_days.loc[init_index:end_index+1]
        journal_entries = journal.shape[0]
        t_journal_entries += journal_entries
        day_index += counter_back+1
        # init trader
        trader = Trader(Position(journal.iloc[0], AD_resume, eROIpb), 
                                 init_budget=init_budget, log_file=log_file,
                                 summary_file=summary_file,positions_dir=positions_dir,
                                 positions_file=positions_file,allow_candidates=False)
        
        out = ("Week counter "+str(week_counter)+". From "+data.dateTest[init_list_index]+
               " to "+data.dateTest[end_list_index]+
               " Number journal entries "+str(journal_entries))
        week_counter += 1
        print(out)
        trader.write_log(out)
        trader.write_summary(out)
        
        load_in_RAM = True
        
        DateTimes, SymbolBids, SymbolAsks, Assets, nEvents = load_in_memory(data.assets, 
                                                                            data.AllAssets,
                                                                            data.dateTest, 
                                                                            init_list_index, 
                                                                            end_list_index,
                                                                            root_dir=root_dir)
    
        # set counters and flags
        n_pos_opened = 0
        secs_counter = 0
        approached = 0
        timeout = 0
        event_idx = 0
        EXIT = 0
        exit_pos = 0
        stoplosses = 0
        takeprofits = 0
        not_entered = 0
        not_entered_av_budget = 0
        not_entered_extention = 0
        not_entered_same_time = 0
        not_entered_secondary = 0
        close_dueto_dirchange = 0
        w = 1-1/20
        
        # get to 
        while event_idx<nEvents:
            rewind = 0
            no_budget = False
            # get time stamp
            DateTime = DateTimes[event_idx].decode("utf-8")
            time_stamp = dt.datetime.strptime(DateTime,
                                              '%Y.%m.%d %H:%M:%S')
            bid = int(np.round(SymbolBids[event_idx]*100000))/100000
            ask = int(np.round(SymbolAsks[event_idx]*100000))/100000
            e_spread = (ask-bid)/ask
            thisAsset = Assets[event_idx].decode("utf-8")
            ass_idx = ass2index_mapping[thisAsset]
            list_idx = trader.map_ass_idx2pos_idx[ass_idx]
            # update bid and ask lists if exist
            if list_idx>-1:
                trader.list_last_bid[list_idx] = bid
                trader.list_last_ask[list_idx] = ask
                em = w*trader.list_EM[list_idx]+(1-w)*bid
#                dev_em = trader.list_EM[list_idx]-em
                trader.list_EM[list_idx] = em
            
            ban_condition = trader.list_is_asset_banned[ass_idx]#trader.check_asset_not_banned()
            if not EXIT:
                sec_cond, sp = trader.check_secondary_contition_for_opening()
            else:
                sec_cond = False
                
            condition_open = (trader.chech_ground_condition_for_opening() and 
                              trader.check_primary_condition_for_opening() and 
                              sec_cond and not ban_condition)
            
            if condition_open:
#                profitability = strategys[name2str_map[trader.next_candidate.strategy
#                                                       ]].get_profitability(
#                                                       trader.next_candidate.p_mc, 
#                                                        trader.next_candidate.p_md, 
#                                                        int(np.abs(trader.next_candidate.bet)-1))
                this_strategy = strategys[name2str_map[trader.next_candidate.strategy]]
                out = (DateTime+" "+
                       thisAsset+
                       " p_mc {0:.3f}".format(trader.next_candidate.p_mc)+
                       " p_md {0:.3f}".format(trader.next_candidate.p_md)+
                      #" pofitability {0:.3f}".format(profitability)+
                      " E_spread {0:.3f}".format(e_spread/trader.pip)+" Bet "+
                      str(int(trader.next_candidate.bet))+
                      " Open Condition met "+" SP = "+str(sp))
                print(out)
                trader.write_log(out)
                
                # open market
                if not trader.is_opened(ass_idx):
                    
                    # assign budget
                    #lot_ratio = 1#/(len(trader.list_opened_positions)+1)
                    lots = trader.assign_lots(DateTime)
                    #print("trader.available_bugdet_in_lots "+str(trader.available_bugdet_in_lots))
    
                    if trader.available_bugdet_in_lots>=lots:
                        # add candidate for opening
                        
                        approached, n_pos_opened, EXIT, rewind = trader.open_position(ass_idx, 
                                                                                      approached, 
                                                                                      n_pos_opened, 
                                                                                      lots, sp)
                    else:
                        no_budget = True
            elif (trader.chech_ground_condition_for_opening() and 
                  trader.check_primary_condition_for_opening() and 
                  not sec_cond):
                #pass
#                profitability = strategys[name2str_map
#                                          [trader.next_candidate.strategy]
#                                          ].get_profitability(
#                                        trader.next_candidate.p_mc, 
#                                        trader.next_candidate.p_md, 
#                                        int(np.abs(trader.next_candidate.bet)-1))
                out = (DateTime+" "+
                       thisAsset+
                       " p_mc {0:.3f}".format(trader.next_candidate.p_mc)+
                       " p_md {0:.3f}".format(trader.next_candidate.p_md)+
                      #" pofitability {0:.3f}".format(profitability)+
                      " E_spread {0:.3f}".format(e_spread/trader.pip)+" Bet "+
                      str(trader.next_candidate.bet)+
                      " Open condition not met")
                print(out)
                trader.write_log(out)
            # check if a position is already opened
            if trader.is_opened(ass_idx):
                trader.count_one_event(ass_idx)
        #        stop_loss = update_stoploss(trader.list_opened_positions[trader.map_ass_idx2pos_idx[ass_idx]], stop_loss)
                exit_pos, stoplosses = trader.is_stoploss_reached(ass_idx, stoplosses)
                if exit_pos:
                    # save original exit time for skipping positions
                    #original_exit_time = trader.list_opened_positions[0].exit_time
                    
                    trader.ban_asset(trader.list_opened_positions[trader.map_ass_idx2pos_idx[ass_idx]],
                                     DateTime, thisAsset, ass_idx, bid, ask)
                    trader.close_position(DateTime, 
                                          thisAsset, 
                                          ass_idx)
                    # reset approched
                    if len(trader.list_opened_positions)==0:
                        approached = 0
#                    EXIT = trader.skip_positions(original_exit_time)
                # check if there is a position change opportunity
                else:
                    if (trader.next_candidate!= None):# and bid==trader.next_candidate.entry_bid and ask==trader.next_candidate.entry_ask
                        
                        # extend position
                        if (trader.check_primary_condition_for_extention(
                                time_stamp, ass_idx)):#next_pos.eGROI>e_spread):
                            #print(time_stamp.strftime('%Y.%m.%d %H:%M:%S')+" "+Assets[event_idx].decode("utf-8")+" Primary condition fulfilled")
                            curr_GROI, _, _, _ = trader.get_rois(ass_idx, date_time=DateTime, roi_ratio=1)
                            # update GROI limit for extension
                            #trader.update_groi_limit(ass_idx, curr_GROI)
                            ext_condition, dir_condition, prods_condition = trader.check_secondary_condition_for_extention(
                                    ass_idx, curr_GROI)
                            if ext_condition:
                                # include third condition for thresholds
                                # extend deadline
                                EXIT, rewind = trader.extend_position(ass_idx, curr_GROI)
                                
                            elif dir_condition: # means that the reason for no extension is not direction change
                                # extend conditon not met
#                                profitability = strategys[name2str_map[
#                                        trader.next_candidate.strategy]
#                                        ].get_profitability(trader.next_candidate.p_mc, 
#                                                            trader.next_candidate.p_md, 
#                                                            int(np.abs(trader.next_candidate.bet)-1))
                                out = (DateTime+" "+
                                       thisAsset+
                                       " p_mc {0:.3f}".format(trader.next_candidate.p_mc)+
                                       " p_md {0:.3f}".format(trader.next_candidate.p_md)+
                                      #" pofitability {0:.3f}".format(profitability)+
                                      " E_spread {0:.3f}".format(e_spread/trader.pip)+" Bet "+
                                      str(trader.next_candidate.bet)+
                                      " cGROI {0:.2f} ".format(100*curr_GROI)+
                                      " Extend condition not met")
                                
                                print(out)
                                trader.write_log(out)
                                
                                EXIT, rewind = trader.update_candidates()
                                
                            
                            elif strategys[name2str_map[
                                        trader.next_candidate.strategy]
                                        ].if_dir_change_close: # if p_mc/p_md greater than min values for market access, then close
                                sp = trader.list_sps[trader.map_ass_idx2pos_idx[ass_idx]]
                                this_strategy = strategys[name2str_map[
                                        trader.next_candidate.strategy]]
                                # check if the new level is as certain as the opening one
                                t = next((i for i,x in enumerate(this_strategy.info_spread_ranges['sp']) if x>=sp), 
                                         len(this_strategy.info_spread_ranges['sp'])-1)
                                
                                prods_condition = trader.next_candidate.p_mc>=\
                                    this_strategy.info_spread_ranges['th'][t][0]+\
                                    this_strategy.info_spread_ranges['mar'][t][0] and\
                                    trader.next_candidate.p_md>=this_strategy.info_spread_ranges['th'][t][1]+\
                                    this_strategy.info_spread_ranges['mar'][t][1]
                                # close due to change direction
#                                close_dueto_dirchange += 1
                                if prods_condition:
                                    out= "WARNING! "+Assets[event_idx].decode("utf-8")+" Close due to change of direction"
                                    print(out)
                                    trader.write_log(out)
    #                                pass
                                    out = (DateTime+" "+
                                           thisAsset+
                                           " p_mc {0:.3f}".format(trader.next_candidate.p_mc)+
                                           " p_md {0:.3f}".format(trader.next_candidate.p_md)+
                                          #" pofitability {0:.3f}".format(profitability)+
                                          " E_spread {0:.3f}".format(e_spread/trader.pip)+" Bet "+
                                          str(trader.next_candidate.bet)+
                                          " cGROI {0:.2f} ".format(100*curr_GROI))
                                    print(out)
                                    trader.write_log(out)
    #                                out= "WARNING! "+Assets[event_idx].decode("utf-8")+" Change of direction."
    ##                                out= "WARNING! "+Assets[event_idx].decode("utf-8")+" NO Change of direction."
    #                                print(out)
    #                                trader.write_log(out)
                                    trader.close_position(DateTime, 
                                                  thisAsset, 
                                                  ass_idx)
                            # reset approched
                            if len(trader.list_opened_positions)==0:
                                approached = 0
#                                change_dir += 1
#                                print(time_stamp.strftime('%Y.%m.%d %H:%M:%S')+" Exit position due to direction change "+Assets[event_idx].decode("utf-8"))
#                                # get lots in case a new one is directly opened
#                                lots = trader.list_lots_per_pos[trader.map_ass_idx2pos_idx[ass_idx]]
#                                trader.close_position(DateTimes[event_idx], Assets[event_idx].decode("utf-8"), ass_idx)
#                                # reset approched
#                                if len(trader.list_opened_positions)==0:
#                                    approached = 0
#                                
#                                if trader.chech_ground_condition_for_opening() and trader.check_primary_condition_for_opening() and trader.check_secondary_contition_for_opening():
#                                    approached, n_entries, n_pos_opened, EXIT, rewind = trader.open_position(ass_idx, approached, n_entries, n_pos_opened, lots=lots)
                        # end of if (next_pos.ADm>this_pos.ADm or (next_pos.ADm==this_pos.ADm and np.abs(next_pos.bet)>np.abs(this_pos.bet))) and Assets[event_idx]!=next_pos.asset:
                    # end of if (time_stamp==next_pos.entry_time and bid==next_pos.entry_bid and ask==next_pos.entry_ask):   
                    
                    # check if it is time to exit market
                    if (trader.map_ass_idx2pos_idx[ass_idx]>-1 and 
                        Assets[event_idx]==trader.list_opened_positions[
                                trader.map_ass_idx2pos_idx[ass_idx]].asset and 
                                bid==trader.list_opened_positions[
                                trader.map_ass_idx2pos_idx[ass_idx]].exit_bid and 
                                ask==trader.list_opened_positions[
                                trader.map_ass_idx2pos_idx[ass_idx]].exit_ask and
                                time_stamp==trader.list_opened_positions[
                                trader.map_ass_idx2pos_idx[ass_idx]].exit_time):#and trader.list_count_events[trader.map_ass_idx2pos_idx[ass_idx]]>=nExS
                        
                        trader.close_position(DateTime, 
                                              thisAsset, 
                                              ass_idx)
                        # reset approched
                        if len(trader.list_opened_positions)==0:
                            approached = 0
                        # if exit was signaled in the previous, then leave
    #                    if EXIT:
    #                        break
                    
                # end of if count_events==nExS or timeout==0 or exit_pos:
            elif trader.chech_ground_condition_for_opening() and \
                trader.check_primary_condition_for_opening() and \
                trader.list_is_asset_banned[ass_idx]:
                #out = "decreasing ban counter"
                #trader.write_log(out)
                #print(out)
                #trader.decrease_ban_counter(ass_idx, 1)
                trader.track_banned_asset(DateTime, thisAsset, ass_idx, bid, ask)
                EXIT, rewind = trader.update_candidates()
            # uptade posiitions if ADm is too low or eGROI is too low too
#            if (trader.chech_ground_condition_for_opening() and 
#                trader.check_primary_condition_for_opening()):
#                print("Primary and sec ok")
#                print(trader.check_secondary_contition_for_opening()[0])
#                print(ban_condition)
#            elif trader.chech_ground_condition_for_opening() and not trader.check_primary_condition_for_opening():
#                print("Primary ok and sec no")
#                print(trader.check_secondary_contition_for_opening()[0])
#                print(ban_condition)
#            elif not trader.chech_ground_condition_for_opening() and trader.check_primary_condition_for_opening():
#                print("Primary no and sec ok")
#                print(trader.check_secondary_contition_for_opening()[0])
#                print(ban_condition)
            if (trader.chech_ground_condition_for_opening() and 
                trader.check_primary_condition_for_opening() and (not 
                trader.check_secondary_contition_for_opening()[0] or not ban_condition)):
                not_entered_secondary += 1
                #print(time_stamp.strftime('%Y.%m.%d %H:%M:%S')+" Secondary condition failed "+Assets[event_idx].decode("utf-8"))
                EXIT, rewind = trader.update_candidates()
        
        #        if EXIT:
        #            break
                if len(trader.list_opened_positions)==0:
                    approached = 0
            if condition_open and no_budget:
                not_entered += 1
                not_entered_av_budget += 1
                out = (DateTime+
                      " Not enough butget "+thisAsset)
                print(out)
                trader.write_log(out)
                if trader.flexible_lot_ratio:
                    raise ValueError(out+" and trader.flexible_lot_ratio== True")
                EXIT, rewind = trader.update_candidates()
                if len(trader.list_opened_positions)==0:
                    approached = 0
            if (trader.next_candidate!= None and 
                trader.next_candidate.entry_time<time_stamp):# and len(trader.list_opened_positions)==0
                out = (DateTime+
                      " ERROR! Next exit time should be "+
                      "later than current time stamp."+" Updating candidates")
                print(out)
                trader.write_log(out)
                #raise ValueError(out)
                # updating candidates
                EXIT, rewind = trader.update_candidates()
#                error()
        #    # update time stamp
        #    else:
            if approached:
                if not rewind:
                    event_idx += 1
                else:
                    original_time = DateTime
#                    out = "WARNING! "+original_time+" Rewind @ index "+str(event_idx)
#                    print(out)
#                    trader.write_log(out)
                    event_idx -= 100
#                    trader.list_banned_counter[trader.list_banned_counter>-1] = \
#                        trader.list_banned_counter[trader.list_banned_counter>-1]+100
                    while DateTimes[event_idx]==original_time:
                        event_idx -= 100
#                        trader.list_banned_counter[trader.list_banned_counter>-1] = \
#                            trader.list_banned_counter[trader.list_banned_counter>-1]+100
                    #print("Rewinded to "+DateTimes[event_idx].decode("utf-8"))
                    rewinded += 1
        
            else:
                if trader.next_candidate!= None:
                    ets = dt.datetime.strftime(trader.next_candidate.entry_time,
                                               '%Y.%m.%d %H:%M:%S').encode('utf-8')
                    if ets<DateTimes[-1]:
                        idx = np.intersect1d(np.where(DateTimes[event_idx:]==ets),
                                np.where(
                                Assets[event_idx:]==trader.next_candidate.asset))
                        if idx.shape[0]>0:
                            event_idx = idx[0]+event_idx
#                            trader.list_banned_counter[trader.list_banned_counter>-1] = \
#                                np.maximum(trader.list_banned_counter[trader.list_banned_counter>-1]-idx[0],-1)
                            approached = 1
                        else:
                            out = " WARNING! Entry not found. Skipping it"
                            print(out)
                            trader.write_log(out)
                            out = journal.iloc[trader.journal_idx:trader.journal_idx+1].to_string()
                            print(out)
                            trader.write_log(out)
                            EXIT, rewind = trader.update_candidates()
                    else:
                        print("Ran out of events. EXIT")
                        break
                else:
                    break
             #end of while event_idx<nEvents:
        
        # close all open positions before week change
        if len(trader.list_opened_positions)>0:
            out = ("WARNING! Exit time not in this week: "+
                   str(trader.list_opened_positions[-1].exit_time))
            trader.write_log(out)
            print(out)
        # get statistics
        t_entries = n_pos_opened+trader.n_pos_extended
        perEntries = t_entries/journal_entries
        if trader.n_entries>0:
            per_net_success = trader.net_successes/trader.n_entries
            average_loss = np.abs(trader.average_loss)/(trader.n_entries-trader.net_successes)
            per_gross_success = trader.gross_successes/trader.n_entries
            perSL = stoplosses/trader.n_entries
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
        
        out = ("GROI = {0:.3f}% ".format(100*GROI)+
               "ROI = {0:.3f}%".format(100*ROI)+
               " Sum GROI = {0:.3f}%".format(100*trader.tGROI_live)+
               " Sum ROI = {0:.3f}%".format(100*trader.tROI_live)+
               " Final budget {0:.2f}E".format(trader.budget)+
               " Earnings {0:.2f}E".format(earnings)+
               " per earnings {0:.3f}%".format(100*(
                      trader.budget-trader.init_budget)/trader.init_budget)+
                " ROI per position {0:.3f}%".format(ROI_per_entry))
        print("\n"+out)
        trader.write_log(out)
        trader.write_summary(out)
        out = ("Number entries "+str(trader.n_entries)+
               " per entries {0:.2f}%".format(100*perEntries)+
               " per net success "+"{0:.3f}%".format(100*per_net_success)+
               " per gross success "+"{0:.3f}%".format(100*per_gross_success)+
               " av loss {0:.3f}%".format(100*average_loss)+
               " per sl {0:.3f}%".format(100*perSL))
        print(out)
        trader.write_log(out)
        trader.write_summary(out)
        out = ("DONE. Time: "+"{0:.2f}".format((time.time()-tic)/60)+" mins")
        print(out)
        trader.write_log(out)
        results.update_results()
        
        init_budget = trader.budget
        
        
        total_entries = int(np.sum(results.number_entries))
        total_successes = int(np.sum(results.net_successes))
        total_failures = total_entries-total_successes
        if total_entries>0:
            per_gross_success = 100*np.sum(results.gross_successes)/total_entries
            per_net_succsess = 100*np.sum(results.net_successes)/total_entries
            av_groi = results.sum_GROI/total_entries
            av_roi = results.sum_ROI/total_entries
        else:
            per_gross_success = 0.0
            per_net_succsess = 0.0
            av_groi = 0.0
            av_roi = 0.0
        average_loss = np.sum(results.total_losses)/((
                total_entries-np.sum(results.net_successes))*trader.pip)
        average_win = np.sum(results.total_wins)/(np.sum(
                results.net_successes)*trader.pip)
        RR = total_successes*average_win/(average_loss*total_failures)
        out = ("Total GROI = {0:.3f}% ".format(results.total_GROI)+
               "Total ROI = {0:.3f}% ".format(results.total_ROI)+
               "Sum GROI = {0:.3f}% ".format(results.sum_GROI)+
               "Sum ROI = {0:.3f}%".format(results.sum_ROI)+
               " Earnings {0:.2f}E ".format(results.total_earnings)+
               "av GROI {0:.4f}% av ROI {1:.4}".format(av_groi,av_groi))
        print("\n"+out)
        trader.write_log(out)
        trader.write_summary(out)
        out = ("Total entries sofar "+str(total_entries)+
           " per entries {0:.2f}".format(100*total_entries/total_journal_entries)+
           " percent gross success {0:.2f}%".format(per_gross_success)+
           " percent nett success {0:.2f}%".format(per_net_succsess)+
           " average loss {0:.2f}p".format(average_loss)+
           " average win {0:.2f}p".format(average_win)+
           " RR 1 to {0:.2f} \n".format(RR))
        print(out)
        trader.write_log(out)
        trader.write_summary(out)
    # end of weeks
    out = ("\nTotal GROI = {0:.3f}% ".format(results.total_GROI)+
           "Total ROI = {0:.3f}% ".format(results.total_ROI)+
           "Sum GROI = {0:.3f}% ".format(results.sum_GROI)+
           "Sum ROI = {0:.3f}% ".format(results.sum_ROI)+
           "Accumulated earnings {0:.2f}E".format(results.total_earnings))
    print(out)
    trader.write_log(out)
    trader.write_summary(out)
    out = ("Total entries "+str(total_entries)+
           " per entries {0:.2f}".format(100*total_entries/total_journal_entries)+
           " percent gross success {0:.2f}%".format(per_gross_success)+
           " percent nett success {0:.2f}%".format(per_net_succsess)+
           " average loss {0:.2f}p".format(average_loss)+
           " average win {0:.2f}p".format(average_win)+
           " RR 1 to {0:.2f}".format(RR))
    print(out)
    trader.write_log(out)
    trader.write_summary(out)
    out = ("DONE. Total time: "+"{0:.2f}".format((time.time()-tic)/60)+" mins")
    print(out)
    trader.write_log(out)
    out = "Results file: "+start_time+"results.p"
    print(out)
    out = "Positions file: "+positions_file
    print(out)
    # build results dictionary
    results_dict = {'dts':results.datetimes,
                    'GROIs':results.GROIs,
                    'ROIs':results.ROIs,
                    'earnings':earnings}
    pickle.dump( results_dict, open( directory+start_time+"_results.p", "wb" ))
    
    results_description = ",TGROI{0:.2f}".format(results.total_GROI)+\
                          "TROI{0:.2f}".format(results.total_ROI)+\
                          "SGROI{0:.2f}".format(results.sum_GROI)+\
                          "SROI{0:.2f}".format(results.sum_ROI)+\
                          "EAR{0:.2f}".format(results.total_earnings)+\
                          "TENT"+str(total_entries)+","+\
                          "PGS{0:.2f}".format(per_gross_success)+\
                          "PNS{0:.2f}".format(per_net_succsess)+\
                          "AVL{0:.2f}".format(average_loss)+\
                          "AVW{0:.2f}".format(average_win)+\
                          "RR{0:.2f}".format(RR)