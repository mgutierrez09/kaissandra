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
from inputs import Data
import pickle

#from TradingManager_v10 import write_log

class Results:
    
    def __init__(self):
        
        self.total_GROI = 0.0
        self.GROIs = np.array([])
        
        self.total_ROI = 0.0
        self.ROIs = np.array([])
        
        self.sum_GROI = 0.0
        self.sum_GROIs = np.array([])
        
        self.sum_ROI = 0.0
        self.sum_ROIs = np.array([])
        
        self.total_earnings = 0.0
        self.earnings = np.array([])
        
        self.number_entries = np.array([])
        
        self.net_successes = np.array([])
        self.total_losses = np.array([])
        self.total_wins = np.array([])
        self.gross_successes = np.array([])
        self.number_stoplosses = np.array([])
        
        self.sl_levels = np.array([5, 10, 15, 20])
        self.n_slots = 10
        self.expectations_matrix = np.zeros((self.sl_levels.shape[0], self.n_slots))
        
    def update_results(self):
        
        self.sum_GROI += 100*trader.tGROI_live
        self.sum_GROIs = np.append(self.sum_GROIs, 100*trader.tGROI_live)
        
        self.sum_ROI += 100*trader.tROI_live
        self.sum_ROIs = np.append(self.sum_ROIs, 100*trader.tROI_live)
        
        self.total_GROI += 100*GROI
        self.GROIs = np.append(self.GROIs, 100*GROI)
        
        self.total_ROI += 100*ROI
        self.ROIs = np.append(self.ROIs, 100*ROI)
        
        self.total_earnings += earnings
        self.earnings = np.append(self.earnings, earnings)
        
        self.number_entries = np.append(self.number_entries, n_entries)
        self.net_successes = np.append(self.net_successes, trader.net_successes)
        self.total_losses = np.append(self.total_losses, np.abs(trader.average_loss))
        self.total_wins = np.append(self.total_wins, np.abs(trader.average_win))
        self.gross_successes = np.append(self.gross_successes, trader.gross_successes)
        self.number_stoplosses = np.append(self.number_stoplosses, stoplosses)
        
    def update_expectations_matrix(self, trader):
        
        return None

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
    
    def __init__(self, direct='', thr_sl=1000, thr_tp=1000, fix_spread=False, 
                 fixed_spread_pips=2, max_lots_per_pos=.1, flexible_lot_ratio=False, 
                 lb_mc_op=0.6, lb_md_op=0.6, lb_mc_ext=0.6, lb_md_ext=0.6, 
                 ub_mc_op=1, ub_md_op=1, ub_mc_ext=1, ub_md_ext=1,
                 if_dir_change_close=False, if_dir_change_extend=False, 
                 name='',use_GRE=False,t_index=3,IDr=None,IDgre=None,epoch='11',
                 weights=np.array([0,1])):
        
        self.name = name
        self.dir_origin = direct
        
        self.thr_sl=thr_sl 
        self.thr_tp=thr_tp
        
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
        
        self.if_dir_change_close = if_dir_change_close
        self.if_dir_change_extend = if_dir_change_extend
        
         # load GRE
        self.use_GRE = use_GRE
        self.IDr = IDr
        self.IDgre = IDgre
        self.epoch = epoch
        self.t_index = t_index
        self.weights = weights
        self._load_GRE()
        
    def _load_GRE(self):
        """ Load strategy efficiency matrix GRE """
        # shape GRE: (model.seq_len+1, len(thresholds_mc), len(thresholds_md), 
        #int((model.size_output_layer-1)/2))
        if self.use_GRE:
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
        else:
            self.GRE = None
    
    def _get_idx(self, p):
        """ Get probability index """
        
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
    
    def get_profitability(self, p_mc, p_md, level):
        '''
        '''
        if self.use_GRE:
            
            return self.GRE[self._get_idx(p_mc), self._get_idx(p_md), level]
        else:
            return 0.0
class Trader:
    
    def __init__(self, next_candidate, init_budget=10000,log_file='',summary_file=''):
        
        self.list_opened_positions = []
        self.map_ass_idx2pos_idx = np.array([-1 for i in range(len(data.AllAssets))])
        self.list_count_events = []
        self.list_stop_losses = []
        self.list_take_profits = []
        self.list_lots_per_pos = []
        self.list_lots_entry = []
        self.list_last_bid = []
        self.list_last_ask = []
        self.list_sl_thr_vector = []
        
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
        
        self.tROI_live = 0.0
        self.tGROI_live = 0.0
        
        self.net_successes = 0 
        self.average_loss = 0.0
        self.average_win = 0.0
        self.gross_successes = 0
        self.n_pos_extended = 0
        
        self.save_log = 1
        if log_file=='':
            start_time = dt.datetime.strftime(dt.datetime.now(),'%y_%m_%d_%H_%M_%S')
            self.log_file = ("../RNN/resultsLive/simulate/trader/"+
                             start_time+"trader_v30.log")
        else:
            self.log_file = log_file
        if summary_file=='':
            self.summary_file = ("../RNN/resultsLive/simulate/trader/"+
                             start_time+"summary.log")
        else:
            self.summary_file = summary_file
    
    def _get_thr_sl_vector(self):
        
        return self.next_candidate.entry_bid*(1-self.next_candidate.direction*
                                              self.sl_thr_vector*self.pip)
    
    def add_position(self, idx, lots):
        self.list_opened_positions.append(self.next_candidate)
        self.list_count_events.append(0)
        self.list_lots_per_pos.append(lots)
        self.list_lots_entry.append(lots)
        self.list_last_bid.append(bid)
        self.list_last_ask.append(ask)
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
#        self.list_sl_thr_vector.append(self._get_thr_sl_vector())
        
        
    def remove_position(self, idx):
#        print(self.map_ass_idx2pos_idx[idx])
#        print(self.list_opened_positions[:self.map_ass_idx2pos_idx[idx]]+self.list_opened_positions[self.map_ass_idx2pos_idx[idx]+1:])
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
        self.list_lots_per_pos = self.list_lots_per_pos[
                :self.map_ass_idx2pos_idx[idx]]+self.list_lots_per_pos[
                        self.map_ass_idx2pos_idx[idx]+1:]
        self.list_lots_entry = self.list_lots_entry[
                :self.map_ass_idx2pos_idx[idx]]+self.list_lots_entry[
                        self.map_ass_idx2pos_idx[idx]+1:]
        self.list_last_bid = self.list_last_bid[
                :self.map_ass_idx2pos_idx[idx]]+self.list_last_bid[
                        self.map_ass_idx2pos_idx[idx]+1:]
        self.list_last_ask = (self.list_last_ask[:self.map_ass_idx2pos_idx[idx]]+
                              self.list_last_ask[self.map_ass_idx2pos_idx[idx]+1:])
        
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
        #entry_time = self.list_opened_positions[self.map_ass_idx2pos_idx[idx]].entry_time
        
        self.list_opened_positions[self.map_ass_idx2pos_idx[idx]] = self.next_candidate
        
        self.list_opened_positions[self.map_ass_idx2pos_idx[idx]].entry_bid = entry_bid
        self.list_opened_positions[self.map_ass_idx2pos_idx[idx]].entry_ask = entry_ask
        self.list_opened_positions[self.map_ass_idx2pos_idx[idx]].direction = direction
        self.list_opened_positions[self.map_ass_idx2pos_idx[idx]].bet = bet
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
        else:
            self.last_fix_spread = strategys[name2str_map[
                    self.next_candidate.strategy]].fix_spread
            self.last_fixed_spread_ratio = strategys[
                    name2str_map[self.next_candidate.strategy]].fixed_spread_ratio
            self.next_candidate = None
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
        
        margin = 0.0
        if strategys[name2str_map[self.next_candidate.strategy]].fix_spread:
            second_condition_open = (self.next_candidate!= None and 
                                     self.next_candidate.p_mc>=strategys[
                                    name2str_map[self.next_candidate.strategy]
                                    ].lb_mc_op and 
                                    self.next_candidate.p_md>=strategys[
                                    name2str_map[self.next_candidate.strategy]
                                    ].lb_md_op and
                                     self.next_candidate.p_mc<strategys[
                                    name2str_map[self.next_candidate.strategy]
                                    ].ub_mc_op and 
                                    self.next_candidate.p_md<strategys[
                                    name2str_map[self.next_candidate.strategy]
                                    ].ub_md_op)
            
        elif (not strategys[name2str_map[self.next_candidate.strategy]
            ].fix_spread and 
            not strategys[name2str_map[self.next_candidate.strategy]].use_GRE):
            second_condition_open = (self.next_candidate!= None and 
                                     e_spread<strategys[name2str_map[
                                    self.next_candidate.strategy]
                                    ].fixed_spread_ratio and 
                                    self.next_candidate.p_mc>=strategys[
                                    name2str_map[self.next_candidate.strategy]
                                    ].lb_mc_op and 
                                     self.next_candidate.p_md>=strategys[
                                    name2str_map[self.next_candidate.strategy]
                                    ].lb_md_op and 
                                    self.next_candidate.p_mc<strategys[
                                    name2str_map[self.next_candidate.strategy]
                                    ].ub_mc_op and 
                                    self.next_candidate.p_md<strategys[
                                    name2str_map[self.next_candidate.strategy]
                                    ].ub_md_op)
            
        elif (not strategys[name2str_map[
            self.next_candidate.strategy]].fix_spread and 
            strategys[name2str_map[self.next_candidate.strategy]].use_GRE):
            second_condition_open = (self.next_candidate!= None and 
                                     strategys[name2str_map[
                                    self.next_candidate.strategy]
                                    ].get_profitability(self.next_candidate.p_mc, 
                                    self.next_candidate.p_md, 
                                    int(np.abs(self.next_candidate.bet
                                    )-1))>e_spread/self.pip+margin)
            
        else:
            print("ERROR: fix_spread cannot be fixed if GRE is in use")
            error()
            
        return second_condition_open
        
    def check_primary_condition_for_extention(self, time_stamp, idx):
        """
        
        """
        return (self.next_candidate.asset==self.list_opened_positions[
                self.map_ass_idx2pos_idx[idx]].asset and 
            time_stamp==self.next_candidate.entry_time)

    def check_secondary_condition_for_extention(self, idx):
        """
        
        """
        margin = 0.5
        if not strategys[name2str_map[self.next_candidate.strategy]].use_GRE:
            condition =  (self.list_opened_positions[
                        self.map_ass_idx2pos_idx[idx]
                        ].direction==self.next_candidate.direction and 
                        self.next_candidate.p_mc>=strategys[name2str_map[
                        self.next_candidate.strategy]].lb_mc_ext and 
                        self.next_candidate.p_md>=strategys[name2str_map[
                        self.next_candidate.strategy]].lb_md_ext and
                        self.next_candidate.p_mc<strategys[name2str_map[
                        self.next_candidate.strategy]].ub_mc_ext and 
                        self.next_candidate.p_md<strategys[name2str_map[
                        self.next_candidate.strategy]].ub_md_ext)
        else:
            condition = (self.next_candidate!= None and 
                         strategys[name2str_map[self.next_candidate.strategy]
                         ].get_profitability(self.next_candidate.p_mc, 
                         self.next_candidate.p_md, 
                         int(np.abs(self.next_candidate.bet)-1))>margin)
        
        
        return condition

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
        if (self.list_opened_positions[self.map_ass_idx2pos_idx[idx]].direction*(
                bid-self.list_stop_losses[self.map_ass_idx2pos_idx[idx]])<=0):
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
    
    def close_position(self, date_time, ass, idx, lot_ratio=None, partial_close=False):
        # update results and exit market
        direction = self.list_opened_positions[self.map_ass_idx2pos_idx[idx]].direction
        
        Bi = self.list_opened_positions[self.map_ass_idx2pos_idx[idx]].entry_bid
        Ai = self.list_opened_positions[self.map_ass_idx2pos_idx[idx]].entry_ask
        Ao = self.list_last_ask[self.map_ass_idx2pos_idx[idx]]
        Bo = self.list_last_bid[self.map_ass_idx2pos_idx[idx]]
        
        # if it's full close, get the raminings of lots as lots ratio
        if not partial_close:
            lot_ratio = 1.0
        roi_ratio = lot_ratio*self.list_lots_per_pos[
                self.map_ass_idx2pos_idx[idx]]/self.list_lots_entry[
                        self.map_ass_idx2pos_idx[idx]]

        if np.isnan(roi_ratio):
            error()
        
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
        
        self.available_budget += self.list_lots_per_pos[
                self.map_ass_idx2pos_idx[idx]]*self.LOT*(lot_ratio+ROI_live)
        self.available_bugdet_in_lots = self.available_budget/self.LOT
        self.budget_in_lots += self.list_lots_per_pos[
                self.map_ass_idx2pos_idx[idx]]*ROI_live
        self.budget += self.list_lots_entry[
                self.map_ass_idx2pos_idx[idx]]*ROI_live*self.LOT
        earnings = self.budget-self.init_budget
        self.gross_earnings += self.list_lots_entry[
                self.map_ass_idx2pos_idx[idx]]*GROI_live*self.LOT
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
        if partial_close:
            partial_string = ' Partial'
        else:
            partial_string = ' Full'
        out =( date_time.decode("utf-8")+partial_string+" close "+ass+" Ratio {0:.2f}".format(lot_ratio)+
              " GROI {2:.3f}% Spread {1:.3f}% ROI = {0:.3f}%".format(
                      100*ROI_live,100*spread,100*GROI_live)+" TGROI {1:.3f}% TROI = {0:.3f}%".format(
                      100*self.tROI_live,100*self.tGROI_live)+" Earnings {0:.2f}".format(earnings)
              +". Remeining open "+str(len(self.list_opened_positions)))
        self.write_log(out)
        print(out)
        assert(lot_ratio<=1.00 and lot_ratio>0)
    
    def open_position(self, idx, approached, n_entries, n_pos_opened, lots):
        """
        
        """
        self.available_budget -= lots*self.LOT
        self.available_bugdet_in_lots -= lots
        approached = 1
        n_entries += 1
        n_pos_opened += 1
        
        # update vector of opened positions
        self.add_position(idx, lots)
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
        
        return approached, n_entries, n_pos_opened, EXIT, rewind
    
    def extend_position(self, idx):
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
               " spread={0:.3f}".format(100*e_spread))
        print(out)
        self.write_log(out)
        print(out)
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
                    error()
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
    
    def write_summary(self, out):
        """
        Write summary into a file
        """
        if self.save_log:
            file = open(self.summary_file,"a")
            file.write(out+"\n")
            file.close()
        return None


def load_in_memory(data, init_list_index, end_list_index,root_dir='D:/SDC/py/Data/'):
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
    for ass in data.assets:
        thisAsset = data.AllAssets[str(ass)]
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
                    if day in data.dateTest[init_list_index:end_list_index+1]:
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


if __name__ == '__main__':
    
    dateTest = ([                                                   '2018.03.09',
                '2018.03.12','2018.03.13','2018.03.14','2018.03.15','2018.03.16',
                '2018.03.19','2018.03.20','2018.03.21','2018.03.22','2018.03.23',
                '2018.03.26','2018.03.27','2018.03.28','2018.03.29','2018.03.30',
                '2018.04.02','2018.04.03','2018.04.04','2018.04.05','2018.04.06',
                '2018.04.09','2018.04.10','2018.04.11','2018.04.12','2018.04.13',
                '2018.04.16','2018.04.17','2018.04.18','2018.04.19','2018.04.20',
                '2018.04.23','2018.04.24','2018.04.25','2018.04.26','2018.04.27',
                '2018.04.30','2018.05.01','2018.05.02','2018.05.03','2018.05.04',
                '2018.05.07','2018.05.08','2018.05.09','2018.05.10','2018.05.11',
                '2018.05.14','2018.05.15','2018.05.16','2018.05.17','2018.05.18',
                '2018.05.21','2018.05.22','2018.05.23','2018.05.24','2018.05.25',
                '2018.05.28','2018.05.29','2018.05.30','2018.05.31','2018.06.01',
                '2018.06.04','2018.06.05','2018.06.06','2018.06.07','2018.06.08',
                '2018.06.11','2018.06.12','2018.06.13','2018.06.14','2018.06.15',
                '2018.06.18','2018.06.19','2018.06.20','2018.06.21','2018.06.22',
                '2018.06.25','2018.06.26','2018.06.27','2018.06.28','2018.06.29',
                '2018.07.02','2018.07.03','2018.07.04','2018.07.05','2018.07.06',
                '2018.07.09','2018.07.10','2018.07.11','2018.07.12','2018.07.13',
                '2018.07.30','2018.07.31','2018.08.01','2018.08.02','2018.08.03',
                '2018.08.06','2018.08.07','2018.08.08','2018.08.09','2018.08.10']+
               ['2018.08.13','2018.08.14','2018.08.15','2018.08.16','2018.08.17',
                '2018.08.20','2018.08.21','2018.08.22','2018.08.23','2018.08.24',
                '2018.08.27','2018.08.28','2018.08.29','2018.08.30','2018.08.31',
                '2018.09.03','2018.09.04','2018.09.05','2018.09.06','2018.09.07',
                '2018.09.10','2018.09.11','2018.09.12','2018.09.13','2018.09.14',
                '2018.09.17','2018.09.18','2018.09.19','2018.09.20','2018.09.21',
                '2018.09.24','2018.09.25','2018.09.26','2018.09.27'])
    # data structure
    data=Data(movingWindow=100,nEventsPerStat=1000,
     dateTest = dateTest)

#    ['2017.11.27','2017.11.28','2017.11.29','2017.11.30','2017.12.01',
#                 '2017.12.04','2017.12.05','2017.12.06','2017.12.07','2017.12.08']+
        
    nExS = data.nEventsPerStat-1
    tic = time.time()
    # init positions vector
    # build asset to index mapping
    ass2index_mapping = {}
    ass_index = 0
    for ass in data.assets:
        ass2index_mapping[data.AllAssets[str(ass)]] = ass_index
        ass_index += 1
    
#    list_epoch = [39,13]
#    list_use_GRE = [True,True]
#    list_t_index = [1,3]
#    list_lb_mc_op = [0.6,0.5]
#    list_lb_md_op = [0.6,0.5]
#    list_lb_mc_ext = [0.6,0.5]
#    list_lb_md_ext = [0.6,0.5]
#    list_ub_mc_op = [1,1]
#    list_ub_md_op = [1,1]
#    list_ub_mc_ext = [1,1]
#    list_ub_md_ext = [1,1]
#    list_thr_sl = [1000,1000]
#    list_thr_tp = [1000,1000]
#    list_fix_spread = [False,False]
#    list_fixed_spread_pips = [4,4]
#    list_max_lots_per_pos = [.1,.1]
#    list_flexible_lot_ratio = [False,False]
#    list_if_dir_change_close = [False,False]
#    list_if_dir_change_extend = [False,False]
#    list_name = ['69','66']
#    list_IDresults = ['100269','100266']
    
    list_IDresults = ['100287','100285','100286','100277fNSRs']#
    list_IDgre = ['100287','100285','100286','100277fNSRs']
    list_name = ['87','85','86','77']
    list_epoch_gre = [6,16,6,13]
    list_epoch_journal = [6,16,6,13]
    list_use_GRE = [True,True,True,True]
    list_weights = [np.array([.5,.5]),np.array([.5,.5]),np.array([.5,.5]),np.array([.5,.5])]
    list_t_index = [2,4,3,3]
    list_lb_mc_op = [.5,.5,.5,.6]
    list_lb_md_op = [.8,.8,.8,.7]
    list_lb_mc_ext = [.5,.5,.5,.5]
    list_lb_md_ext = [.6,.6,.6,.6]
    list_ub_mc_op = [1,1,1,1]
    list_ub_md_op = [1,1,1,1]
    list_ub_mc_ext = [1,1,1,1]
    list_ub_md_ext = [1,1,1,1]
    list_thr_sl = [1000,1000,1000,1000]
    list_thr_tp = [1000,1000,1000,1000]
    list_fix_spread = [False,False,False,False]
    list_fixed_spread_pips = [4,4,4,4]
    list_max_lots_per_pos = [.1,.1,.1,.1]
    list_flexible_lot_ratio = [False,False,False,False]
    list_if_dir_change_close = [False,False,False,False]
    list_if_dir_change_extend = [False,False,False,False]
    
#    list_IDresults = ['100287']#
#    list_IDgre = ['100287']
#    list_name = ['87']
#    list_epoch_gre = [6]
#    list_epoch_journal = [6]
#    list_use_GRE = [True]
#    list_weights = [np.array([.5,.5])]
#    list_t_index = [2]
#    list_lb_mc_op = [.6]
#    list_lb_md_op = [.8]
#    list_lb_mc_ext = [.5]
#    list_lb_md_ext = [.6]
#    list_ub_mc_op = [1]
#    list_ub_md_op = [1]
#    list_ub_mc_ext = [1]
#    list_ub_md_ext = [1]
#    list_thr_sl = [1000]
#    list_thr_tp = [1000]
#    list_fix_spread = [False]
#    list_fixed_spread_pips = [4]
#    list_max_lots_per_pos = [.1]
#    list_flexible_lot_ratio = [False]
#    list_if_dir_change_close = [False]
#    list_if_dir_change_extend = [False]
    
    
    strategys = [Strategy(direct='../RNN/results/',thr_sl=list_thr_sl[i], 
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
                          name=list_name[i],use_GRE=list_use_GRE[i],
                          t_index=list_t_index[i],IDr=list_IDresults[i],
                          IDgre=list_IDgre[i],
                          epoch=str(list_epoch_gre[i]),weights=list_weights[i]) 
                          for i in range(len(list_t_index))]
    
    name2str_map = {}
    for n in range(len(list_name)):
        name2str_map[strategys[n].name] = n
    
    load_from_live = False
    if not load_from_live:
        
        resultsDir = "../RNN/results/"
        list_journal_dir = [resultsDir+list_IDresults[i]+"/t"+
                            str(list_t_index[i])+"/"+list_IDresults[i]+"t"+
                            str(list_t_index[i])+"mc"+str(list_lb_mc_ext[i])+
                            "/"+list_IDresults[i]+"t"+str(list_t_index[i])+
                            "mc"+str(list_lb_mc_ext[i])+"md"+str(list_lb_md_ext[i])+
                            "/" for i in range(len(list_t_index))]
        list_journal_name = ["J"+list_IDresults[i]+"t"+str(list_t_index[i])+"mc"+
                             str(list_lb_mc_ext[i])+"md"+str(list_lb_md_ext[i])+
                             "e"+str(list_epoch_journal[i])+".txt" 
                             for i in range(len(list_t_index))]
        entry_time_column = 'DT1'#'Entry Time
        exit_time_column = 'DT2'#'Exit Time
        entry_bid_column = 'B1'
        entry_ask_column = 'A1'
        exit_ask_column = 'A2'
        exit_bid_column = 'B2'
        root_dir = 'D:/SDC/py/Data/'
        list_journal_all_days = [pd.read_csv(list_journal_dir[i]+
                                             list_journal_name[i], 
                                             sep='\t').sort_values(
                                            by=[entry_time_column]).reset_index(
                                                    ).drop(labels='level_0',
                                                    axis=1).assign(
                                                    strategy=list_name[i]) 
                                            for i in range(len(list_t_index))]# .drop(labels='level_0',axis=1)
    else:
        
        resultsDir = "../RNN/resultsLive/back_test/"
        IDresults = ["100248GREN2"]
        list_journal_dir = [resultsDir+IDresults[0]+'T'+
                            str(list_t_index[0])+
                            'E'+str(list_epoch_journal[0])+'/']
        list_journal_name = [IDresults[0]+'T'+str(list_t_index[0])+
                             'E'+str(list_epoch_journal[0])+'.txt']
        entry_time_column = 'Entry Time'#'Entry Time
        exit_time_column = 'Exit Time'#'Exit Time
        entry_bid_column = 'Bi'
        entry_ask_column = 'Ai'
        exit_ask_column = 'Ao'
        exit_bid_column = 'Bo'
        root_dir = 'D:/SDC/py/Data_test/'#'D:/SDC/py/Data_aws_3/'#'D:/SDC/py/Data_DL3/'
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
    start_time = dt.datetime.strftime(dt.datetime.now(),'%y_%m_%d_%H_%M_%S')
    log_file = "../RNN/resultsLive/simulate/trader/"+start_time+"trader_v30.log"
    summary_file = "../RNN/resultsLive/simulate/trader/"+start_time+"summary.log"
    ##### loop over different day groups #####
    day_index = 0
    t_journal_entries = 0
    change_dir = 0
    rewinded = 0
    while day_index<len(data.dateTest):
        counter_back = 0
        init_list_index = day_index#data.dateTest[day_index]
        pd_init_index = journal_all_days[journal_all_days[entry_time_column].str.find(data.dateTest[day_index])>-1]
        while day_index<len(data.dateTest)-1 and pd_init_index.shape[0]==0:
            day_index += 1
            pd_init_index = journal_all_days[journal_all_days[entry_time_column].str.find(data.dateTest[day_index])>-1]
        init_index = pd_init_index.index[0]
        # find sequence of consecutive days in test days
        while day_index<len(data.dateTest)-1 and dt.datetime.strptime(data.dateTest[day_index],'%Y.%m.%d')+dt.timedelta(1)==dt.datetime.strptime(data.dateTest[day_index+1],'%Y.%m.%d'):
            day_index += 1
        pd_end_index = journal_all_days[journal_all_days[exit_time_column].str.find(data.dateTest[day_index])>-1]
        while day_index>init_list_index and pd_end_index.shape[0]==0:
            day_index -= 1
            counter_back += 1
            pd_end_index = journal_all_days[journal_all_days[exit_time_column].str.find(data.dateTest[day_index])>-1]
        end_index = pd_end_index.index[-1]
        end_list_index = day_index+counter_back
        journal = journal_all_days.loc[init_index:end_index+1]
        journal_entries = journal.shape[0]
        t_journal_entries += journal_entries
        day_index += counter_back+1
        # init trader
        trader = Trader(Position(journal.iloc[0], AD_resume, eROIpb), 
                                 init_budget=init_budget, log_file=log_file,
                                 summary_file=summary_file)
        
        out = ("Week from "+data.dateTest[init_list_index]+
               " to "+data.dateTest[end_list_index]+
               " Number journal entries "+str(journal_entries))
        print(out)
        trader.write_log(out)
            
        load_in_RAM = True
        DateTimes, SymbolBids, SymbolAsks, Assets, nEvents = load_in_memory(data, 
                                                                            init_list_index, 
                                                                            end_list_index,
                                                                            root_dir=root_dir)
    
        # set counters and flags
        n_entries = 0
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
        
        # get to 
        while event_idx<nEvents:
            rewind = 0
            no_budget = False
            # get time stamp
            time_stamp = dt.datetime.strptime(DateTimes[event_idx].decode("utf-8"),
                                              '%Y.%m.%d %H:%M:%S')
            bid = int(np.round(SymbolBids[event_idx]*100000))/100000
            ask = int(np.round(SymbolAsks[event_idx]*100000))/100000
            e_spread = (ask-bid)/ask
            ass_idx = ass2index_mapping[Assets[event_idx].decode("utf-8")]
            list_idx = trader.map_ass_idx2pos_idx[ass_idx]
            # update bid and ask lists if exist
            if list_idx>-1:
                trader.list_last_bid[list_idx] = bid
                trader.list_last_ask[list_idx] = ask
            
            condition_open = (trader.chech_ground_condition_for_opening() and 
                              trader.check_primary_condition_for_opening() and 
                              trader.check_secondary_contition_for_opening())
    
            if condition_open:
                profitability = strategys[name2str_map[trader.next_candidate.strategy
                                                       ]].get_profitability(
                                                       trader.next_candidate.p_mc, 
                                                        trader.next_candidate.p_md, 
                                                        int(np.abs(trader.next_candidate.bet)-1))
                out = (time_stamp.strftime('%Y.%m.%d %H:%M:%S')+" "+
                       Assets[event_idx].decode("utf-8")+
                       " p_mc {0:.3f}".format(trader.next_candidate.p_mc)+
                       " p_md {0:.3f}".format(trader.next_candidate.p_md)+
                      " pofitability {0:.3f}".format(profitability)+
                      " E_spread {0:.3f}".format(e_spread/trader.pip)+" Bet "+
                      str(int(np.abs(trader.next_candidate.bet)-1))+
                      " Open Condition met")
                print(out)
                trader.write_log(out)
                
                # open market
                if not trader.is_opened(ass_idx):
                    
                    # assign budget
                    #lot_ratio = 1#/(len(trader.list_opened_positions)+1)
                    lots = trader.assign_lots(DateTimes[event_idx])
                    #print("trader.available_bugdet_in_lots "+str(trader.available_bugdet_in_lots))
    
                    if trader.available_bugdet_in_lots>=lots:
                        approached, n_entries, n_pos_opened, EXIT, rewind = trader.open_position(ass_idx, 
                                                                                                 approached, 
                                                                                                 n_entries, 
                                                                                                 n_pos_opened, 
                                                                                                 lots=lots)
                    else:
                        no_budget = True
            elif (trader.chech_ground_condition_for_opening() and 
                  trader.check_primary_condition_for_opening() and 
                  not trader.check_secondary_contition_for_opening()):
                #pass
                profitability = strategys[name2str_map
                                          [trader.next_candidate.strategy]
                                          ].get_profitability(
                                        trader.next_candidate.p_mc, 
                                        trader.next_candidate.p_md, 
                                        int(np.abs(trader.next_candidate.bet)-1))
                out = (time_stamp.strftime('%Y.%m.%d %H:%M:%S')+" "+
                       Assets[event_idx].decode("utf-8")+
                       " p_mc {0:.3f}".format(trader.next_candidate.p_mc)+
                       " p_md {0:.3f}".format(trader.next_candidate.p_md)+
                      " pofitability {0:.3f}".format(profitability)+
                      " E_spread {0:.3f}".format(e_spread/trader.pip)+" Bet "+
                      str(int(np.abs(trader.next_candidate.bet)-1))+
                      " Open condition not met")
                print(out)
                trader.write_log(out)
            # check if a position is already opened
            if trader.is_opened(ass_idx):
                trader.count_one_event(ass_idx)
                results.update_expectations_matrix(trader)
        #        stop_loss = update_stoploss(trader.list_opened_positions[trader.map_ass_idx2pos_idx[ass_idx]], stop_loss)
                exit_pos, stoplosses = trader.is_stoploss_reached(ass_idx, stoplosses)
                if exit_pos:
                    # save original exit time for skipping positions
                    #original_exit_time = trader.list_opened_positions[0].exit_time
                    trader.close_position(DateTimes[event_idx], 
                                          Assets[event_idx].decode("utf-8"), 
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
                            if trader.check_secondary_condition_for_extention(
                                    ass_idx):
                                # include third condition for thresholds
                                # extend deadline
                                EXIT, rewind = trader.extend_position(ass_idx)
                                
                            else:
                                # extend conditon not met
                                profitability = strategys[name2str_map[
                                        trader.next_candidate.strategy]
                                        ].get_profitability(trader.next_candidate.p_mc, 
                                                            trader.next_candidate.p_md, 
                                                            int(np.abs(trader.next_candidate.bet)-1))
                                out = (time_stamp.strftime('%Y.%m.%d %H:%M:%S')+
                                       " "+Assets[event_idx].decode("utf-8")+
                                       " p_mc "+str(trader.next_candidate.p_mc)+
                                       " p_md "+str(trader.next_candidate.p_md)+
                                      " pofitability "+str(profitability)+
                                      " E_spread "+str(e_spread/trader.pip)+
                                      " Bet "+str(int(np.abs(trader.next_candidate.bet)-1))+
                                      " Extend Condition not met")
                                print(out)
                                trader.write_log(out)
                                
                                EXIT, rewind = trader.update_candidates()
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
                        
                        trader.close_position(DateTimes[event_idx], 
                                              Assets[event_idx].decode("utf-8"), 
                                              ass_idx)
                        # reset approched
                        if len(trader.list_opened_positions)==0:
                            approached = 0
                        # if exit was signaled in the previous, then leave
    #                    if EXIT:
    #                        break
                    
                # end of if count_events==nExS or timeout==0 or exit_pos:
            # uptade posiitions if ADm is too low or eGROI is too low too
            if (trader.chech_ground_condition_for_opening() and 
                trader.check_primary_condition_for_opening() and not 
                trader.check_secondary_contition_for_opening()):
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
                print(time_stamp.strftime('%Y.%m.%d %H:%M:%S')+
                      " Not enough butget "+Assets[event_idx].decode("utf-8"))
                if trader.flexible_lot_ratio:
                    error()
                EXIT, rewind = trader.update_candidates()
                if len(trader.list_opened_positions)==0:
                    approached = 0
            if (trader.next_candidate!= None and 
                trader.next_candidate.entry_time<time_stamp):# and len(trader.list_opened_positions)==0
                out = (time_stamp.strftime('%Y.%m.%d %H:%M:%S')+
                      " WARNING! Next exit time should be "+
                      "later than current time stamp."+" Updating candidates")
                print(out)
                trader.write_log(out)
                error()
                # updating candidates
                EXIT, rewind = trader.update_candidates()
#                error()
        #    # update time stamp
        #    else:
            if approached:
                if not rewind:
                    event_idx += 1
                else:
                    original_time = time_stamp.strftime('%Y.%m.%d %H:%M:%S')
                    out = "WARNING! "+original_time+" Rewind @ index "+str(event_idx)
                    print(out)
                    trader.write_log(out)
                    event_idx -= 100
                    while DateTimes[event_idx].decode("utf-8")==original_time:
                        event_idx -= 100
                    print("Rewinded to "+DateTimes[event_idx].decode("utf-8"))
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
                        print("Run out of events. EXIT")
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
        if n_entries>0:
            per_net_success = trader.net_successes/n_entries
            average_loss = np.abs(trader.average_loss)/(n_entries-trader.net_successes)
            per_gross_success = trader.gross_successes/n_entries
            perSL = stoplosses/n_entries
            ROI_per_entry = 100*trader.tROI_live/n_entries
        else:
            per_net_success = 0.0
            average_loss = 0.0
            per_gross_success = 0.0
            perSL = 0.0
            ROI_per_entry = 0.0
        
        GROI = trader.gross_earnings/trader.init_budget
        earnings = trader.budget-trader.init_budget
        ROI = earnings/trader.init_budget
        
        out = ("\nGROI = {0:.3f}% ".format(100*GROI)+
               "ROI = {0:.3f}%".format(100*ROI)+
               " Sum GROI = {0:.3f}%".format(100*trader.tGROI_live)+
               " Sum ROI = {0:.3f}%".format(100*trader.tROI_live)+
               " Final budget {0:.2f}E".format(trader.budget)+
               " Earnings {0:.2f}E".format(earnings)+
               " per earnings {0:.3f}%".format(100*(
                      trader.budget-trader.init_budget)/trader.init_budget)+
                " ROI per position {0:.3f}%".format(ROI_per_entry))
        print(out)
        trader.write_log(out)
        trader.write_summary(out)
        out = ("Number entries "+str(n_entries)+
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
        trader.write_summary(out)
        results.update_results()
        
        init_budget = trader.budget
        
        out = ("\nTotal GROI = {0:.3f}% ".format(results.total_GROI)+
               "Total ROI = {0:.3f}% ".format(results.total_ROI)+
               "Sum GROI = {0:.3f}% ".format(results.sum_GROI)+
               "Sum ROI = {0:.3f}%".format(results.sum_ROI)+
               " Accumulated earnings {0:.2f}E\n".format(results.total_earnings))
        print(out)
        trader.write_log(out)
        trader.write_summary(out)
        total_entries = int(np.sum(results.number_entries))
        total_successes = int(np.sum(results.net_successes))
        total_failures = total_entries-total_successes
        per_gross_success = 100*np.sum(results.gross_successes)/total_entries
        per_net_succsess = 100*np.sum(results.net_successes)/total_entries
        average_loss = np.sum(results.total_losses)/((
                total_entries-np.sum(results.net_successes))*trader.pip)
        average_win = np.sum(results.total_wins)/(np.sum(
                results.net_successes)*trader.pip)
        RR = total_successes*average_win/(average_loss*total_failures)
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
    out = ("DONE. Total time: "+"{0:.2f}".format((time.time()-tic)/60)+" mins\n")
    print(out)
    trader.write_log(out)

# Combine .6/.7/.6/.6 .7/.7/.6/.6 real spread<.03 fix invest to .1 vol epoch 11 t_index 3 till 180810 IDr 100248/100246 not closing not extending if direction changes 
#Total GROI = 11.064% Total ROI = 7.700% Sum GROI = 11.411% Sum ROI = 7.923% Accumulated earnings 792.35E
#Total entries 198 percent gross success 69.70% percent nett success 59.60% average loss 7.48p average win 11.79p RR 1 to 2.32
#DONE. Total time: 52.51 mins
    
# Combine .6/.7/.6/.6 .7/.7/.6/.6 .6/.6 real spread<.03x2 real spread<.01 fix invest to .1 vol epoch 11/10/15 t_index 3/1/15 till 180810 IDr 100248/100246/251 not closing not extending if direction changes 
#Total GROI = 20.678% Total ROI = 7.213% Sum GROI = 21.567% Sum ROI = 7.378% Accumulated earnings 737.79E
#Total entries 1083 per entries 3.08 percent gross success 59.83% percent nett success 52.35% average loss 5.91p average win 6.68p RR 1 to 1.24
#DONE. Total time: 100.18 mins

# .6/.6 fixed spread .01 fix invest to .1 vol epoch 15 t_index 3 till 180810 IDr 100251 not closing not extending if direction changes 
#Total GROI = 51.820% Total ROI = 13.550% Sum GROI = 55.015% Sum ROI = 14.325% Accumulated earnings 1432.50E
#Total entries 4118 per entries 14.43 percent gross success 60.17% percent nett success 54.15% average loss 5.50p average win 5.29p RR 1 to 1.14
#DONE. Total time: 44.09 mins

# .6/.6_.7 real spread<.01 fix invest to .1 vol epoch 15 t_index 3 till 180810 IDr 100251 not closing not extending if direction changes 
#Total GROI = 11.696% Total ROI = 0.306% Sum GROI = 11.651% Sum ROI = 0.287% Accumulated earnings 28.73E
#Total entries 951 per entries 3.33 percent gross success 57.62% percent nett success 50.89% average loss 5.59p average win 5.45p RR 1 to 1.01
#DONE. Total time: 234.00 mins

# .6/.7/.6/.6 real spread<.04 fix invest to .1 vol epoch 11 t_index 3 till 180810 IDr 100248GRE not closing not extending if direction changes 
#Total GROI = 10.412% Total ROI = 6.630% Sum GROI = 10.656% Sum ROI = 6.775% Accumulated earnings 677.49E
#Total entries 184 per entries 6.46 percent gross success 69.02% percent nett success 60.87% average loss 8.19p average win 11.32p RR 1 to 2.15
#DONE. Total time: 14.47 mins

# GRE fix invest to .1 vol epoch 11 t_index 3 till 180810 IDr 100248GRE not closing not extending if direction changes 
#Total GROI = 9.656% Total ROI = 7.466% Sum GROI = 9.944% Sum ROI = 7.671% Accumulated earnings 767.05E
#Total entries 143 per entries 5.02 percent gross success 69.23% percent nett success 63.64% average loss 6.18p average win 11.96p RR 1 to 3.39
#DONE. Total time: 16.45 mins
    
# GRE fix invest to .1 vol epoch 11 t_index 3 till 180810 IDr 100248GRE extension margin .5 pips not closing not extending if direction changes 
#Total GROI = 10.008% Total ROI = 7.841% Sum GROI = 10.313% Sum ROI = 8.056% Accumulated earnings 805.58E
#Total entries 144 per entries 5.05 percent gross success 70.14% percent nett success 63.19% average loss 6.08p average win 12.40p RR 1 to 3.50
#DONE. Total time: 17.92 mins

# GRE fix invest to .1 vol epoch 11 t_index 3 till 180810 IDr 100248GREN2 extension margin .5 pips not closing not extending if direction changes 
#Total GROI = 10.079% Total ROI = 7.925% Sum GROI = 10.391% Sum ROI = 8.146% Accumulated earnings 814.58E
#Total entries 144 per entries 5.05 percent gross success 70.83% percent nett success 64.58% average loss 6.22p average win 12.17p RR 1 to 3.57
#DONE. Total time: 19.92 mins
    
# GRE fix invest to .1 vol epoch 11 t_index 3 till 180810 IDr 100248GREN2 extension margin .5 pips rewind 10 not closing not extending if direction changes 
#Total GROI = 9.373% Total ROI = 6.960% Sum GROI = 9.627% Sum ROI = 7.131% Accumulated earnings 713.14E
#Total entries 159 per entries 5.58 percent gross success 66.04% percent nett success 60.38% average loss 6.69p average win 11.82p RR 1 to 2.69
#DONE. Total time: 31.92 mins
    
# GRE fix invest to .1 vol epoch 11 t_index 3 till 180810 IDr 100248GREN2 extension margin .5 pips rewind 0 not closing not extending if direction changes 
#Total GROI = 10.062% Total ROI = 7.650% Sum GROI = 10.363% Sum ROI = 7.865% Accumulated earnings 786.50E
#Total entries 162 per entries 5.69 percent gross success 67.28% percent nett success 61.73% average loss 6.66p average win 12.00p RR 1 to 2.90
#DONE. Total time: 16.11 mins
    
# GRE fix invest to .1 vol epoch 11 t_index 3 till 180810 IDr 100248GREN2 extension margin .3 and .2 pips rewind 0 not closing not extending if direction changes 
#Total GROI = 10.287% Total ROI = 7.960% Sum GROI = 10.600% Sum ROI = 8.196% Accumulated earnings 819.63E
#Total entries 155 per entries 5.44 percent gross success 69.68% percent nett success 65.16% average loss 7.51p average win 12.13p RR 1 to 3.02
#DONE. Total time: 16.44 mins

# GRE fix invest to .1 vol epoch 11 t_index 3 till 180810 IDr 100248GREN not closing not extending if direction changes 
#Total GROI = 12.172% Total ROI = 6.432% Sum GROI = 12.482% Sum ROI = 6.554% Accumulated earnings 655.38E
#Total entries 356 per entries 12.50 percent gross success 62.92% percent nett success 54.49% average loss 7.47p average win 9.62p RR 1 to 1.54
#DONE. Total time: 15.48 mins

# GRE fix invest to .1 vol epoch 11 t_index 3 till 180810 IDr 100248GREN2 extention margin .5 pips not closing not extending if direction changes 
#Total GROI = 10.191% Total ROI = 8.035% Sum GROI = 10.514% Sum ROI = 8.265% Accumulated earnings 826.49E
#Total entries 144 per entries 5.12 percent gross success 71.53% percent nett success 64.58% average loss 6.12p average win 12.24p RR 1 to 3.65
#DONE. Total time: 15.75 mins
    
# GRE fix invest to .1 vol epoch 11 t_index 3 till 180810 IDr 100266 extention margin .5 pips not closing not extending if direction changes 
#Total GROI = 6.407% Total ROI = 5.029% Sum GROI = 6.540% Sum ROI = 5.127% Accumulated earnings 512.69E +
# Total GROI = 1.050% Total ROI = 0.156% Sum GROI = 1.049% Sum ROI = 0.153% Accumulated earnings 15.33E
# Sum ROI = 5.278%

# GRE fix invest to .1 vol epoch 11 t_index 3 till 180810 IDr 100248GREN2 100266 extention margin .5 pips only 2018
#Total GROI = 8.132% Total ROI = 5.234% Sum GROI = 8.350% Sum ROI = 5.352% Accumulated earnings 535.19E
#Total entries 256 per entries 0.37 percent gross success 65.23% percent nett success 57.42% average loss 7.07p average win 8.89p RR 1 to 1.69
#DONE. Total time: 113.64 mins
    
# GRE fix invest to .1 vol epoch 39 t_index 3 till 180810 IDr 100269 extention margin .5 pips only 2018
#Total GROI = 5.563% Total ROI = 3.070% Sum GROI = 5.597% Sum ROI = 3.085% Accumulated earnings 308.51E
#Total entries 203 per entries 0.59 percent gross success 59.11% percent nett success 51.23% average loss 6.63p average win 9.28p RR 1 to 1.47
#DONE. Total time: 96.70 mins
    
# GRE fix invest to .1 vol epoch 39 t_index 3 till 180810 IDr 100269 100266 extention margin .5 pips only 2018
#Total GROI = 10.562% Total ROI = 6.488% Sum GROI = 10.882% Sum ROI = 6.642% Accumulated earnings 664.23E
#Total entries 361 per entries 0.56 percent gross success 59.28% percent nett success 52.63% average loss 6.64p average win 9.47p RR 1 to 1.59
#DONE. Total time: 142.83 mins

# GRE fix invest to .1 vol epoch 13 t_index 3 IDr 100277 extention margin .5 pips from 2018.3.9 to .9.27
#Total GROI = 7.101% Total ROI = 5.069% Sum GROI = 7.246% Sum ROI = 5.160% Accumulated earnings 515.96E
#Total entries 172 per entries 10.46 percent gross success 64.53% percent nett success 61.05% average loss 7.58p average win 9.75p RR 1 to 2.02
#DONE. Total time: 12.19 mins

# .5/.8_.6 real spread<4 pips fix invest to .1 vol epoch 6 t_index 3 IDr 100286 from 2018.3.9 to .9.27
#Total GROI = 9.121% Total ROI = 5.625% Sum GROI = 9.353% Sum ROI = 5.760% Accumulated earnings 576.00E
#Total entries 197 per entries 1.72 percent gross success 71.07% percent nett success 62.94% average loss 7.67p average win 9.16p RR 1 to 2.03
#DONE. Total time: 40.46 mins
    
# GRE fix invest to .1 vol epoch 6 t_index 3 IDr 100286 from 2018.3.9 to .9.27    
#Total GROI = 4.786% Total ROI = 3.307% Sum GROI = 4.843% Sum ROI = 3.334% Accumulated earnings 333.39E
#Total entries 172 per entries 1.50 percent gross success 58.72% percent nett success 56.40% average loss 6.74p average win 8.64p RR 1 to 1.66
#DONE. Total time: 36.09 mins

# GREex fix invest to .1 vol epoch 6 t_index 3 IDr 100286 from 2018.3.9 to .9.27    
#Total GROI = 16.249% Total ROI = 6.847% Sum GROI = 16.836% Sum ROI = 7.032% Accumulated earnings 703.17E
#Total entries 726 per entries 6.33 percent gross success 61.85% percent nett success 54.13% average loss 7.00p average win 7.72p RR 1 to 1.30
#DONE. Total time: 37.07 mins

# wGRE=[.5,.5] fix invest to .1 vol epoch 6 t_index 3 IDr 100286 from 2018.3.9 to .9.27
#Total GROI = 11.014% Total ROI = 6.789% Sum GROI = 11.361% Sum ROI = 6.954% Accumulated earnings 695.43E
#Total entries 376 per entries 3.28 percent gross success 63.03% percent nett success 59.84% average loss 8.15p average win 8.56p RR 1 to 1.56
#DONE. Total time: 40.00 mins

# wGRE=[.5,.5] fix invest to .1 vol epoch 13 t_index 3 IDr 100277 from 2018.3.9 to .9.27 new AD_resune
#Total GROI = 12.966% Total ROI = 6.641% Sum GROI = 13.341% Sum ROI = 6.786% Accumulated earnings 678.56E
#Total entries 465 per entries 5.01 percent gross success 61.29% percent nett success 52.90% average loss 6.25p average win 8.32p RR 1 to 1.50
#DONE. Total time: 32.67 mins

# wGRE=[.5,.5]/[.5,.5] fix invest to .1 vol epoch 13/6 t_index 3 IDr 100277 100286 from 2018.3.9 to .9.27 new AD_resune
#Total GROI = 13.414% Total ROI = 6.131% Sum GROI = 13.782% Sum ROI = 6.233% Accumulated earnings 623.29E
#Total entries 568 per entries 2.74 percent gross success 62.68% percent nett success 54.58% average loss 7.47p average win 8.23p RR 1 to 1.32
#DONE. Total time: 53.02 mins

# wGRE=[.5,.5] fix invest to .1 vol epoch 6 t_index 16 IDr 100285 from 2018.3.9 to .9.27
#Total GROI = 13.119% Total ROI = 6.259% Sum GROI = 13.509% Sum ROI = 6.378% Accumulated earnings 637.82E
#Total entries 577 per entries 6.30 percent gross success 59.97% percent nett success 53.38% average loss 7.31p average win 8.46p RR 1 to 1.32
#DONE. Total time: 29.99 mins
    
# wGRE=[0,1] fix invest to .1 vol epoch 6 t_index 16 IDr 100285 from 2018.3.9 to .9.27
#Total GROI = 19.277% Total ROI = 6.691% Sum GROI = 19.941% Sum ROI = 6.815% Accumulated earnings 681.53E
#Total entries 888 per entries 9.70 percent gross success 59.80% percent nett success 52.14% average loss 7.26p average win 8.14p RR 1 to 1.22
#DONE. Total time: 29.46 mins

# wGRE=[0,1]/[0,1] fix invest to .1 vol epoch 16/6 t_index 3 IDr 100285 100286 from 2018.3.9 to .9.27 new AD_resune
#Total GROI = 22.585% Total ROI = 6.946% Sum GROI = 23.473% Sum ROI = 7.096% Accumulated earnings 709.62E
#Total entries 1117 per entries 5.41 percent gross success 60.34% percent nett success 52.91% average loss 7.31p average win 7.71p RR 1 to 1.18
#DONE. Total time: 42.84 mins

# wGRE=[0,1]/[0,1]/[0,1] fix invest to .1 vol epoch 16/6/13 t_index 3 IDr 100285 100286 100277 from 2018.3.9 to .9.27 new AD_resune
#Total GROI = 25.602% Total ROI = 7.277% Sum GROI = 26.696% Sum ROI = 7.409% Accumulated earnings 740.89E
#Total entries 1267 per entries 4.23 percent gross success 59.83% percent nett success 51.46% average loss 7.09p average win 7.82p RR 1 to 1.17
#DONE. Total time: 41.57 mins

# wGRE=[.5,.5]/[.5,.5]/[.5,.5] fix invest to .1 vol epoch 16/6/13 t_index 3 IDr 100285 100286 100277 from 2018.3.9 to .9.27 new AD_resune
#Total GROI = 17.811% Total ROI = 8.128% Sum GROI = 18.595% Sum ROI = 8.331% Accumulated earnings 833.14E
#Total entries 784 per entries 2.62 percent gross success 61.35% percent nett success 55.61% average loss 7.87p average win 8.19p RR 1 to 1.30
#DONE. Total time: 46.10 mins

# wGRE=[.5,.5] fix invest to .1 vol epoch 6 t_index 2 IDr 100287 from 2018.3.9 to .9.27
#Total GROI = 13.371% Total ROI = 7.186% Sum GROI = 13.779% Sum ROI = 7.383% Accumulated earnings 738.32E
#Total entries 542 per entries 5.13 percent gross success 60.89% percent nett success 55.90% average loss 7.61p average win 8.44p RR 1 to 1.41
#DONE. Total time: 41.07 mins
# ULTIMATELY!!!!!!!!!!!!!!!!!!!