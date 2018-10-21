# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 11:40:34 2017

@author: mgutierrez

This script merges the sequential run of the entire chain (Fetcher/Dispatcher/live RNN/Trader)
with the option of getting trader's input from a pre-saved Journal
"""
import time
import os
import numpy as np
import pandas as pd
import datetime as dt
import pickle
import h5py
import tensorflow as tf
from simulate_trader_eventbased_v20 import load_in_memory
from data_manager_hdf5 import Data, load_stats, initFeaturesLive, extractFeaturesLive
from RNN import modelRNN
import shutil


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
        
    def update_results(self, GROI, earnings, ROI, n_entries, stoplosses):
        
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
        
#        if trader.list_opened_positions[trader.map_ass_idx2pos_idx[idx]].direction*(bid-trader.list_stop_losses[trader.map_ass_idx2pos_idx[idx]])<=0:
#            pass
        
        return None

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
        self.direction = np.sign(self.bet)
        self.level = int(np.abs(self.bet)-1)
        self.p_mc = float(journal_entry['P_mc'])
        self.p_md = float(journal_entry['P_md'])
        self.e_spread = float(journal_entry['E_spread'])
        self.deadline = int(journal_entry['Deadline'])
        self.strategy = strategy
        self.profitability = self.strategy.get_profitability(self.p_mc, self.p_md, self.level)
        

class Strategy():
    
    def __init__(self, direct='',thr_sl=1000, thr_tp=1000, fix_spread=False, fixed_spread_pips=2, max_lots_per_pos=.1, flexible_lot_ratio=False, 
                 lb_mc_op=0.6, lb_md_op=0.6, lb_mc_ext=0.6, lb_md_ext=0.6, ub_mc_op=1, ub_md_op=1, ub_mc_ext=1, ub_md_ext=1,
                 if_dir_change_close=False, if_dir_change_extend=False, name='',t_index=3,use_GRE=False,IDr=None,epoch='11'):
        
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
        self.use_GRE = use_GRE
        self.IDr = IDr
        self.epoch = epoch
        self.t_index = t_index
        self._load_GRE()
        
    def _load_GRE(self):
        '''
        '''
        # shape GRE: (model.seq_len+1, len(thresholds_mc), len(thresholds_md), int((model.size_output_layer-1)/2))
        if self.use_GRE:
            allGREs = pickle.load( open( self.dir_origin+self.IDr+"/GRE_e"+self.epoch+".p", "rb" ))
            self.GRE = allGREs[self.t_index, :, :, :]/self.pip+20
            print("GRE level 1:")
            print(self.GRE[:,:,0])
            print("GRE level 2:")
            print(self.GRE[:,:,1])
        else:
            self.GRE = None
    
    def _get_idx(self, p):
        '''
        '''
        
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
    
    def get_profitability(self, p_mc, p_md, level):
        '''
        '''
        if self.use_GRE:
            return self.GRE[self._get_idx(p_mc), self._get_idx(p_md), level]
        else:
            return None
        
 
class Trader:
    
    def __init__(self, results_dir="../RNN/resultsLive/back_test/trader/", init_budget=10000, log_file_time=''):
        
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
        self.list_deadlines = []
        
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
        self.results_dir = results_dir
        if log_file_time=='':
            
            start_time = dt.datetime.strftime(dt.datetime.now(),'%y_%m_%d_%H_%M_%S')
            self.log_file = self.results_dir+start_time+"_BT_trader_v30.log"
            self.log_summary = self.results_dir+start_time+"_BT_summary_v30.log"
        else:
            if run_back_test:
                tag = '_BT_'
            else:
                tag = '_LI_'
            self.log_file = self.results_dir+log_file_time+tag+"trader_v30.log"
            self.log_summary = self.results_dir+log_file_time+tag+"summary_v30.log"
        
        if not os.path.exists(self.results_dir):
            os.mkdir(self.results_dir)
        # results tracking
        resultsInfoHeader = "Asset,Entry Time,Exit Time,Position,Bi,Ai,Bo,Ao,ticks_d,GROI,Spread,ROI,Profit"
        file = open(self.log_summary,"a")
        file.write(resultsInfoHeader+"\n")
        file.close()
        
        # flow control
        self.EXIT = 0
        self.rewind = 0
        self.approached = 0
        self.swap_pending = 0
    
    def _get_thr_sl_vector(self):
        '''
        '''
        return self.next_candidate.entry_bid*(1-self.next_candidate.direction*self.sl_thr_vector)
    
    def add_new_candidate(self, position):
        '''
        '''
        self.next_candidate = position
    
    def add_position(self, idx, lots, bid, ask, deadline):
        '''
        '''
        self.list_opened_positions.append(self.next_candidate)
        self.list_count_events.append(0)
        self.list_lots_per_pos.append(lots)
        self.list_lots_entry.append(lots)
        self.list_last_bid.append(bid)
        self.list_last_ask.append(ask)
        self.map_ass_idx2pos_idx[idx] = len(self.list_count_events)-1
        self.list_stop_losses.append(self.next_candidate.entry_bid*(1-self.next_candidate.direction*self.next_candidate.strategy.thr_sl))
        self.list_take_profits.append(self.next_candidate.entry_bid*(1+self.next_candidate.direction*self.next_candidate.strategy.thr_tp))
        self.list_deadlines.append(deadline)
        
        
    def remove_position(self, idx):
        '''
        '''
        self.list_opened_positions = self.list_opened_positions[:self.map_ass_idx2pos_idx[idx]]+self.list_opened_positions[self.map_ass_idx2pos_idx[idx]+1:]
        self.list_count_events = self.list_count_events[:self.map_ass_idx2pos_idx[idx]]+self.list_count_events[self.map_ass_idx2pos_idx[idx]+1:]
        self.list_stop_losses = self.list_stop_losses[:self.map_ass_idx2pos_idx[idx]]+self.list_stop_losses[self.map_ass_idx2pos_idx[idx]+1:]
        self.list_take_profits = self.list_take_profits[:self.map_ass_idx2pos_idx[idx]]+self.list_take_profits[self.map_ass_idx2pos_idx[idx]+1:]
        self.list_lots_per_pos = self.list_lots_per_pos[:self.map_ass_idx2pos_idx[idx]]+self.list_lots_per_pos[self.map_ass_idx2pos_idx[idx]+1:]
        self.list_lots_entry = self.list_lots_entry[:self.map_ass_idx2pos_idx[idx]]+self.list_lots_entry[self.map_ass_idx2pos_idx[idx]+1:]
        self.list_last_bid = self.list_last_bid[:self.map_ass_idx2pos_idx[idx]]+self.list_last_bid[self.map_ass_idx2pos_idx[idx]+1:]
        self.list_last_ask = self.list_last_ask[:self.map_ass_idx2pos_idx[idx]]+self.list_last_ask[self.map_ass_idx2pos_idx[idx]+1:]
        self.list_deadlines = self.list_deadlines[:self.map_ass_idx2pos_idx[idx]]+self.list_deadlines[self.map_ass_idx2pos_idx[idx]+1:]
        
        mask = self.map_ass_idx2pos_idx>self.map_ass_idx2pos_idx[idx]
        self.map_ass_idx2pos_idx[idx] = -1
        self.map_ass_idx2pos_idx = self.map_ass_idx2pos_idx-mask*1#np.maximum(,-1)
        
    def update_position(self, idx):
        '''
        '''
        # reset counter
        self.list_count_events[self.map_ass_idx2pos_idx[idx]] = 0
        
        entry_bid = self.list_opened_positions[self.map_ass_idx2pos_idx[idx]].entry_bid
        entry_ask = self.list_opened_positions[self.map_ass_idx2pos_idx[idx]].entry_ask
        direction = self.list_opened_positions[self.map_ass_idx2pos_idx[idx]].direction
        bet = self.list_opened_positions[self.map_ass_idx2pos_idx[idx]].bet
        entry_time = self.list_opened_positions[self.map_ass_idx2pos_idx[idx]].entry_time
        
        self.list_opened_positions[self.map_ass_idx2pos_idx[idx]] = self.next_candidate
        
        self.list_opened_positions[self.map_ass_idx2pos_idx[idx]].entry_bid = entry_bid
        self.list_opened_positions[self.map_ass_idx2pos_idx[idx]].entry_ask = entry_ask
        self.list_opened_positions[self.map_ass_idx2pos_idx[idx]].direction = direction
        self.list_opened_positions[self.map_ass_idx2pos_idx[idx]].bet = bet
        self.list_opened_positions[self.map_ass_idx2pos_idx[idx]].entry_time = entry_time
    
    def is_opened(self, idx):
        '''
        '''
        if self.map_ass_idx2pos_idx[idx]>=0:
            return True
        else:
            return False
    
    def count_one_event(self, idx):
        self.list_count_events[self.map_ass_idx2pos_idx[idx]] += 1
    
    def check_contition_for_opening(self):
        '''
        '''
        this_strategy = self.next_candidate.strategy
        e_spread = self.next_candidate.e_spread
        margin = 0.0
        if this_strategy.fix_spread and not this_strategy.use_GRE:
            condition_open = (self.next_candidate.p_mc>=this_strategy.lb_mc_op and self.next_candidate.p_md>=this_strategy.lb_md_op and
                                         self.next_candidate.p_mc<this_strategy.ub_mc_op and self.next_candidate.p_md<this_strategy.ub_md_op)
        elif not this_strategy.fix_spread and not this_strategy.use_GRE:
            condition_open = (self.next_candidate!= None and e_spread<this_strategy.fixed_spread_ratio and self.next_candidate.p_mc>=this_strategy.lb_mc_op and 
                                         self.next_candidate.p_md>=this_strategy.lb_md_op and self.next_candidate.p_mc<this_strategy.ub_mc_op and 
                                         self.next_candidate.p_md<this_strategy.ub_md_op)
        elif not this_strategy.fix_spread and this_strategy.use_GRE:
            condition_open = self.next_candidate.profitability>e_spread+margin
        else:
            print("ERROR: fix_spread cannot be fixed if GRE is in use")
            error()
            
        return condition_open
    
    def check_primary_condition_for_extention(self, idx):
        
        return self.list_opened_positions[self.map_ass_idx2pos_idx[idx]].direction==self.next_candidate.direction
        
        

    def check_secondary_condition_for_extention(self):
        '''
        '''
        this_strategy = self.next_candidate.strategy
        margin = 0.5
        if not this_strategy.use_GRE:
            condition_open = (self.next_candidate.p_mc>=this_strategy.lb_mc_ext and self.next_candidate.p_md>=this_strategy.lb_md_ext and
                self.next_candidate.p_mc<this_strategy.ub_mc_ext and self.next_candidate.p_md<this_strategy.ub_md_ext)
        else:
            condition_open = self.next_candidate.profitability>margin
        return condition_open

    def update_stoploss(self, idx, bid):
        # update stoploss
        this_strategy = self.next_candidate.strategy
        if self.list_opened_positions[self.map_ass_idx2pos_idx[idx]].direction == 1:
            self.list_stop_losses[self.map_ass_idx2pos_idx[idx]] = max(self.list_stop_losses[self.map_ass_idx2pos_idx[idx]],
                                 bid*(1-self.list_opened_positions[self.map_ass_idx2pos_idx[idx]].direction*this_strategy.thr_sl*this_strategy.fixed_spread_ratio))
        else:
            self.list_stop_losses[self.map_ass_idx2pos_idx[idx]] = min(self.list_stop_losses[self.map_ass_idx2pos_idx[idx]],
                                 bid*(1-self.list_opened_positions[self.map_ass_idx2pos_idx[idx]].direction*this_strategy.thr_sl*this_strategy.fixed_spread_ratio))
    
    def is_stoploss_reached(self, idx, bid, event_idx):
        # check stop loss reachead
        #print(self.list_stop_losses)
        if self.list_opened_positions[self.map_ass_idx2pos_idx[idx]].direction*(bid-self.list_stop_losses[self.map_ass_idx2pos_idx[idx]])<=0:
            # exit position due to stop loss
            exit_pos = 1
            self.stoplosses += 1
            out = "Exit position due to stop loss @event idx "+str(event_idx)+" bid="+str(bid)+" sl="+str(self.list_stop_losses[self.map_ass_idx2pos_idx[idx]])
            print("\r"+out)
            self.write_log(out)
        else:
            exit_pos = 0
        return exit_pos
    
    def is_takeprofit_reached(self, this_pos, take_profit, takeprofits, bid, event_idx):
        
        # check take profit reachead
        if this_pos.direction*(bid-take_profit)>=0:
            # exit position due to stop loss
            exit_pos = 1
            takeprofits += 1
            out = "Exit position due to take profit @event idx "+str(event_idx)+". tp="+str(take_profit)
            print("\r"+out)
            self.write_log(out)
        else:
            exit_pos = 0
        return exit_pos, takeprofits
    
    def close_position(self, date_time, ass, idx, lot_ratio=None, partial_close=False):
        """
        
        """
        
        # update results and exit market
        direction = self.list_opened_positions[self.map_ass_idx2pos_idx[idx]].direction
        Ti = self.list_opened_positions[self.map_ass_idx2pos_idx[idx]].entry_time
        bet = self.list_opened_positions[self.map_ass_idx2pos_idx[idx]].bet
        Bi = self.list_opened_positions[self.map_ass_idx2pos_idx[idx]].entry_bid
        Ai = self.list_opened_positions[self.map_ass_idx2pos_idx[idx]].entry_ask
        Ao = self.list_last_ask[self.map_ass_idx2pos_idx[idx]]
        Bo = self.list_last_bid[self.map_ass_idx2pos_idx[idx]]
        # if it's full close, get the raminings of lots as lots ratio
        if not partial_close:
            lot_ratio = 1.0
            
        roi_ratio = lot_ratio*self.list_lots_per_pos[self.map_ass_idx2pos_idx[idx]]/self.list_lots_entry[self.map_ass_idx2pos_idx[idx]]
#        print("roi_ratio "+str(roi_ratio))
#        print("lot_ratio "+str(lot_ratio))
        if np.isnan(roi_ratio):
            error()
        
#        GROI_live = roi_ratio*direction*(this_bid-entry_bid)/entry_bid
        
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
        
        self.available_budget += self.list_lots_per_pos[self.map_ass_idx2pos_idx[idx]]*self.LOT*(lot_ratio+ROI_live)
        self.available_bugdet_in_lots = self.available_budget/self.LOT
        
        self.budget_in_lots += self.list_lots_per_pos[self.map_ass_idx2pos_idx[idx]]*ROI_live
        self.budget += self.list_lots_entry[self.map_ass_idx2pos_idx[idx]]*ROI_live*self.LOT
        earnings = self.budget-self.init_budget
        profit = self.list_lots_entry[self.map_ass_idx2pos_idx[idx]]*ROI_live*self.LOT
        self.gross_earnings += self.list_lots_entry[self.map_ass_idx2pos_idx[idx]]*GROI_live*self.LOT
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
        
        # write output to trader summary
        info_close = (ass+","+Ti+","+date_time+","+str(bet)+","+
                  str(Bi)+","+str(Ai)+","+str(Bo)+","+
                  str(Ao)+","+"0"+","+str(100*GROI_live)+","+str(100*spread)+","+str(100*ROI_live)+","+str(profit))
        
        file = open(self.log_summary,"a")
        file.write(info_close+"\n")
        file.close()
        
        if not partial_close:
            self.remove_position(idx)
        else:
            # decrease the lot ratio in case the position is not fully closed
            self.list_lots_per_pos[self.map_ass_idx2pos_idx[idx]] = self.list_lots_per_pos[self.map_ass_idx2pos_idx[idx]]*(1-lot_ratio)

        if partial_close:
            partial_string = ' Partial'
        else:
            partial_string = ' Full'
        
        out =( date_time+partial_string+" close "+ass+" Ratio {0:.2f}".format(lot_ratio)+
              " GROI {2:.3f}% Spread {1:.3f}% ROI = {0:.3f}%".format(
                      100*ROI_live,100*spread,100*GROI_live)+" TGROI {1:.3f}% TROI = {0:.3f}%".format(
                      100*self.tROI_live,100*self.tGROI_live)+" Earnings {0:.2f}".format(earnings)
              +". Remeining open "+str(len(self.list_opened_positions)))
        self.write_log(out)
        print("\r"+out)
        
        assert(lot_ratio<=1.00 and lot_ratio>0)
    
    def open_position(self, idx, lots, DateTime, e_spread, bid, ask, deadline):
        """
        
        """
        self.available_budget -= lots*self.LOT
        self.available_bugdet_in_lots -= lots
        self.n_entries += 1
        self.n_pos_opened += 1
        
        # update vector of opened positions
        self.add_position(idx, lots, bid, ask, deadline)
            
        out = (DateTime+" Open "+self.list_opened_positions[-1].asset+
              " Lots {0:.1f}".format(lots)+" "+str(self.list_opened_positions[-1].bet)+
              " p_mc={0:.2f}".format(self.list_opened_positions[-1].p_mc)+" p_md={0:.2f}".format(self.list_opened_positions[-1].p_md)+ " spread={0:.3f}".format(e_spread))
        print("\r"+out)
        self.write_log(out)
        
        
        return None
    
    def assign_lots(self, date_time):#, date_time, ass, idx
        '''
        '''
        this_strategy = self.next_candidate.strategy
        if not this_strategy.flexible_lot_ratio:
            if self.available_bugdet_in_lots>0:
                open_lots = min(this_strategy.max_lots_per_pos, self.available_bugdet_in_lots)
            else:
                open_lots = this_strategy.max_lots_per_pos
        else:
            # lots ratio to asssign to new asset
            open_lots = min(self.budget_in_lots/(len(self.list_opened_positions)+1),this_strategy.max_lots_per_pos)
#            print("open_lots "+str(open_lots))
#            print("self.budget_in_lots "+str(self.budget_in_lots))
#            print("self.available_bugdet_in_lots "+str(self.available_bugdet_in_lots))
#            print("self.available_bugdet_in_lots "+str(self.budget_in_lots))
            margin = 0.0001
            # check if necessary lots for opening are available
            if open_lots>self.available_bugdet_in_lots+margin:
                close_lots = (open_lots-self.available_bugdet_in_lots)/len(self.list_opened_positions)
                print("close_lots "+str(close_lots))
                if close_lots==np.inf:
                    error()
                # loop over open positions to close th close_lot_ratio ratio to allocate the new position
                for pos in range(len(self.list_opened_positions)):
                    # lots ratio to be closed for this asset
                    ass = self.list_opened_positions[pos].asset.decode("utf-8")
#                    print(ass)
#                    print("self.list_lots_per_pos[pos] "+str(self.list_lots_per_pos[pos]))
                    close_lot_ratio = close_lots/self.list_lots_per_pos[pos]
#                    print("close_lot_ratio "+str(close_lot_ratio))
                    idx = running_assets[ass2index_mapping[ass]]
                    self.close_position(date_time, ass, idx, lot_ratio = close_lot_ratio, partial_close = True)
            
            # make sure the available resources are smaller or equal than slots to open
            open_lots = min(open_lots,self.available_bugdet_in_lots)
        #print("open_lots corrected "+str(open_lots))
            
        return open_lots
    
    def get_new_entry(self, inputs, thisAsset):
        
        soft_tilde = inputs[0][0]
        e_spread = inputs[0][1]
        DateTime = inputs[0][2]
        Bi = inputs[0][3]
        Ai = inputs[0][4]
        deadline = inputs[0][5]
            
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
        #p_mg = soft_tilde[Y_tilde]
        
        new_entry = {}
        new_entry[entry_time_column] = DateTime#dt.datetime.strftime(dt.datetime.now(),'%Y.%m.%d %H:%M:%S')
        new_entry['Asset'] = thisAsset
        new_entry['Bet'] = Y_tilde
        new_entry['P_mc'] = p_mc
        new_entry['P_md'] = p_md
        new_entry[entry_bid_column] = Bi
        new_entry[entry_ask_column] = Ai
        new_entry['E_spread'] = e_spread/self.pip
        new_entry['Deadline'] = deadline
        
        return new_entry
    
    def check_new_inputs(self, inputs, thisAsset, directory_MT5_ass=''):
        '''
        <DocString>
        '''
        # TODO: make it compatible for multiple inputs at the same time
        
        new_entry = self.get_new_entry(inputs, thisAsset)
        
        out = ("New entry @ "+new_entry[entry_time_column]+" "+new_entry['Asset']+
               " P_mc "+str(new_entry['P_mc'])+" P_md "+str(new_entry['P_md'])+" Bet "+
               str(new_entry['Bet'])+" E_spread "+str(new_entry['E_spread']))
        print("\r"+out)
        self.write_log(out)
        
        position = Position(new_entry, strategys[0])
        
        self.add_new_candidate(position)
        
        ass_id = running_assets[ass2index_mapping[thisAsset]]
#        list_id = self.map_ass_idx2pos_idx[ass_id]
#        if not run_back_test and list_id>-1:
#            self.list_last_bid[list_id] = self.next_candidate.entry_bid
#            self.list_last_ask[list_id] = self.next_candidate.entry_ask

        # open market
        if not self.is_opened(ass_id):
            # check if condition for opening is met
            condition_open = self.check_contition_for_opening()
            if condition_open:
                # assign budget
                lots = self.assign_lots(new_entry[entry_time_column])
                # check if there is enough budget
                if self.available_bugdet_in_lots>=lots:
                    if not run_back_test:
                        self.send_open_command(directory_MT5_ass)
                    self.open_position(ass_id, lots, new_entry[entry_time_column], 
                                       self.next_candidate.e_spread, self.next_candidate.entry_bid, 
                                       self.next_candidate.entry_ask, self.next_candidate.deadline)
                else: # no opening due to budget lack
                    out = "Not enough budget"
                    print("\r"+out)
                    self.write_log(out)
                    # check swap of resources
                    if self.next_candidate.strategy.use_GRE and self.check_resources_swap():
                        # lauch swap of resourves
                        self.initialize_resources_swap(directory_MT5_ass)
            else:
                pass
#                print(" Condition not met")
        else: # position is opened
            # check for extension
            if self.check_primary_condition_for_extention(ass_id):
                if self.check_secondary_condition_for_extention():    
                    # include third condition for thresholds
                    # extend deadline
                    if not run_back_test:
                        self.send_open_command(directory_MT5_ass)
                    self.update_position(ass_id)
                    self.n_pos_extended += 1
                    out = (new_entry[entry_time_column]+" Extended "+thisAsset+
                       " Lots {0:.1f}".format(self.list_lots_per_pos[self.map_ass_idx2pos_idx[ass_id]])+
                       " "+str(new_entry['Bet'])+" p_mc={0:.2f}".format(new_entry['P_mc'])+
                       " p_md={0:.2f}".format(new_entry['P_md'])+ 
                       " spread={0:.3f}".format(new_entry['E_spread']))
                    #out = new_entry[entry_time_column]+" Extended "+thisAsset
                    print("\r"+out)
                    self.write_log(out)
                else: # if candidate for extension does not meet requirements
                    # TODO: link it with the stratetgy
                    pass
    #                               change_dir += 1
    #                               print(time_stamp.strftime('%Y.%m.%d %H:%M:%S')+" Exit position due to direction change "+Assets[event_idx].decode("utf-8"))
    #                                # get lots in case a new one is directly opened
    #                                lots = trader.list_lots_per_pos[trader.map_ass_idx2pos_idx[ass_idx]]
    #                                trader.close_position(DateTimes[event_idx], Assets[event_idx].decode("utf-8"), ass_idx)
    #                                # reset approched
    #                                if len(trader.list_opened_positions)==0:
    #                                    approached = 0
    #                                
    #                                if trader.chech_ground_condition_for_opening() and trader.check_primary_condition_for_opening() and trader.check_secondary_contition_for_opening():
    #                                    approached, n_entries, n_pos_opened, EXIT, rewind = trader.open_position(ass_idx, approached, n_entries, n_pos_opened, lots=lots)
            else: # if direction is different
                # if new position has higher GRE, close
                if self.next_candidate.profitability>=self.list_opened_positions[self.map_ass_idx2pos_idx[ass_id]].profitability:
                    if not run_back_test:
                        self.send_close_command(directory_MT5_ass)
                        # TODO: study the option of not only closing postiion but also changing direction
                    else:
                        # TODO: implement it for back test
                        pass
            # end of extention options
            
        return None
    
    def write_log(self, log):
        """
        <DocString>
        """
        if self.save_log:
            file = open(self.log_file,"a")
            file.write(log+"\n")
            file.close()
        return None
    
    def send_open_command(self, directory_MT5_ass):
        """
        <DocString>    
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
            except PermissionError:
                print("Error writing TT")
        
    def send_close_command(self, dirDes):
        """
        <DocString>
        """
        success = 0
        # load network output
        while not success:
            try:
                fh = open(dirDes+"LC","w", encoding='utf_16_le')
                fh.close()
                success = 1
            except PermissionError:
                print("Error writing LC")
                
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
            out = "Swap "+self.next_candidate.asset+" with "+self.list_opened_positions[op].asset
            self.write_log(out)
            print(out)
            return True
        else:
            return False
    
    def initialize_resources_swap(self, directory_MT5_ass):
        """
        <DocString>
        """
        out = self.next_candidate.entry_time+" "+self.next_candidate.asset+" initialize swap"
        self.write_log(out)
        print(out)
        #send close command if running live
        if not run_back_test:
            directory_MT5_ass2close = directory_MT5+self.list_opened_positions[self.swap_pos].asset+"/"
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
        out = self.new_swap_candidate.entry_time+" "+self.new_swap_candidate.asset+" finlaize swap"
        self.write_log(out)
        print(out)
        # open new position
        lots = self.assign_lots(self.new_swap_candidate.entry_time)
        ass_idx = running_assets[ass2index_mapping[self.new_swap_candidate.asset]]
        # check if there is enough budget
        if self.available_bugdet_in_lots>=lots:
            # send open command to MT5 if running live
            if not run_back_test:
                self.send_open_command(self.swap_directory_MT5_ass)
            # open position
            self.open_position(ass_idx, lots, self.new_swap_candidate.entry_time, 
                               self.new_swap_candidate.e_spread, self.new_swap_candidate.entry_bid, 
                               self.new_swap_candidate.entry_ask, self.new_swap_candidate.deadline)
            self.swap_pending = 0
    
    
def runRNNliveFun(tradeInfoLive, listFillingX, init, listFeaturesLive,listParSarStruct,
                  listEM,listAllFeatsLive,list_X_i, means_in,phase_shift,stds_in, 
                  stds_out,AD, thisAsset, netName,listCountPos,list_weights_matrix,
                  list_time_to_entry,list_list_soft_tildes, list_Ylive,list_Pmc_live,
                  list_Pmd_live,list_Pmg_live,EOF,countOuts,t_index, c, results_dir, 
                  results_file, model):
    """
    <DocString>
    """
    
    #print("\r"+tradeInfoLive.DateTime.iloc[-1]+" "+thisAsset+netName+" New input")
    nChannels = int(data.nEventsPerStat/data.movingWindow)
    file = 0
    #deli = "_"
    
    countIndex = 0
    countOut = 0
    count_t = 0
    
    output = []# init as an empy position
    
    # sub channel
    sc = c%phase_shift
    #rc = int(np.floor(c/phase_shift))
    
    if listFillingX[sc] and verbose_RNN:# and not simulate
        print("\r"+tradeInfoLive.DateTime.iloc[-1]+" "+thisAsset+netName+"C"+str(c)+"F"+str(file))
                
    # launch features extraction
    if init[sc]==False:
        if verbose_RNN:
            print("\r"+tradeInfoLive.DateTime.iloc[-1]+" "+thisAsset+netName+" Features inited")
        listFeaturesLive[sc],listParSarStruct[sc],listEM[sc] = initFeaturesLive(data,tradeInfoLive)
        init[sc] = True

    listFeaturesLive[sc],listParSarStruct[sc],listEM[sc] = extractFeaturesLive(tradeInfoLive,data,listFeaturesLive[sc],listParSarStruct[sc],listEM[sc])
    # check if variations can be calculated or have to wait
    #allFeats = np.append(allFeats,features,axis=1)
    listAllFeatsLive[sc] = np.append(listAllFeatsLive[sc],listFeaturesLive[sc],axis=1)
    
    if listAllFeatsLive[sc].shape[1]>nChannels:
        
        # Shift old samples forward and leave space for new one
        list_X_i[sc][0,:-1,:] = list_X_i[sc][0,1:,:]
        variationLive = np.zeros((data.nFeatures,len(data.channels)))
        for r in range(len(data.channels)):
            variationLive[:,r] = listAllFeatsLive[sc][:,-1]-listAllFeatsLive[sc][:,-(data.channels[r]+2)]#-np.min([allFeatsLive.shape[1],channsInX[r]+2])
            variationLive[nonVarIdx,0] = listAllFeatsLive[sc][data.noVarFeats,-(data.channels[r]+2)]#-np.min([allFeatsLive.shape[1],channsInX[r]+2])
            varNormed = np.minimum(np.maximum((variationLive.T-means_in[data.channels[r],:])/stds_in[data.channels[r],:],-10),10)
            # copy new entry in last pos of X
            list_X_i[sc][0,-1,r*data.nFeatures:(r+1)*data.nFeatures] = varNormed
            
        #delete old infos
        listAllFeatsLive[sc] = listAllFeatsLive[sc][:,-100:]
        listCountPos[sc]+=1
        
        if listFillingX[sc]:
            if verbose_RNN:
                print("\r"+tradeInfoLive.DateTime.iloc[-1]+" "+thisAsset+netName+" Filling X sc "+str(sc))
                
            if listCountPos[sc]>=model.seq_len-1:
                if verbose_RNN:
                    print("\r"+tradeInfoLive.DateTime.iloc[-1]+" "+thisAsset+netName+" Filling X sc "+str(sc)+" done. Waiting for output...")
                listFillingX[sc] = False
        else:
########################################### Predict #########################################################################              
            if 1:#verbose_RNN:
                print("\r"+tradeInfoLive.DateTime.iloc[-1]+" "+thisAsset+netName, sep=' ', end='', flush=True)
            
            soft_tilde = model.run_live_session(list_X_i[sc])
            # if only one t_index mode
            if type(AD)==type(None):
                soft_tilde_t = soft_tilde[:,-(model.seq_len-t_index),:]
            else:
                # if MRC mode
                for t in range(model.seq_len):
                    # get weight from AD
                    mc_idx = (np.floor(soft_tilde[0,t,0]*10)-lb_level)*thr_levels
                    if mc_idx>=0:
                        idx_AD = int(mc_idx+(np.floor(np.max(soft_tilde[0,t,1:3])*10)-lb_level))
                        w = AD[t, max(idx_AD,first_nonzero), 0]
                    else:
                        w = AD[t, first_nonzero, 0]# temp. First non-zero level is 6
                    #print(idx_AD)
#                      print(AD.shape)
#                      print(first_nonzero)
#                   update t-th matrix row from lower row
                    if t<model.seq_len-1:
                        list_weights_matrix[sc][t,:] = list_weights_matrix[sc][t+1,:]
                        # update t matrix entry with weight
#                       print(w)
                                
                    list_weights_matrix[sc][t,t] = w
                        
                    # expand dimensions of one row to fit network output
                    weights = np.expand_dims(np.expand_dims(list_weights_matrix[sc][0,:]/np.sum(list_weights_matrix[sc][0,:]),axis=0),axis=2)
                    # MRC
#                      print(weights_matrix)
#                      print(weights)
            soft_tilde_t = np.sum(weights*soft_tilde, axis=1)
                            
#############################################################################################################################  
                        
################################# Send prediciton to trader ################################################################# 
            # get condition to 
            condition = soft_tilde_t[0,0]>thr_mc
      
            # set countdown to enter market
            if condition:
                list_time_to_entry[sc].append(t_index)
                list_list_soft_tildes[sc].append(soft_tilde_t[0,:])
                if verbose_RNN:
                    print("\r"+thisAsset+netName+" ToBe sent DateTime "+tradeInfoLive.DateTime.iloc[-1]+
                                              " P_mc "+str(soft_tilde_t[0,0])+" P_md "+str(np.max(soft_tilde_t[0,1:3])))
            elif test and verbose_RNN:
                print("\r"+thisAsset+netName+" DateTime "+tradeInfoLive.DateTime.iloc[-1]+
                      " P_mc "+str(soft_tilde_t[0,0])+" P_md "+str(np.max(soft_tilde_t[0,1:3])))
                #print("Prediction in market change")
                # Snd prediction to trading robot
            if len(list_time_to_entry[sc])>0 and list_time_to_entry[sc][0]==0:
                count_t += 1 
                e_spread = (tradeInfoLive.SymbolAsk.iloc[-1]-tradeInfoLive.SymbolBid.iloc[-1])/tradeInfoLive.SymbolAsk.iloc[-1]
#                tradeManager([list_list_soft_tildes[sc][0], e_spread])
                output = [list_list_soft_tildes[sc][0], e_spread, tradeInfoLive.DateTime.iloc[-1],
                          tradeInfoLive.SymbolBid.iloc[-1], tradeInfoLive.SymbolAsk.iloc[-1], data.nEventsPerStat]
                
                if verbose_RNN:
                    print(thisAsset+netName+" Sent DateTime "+tradeInfoLive.DateTime.iloc[-1]+
                                              " P_mc "+str(list_list_soft_tildes[sc][0][0])+" P_md "+str(np.max(list_list_soft_tildes[sc][0][1:3])))
                
                list_time_to_entry[sc] = list_time_to_entry[sc][1:]
                list_list_soft_tildes[sc] = list_list_soft_tildes[sc][1:]
                
            list_time_to_entry[sc] = [list_time_to_entry[sc][i]-1 for i in range(len(list_time_to_entry[sc]))]
###############################################################################################################################
                        
################################################# Evaluation ##################################################################
            
            
            prob_mc = np.array([soft_tilde_t[0,0]])
                        
            if prob_mc>thr_mc:
                Y_tilde_idx = np.argmax(soft_tilde_t[0,3:])#np.argmax(soft_tilde_t[0,1:3])#np.array([])
            else:
                Y_tilde_idx = int((model.size_output_layer-1)/2) # zero index
                            
            Y_tilde = np.array([Y_tilde_idx-(model.size_output_layer-1)/2]).astype(int)
            prob_md = np.array([np.max([soft_tilde_t[0,1],soft_tilde_t[0,2]])])
            prob_mg = soft_tilde_t[0:,np.argmax(soft_tilde_t[0,3:])]
            
            # Check performance. Evaluate prediction
            list_Ylive[sc] = np.append(list_Ylive[sc],Y_tilde,axis=0)
            list_Pmc_live[sc] = np.append(list_Pmc_live[sc],prob_mc,axis=0)
            list_Pmd_live[sc] = np.append(list_Pmd_live[sc],prob_md,axis=0)
            list_Pmg_live[sc] = np.append(list_Pmg_live[sc],prob_mg,axis=0)
            
            # wait for output to come
            if listCountPos[sc]>nChannels+model.seq_len+t_index-1:
                            
                Output_i=(tradeInfoLive.SymbolBid.iloc[-2]-EOF.SymbolBid.iloc[c])/stds_out[0,data.lookAheadIndex]
                countOut+=1
                            
                Y = (np.minimum(np.maximum(np.sign(Output_i)*np.round(abs(Output_i)*model.outputGain),-
                            (model.size_output_layer-1)/2),(model.size_output_layer-1)/2)).astype(int)
                            
                look_back_index = -nChannels-t_index-1
                # compare prediction with actual output 
                pred = list_Ylive[sc][look_back_index]
                p_mc = list_Pmc_live[sc][look_back_index]
                p_md = list_Pmd_live[sc][look_back_index]
                p_mg = list_Pmg_live[sc][look_back_index]
                # delete older entries if real time test is running
                #if 1:#not simulate:
                list_Ylive[sc] = list_Ylive[sc][look_back_index:]
                list_Pmc_live[sc] = list_Pmc_live[sc][look_back_index:]
                list_Pmd_live[sc] = list_Pmd_live[sc][look_back_index:]
                list_Pmg_live[sc] = list_Pmg_live[sc][look_back_index:]
                

                countOuts[c]+=1
                                
                # if prediction is different than 0, evaluate
                if pred!=0:
                    
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
                    
                    columnsResultInfo = ["Asset","Entry Time","Exit Time","Bet","Outcome","Diff","Bi","Ai","Bo","Ao","GROI","Spread","ROI","P_mc","P_md","P_mg"]
                    #resultInfo = pd.DataFrame(columns = columnsResultInfo)
                    resultInfo = pd.DataFrame(newEntry,index=[0])[pd.DataFrame(columns = columnsResultInfo).columns.tolist()]
                    #if not test:
                    resultInfo.to_csv(results_dir+results_file,mode="a",header=False,index=False,sep='\t',float_format='%.5f')
                    # print entry
                    if verbose_RNN:
                        print("\r"+netName+resultInfo.to_string(index=False,header=False))#thisAsset+netName+" "+str(counterFeats)+":\n"+

        countIndex+=1
    # end of else if fillingX:
    EOF.iloc[c] = tradeInfoLive.iloc[-1]
    
    return output
    
def dispatch(buffer, ass_id, ass_idx):
    #AllAssets, assets, running_assets, nCxAxN, buffSizes, simulation, delays, PA, verbose
    '''
    inputs: AllAssets
            assets
            nCxAss: matrix containning number of channels per asset and per network
            buffSizes: matrix containning max size of buffer for each channel
    '''

    #print("Dispatcher running and ready.")

    thisAsset = data.AllAssets[str(ass_id)]
    
    tradeInfo = buffer
    
    outputs = []
    new_outputs = 0
    # add trade info to all networks of this asset
    for nn in range(nNets):
        # loop over buffers to add new info
        for ch in range(int(nCxAxN[ass_idx,nn])):
            # add info to buffer
            if buffersCounter[ass_idx][nn][ch]>=0:
                
                buffers[ass_idx][nn][ch] = buffers[ass_idx][nn][ch].append(tradeInfo,ignore_index=True)
            buffersCounter[ass_idx][nn][ch] += tradeInfo.shape[0]
            # check if buffer reached max
            if buffers[ass_idx][nn][ch].shape[0]==buffSizes[ass_idx,nn]:
                
                ### Dispatch buffer to corresponding network ###
                ## TODO Several models
                output = runRNNliveFun(buffers[ass_idx][nn][ch],listFillingXs[ass_idx][nn],inits[ass_idx][nn],list_listFeaturesLive[ass_idx][nn],list_listParSarStruct[ass_idx][nn],
                              list_listEM[ass_idx][nn],list_listAllFeatsLive[ass_idx][nn],list_list_X_i[ass_idx][nn],list_means_in[ass_idx][nn],phase_shifts[nn],
                              list_stds_in[ass_idx][nn],list_stds_out[ass_idx][nn],ADs[nn], thisAsset,netNames[nn],listCountPoss[ass_idx][nn],list_list_weights_matrix[ass_idx][nn],
                              list_list_time_to_entry[ass_idx][nn],list_list_list_soft_tildes[ass_idx][nn],list_list_Ylive[ass_idx][nn],list_list_Pmc_live[ass_idx][nn],
                              list_list_Pmd_live[ass_idx][nn],list_list_Pmg_live[ass_idx][nn],EOFs[ass_idx][nn],countOutss[ass_idx][nn],
                              t_indexs[nn],ch, resultsDir[nn],results_files[nn], list_models[0])
                if len(output)>0:
                    outputs.append(output)
                    new_outputs = 1
                
                #reset buffer
                buffers[ass_idx][nn][ch] = pd.DataFrame()
                buffersCounter[ass_idx][nn][ch] = 0
                # update file extension
                bufferExt[ass_idx][nn][ch] = (bufferExt[ass_idx][nn][ch]+1)%2
                            
            elif buffers[ass_idx][nn][ch].shape[0]>buffSizes[ass_idx,nn]:
                print(thisAsset+" buffer size "+str(buffers[ass_idx][nn][ch].shape[0]))
                print("Error. Buffer cannot be greater than max buff size")
                error()
                
    if len(outputs)>1:
        print("WARNING! Outputs length="+str(len(outputs))+". No support for multiple outputs at the same time yet.") 
    return outputs, new_outputs

def fetch(budget):
    """
    <DocString>
    """
    
    for ass_id in running_assets:
        thisAsset = data.AllAssets[str(ass_id)]
        
        directory_MT5_ass = directory_MT5+thisAsset+"/"
        
        try:
            shutil.rmtree(directory_MT5_ass)
        except:
            print(thisAsset+" Warning. Error when renewing directory")
            
        if not os.path.exists(directory_MT5_ass):
            try:
                os.mkdir(directory_MT5_ass)
                print(thisAsset+" Directiory created")
            except:
                print("Warning. Error when creating directory")
    
    print("Fetcher lauched")

    extension = ".txt"
    deli = "_"
    fileExt = [0 for ass in range(nAssets)]
    nFiles = 10
    
    first_info_fetched = False
    
    flag_cl_name = "CL"
    flag_sl_name = "SL"
    
    while 1:
        
        for ass_idx, ass_id in enumerate(running_assets):
            
            thisAsset = data.AllAssets[str(ass_id)]
            
            directory_MT5_ass = directory_MT5+thisAsset+"/"
            
            # Fetching buffers
            fileID = thisAsset+deli+str(fileExt[ass_idx])+extension
            success = 0
            try:
                
                buffer = pd.read_csv(directory_MT5_ass+fileID, encoding='utf_16_le')#
                #print(thisAsset+" new buffer received")
                os.remove(directory_MT5_ass+fileID)
                success = 1
                
                if not first_info_fetched:
                    print("First info fetched")
                    first_info_fetched = True
                
            except (FileNotFoundError,PermissionError):
                pass
            
            # update file extension
            if success:
                fileExt[ass_idx] = (fileExt[ass_idx]+1)%nFiles
                outputs, new_outputs = dispatch(buffer, ass_id, ass_idx)
                ################# Trader ##################
                if new_outputs and not trader.swap_pending:
                    #print(outputs)
                    trader.check_new_inputs(outputs, thisAsset, directory_MT5_ass=directory_MT5_ass)
            
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
                            print("Error in reading file. Length "+str(len(info_close)))
                    except (FileNotFoundError,PermissionError):
                        pass
                
                
            # i stoploss
            if flag_cl:
                # update postiions vector
                info_split = info_close.split(",")
                list_idx = trader.map_ass_idx2pos_idx[ass_id]
                bid = float(info_split[6])
                ask = float(info_split[7])
                DateTime = info_split[2]
                # update bid and ask lists if exist
                if list_idx>-1:
                    trader.list_last_bid[list_idx] = bid
                    trader.list_last_ask[list_idx] = ask
                else:
                    error()
                    
                trader.close_position(DateTime, thisAsset, ass_id)
                
                file = open(trader.log_summary,"a")
                file.write(info_close+"\n")
                file.close()
                # open position if swap process is on
                if trader.swap_pending:
                    trader.finalize_resources_swap()
                
                
            elif flag_sl:
                # update postiions vector
                info_split = info_close.split(",")
                list_idx = trader.map_ass_idx2pos_idx[ass_id]
                bid = float(info_split[6])
                ask = float(info_split[7])
                DateTime = info_split[2]
                # update bid and ask lists if exist
                if list_idx>-1:
                    trader.list_last_bid[list_idx] = bid
                    trader.list_last_ask[list_idx] = ask
                else:
                    error()
                    
                trader.close_position(DateTime, thisAsset, ass_id)
                
                trader.stoplosses += 1
                out = "Exit position due to stop loss "+" sl="+str(trader.list_stop_losses[trader.map_ass_idx2pos_idx[ass_id]])
                print("\r"+out)
                trader.write_log(out)
                
                #if not simulate:
                file = open(trader.log_summary,"a")
                file.write(info_close+"\n")
                file.close()
                
    #            trader.ban_currencies(data, thisAsset, rootDirOr, netName)
            # end of elifs
    
    return budget

def back_test(DateTimes, SymbolBids, SymbolAsks, Assets, nEvents ,data, budget):
    """
    <DocString>
    """
    nAssets = len(data.assets)
    print("Fetcher lauched")
    # number of events per file
    n_files = 10
    n_samps_buffer = 10
    

    
#    nChannels = int(data.nEventsPerStat/data.movingWindow)
    init_row = ['d',0.0,0.0]
    #init_df = pd.DataFrame(data=[init_row for i in range(n_samps_buffer)],columns=['DateTime','SymbolBid','SymbolAsk'])
    fileIDs = [0 for ass in range(nAssets)]
    buffers = [pd.DataFrame(data=[init_row for i in range(n_samps_buffer)],
                                  columns=['DateTime','SymbolBid','SymbolAsk']) for ass in range(nAssets)]
    
    buffersCounter = [0 for ass in range(nAssets)]
    
    event_idx = 0
    
    tic = time.time()
    
    while event_idx<nEvents:
        
        outputs = []
        thisAsset = Assets[event_idx].decode("utf-8")
        ass_idx = ass2index_mapping[thisAsset]
        ass_id = running_assets[ass_idx]
        list_idx = trader.map_ass_idx2pos_idx[ass_id]
        DateTime = DateTimes[event_idx].decode("utf-8")
        bid = int(np.round(SymbolBids[event_idx]*100000))/100000
        ask = int(np.round(SymbolAsks[event_idx]*100000))/100000
        # update bid and ask lists if exist
        if list_idx>-1:
            trader.list_last_bid[list_idx] = bid
            trader.list_last_ask[list_idx] = ask
            
        new_outputs = 0
#        print("\r"+DateTime+" "+thisAsset, sep=' ', end='', flush=True)
        # Run RNN
        if 1:
            # add new entry to buffer
            (buffers[ass_idx]).iloc[buffersCounter[ass_idx]] = [DateTime, bid, ask]
            buffersCounter[ass_idx] = (buffersCounter[ass_idx]+1)%n_samps_buffer
            
            if buffersCounter[ass_idx]==0:

                outputs, new_outputs = dispatch(buffers[ass_idx], ass_id, ass_idx)
    
                ####### Update counters and buffers ##########
                fileIDs[ass_idx] = (fileIDs[ass_idx]+1)%n_files
                # reset buffer
                buffers[ass_idx] = pd.DataFrame(data=[init_row for i in range(n_samps_buffer)],columns=['DateTime','SymbolBid','SymbolAsk'])
        
        
        ################# Trader ##################
        if new_outputs:
            #print(outputs)
            trader.check_new_inputs(outputs, thisAsset)
        
        ################ MT5 simulator ############
        # check if a position is already opened
        if trader.is_opened(ass_idx):
            
            trader.count_one_event(ass_idx)
            exit_pos  = trader.is_stoploss_reached(ass_id, bid, event_idx)
                
            if exit_pos:
                # save original exit time for skipping positions
                trader.close_position(DateTime, thisAsset, ass_id)
            elif trader.list_count_events[trader.map_ass_idx2pos_idx[ass_id]]==trader.list_deadlines[trader.map_ass_idx2pos_idx[ass_id]]:# check for closing
                # close position
                trader.close_position(DateTime, thisAsset, ass_id)
        
        ###################### End of Trader ###########################
        event_idx += 1
    
    # get statistics
    #t_entries = trader.n_pos_opened+trader.n_pos_extended
    perEntries = 0#t_entries/journal_entries
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
    budget = trader.budget
    
    out = ("\nGROI = {0:.3f}% ".format(100*GROI)+"ROI = {0:.3f}%".format(100*ROI)+" Sum GROI = {0:.3f}%".format(100*trader.tGROI_live)+" Sum ROI = {0:.3f}%".format(100*trader.tROI_live)+
          " Final budget {0:.2f}E".format(trader.budget)+" Earnings {0:.2f}E".format(earnings)+
          " per earnings {0:.3f}%".format(100*(trader.budget-trader.init_budget)/trader.init_budget)+" ROI per position {0:.3f}%".format(ROI_per_entry))
    trader.write_log(out)
    print(out)
    out = ("Number entries "+str(trader.n_entries)+" per entries {0:.2f}%".format(100*perEntries)+" per net success "+"{0:.3f}%".format(100*per_net_success)+
          " per gross success "+"{0:.3f}%".format(100*per_gross_success)+" av loss {0:.3f}%".format(100*average_loss)+" per sl {0:.3f}%".format(100*perSL))
    trader.write_log(out)
    print(out)
    out = "DONE. Time: "+"{0:.2f}".format((time.time()-tic)/60)+" mins"
    trader.write_log(out)
    print(out)
    
    results.update_results(GROI, earnings, ROI, trader.n_entries, trader.stoplosses)
    # TODO: update badget
#    init_budget = trader.budget
        
    out = ("\nTotal GROI = {0:.3f}% ".format(results.total_GROI)+"Total ROI = {0:.3f}% ".format(results.total_ROI)+"Sum GROI = {0:.3f}% ".format(results.sum_GROI)+"Sum ROI = {0:.3f}%".format(results.sum_ROI)+
      " Accumulated earnings {0:.2f}E\n".format(results.total_earnings))
    print(out)
    trader.write_log(out)
    
    return budget


if __name__ == '__main__':
    
    test = True
    run_back_test = True
    
    data=Data(movingWindow=100,nEventsPerStat=1000,lB=1200,dateStart='2017.01.01',
              dateTest = ['2018.10.18']
                )

#    ['2018.10.01','2018.10.02','2018.10.03','2018.10.04','2018.10.05',
#                          '2018.10.08','2018.10.09','2018.10.10','2018.10.11','2018.10.12']
#    ['2017.11.27','2017.11.28','2017.11.29','2017.11.30','2017.12.01',
#                 '2017.12.04','2017.12.05','2017.12.06','2017.12.07','2017.12.08']+[
#                 '2018.03.19','2018.03.20','2018.03.21','2018.03.22','2018.03.23',
#                 '2018.03.26','2018.03.27','2018.03.28','2018.03.29','2018.03.30',
#                 '2018.04.02','2018.04.03','2018.04.04','2018.04.05','2018.04.06',
#                 '2018.04.09','2018.04.10','2018.04.11','2018.04.12','2018.04.13',
#                 '2018.04.16','2018.04.17','2018.04.18','2018.04.19','2018.04.20',
#                 '2018.04.23','2018.04.24','2018.04.25','2018.04.26','2018.04.27',
#                 '2018.04.30','2018.05.01','2018.05.02','2018.05.03','2018.05.04',
#                 '2018.05.07','2018.05.08','2018.05.09','2018.05.10','2018.05.11',
#                 '2018.05.14','2018.05.15','2018.05.16','2018.05.17','2018.05.18',
#                 '2018.05.21','2018.05.22','2018.05.23','2018.05.24','2018.05.25',
#                 '2018.05.28','2018.05.29','2018.05.30','2018.05.31','2018.06.01',
#                 '2018.06.04','2018.06.05','2018.06.06','2018.06.07','2018.06.08',
#                 '2018.06.11','2018.06.12','2018.06.13','2018.06.14','2018.06.15',
#                 '2018.06.18','2018.06.19','2018.06.20','2018.06.21','2018.06.22',
#                 '2018.06.25','2018.06.26','2018.06.27','2018.06.28','2018.06.29',
#                 '2018.07.02','2018.07.03','2018.07.04','2018.07.05','2018.07.06',
#                 '2018.07.09','2018.07.10','2018.07.11','2018.07.12','2018.07.13',
#                 '2018.07.30','2018.07.31','2018.08.01','2018.08.02','2018.08.03',
#                 '2018.08.06','2018.08.07','2018.08.08','2018.08.09','2018.08.10']

#                    '2018.03.09',
#                 '2018.03.12','2018.03.13','2018.03.14','2018.03.15','2018.03.16',
                    
#    ['2018.09.03','2018.09.04','2018.09.05','2018.09.06','2018.09.07','2018.09.10']

    IDepoch = ["13"]
    IDweights = ["000266"]
    IDresults = ["100266"]#
    delays = [0]
    t_indexs = [2] # time index to use as output. Value between {0,...,model.seq_len-1}
    MRC = [True]
    mWs = [100]
    nExSs = [1000]
    lBs = [1200]
    netNames = ["000266"]
    phase_shifts = [10] # phase shift
    
    list_use_GRE = [True]
    list_lb_mc_op = [0.5]
    list_lb_md_op = [0.5]
    list_lb_mc_ext = [2]
    list_lb_md_ext = [2]
    list_ub_mc_op = [1]
    list_ub_md_op = [1]
    list_ub_mc_ext = [1]
    list_ub_md_ext = [1]
    list_thr_sl = [1000]
    list_thr_tp = [1000]
    list_fix_spread = [False]
    list_fixed_spread_pips = [4]
    list_max_lots_per_pos = [.1]
    list_flexible_lot_ratio = [False]
    list_if_dir_change_close = [False]
    list_if_dir_change_extend = [False]
    list_name = ['66']
    
#    IDepoch = ["5"]#
#    IDweights = ["000233"]#
#    IDresults = ["100233"]
#    shifts = [0]
#    t_index = [2] # time index to use as output. Value between {0,...,model.seq_len-1}
#    MRC = [False]
#    mWs = [10]
#    nExSs = [100]
#    lBs = [120]
#    netNames = ["100"]
#    
    verbose_RNN = True
    verbose_trader = True
    ti_name = []
    for i in range(len(MRC)):
        if MRC[i]==True:
            ti_name.append(3)
        else:
            ti_name.append(t_indexs[i])
            
    
    resultsDir = "../RNN/results/"
    
    ADs = []
    for i in range(len(IDepoch)):
        
        if MRC[i]==True:
            if t_indexs[i]!=2:
                error()
            ADs.append(pickle.load( open( resultsDir+IDresults[i]+"/AD_e"+IDepoch[i]+".p", "rb" )))
        else:
            ADs.append(None)

    nChans = (np.array(nExSs)/np.array(mWs)).astype(int).tolist()
    
    
    thr_mc = 0.5
    thr_md = 0.5
    assets = [1, 2, 3, 4, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 19, 27, 28, 29, 30, 31, 32]#
    running_assets = assets
    numberNetworks = len(nChans)
    
    if run_back_test:
        resultsDir = ["../RNN/resultsLive/back_test/"+IDresults[nn]+"T"+str(ti_name[nn])+"E"+IDepoch[nn]+"/" for nn in range(numberNetworks)]
    else:
        resultsDir = ["../RNN/resultsLive/live/"+IDresults[nn]+"T"+str(ti_name[nn])+"E"+IDepoch[nn]+"/" for nn in range(numberNetworks)]
    
    results_files = [IDresults[nn]+"T"+str(ti_name[nn])+"E"+IDepoch[nn]+".txt" for nn in range(numberNetworks)]
    dataOr=Data(assets=assets,dateEnd='2018.01.12')
    nCxAxN = np.zeros((len(running_assets),numberNetworks))
    columnsResultInfo = ["Asset","Entry Time","Exit Time","Bet","Outcome","Diff","Bi","Ai","Bo","Ao","GROI","Spread","ROI","P_mc","P_md","P_mg"]
    
    for nn in range(numberNetworks):
        if phase_shifts[nn] != 0:
            nCxAxN[:,nn] = int(nChans[nn]*phase_shifts[nn])
        else:
            nCxAxN[:,nn] = int(nChans[nn])
            
        if not os.path.exists(resultsDir[nn]):
            
            try:
                os.mkdir(resultsDir[nn])
                
            except:
                print("Warning. Error when creating directory")
        
        if not os.path.exists(resultsDir[nn]+results_files[nn]):
            pd.DataFrame(columns = columnsResultInfo).to_csv(resultsDir[nn]+results_files[nn],mode="w",index=False,sep='\t')
    
    buffSizes = nExSs+np.zeros((len(running_assets),numberNetworks)).astype(int)
    
    ############################################
    # init bufers and file extensions (n buffers = n channels per network and asset)
    nNets = nCxAxN.shape[1]
    nAssets = nCxAxN.shape[0]
    buffers = [[[pd.DataFrame() for k in range(int(nCxAxN[ass,nn]))] 
                                for nn in range(nNets)] 
                                for ass in range(nAssets)]
    buffersCounter = [[[-k*buffSizes[ass,nn]/nCxAxN[ass,nn]-delays[nn] for k in range(int(nCxAxN[ass,nn]))] 
                                                                       for nn in range(nNets)] 
                                                                       for ass in range(nAssets)]
    bufferExt = [[[0 for k in range(int(nCxAxN[ass,nn]))] for nn in range(nNets)] 
                                                          for ass in range(nAssets)]
    #########################################
    
    # init stats structures
    hdf5_directory = 'D:/SDC/py/HDF5/'#'../HDF5/'#
    thr_levels = 5
    lb_level = 5
    first_nonzero = 0
    
    # load stats
    list_stats = [[load_stats(data, data.AllAssets[str(running_assets[ass])], h5py.File(hdf5_directory+'IO_mW'+str(mWs[nn])
                    +'_nE'+str(nExSs[nn])+'_nF'+str(data.nFeatures)+'.hdf5','r')[data.AllAssets[str(running_assets[ass])]], 
                    0, from_stats_file=True, hdf5_directory=hdf5_directory+'stats/') for nn in range(nNets)] for ass in range(nAssets)]
    
    if test:
        gain = .000000001
        
    else: 
        gain = 1
    
    list_means_in =  [[list_stats[ass][nn]['means_t_in'] for nn in range(nNets)] for ass in range(nAssets)]
    list_stds_in =  [[gain*list_stats[ass][nn]['stds_t_in'] for nn in range(nNets)] for ass in range(nAssets)]
    list_stds_out =  [[gain*list_stats[ass][nn]['stds_t_out'] for nn in range(nNets)] for ass in range(nAssets)]
    # pre allocate memory size
    
    # init non-variation features
    nonVarFeats = np.intersect1d(data.noVarFeats,data.features)
    nonVarIdx = np.zeros((len(nonVarFeats))).astype(int)
    nv = 0
    for allnv in range(data.nFeatures):
        if data.features[allnv] in nonVarFeats:
            nonVarIdx[nv] = int(allnv)
            nv += 1
    
    ass2index_mapping = {}
    ass_index = 0
    
    for ass in running_assets:
        ass2index_mapping[data.AllAssets[str(ass)]] = ass_index
        ass_index += 1
    
    ################# Trader #############################
    entry_time_column = 'Entry Time'#'Entry Time
    exit_time_column = 'Exit Time'#'Exit Time
    entry_bid_column = 'Bi'
    entry_ask_column = 'Ai'
    exit_ask_column = 'Ao'
    exit_bid_column = 'Bo'
    
    strategys = [Strategy(direct='../RNN/results/',thr_sl=list_thr_sl[i], thr_tp=list_thr_tp[i], fix_spread=list_fix_spread[i], fixed_spread_pips=list_fixed_spread_pips[i], max_lots_per_pos=list_max_lots_per_pos[i], flexible_lot_ratio=list_flexible_lot_ratio[i], 
                 lb_mc_op=list_lb_mc_op[i], lb_md_op=list_lb_md_op[i], lb_mc_ext=list_lb_mc_ext[i], lb_md_ext=list_lb_md_ext[i], ub_mc_op=list_ub_mc_op[i], ub_md_op=list_ub_md_op[i], ub_mc_ext=list_ub_mc_ext[i], ub_md_ext=list_ub_md_ext[i],
                 if_dir_change_close=list_if_dir_change_close[i], if_dir_change_extend=list_if_dir_change_extend[i], name=list_name[i],t_index=ti_name[i],use_GRE=list_use_GRE[i],IDr=IDresults[i],epoch=IDepoch[i]) for i in range(len(ti_name))]
    
    root_dir = 'D:/SDC/py/Data_test/'#'D:/SDC/py/Data_aws_3/'#'D:/SDC/py/Data/'
    directory_MT5 = "C:/Users/mgutierrez/AppData/Roaming/MetaQuotes/Terminal/D0E8209F77C8CF37AD8BF550E51FF075/MQL5/Files/IOlive/"
    results = Results()
    
    tic = time.time()
    
    ################### RNN ###############################
    
    tf.reset_default_graph()
    with tf.Session() as sess:
        
        list_models = [modelRNN(data,
                    size_hidden_layer=100,
                    L=3,
                    size_output_layer=5,
                    keep_prob_dropout=1,
                    miniBatchSize=32,
                    outputGain=0.6,
                    lR0=0.0001,
                    IDgraph=IDweights[0]+IDepoch[0],
                    sess=sess)]
        
    ##########################################################
        init_budget = 10000.0
        start_time = dt.datetime.strftime(dt.datetime.now(),'%y_%m_%d_%H_%M_%S')
        
        if run_back_test:
            day_index = 0
            t_journal_entries = 0
            
            
    
            while day_index<len(data.dateTest):
                counter_back = 0
                init_list_index = day_index#data.dateTest[day_index]
                
                # find sequence of consecutive days in test days
                while day_index<len(data.dateTest)-1 and dt.datetime.strptime(data.dateTest[day_index],'%Y.%m.%d')+dt.timedelta(1)==dt.datetime.strptime(data.dateTest[day_index+1],'%Y.%m.%d'):
                    day_index += 1
                
                end_list_index = day_index+counter_back
                print("Week from "+data.dateTest[init_list_index]+" to "+data.dateTest[end_list_index])
                
                
                day_index += counter_back+1
                
                # init structures RNN
                inits = [[[False for ps in range(phase_shifts[nn])] for nn in range(nNets)] for ass in range(nAssets)]
                listFillingXs = [[[True for ps in range(phase_shifts[nn])] for nn in range(nNets)] for ass in range(nAssets)]
                listCountPoss = [[[0 for ps in range(phase_shifts[nn])] for nn in range(nNets)] for ass in range(nAssets)]
                countOutss = [[np.zeros((int(nChans[nn]*phase_shifts[nn]))).astype(int) for nn in range(nNets)] for ass in range(nAssets)]
                EOFs = [[pd.DataFrame(columns=['DateTime','SymbolBid','SymbolAsk'], index=range(int(nChans[nn]*phase_shifts[nn]))) for nn in range(nNets)] for ass in range(nAssets)]
                list_list_X_i = [[[np.zeros((1, int((lBs[nn]-nExSs[nn])/mWs[nn]+1), data.nFeatures)) for ps in range(phase_shifts[nn])] for nn in range(nNets)] for ass in range(nAssets)]
                #Xlive = np.zeros((alloc, model.seq_len, model.nFeatures))
                list_list_Ylive = [[[np.zeros((0,)) for ps in range(phase_shifts[nn])] for nn in range(nNets)] for ass in range(nAssets)]
                list_list_Pmc_live = [[[np.zeros((0,)) for ps in range(phase_shifts[nn])] for nn in range(nNets)] for ass in range(nAssets)]
                list_list_Pmd_live = [[[np.zeros((0,)) for ps in range(phase_shifts[nn])] for nn in range(nNets)] for ass in range(nAssets)]
                list_list_Pmg_live = [[[np.zeros((0,)) for ps in range(phase_shifts[nn])] for nn in range(nNets)] for ass in range(nAssets)]
                list_listAllFeatsLive = [[[np.zeros((data.nFeatures,0)) for ps in range(phase_shifts[nn])] for nn in range(nNets)] for ass in range(nAssets)]
                list_listFeaturesLive = [[[None for ps in range(phase_shifts[nn])] for nn in range(nNets)] for ass in range(nAssets)]
                list_listParSarStruct = [[[None for ps in range(phase_shifts[nn])] for nn in range(nNets)] for ass in range(nAssets)]
                list_listEM = [[[None for ps in range(phase_shifts[nn])] for nn in range(nNets)] for ass in range(nAssets)]
                # init condition vector to open market
                #condition = np.zeros((model.seq_len))
                list_list_time_to_entry = [[[[] for ps in range(phase_shifts[nn])] for nn in range(nNets)] for ass in range(nAssets)] # list tracking times to entry the market
                list_list_list_soft_tildes = [[[[] for ps in range(phase_shifts[nn])] for nn in range(nNets)] for ass in range(nAssets)]
                # upper diagonal matrix containing latest weight values
                list_list_weights_matrix = [[[np.zeros((int((lBs[nn]-nExSs[nn])/mWs[nn]+1),int((lBs[nn]-nExSs[nn])/mWs[nn]+1))) for ps in range(phase_shifts[nn])] for nn in range(nNets)] for ass in range(nAssets)]
                
    #             init trader
    #            trader = Trader(Position(journal.iloc[0], AD_resume, eROIpb), init_budget=init_budget)
                trader = Trader(results_dir="../RNN/resultsLive/back_test/trader/", init_budget=init_budget, log_file_time=start_time)
                
                DateTimes, SymbolBids, SymbolAsks, Assets, nEvents = load_in_memory(data, init_list_index, end_list_index,root_dir=root_dir)
                init_budget = back_test(DateTimes, SymbolBids, SymbolAsks, Assets, nEvents ,data, init_budget)
        else:
            # init lists
            # init structures RNN
            inits = [[[False for ps in range(phase_shifts[nn])] for nn in range(nNets)] for ass in range(nAssets)]
            listFillingXs = [[[True for ps in range(phase_shifts[nn])] for nn in range(nNets)] for ass in range(nAssets)]
            listCountPoss = [[[0 for ps in range(phase_shifts[nn])] for nn in range(nNets)] for ass in range(nAssets)]
            countOutss = [[np.zeros((int(nChans[nn]*phase_shifts[nn]))).astype(int) for nn in range(nNets)] for ass in range(nAssets)]
            EOFs = [[pd.DataFrame(columns=['DateTime','SymbolBid','SymbolAsk'], index=range(int(nChans[nn]*phase_shifts[nn]))) for nn in range(nNets)] for ass in range(nAssets)]
            list_list_X_i = [[[np.zeros((1, int((lBs[nn]-nExSs[nn])/mWs[nn]+1), data.nFeatures)) for ps in range(phase_shifts[nn])] for nn in range(nNets)] for ass in range(nAssets)]
            #Xlive = np.zeros((alloc, model.seq_len, model.nFeatures))
            list_list_Ylive = [[[np.zeros((0,)) for ps in range(phase_shifts[nn])] for nn in range(nNets)] for ass in range(nAssets)]
            list_list_Pmc_live = [[[np.zeros((0,)) for ps in range(phase_shifts[nn])] for nn in range(nNets)] for ass in range(nAssets)]
            list_list_Pmd_live = [[[np.zeros((0,)) for ps in range(phase_shifts[nn])] for nn in range(nNets)] for ass in range(nAssets)]
            list_list_Pmg_live = [[[np.zeros((0,)) for ps in range(phase_shifts[nn])] for nn in range(nNets)] for ass in range(nAssets)]
            list_listAllFeatsLive = [[[np.zeros((data.nFeatures,0)) for ps in range(phase_shifts[nn])] for nn in range(nNets)] for ass in range(nAssets)]
            list_listFeaturesLive = [[[None for ps in range(phase_shifts[nn])] for nn in range(nNets)] for ass in range(nAssets)]
            list_listParSarStruct = [[[None for ps in range(phase_shifts[nn])] for nn in range(nNets)] for ass in range(nAssets)]
            list_listEM = [[[None for ps in range(phase_shifts[nn])] for nn in range(nNets)] for ass in range(nAssets)]
            # init condition vector to open market
            #condition = np.zeros((model.seq_len))
            list_list_time_to_entry = [[[[] for ps in range(phase_shifts[nn])] for nn in range(nNets)] for ass in range(nAssets)] # list tracking times to entry the market
            list_list_list_soft_tildes = [[[[] for ps in range(phase_shifts[nn])] for nn in range(nNets)] for ass in range(nAssets)]
            # upper diagonal matrix containing latest weight values
            list_list_weights_matrix = [[[np.zeros((int((lBs[nn]-nExSs[nn])/mWs[nn]+1),int((lBs[nn]-nExSs[nn])/mWs[nn]+1))) for ps in range(phase_shifts[nn])] for nn in range(nNets)] for ass in range(nAssets)]
                
            # init trader
            trader = Trader(results_dir="../RNN/resultsLive/live/trader/", init_budget=init_budget, log_file_time=start_time)
            init_budget = fetch(init_budget)
            # launch fetcher
            
            
        # gather results
        total_entries = int(np.sum(results.number_entries))
        total_successes = int(np.sum(results.net_successes))
        total_failures = total_entries-total_successes
        per_gross_success = 100*np.sum(results.gross_successes)/total_entries
        per_net_succsess = 100*np.sum(results.net_successes)/total_entries
        average_loss = np.sum(results.total_losses)/((total_entries-np.sum(results.net_successes))*trader.pip)
        average_win = np.sum(results.total_wins)/(np.sum(results.net_successes)*trader.pip)#results.total_ROI/np.sum(results.net_successes)/(100*trader.pip)
        RR = total_successes*average_win/(average_loss*total_failures)
        
        out = ("\nTotal GROI = {0:.3f}% ".format(results.total_GROI)+"Total ROI = {0:.3f}% ".format(results.total_ROI)+"Sum GROI = {0:.3f}% ".format(results.sum_GROI)+"Sum ROI = {0:.3f}%".format(results.sum_ROI)+
          " Accumulated earnings {0:.2f}E".format(results.total_earnings))
        print(out)
        trader.write_log(out)
        out = ("Total entries "+str(total_entries)+" percent gross success {0:.2f}%".format(per_gross_success)+
              " percent nett success {0:.2f}%".format(per_net_succsess)+" average loss {0:.2f}p".format(average_loss)+" average win {0:.2f}p".format(average_win)+" RR 1 to {0:.2f}".format(RR))
        print(out)
        trader.write_log(out)
        out = ("DONE. Total time: "+"{0:.2f}".format((time.time()-tic)/60)+" mins\n")
        print(out)
        trader.write_log(out)
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