# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 09:34:58 2019

@author: magut
"""

from datetime import datetime
import MetaTrader5 as mt5
from pytz import timezone
import matplotlib.pyplot as plt
from kaissandra import config as C

utc_tz = timezone('UTC')
 
 
 # connect to MetaTrader 5
mt5.MT5Initialize()
# wait till MetaTrader 5 establishes connection to the trade server and synchronizes the environment
mt5.MT5WaitForTerminal()
 
 # request connection status and parameters
print(mt5.MT5TerminalInfo())
# get data on MetaTrader 5 version
print(mt5.MT5Version())
 
# request ticks from AUDUSD within 2019.04.01 13:00 - 2019.04.02 13:00 
audusd_ticks = mt5.MT5CopyTicksRange("AUDUSD", datetime(2013,2,4,13), datetime(2013,2,7,13), mt5.MT5_COPY_TICKS_ALL)
 
# shut down connection to MetaTrader 5
mt5.MT5Shutdown()
#PLOTTING
x_time = [x.time.astimezone(utc_tz) for x in audusd_ticks]
# prepare Bid and Ask arrays
bid = [y.bid for y in audusd_ticks]
ask = [y.ask for y in audusd_ticks]
 
# draw ticks on the chart
plt.plot(x_time, ask,'r-', label='ask')
plt.plot(x_time, bid,'g-', label='bid')
# display legends 
plt.legend(loc='upper left')
# display header 
plt.title('EURAUD ticks')
# display the chart
plt.show()