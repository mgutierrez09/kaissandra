# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 13:52:56 2020

@author: mgutierrez

Kaissandra's module containing all functions and classes related to logging.
"""

import logging
import logging.handlers
from kaissandra.local_config import local_vars as LV

def config_logger_online(directory=LV.log_directory, filename='online.log', mode='a', 
                         maxBytes=1000000, backupCount=10, level=logging.INFO):
    root = logging.getLogger()
    root.setLevel(level)
    h = logging.handlers.RotatingFileHandler(directory+filename, mode, 
                                             maxBytes=maxBytes, backupCount=backupCount)
    # '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
    #'%(asctime)s %(processName)-10s %(name)s %(levelname)-8s %(message)s' 
    f = logging.Formatter('%(asctime)s %(processName)-10s %(name)s %(levelname)-8s %(message)s' )
    h.setFormatter(f)
    root.addHandler(h)
    root.info("\n\nNew online session launched")
    
def worker_configurer_online(queue):
    h = logging.handlers.QueueHandler(queue)  # Just the one handler needed
    root = logging.getLogger()
    root.addHandler(h)
    # send all messages, for demo; no other level or filter logic applied.
    root.setLevel(logging.INFO)