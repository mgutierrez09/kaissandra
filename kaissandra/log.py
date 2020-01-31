# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 13:52:56 2020

@author: mgutierrez
"""

import os
import logging
import logging.handlers
from kaissandra.local_config import local_vars as LV

class Logger():
    
    def __init__(self, name='logger', filename='kaissandra.log', directory=LV.log_directory):
        """  """
        # create directory if does not exist
        if not os.path.exists(directory):
            os.makedirs(directory)
        # create loggers
        logger = logging.getLogger(name)
        
        # create console handler and set level to debug
        fl = logging.handlers.RotatingFileHandler(directory+filename, maxBytes=10240,
                                              backupCount=10)
        fl.setLevel(logging.DEBUG)
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        
        # create formatter
        formatter = logging.Formatter('%(asctime)s %(name)s: %(levelname)s - %(message)s')#'%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
        
        
        # add formatter to ch
        fl.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        # add ch to logger
        logger.addHandler(fl)
        logger.addHandler(ch)
        
        self.logger = logger

if __name__=='__main__':
        
    logger = Logger()
    logger.logger.info('this is some info')