# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 14:23:35 2019

@author: mgutierrez
"""
import os

class Config(object):
    """ Configuration object providing necessary fields to connect with server """
    URL = os.environ.get('URL') #or 'https://kaissandra-webapp.herokuapp.com/api/'#os.environ.get('URL') or 'https://localhost:5000/api/'#
    USERNAME = os.environ.get('USER') or 'kaissandra'
    PASSWORD = os.environ.get('PASSWORD') or "kaissandra"
    TRADERNAME = os.environ.get('TRADERNAME') #or 'farnamstreet'
    MACHINE = os.environ.get('MACHINE')# or 'aws_i-0db4c8daa833808b4'
    MAGICNUMBER = os.environ.get('MAGICNUMBER') or 123456
    if URL == None:
        raise ValueError("URL environment variable cannot be None.")
    if TRADERNAME == None:
        raise ValueError("TRADERNAME environment variable cannot be None.")