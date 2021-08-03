#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  2 14:14:44 2021

@author: davi
"""

import wget
import os
from pandas import read_csv

# recebe url do csv, baixa dado do link e retorna um dataframe
def get_data(url):  
    # verifica se já tem o arquivo (pra não ficar criando várias cópias)
    path = '/home/davi/ncovid_mod/data'
    filename = path + '/' + os.path.basename(url) # get the full path of the file
    if os.path.exists(filename):
        os.remove(filename) # if exist, remove it directly
    wget.download(url, out=filename) # download it to the specific path.
    
    df = read_csv(path + '/' + os.path.basename(url))
    
    return df