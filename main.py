#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  2 15:29:45 2021

@author: davi
"""

from get_data import get_data
from train_model import model_configs
from numpy import array
from grid_search import grid_search

url = "https://raw.githubusercontent.com/Natalnet/ncovid-air-paper/master/df_spsp_pred.csv"

df = get_data(url)

print(df.head())

data = [df['deaths'].values, df['deaths'].values, df['cases'].values, df['aqi'].values, df['temperature'].values, df['humidity'].values]
data = array(data).T

print(data.shape)

n_weeks = 6

n_test = 7*n_weeks
n_repeats = 2

config = model_configs()

print(config)

scores_grid = list()
models = list()
models_test = list()

grid_search(data, n_test, config, n_repeats, models, scores_grid, models_test)
print('done')
# list top 3 configs
for cfg, error in scores_grid[:3]:
  print(cfg, error)