#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 13:47:25 2021

@author: davi
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import data_manner
import evaluator_manner
import plot_manner

def read_csv_file(path, column_date, last_day, first_date=None):
    df = pd.read_csv(path)
    if first_date is not None:
        return df[(df[column_date] < last_day) & (df[column_date] >= first_date)]
    return df[(df[column_date] < last_day)][1:]

data_plot = list()

db_folder = '../dbs/'
last_date = '2021-03-21'
df_araraquara = read_csv_file(db_folder + 'df_araraquara.csv', 'date', last_date, None)
df_araraquara = pd.read_csv(db_folder + 'df_araraquara.csv')

# acumulado
data_plot.append(df_araraquara.deaths.values.copy())

# média móvel 7 dias
data_plot.append(df_araraquara.deaths.diff(periods=7).copy())

print(df_araraquara.deaths.values)

# média móvel 14 dias
df_araraquara.deaths = df_araraquara.deaths.diff(periods=1)
data_plot.append(df_araraquara.deaths.rolling(14).sum())

for data_series in data_plot:
  plt.plot(data_series)
plt.show()

# test.x uma coisa
# train.x outra coisa
