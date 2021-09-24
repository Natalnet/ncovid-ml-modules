#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 10:55:32 2021

@author: davi
"""

import pandas as pd
import matplotlib.pyplot as plt

import data_manner
import evaluator_manner
import plot_manner

def read_csv_file(path, column_date, last_day, first_date=None):
    df = pd.read_csv(path)
    if first_date is not None:
        return df[(df[column_date] < last_day) & (df[column_date] >= first_date)]
    return df[(df[column_date] < last_day)][1:]


db_folder = '../dbs/'
last_date = '2021-03-21'
df_araraquara = read_csv_file(db_folder + 'df_araraquara.csv', 'date', last_date, None)
df_araraquara.confirmed = df_araraquara.confirmed.diff(7).dropna()
df_araraquara.deaths = df_araraquara.deaths.diff(7).dropna()
df_araraquara = df_araraquara.dropna()

# ---------------------


data = [df_araraquara.deaths.values,
        df_araraquara.deaths.values]

week_size = 7
train, test = data_manner.build_data(data, step_size=week_size, size_data_test=21)

# model_config: [n_input, n_lstm_nodes, dropout, n_features]
model_config = [week_size, 200, 0.0, train.x.shape[2]]

n_repeat = 2

model_config_dict = {'n_input': [7],
                    'n_lstm_nodes': [100],
                    'dropout': [0.0],
                    'n_features': [train.x.shape[2]],
                    'epochs': [5],
                    'batch_size': [32]
                    }
# grid_search_params (n_repeat, model_config_dict, data, verbose=0)
gridsearch_result_list = evaluator_manner.grid_search_params(n_repeat, model_config_dict, data, verbose=0)

for i in gridsearch_result_list:
    print(i)
    
    
# (gridsearch_result_list, n_repeat, min_value, max_value)
# parameters min_value and max_value adjust the boxplot y-axis range
plot_manner.boxplot_experiments(gridsearch_result_list, n_repeat)

print(gridsearch_result_list)

y_hat = gridsearch_result_list[0]['model_list'][0].model.predict(test.x, 1)

print(y_hat)

print(test.y.shape)
print(y_hat.shape)

y_hat = plot_manner.build_data_from_trained_model(test, gridsearch_result_list[0]['model_list'][0].model)

print(plot_manner.build_data_from_trained_model(test, gridsearch_result_list[0]['model_list'][0].model))

model_example = gridsearch_result_list[0]['model_list'][0].model

plot_manner.plot_predictions(test, model_example, last_date)

plt.plot(train.x.flatten())
plt.plot(data[0])
plt.show()
print(train.x)

plot_manner.plot_predictions(train, model_example, last_date)
