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

import datetime

dt = datetime.datetime(2021, 12, 1)
step = datetime.timedelta(days=1)
print(dt.strftime('%Y-%m-%d'))

for i in range(10):
  print(dt - datetime.timedelta(days=i))

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
        df_araraquara.confirmed.values]

week_size = 7
train, test = data_manner.build_data(data, step_size=week_size, size_data_test=21)

# model_config: [n_input, n_lstm_nodes, dropout, n_features]
model_config = [week_size, 200, 0.0, train.x.shape[2]]

# # ---------- Sample 1 - Without griding-search
# regressor = model_manner.build_model(model_config)
#
# regressor.fit_model(train, epochs=1, batch_size=16, verbose=0)
#
# train.y_hat, train.rmse = evaluator_manner.evaluate_model(regressor.model, train)
# test.y_hat, test.rmse = evaluator_manner.evaluate_model(regressor.model, test)
#
# print(sum(test.rmse))
#
# plt.rcParams["figure.figsize"] = (20, 5)
# plt.plot(train.x_label.reshape(train.x_label.shape[0], train.x_label.shape[1])[:, :1], label='original')
# plt.plot(train.y_hat.reshape(train.y_hat.shape[0], train.y_hat.shape[1])[:, :1], label='predicted')
# plt.show()

# ----------- Sample 2 - griding-search for a single configuration model based over the train data

n_repeat = 2
#grid_result_list = evaluator_manner.evaluate_model_n_repeat(n_repeat, model_config, train, epochs=10, batch_size=32, verbose=0)

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

print(gridsearch_result_list[0]['model_list'][0].model.predict(test.x, 1))

print(test.y.shape)
print(y_hat.shape)

plt.plot(test.y.flatten())
plt.plot(y_hat.flatten())
plt.show()

y_hat = plot_manner.build_data_from_trained_model(test, gridsearch_result_list[0]['model_list'][0].model)

print(plot_manner.build_data_from_trained_model(test, gridsearch_result_list[0]['model_list'][0].model))

model_example = gridsearch_result_list[0]['model_list'][0].model

plot_manner.plot_predictions(train, model_example, last_date)
