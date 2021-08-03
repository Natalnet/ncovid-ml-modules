#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  2 15:13:49 2021

@author: davi
"""

from numpy import array
from normalize_data import normalize_data
from prepare_data import split_dataset
from train_model import build_model
import numpy as np
from normalize_data import desnormalize_data
from evaluate_model import evaluate_forecasts

# make a forecast
def forecast(model, history, config):
  n_input, *_ = config
  # flatten data
  data = array(history)
  data = data.reshape((data.shape[0]*data.shape[1], data.shape[2]))
  # retrieve last observations for input data
  input_x = data[-n_input:, 1:]
  # reshape into [1, n_input, 1]
  input_x = input_x.reshape((1, input_x.shape[0], input_x.shape[1]))
  # forecast the next week
  yhat = model.predict(input_x, verbose=0)
  # we only want the vector forecast
  yhat = yhat[0]
  return yhat

# evaluate a single model
def evaluate_model(data, n_test, cfg, models, models_test):
  n_input, *_, type_norm = cfg
  if type_norm != None: 
    data_norm, scaler = normalize_data(data, type_norm)
  else:
    data_norm = data
    scaler = None
  train, test = split_dataset(data_norm, n_test, n_input)
  # fit model
  model = build_model(train, cfg)
  all_data = np.vstack((train,test))
  # history is a list of weekly data
  history = [all_data[0]]
  # walk-forward validation over each week
  predictions = list()
  for i in range(1, len(all_data)):
    # predict the week
    yhat_sequence = forecast(model, history, cfg)
    # store the predictions
    predictions.append(yhat_sequence)
    # get real observation and add to history for predicting the next week
    history.append(all_data[i, :])
  # evaluate predictions days for each week
  predictions = array(predictions)

  # 'denormalize' data, if necessary
  if type_norm != None:
    shape = predictions.shape
    predictions = desnormalize_data(predictions, scaler)
    predictions = predictions.reshape(shape[0], shape[1])
    shape = all_data.shape
    all_data = all_data[:, :, 0].flatten()
    all_data = desnormalize_data(all_data, scaler)
    all_data = all_data.reshape(shape[0], shape[1])
    all_data = all_data[1:, :]

  if type_norm == None:
    all_data = all_data[1:, :, 0]

  score, scores, score_test = evaluate_forecasts(all_data, predictions, n_input, n_test)

  # store the 10 best models to plot forecast curve
  if len(models) < 10:
    models.append((score, cfg, model, scaler, score_test))
    models.sort()

  if (len(models) >= 10) & (score < models[-1][0]):
    models[-1] = (score, cfg, model, scaler, score_test)
    models.sort()

  print(cfg, score, score_test)

  return (score, score_test)

# evaluate a single model
def forecast_model(model, data, n_test, cfg):
  n_input, *_, type_norm = cfg
  if type_norm != None: 
    data_norm, scaler = normalize_data(data, type_norm)
  else:
    data_norm = data
    scaler = None
  train, test = split_dataset(data_norm, n_test, n_input)
  # history is a list of weekly data
  history = [x for x in train]
  # walk-forward validation over each week
  predictions = list()
  for i in range(len(test)):
    # predict the week
    yhat_sequence = forecast(model, history, cfg)
    # store the predictions
    predictions.append(yhat_sequence)
    # get real observation and add to history for predicting the next week
    history.append(test[i, :])
  # evaluate predictions days for each week
  predictions = array(predictions)

  # 'denormalize' data, if necessary
  if type_norm != None:
    shape = predictions.shape
    predictions = desnormalize_data(predictions, scaler)
    predictions = predictions.reshape(shape[0], shape[1])
    shape = test.shape
    test = test[:, :, 0].flatten()
    test = desnormalize_data(test, scaler)
    test = test.reshape(shape[0], shape[1])

  if type_norm == None:
    test = test[:, :, 0]

  # returns real data from feature 0 and its forecast
  return test[:,:], predictions