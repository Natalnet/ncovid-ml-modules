#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  2 15:13:49 2021

@author: davi
"""

from sklearn.metrics import mean_squared_error
from math import sqrt

# evaluate one or more weekly forecasts against expected values
def evaluate_forecasts(actual, predicted, n_input, n_test):
  scores = list()
  # calculate an RMSE score for each day
  for i in range(actual.shape[1]):
    # calculate mse
    mse = mean_squared_error(actual[:, i], predicted[:, i])
    # calculate rmse
    rmse = sqrt(mse)
    # store
    scores.append(rmse)
  # calculate overall RMSE
  s = 0
  s_test = 0
  for row in range(actual.shape[0]):
    for col in range(actual.shape[1]):
      s += (actual[row, col] - predicted[row, col])**2
      if row > actual.shape[0]-(n_test/n_input):
        s_test += (actual[row, col] - predicted[row, col])**2
  score = sqrt(s / actual.shape[1])
  score_test = sqrt(s_test / actual.shape[1])
  return score, scores, score_test

# summarize scores
def summarize_scores(name, score, scores):
  s_scores = ', '.join(['%.1f' % s for s in scores])
  print('%s: [%.3f] %s' % (name, score, s_scores))