#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  2 15:21:09 2021

@author: davi
"""

from validate_model import evaluate_model

# return average of prediction errors for multiple repeats of same config
def repeat_evaluate(data, n_test, config, models, models_test, n_repeats=3):
  # fit and evaluate the model n times
  scores = [evaluate_model(data, n_test, config, models, models_test) for _ in range(n_repeats)]
  # convert config to a key
  key = str(config)
  #result = mean(scores)
  return (key, scores)

# grid search configs
def grid_search(data, n_test, cfg_list, n_repeats, models, scores, models_test):
  # evaluate configs
  for cfg in cfg_list:
    scores.append(repeat_evaluate(data, n_test, cfg, models, models_test, n_repeats))
  # sort configs by error, asc
  scores.sort(key=lambda tup: tup[1])
  return scores