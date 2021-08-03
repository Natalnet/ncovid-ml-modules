#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  2 15:13:49 2021

@author: davi
"""

from prepare_data import to_supervised
from build_model import define_model_space
from keras.callbacks import EarlyStopping

# create a list of configs to try
def model_configs():
  # define scope of configs
  n_input = [7]
  n_nodes = [50]
  n_epochs = [50]
  n_batch = [16]
  dropout = [0.0]
  normalization = [None]
  # create configs
  configs = list()
  for i in n_input:
    for j in n_nodes:
      for k in n_epochs:
        for l in n_batch:
          for o in dropout:
            for p in normalization:
              cfg = [i, j, k, l, o, p]
              configs.append(cfg)
  print('Total configs: %d' % len(configs))
  return configs

# train the model
def build_model(train, config):
  # unpack config
  n_input, n_nodes, n_epochs, n_batch, dropout, *_ = config
  n_output = n_input
  # prepare data
  train_x, train_y = to_supervised(train, n_input, n_output)
  n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]
  train_y = train_y.reshape((train_y.shape[0], train_y.shape[1], 1))
  # define model space (encoder - decoder)
  model = define_model_space(n_nodes, n_timesteps, n_features, n_outputs, dropout)
  # early stop
  es = EarlyStopping(monitor='loss', mode='min', verbose=0, patience=100)
  # fit model
  model.fit(train_x, train_y, epochs=n_epochs, batch_size=n_batch, verbose=0)
  return model