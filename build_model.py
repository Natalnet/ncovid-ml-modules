#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  2 15:13:33 2021

@author: davi
"""

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.constraints import maxnorm
import numpy as np

def define_model_space(n_nodes, n_timesteps, n_features, n_outputs, dropout):
  # create model (encoder - decoder)
  model = Sequential()
  model.add(LSTM(n_nodes, activation= 'relu' , input_shape=(n_timesteps, n_features), kernel_constraint=maxnorm(3)))
  model.add(Dropout(dropout))
  model.add(RepeatVector(n_outputs))
  model.add(LSTM(n_nodes, activation= 'relu' , return_sequences=True, kernel_constraint=maxnorm(3)))
  model.add(Dropout(dropout))
  model.add(TimeDistributed(Dense(np.floor(n_nodes/2), activation= 'relu')))
  model.add(TimeDistributed(Dense(1)))
  model.compile(loss= 'mse', optimizer= 'adam')
  return model