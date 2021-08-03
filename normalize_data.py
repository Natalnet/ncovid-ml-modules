#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  2 15:20:18 2021

@author: davi
"""

import numpy as np
from sklearn.preprocessing import MinMaxScaler

# grid search configs
def normalize_data(data, type_norm):
  scaler = None
  norm_data = np.array([])
  if not type_norm:
    return data, None

  if type_norm == 'MinMax':
    samples = data.shape[0]
    columns = data.shape[1]
    data_ = data.flatten(order='A')
    data_ = data_.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaler.fit(data_)
    data_ = scaler.fit_transform(data_)
    for i in range(columns):
      a = data_[i*samples:(i+1)*samples]
      if norm_data.shape[0] == 0:
        norm_data = a
      else:
        norm_data = np.hstack((norm_data, a))
  return norm_data, scaler

# grid search configs
def desnormalize_data(data, scaler):
  data_ = data.flatten()
  data_ = scaler.inverse_transform(data_.reshape(-1,1))
  return data_