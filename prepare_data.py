#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  2 15:10:23 2021

@author: davi
"""

from numpy import split
from numpy import array

# split a univariate dataset into train/test sets
def split_dataset(data, n_test, n_days):
  # makes dataset multiple of n_days
  data = data[data.shape[0] % n_days:]
  # make test set multiple of n_days
  n_test -= n_test % n_days
  # split into standard weeks
  train, test = data[:-n_test], data[-n_test:]
  # restructure into windows of weekly data
  train = array(split(train, len(train)/n_days))
  test = array(split(test, len(test)/n_days))
  return train, test

# convert history into inputs and outputs
def to_supervised(train, n_input, n_out=7):
  # flatten data
  data = train.reshape((train.shape[0]*train.shape[1], train.shape[2]))
  X, y = list(), list()
  in_start = 0
  # step over the entire history one time step at a time
  for _ in range(len(data)):
    # define the end of the input sequence
    in_end = in_start + n_input
    out_end = in_end + n_input
    # ensure we have enough data for this instance
    if out_end < len(data):
      # use all features, execept the first
      X.append(data[in_start:in_end, 1:])
      # to predict feature zero
      y.append(data[in_end:out_end, 0])
    # move along one time step
    in_start += 1
  return array(X), array(y)