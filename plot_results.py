#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  2 15:21:20 2021

@author: davi
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon
from validate_model import forecast_model
import pandas as pd
from scipy.signal import savgol_filter

def boxplot_experiments(data, key, n_repeats, max_value, min_value):
  random_dists = key

  fig, ax1 = plt.subplots(figsize=(10, 6))
  fig.canvas.manager.set_window_title('A Boxplot Example')
  fig.subplots_adjust(left=0.075, right=0.95, top=0.9, bottom=0.25)

  bp = ax1.boxplot(data, notch=0, sym='+', vert=1, whis=1.5)
  plt.setp(bp['boxes'], color='black')
  plt.setp(bp['whiskers'], color='black')
  plt.setp(bp['fliers'], color='red', marker='+')

  ax1.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
                alpha=0.5)

  ax1.set(
      axisbelow=True,  # Hide the grid behind plot objects
      title='Average performance of %d models for each configuration' % n_repeats,
      xlabel='Model configurations',
      ylabel='RMSE - Deaths forecast',
  )

  box_colors = ['darkkhaki', 'royalblue']
  num_boxes = len(data)
  medians = np.empty(num_boxes)
  for i in range(num_boxes):
      box = bp['boxes'][i]
      box_x = []
      box_y = []
      for j in range(5):
          box_x.append(box.get_xdata()[j])
          box_y.append(box.get_ydata()[j])
      box_coords = np.column_stack([box_x, box_y])
      # Alternate between Dark Khaki and Royal Blue
      ax1.add_patch(Polygon(box_coords, facecolor=box_colors[i % 2]))
      # Now draw the median lines back over what we just filled in
      med = bp['medians'][i]
      median_x = []
      median_y = []
      for j in range(2):
          median_x.append(med.get_xdata()[j])
          median_y.append(med.get_ydata()[j])
          ax1.plot(median_x, median_y, 'k')
      medians[i] = median_y[0]
      # Finally, overplot the sample averages, with horizontal alignment
      # in the center of each box
      ax1.plot(np.average(med.get_xdata()), np.average(data[i]),
              color='w', marker='*', markeredgecolor='k')

  # Due to the Y-axis scale being different across samples, it can be
  # hard to compare differences in medians across the samples. Add upper
  # X-axis tick labels with the sample medians to aid in comparison
  # (just use two decimal places of precision)
  pos = np.arange(num_boxes) + 1
  upper_labels = [str(round(s, 2)) for s in medians]
  weights = ['bold', 'semibold']
  for tick, label in zip(range(num_boxes), ax1.get_xticklabels()):
      k = tick % 2
      ax1.text(pos[tick], .95, upper_labels[tick],
              transform=ax1.get_xaxis_transform(),
              horizontalalignment='center', size='x-small',
              weight=weights[k], color=box_colors[k])

  # Set the axes ranges and axes labels
  ax1.set_xlim(0.5, num_boxes + 0.5)
  top = max(max(data))+max_value
  bottom = min(min(data))+min_value
  ax1.set_ylim(bottom, top)
  ax1.set_xticklabels(np.repeat(random_dists, 1),
                      rotation=70, fontsize=8)

  plt.show()

def boxplot_experiments_best(scores_grid, n_best, n_repeats, max_value, min_value):
  # get errors
  scores_list = list()
  scores_list_test = list()
  for key, cfg in scores_grid:
    scores = list()
    scores_test = list()
    for error, error_test  in cfg:
      # erro de ajuste
      scores.append(error)
      # erro de predição
      scores_test.append(error_test)
    scores_list.append((key, scores))
    scores_list_test.append((key, scores_test))

  list_scores_list = list()

  for list_score in [scores_list, scores_list_test]:
    # scores_list contém key da cfg e erros
    scores_array = np.array(list_score, dtype=object)
    scores_dict = dict(zip(scores_array[:,0], scores_array[:,1]))

    # sort configs by mean error
    config_averages = ((sum(scores) / len(scores), scores, s) for s, scores in scores_dict.items())

    score_list = list()
    key_list = list()
    average_list = list()

    for average, scores, config in sorted(config_averages, reverse=False):
      average_list.append(average)
      score_list.append(scores)
      key_list.append(config)

    boxplot_experiments(score_list[:n_best], key_list[:n_best], n_repeats, max_value, min_value)
    list_scores_list.append(score_list)

  return list_scores_list, key_list

# plot best forecast curve for the best models
def plot_predictions(models, data, n_models, n_test, date_, t_average):

  # array com todos os caras ordenados
  for model in models[:n_models]:
    model_ = model[2]
    cfg_array = model[1]
    cfg_ = str(cfg_array)
    n_input = cfg_array[0]

    # forecast_model
    test, predictions = forecast_model(model_, data, data.shape[0]-n_input, cfg_array)
    predictions = predictions.flatten()
    predictions = predictions.reshape(-1,1)
    test = test.flatten()
    test = test.reshape(-1,1)

    # copy this variable beacuse python ResidentSleeper
    date = date_.copy()

    date = date[:test.shape[0]]

    x_tick = list()
    # 10 values in axis x (9 + the last one)
    n_ticks = np.ceil(date.shape[0]/9)

    for i in range(date.shape[0]):
      if i % n_ticks == 0:
        x_tick.append(date[i])
    x_tick.append(date[-1])

    # calculates the series of accumulated deaths
    predictions_acc = moving_average_to_acc(predictions, t_average)
    test_acc = moving_average_to_acc(test, t_average)

    error_acc = (-test_acc[-1] + predictions_acc[-1]) * 100 / test_acc[-1]

    plt.rcParams["figure.figsize"] = (15,5)
    plt.plot(date, test, label='Real Data')
    plt.plot(date, predictions, label='Predicted')
    plt.axvspan(test.shape[0]-n_test, test.shape[0]-1, color='red', alpha=0.1)
    plt.xlabel("Days")
    plt.ylabel("Biweekly moving average")
    plt.title("Forecast of model (%.2f, %.2f) from config " % (model[0], model[4]) + cfg_)
    plt.legend()
    plt.xticks(x_tick)
    plt.grid()
    plt.show()

    plt.rcParams["figure.figsize"] = (15,5)
    plt.plot(date, test_acc, label='Real Data')
    plt.plot(date, predictions_acc, label='Predicted')
    plt.axvspan(test_acc.shape[0]-n_test, test_acc.shape[0]-1, color='red', alpha=0.1)
    plt.xlabel("Days")
    plt.ylabel("Biweekly moving average")
    plt.title("Accumulated Forecast with error %.2f %% from config " % error_acc + cfg_)
    plt.legend()
    plt.xticks(x_tick)
    plt.grid()
    plt.show()

def moving_average_to_acc(data, t_average):
  # transform to dataframe
  df_data = pd.DataFrame(data=data[:,0], columns=['deaths'])
  # copy this variable beacuse python ResidentSleeper
  df_data = df_data.copy()

  flag = False

  for i in range(0, data.shape[0]-1):
    sum = 0
    if i >= t_average-1:
      for j in range(1,t_average):
        sum = sum + df_data['deaths'][i-j]
      # convert from biweekly moving average to daily deaths
      df_data['deaths'][i] = df_data['deaths'][i+1]*t_average - sum
      # check if theres negative numbers (sometimes this happens if the model is bad)
      if df_data['deaths'][i+1]*t_average < sum:
        flag = True
  # accumulate daily deaths
  df_data["deaths"] = df_data["deaths"].cumsum()
  # shift
  df_data['deaths'] = df_data['deaths'].shift(periods=1)
  # remove nans
  df_data['deaths'] = df_data['deaths'].fillna(0)
  # filter acc data from bad model (avoid weird behavior)
  if flag:
    df_data['deaths'] = savgol_filter(df_data['deaths'].values, 71, 2)
  return df_data['deaths'].values