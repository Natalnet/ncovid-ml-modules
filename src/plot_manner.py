#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  6 15:47:41 2021

@author: davi
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon
import datetime

def boxplot_grid_search(data, key, n_repeats):
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
  top = np.max(data)+np.max(data)*0.05
  bottom = np.min(data)-np.min(data)*0.05
  print(top, bottom)
  ax1.set_ylim(bottom, top)
  ax1.set_xticklabels(np.repeat(random_dists, 1),
                      rotation=70, fontsize=8)

  plt.show()
  
def boxplot_experiments(gridsearch_result_list, n_repeats):
  
  # parse list to boxplot_gridsearch input
  # call it
  data_list = list()
  key_list = list()
  for experiment in gridsearch_result_list:
    key_list.append(str(experiment['config']))
    data_list.append(experiment['n_rmse_distribution'])
    
  boxplot_grid_search(data_list, key_list, n_repeats)
    
def build_data_from_trained_model(data, model):
  return model.predict(data.x)

def create_date_axis(last_date, n_days):
  dt = datetime.datetime.strptime(last_date, '%Y-%m-%d')
  date_axis = list()

  for i in range(n_days):
    date_ = dt - datetime.timedelta(days=i)
    date_axis.append(date_.strftime('%Y-%m-%d'))
    
  return date_axis[::-1]

def create_x_tick(dates):
  size = len(dates)
  x_tick = list()
  # 10 values in axis x (9 + the last one)
  n_ticks = np.ceil(size/9)

  for i in range(size):
    if i % n_ticks == 0:
      x_tick.append(dates[i])
  x_tick.append(dates[-1])
  return x_tick

def plot_predictions(data, model, last_date):
  
  # get model prediction from data
  predictions = build_data_from_trained_model(data, model)
   
  data_y_plot = data.y.flatten()
  predictions_plot = predictions.flatten()
  # create x-axis from last_date
  date_series_plot = create_date_axis(last_date, data_y_plot.shape[0])
  
  x_tick = create_x_tick(date_series_plot)
  
  plt.rcParams["figure.figsize"] = (15,5)
  plt.plot(date_series_plot, data_y_plot, label='Real Data')
  plt.plot(date_series_plot, predictions_plot, label='Predicted')
  #plt.axvspan(test.shape[0]-n_test, test.shape[0]-1, color='red', alpha=0.1)
  plt.xlabel("Days")
  plt.ylabel("Biweekly moving average")
  #plt.title("Forecast of model (%.2f, %.2f) from config " % (model[0], model[4]) + cfg_)
  plt.legend()
  plt.xticks(x_tick)
  plt.grid()
  plt.show()