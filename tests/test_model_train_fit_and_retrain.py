import sys
sys.path.append("../src")

import data_manner
from models.artificial import lstm_manner
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt

# The data has only values from 2020-03-14 to 2021-08-22
# --------- MANIPULATING DATA
repo = "p971074907"
path = "brl:rn"
feature = "date:NewDeaths:newCases:"
# Date to train and test, the last 2 months will be used to re-train e re-validation
begin = "2020-03-14"
end = "2021-06-22"

# construtor_dados = data_manner.DataConstructor()
# data_rn = construtor_dados.collect_dataframe(path, repo, feature, begin, end)
# train, test = construtor_dados.build_train_test(data_rn)

# #Creating and training the LSTM model
# lstm_model = lstm_manner.ModelLSTM(path)
# lstm_model.creating()
# lstm_model.fiting(train.x, train.y, verbose=0)
# lstm_model.saving()
 
 
# #Plotting 
# data_contructor = data_manner.DataConstructor()
# data_rn = data_contructor.collect_dataframe(path, repo, feature, begin, end)
# data_contructor.is_predicting = True
# test_entire = data_contructor.build_test(data_rn) 
 
# lstm_model = lstm_manner.ModelLSTM(path)
# lstm_model.loading()

# yhat = lstm_model.predicting(test_entire)

# real = test_entire.y.reshape(-1)
# pred = yhat[0].reshape(-1)

# entire_rmse = sqrt(mean_squared_error(real, pred))
# train_rmse = sqrt(mean_squared_error(real[:-42], pred[:-42]))
# test_rmse = sqrt(mean_squared_error(real[-42:], pred[-42:]))

# plot_begin = "2020-03-21"
# date = pd.date_range(plot_begin, end)[:len(real)]

# plt.figure(figsize=(15,8))
# plt.title(f"RMSE Total: %.3f - Train: %.3f - Test: %.3f" % (entire_rmse, train_rmse, test_rmse))
# plt.plot(date, real, label='real')
# plt.plot(date, pred, label='pred')
# plt.axvspan(date[-43], date[-1], label='test', color='g', alpha=0.2)
# plt.legend(loc='best')
# plt.grid(True)
# plt.tight_layout()
# plt.savefig("develop_plot")
# plt.show()


# Retraining

# date for retrain data, include test data from previous training (aprox)
begin = "2021-05-12"
end = "2021-08-22"

data_constructor = data_manner.DataConstructor()
data_rn_retrain = data_constructor.collect_dataframe(path, repo, feature, begin, end)

retrain_len = len(data_rn_retrain[0])

data_constructor.is_predicting = True
retrain_data = data_constructor.build_train(data_rn_retrain) 

lstm_model = lstm_manner.ModelLSTM(path)
lstm_model.loading()

# Retraining

print(lstm_model.model.summary())

layer = lstm_model.model.layers[6]

#layer.trainable = False

#print("ALL TRAINABLE", layer.trainable_weights)
import tensorflow as tf
vector = layer.trainable_weights[0]

randon_len = int(np.random.randint(vector.shape[0], size=1))

non_train = tf.gather(vector, indices=np.random.randint(vector.shape[0], size=randon_len)).numpy()

setattr(layer, '_non_trainable_weights', layer.trainable_weights)

layer.non_trainable_weights[0].assign(np.pad(non_train.reshape(-1), (0, vector.shape[0] - non_train.shape[0]), 'mean').reshape(vector.shape[0], 1))

#print("NEW NONTRAINABLE", layer.non_trainable_weights)

# #non_train = layer.non_trainable_weights
# print("NEW NONTRAINABLE", layer.non_trainable_weights)

#print("NON TRAINABLE", non_train)

# setattr(layer, '_non_trainable_weights', non_train)

# print("non-tainable", layer.non_trainable_weights)

print(lstm_model.model.summary())

    
# from tensorflow.keras.optimizers import Adam

# opt = Adam(learning_rate=0.0001)
# lstm_model.model.compile(loss="mse", optimizer=opt)

# lstm_model.fiting(retrain_data.x, retrain_data.y, verbose=0)

# # Plot the retrain model for all data
# # Date to all data
# begin = "2020-03-14"
# end = "2021-08-22"

# data_constructor = data_manner.DataConstructor()
# data_rn = data_constructor.collect_dataframe(path, repo, feature, begin, end)

# data_constructor.is_predicting = True
# test_entire = data_constructor.build_test(data_rn) 

# yhat = lstm_model.predicting(test_entire)

# real = test_entire.y.reshape(-1)
# pred = yhat[0].reshape(-1)

# entire_rmse = sqrt(mean_squared_error(real, pred))
# retrain_rmse = sqrt(mean_squared_error(real[:-retrain_len], pred[:-retrain_len]))

# plot_begin = "2020-03-21"
# date = pd.date_range(plot_begin, end)[:len(real)]

# plt.figure(figsize=(15,8))
# plt.title(f"RMSE Total: %.3f - Retrain: %.3f " % (entire_rmse, retrain_rmse))
# plt.plot(date, real, label='real')
# plt.plot(date, pred, label='pred')
# plt.axvspan(date[-retrain_len], date[-1], label='test', alpha=0.2)
# plt.legend(loc='best')
# plt.grid(True)
# plt.tight_layout()
# plt.savefig("plots/retrain_plot")
# plt.show()