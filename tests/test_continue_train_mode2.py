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
# --------- Rquest data params ------------
repo = "p971074907"
path = "brl:rn"
feature = "date:NewDeaths:newCases:"
# Date used to train and test, the last 2 months will be used to continue to train
end = "2021-06-22"

# Retraining

# date for continue to train data, include test data from previous training (aprox)
begin = "2021-05-12"
end = "2021-08-22"

data_constructor = data_manner.DataConstructor()
data_rn_retrain = data_constructor.collect_dataframe(path, repo, feature, begin, end)

retrain_len = len(data_rn_retrain[0])

data_constructor.is_predicting = True
retrain_data = data_constructor.build_train(data_rn_retrain) 

# Loading a trained model
lstm_model = lstm_manner.ModelLSTM(path)
lstm_model.loading()

# Continue train #MODE2

# All layer are not trainable, except the chosen. 
layer_trainable_index = 6 
 
for l, layer in enumerate(lstm_model.model.layers):
    if l != layer_trainable_index:
        layer.trainable = False

from tensorflow.keras.optimizers import Adam

opt = Adam(learning_rate=0.0005)
lstm_model.model.compile(loss="mse", optimizer=opt)

# Continue to train

lstm_model.fiting(retrain_data.x, retrain_data.y, verbose=0)

# Plot the retrain model for all data
# Date to all data
begin = "2020-03-14"
end = "2021-08-22"

# Data for plot
data_constructor = data_manner.DataConstructor()
data_rn = data_constructor.collect_dataframe(path, repo, feature, begin, end)

data_constructor.is_predicting = True
test_entire = data_constructor.build_test(data_rn) 

yhat = lstm_model.predicting(test_entire)

real = test_entire.y.reshape(-1)
pred = yhat[0].reshape(-1)

entire_rmse = sqrt(mean_squared_error(real, pred))
retrain_rmse = sqrt(mean_squared_error(real[:-retrain_len], pred[:-retrain_len]))

date_plot_begin = "2020-03-21"
date = pd.date_range(date_plot_begin, end)[:len(real)]

plt.figure(figsize=(15,8))
plt.title(f"RMSE Total: %.3f - Retrain: %.3f " % (entire_rmse, retrain_rmse))
plt.plot(date, real, label='real')
plt.plot(date, pred, label='pred')
plt.axvspan(date[-retrain_len], date[-1], label='test', alpha=0.2)
plt.legend(loc='best')
plt.grid(True)
plt.tight_layout()
plt.savefig("plots/MODE2")
plt.show()