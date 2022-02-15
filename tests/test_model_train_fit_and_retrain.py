import sys
sys.path.append("../src")

import data_manner
from models.artificial import lstm_manner
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt
import datetime

# The data has only values from 2020-03-14 to 2021-08-22
# --------- MANIPULATING DATA
repo = "p971074907"
path = "brl:rn"
feature = "date:NewDeaths:newCases:"
# Date to train and test, the last 2 months will be used to re-train e re-validation
begin = "2020-03-14"
end = "2021-06-22"

construtor_dados = data_manner.DataConstructor()
data_rn = construtor_dados.collect_dataframe(path, repo, feature, begin, end)
train, test = construtor_dados.build_train_test(data_rn)

# Creating and training the LSTM model
# lstm_model = lstm_manner.ModelLSTM(path)

# lstm_model.fiting(train.x, train.y, verbose=0)
# lstm_model.saving()
 
 
#Plotting 
data_contructor = data_manner.DataConstructor()
data_rn = data_contructor.collect_dataframe(path, repo, feature, begin, end)
data_contructor.is_predicting = True
test_entire = data_contructor.build_test(data_rn) 
 
lstm_model = lstm_manner.ModelLSTM(path)
lstm_model.loading()

test_entire.y_hat, _ = lstm_model.predicting(test_entire)

real = test_entire.y.reshape(-1)[:-7]
pred = test_entire.y_hat.reshape(-1)[7:]

entire_rmse = sqrt(mean_squared_error(real, pred))
train_rmse = sqrt(mean_squared_error(real[:-42], pred[:-42]))
test_rmse = sqrt(mean_squared_error(real[-42:], pred[-42:]))

plot_begin = "2020-03-21"
date = pd.date_range(plot_begin, end)[:len(real)]

plt.figure(figsize=(15,8))
plt.title(f"RMSE Total: %.3f - Train: %.3f - Test: %.3f" % (entire_rmse, train_rmse, test_rmse))
plt.plot(date, real, label='real')
plt.plot(date, pred, label='pred')
plt.axvspan(date[-43], date[-1], label='test', color='g', alpha=0.2)
plt.legend(loc='best')
plt.grid(True)
plt.tight_layout()
plt.savefig("plot")
plt.show()


