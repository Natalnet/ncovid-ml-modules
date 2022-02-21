import sys
sys.path.append("../src")
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from math import sqrt
import numpy as np

import data_manner
from models.artificial import lstm_manner

# --------- MANIPULATING DATA
repo = "p971074907"
path = "brl:rn"
feature = "date:NewDeaths:newCases:"
begin = "2020-05-01"
end = "2021-07-01"


#Training the model

# construtor_dados = data_manner.DataConstructor()
# data_araraquara = construtor_dados.collect_dataframe(path, repo, feature, begin, end)
# train, test = construtor_dados.build_train_test(data_araraquara)

# lstm_model = lstm_manner.ModelLSTM(path)
# lstm_model.fiting(train.x, train.y, verbose=0)
# lstm_model.saving()

# y_hat, rmse = lstm_model.predicting(test)

# real = test.y.reshape(-1)
# pred = y_hat.reshape(-1)

# print("fucntion rmse: ", sum(rmse)/len(rmse))
 
# cal_rmse = sqrt(mean_squared_error(real, pred))
# print("calculated rmse: ", cal_rmse)

# plt.plot(real, label='real')
# plt.plot(pred, label='pred')
# plt.legend(loc='best')
# plt.show()

#Loading the model

data_contructor = data_manner.DataConstructor()
data_araraquara = data_contructor.collect_dataframe(path, repo, feature, begin, end)
data_contructor.is_predicting = True
test_entire = data_contructor.build_test(data_araraquara)

lstm_model = lstm_manner.ModelLSTM(path)

lstm_model.loading()


test_entire.y_hat, test_entire.rmse = lstm_model.predicting(test_entire)

real = test_entire.y.reshape(-1)[:-7]
pred = test_entire.y_hat.reshape(-1)[7:]

print("function rmse: ", np.mean(test_entire.rmse))
 
cal_rmse = sqrt(mean_squared_error(real, pred))
print("calculated rmse: ", cal_rmse)

plt.plot(real, label='real')
plt.plot(pred, label='pred')
plt.legend(loc='best')
plt.show()