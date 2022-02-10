import matplotlib.pyplot as plt
from statistics import mean

import data_manner
from models.artificial import lstm_manner

# --------- EXTRACT DATA
repo = "p971074907"
path = "brl:rn"
feature = "date:newDeaths:newCases:"
begin = "2020-05-01"
end = "2021-07-01"

construtor_dados = data_manner.DataConstructor()
data_araraquara = construtor_dados.collect_dataframe(path, repo, feature, begin, end)

# --------- BUILDING TRAIN AND TEST
train, test = construtor_dados.build_train_test(data_araraquara)
print(train.x.shape, train.y.shape)
print(test.x.shape, test.y.shape)

# --------- MODEL: CREATE NEW - TRAIN - SAVE
lstm_model_local_2 = lstm_manner.ModelLSTM(path)
lstm_model_local_2.creating()
lstm_model_local_2.fiting(train.x, train.y, verbose=0)
lstm_model_local_2.saving()


test.y_hat, test.rmse = lstm_model_local_2.predicting(test)
print(mean(test.rmse))

plt.plot(test.y.reshape(-1)[:-7], label="real")
plt.plot(test.y_hat.reshape(-1)[7:], label="pred")
plt.legend(loc="best")
plt.show()
print()
