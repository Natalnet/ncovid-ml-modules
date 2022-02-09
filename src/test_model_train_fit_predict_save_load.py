import matplotlib.pyplot as plt

from statistics import mean
import data_manner
from models.artificial import lstm_manner

# --------- EXTRACT DATA
repo = "p971074907"
path = "brl:rn"
feature = "date:deaths:newCases:"
begin = "2020-05-01"
end = "2021-07-01"

construtor_dados = data_manner.DataConstructor()
data_araraquara = construtor_dados.collect_dataframe(path, repo, feature, begin, end)

# --------- BUILDING TRAIN AND TEST
train, test = construtor_dados.build_train_test(data_araraquara)
print(train.x.shape, train.y.shape)
print(test.x.shape, test.y.shape)

# --------- MODEL: CREATE - TRAIN - SAVE
lstm_model = lstm_manner.ModelLSTM("brl:rn")
lstm_model.loading()
if lstm_model.fiting(train.x, train.y, verbose=0):
    test.y_hat, test.rmse = lstm_model.predicting(test)
    print(mean(test.rmse))

lstm_model.saving()

construtor_dados_2 = data_manner.DataConstructor()
construtor_dados_2.is_predicting = True

# --------- BUILDING TEST
test_2 = construtor_dados_2.build_test(data_araraquara)
print(test_2.x.shape, test_2.y_hat.shape)

# --------- MODEL: LOAD - PREDICT
lstm_model_2 = lstm_manner.ModelLSTM(path)
lstm_model_2.loading()
test_2.y_hat, test_2.rmse = lstm_model_2.predicting(test)
