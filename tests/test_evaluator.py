import sys

sys.path.append("../src")

import matplotlib.pyplot as plt
import numpy as np
from statistics import mean
from math import sqrt
from sklearn.metrics import mean_squared_error

import data_manner
import evaluator_manner
from models.artificial import lstm_manner
import configs_manner

# --------- EXTRACT DATA
repo = "p971074907"
path = "brl:rn"
feature = "date:newDeaths:newCases:"
begin = "2020-03-14"
end = "2021-06-22"

construtor_dados = data_manner.DataConstructor()
data_araraquara = construtor_dados.collect_dataframe(path, repo, feature, begin, end)

# --------- BUILDING TRAIN AND TEST
train, test = construtor_dados.build_train_test(data_araraquara)
print("Tain X and Y shape: ", train.x.shape, train.y.shape)
print("Test X and Y shape: ", test.x.shape, test.y.shape)
# --------- MODEL: CREATE NEW / ->{LOAD LOCALY}<- / LOAD REMOTELY - TRAIN - SAVE

lstm_model_local = lstm_manner.ModelLSTM(path)
lstm_model_local.loading()


# Evaluating Model

data_constructor = data_manner.DataConstructor()
data_rn = data_constructor.collect_dataframe(path, repo, feature, begin, end)

data_constructor.is_predicting = True
test_entire = data_constructor.build_test(data_rn) 

metrics = ['rmse', 'mse']

evaluator = evaluator_manner.Evaluator()
evaluated = evaluator.evaluate_model(lstm_model_local, test_entire, metrics=metrics)

print(evaluated)
