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

data_constructor = data_manner.DataConstructor()
data_repo = data_constructor.collect_dataframe(path, repo, feature, begin, end)
data_constructor.is_predicting = True
all_data = data_constructor.build_test(data_repo) 

# --------- MODEL: CREATE NEW / ->{LOAD LOCALY}<- / LOAD REMOTELY - TRAIN - SAVE

lstm_model_local = lstm_manner.ModelLSTM(path)
lstm_model_local.loading()

# Evaluating Model

# If you want specify the metrics
#metrics = ['rmse', 'mse']

# Do this mode
evaluator = evaluator_manner.Evaluator(model=lstm_model_local, all_data=all_data)
evaluated = evaluator.evaluate_model()
print(evaluated)

# Or do this mode
evaluator = evaluator_manner.Evaluator()
evaluated = evaluator.evaluate_model(lstm_model_local, all_data, save=True)
print(evaluated)