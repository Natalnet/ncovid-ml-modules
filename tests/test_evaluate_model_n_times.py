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

# --------- BUILDING TRAIN AND TEST AND EVALUATE DATA
train, test = data_constructor.build_train_test(data_repo)

data_constructor.is_predicting = True
all_data = data_constructor.build_test(data_repo) 

# Evaluating Model N times

# Do this mode
n_repeat = 2
model_list = [lstm_manner.ModelLSTM(path) for _ in range(n_repeat)]

evaluator = evaluator_manner.Evaluator()
evaluated = evaluator.evaluate_model_n_times(model_list, train, all_data, n_repeat=n_repeat, save=True)

print("\n")
print("WAY 1", evaluated)
print("\n")
    
#Or do this way

# where 'path' is required to load the model_manner from the configure.json
evaluator = evaluator_manner.Evaluator()
evaluated = evaluator.evaluate_model_n_times(path, train, all_data, n_repeat=n_repeat)

print("WAY 2", evaluated)
print("\n")

# Also like evaluate_model() you can pass the infos on the constructor call

# where 'path' is required to load the model_manner from the configure.json
evaluator = evaluator_manner.Evaluator(path, train, all_data, n_repeat=n_repeat)
evaluated = evaluator.evaluate_model_n_times()

print("WAY 3", evaluated)
print("\n")

# or you can give a list of models
evaluator = evaluator_manner.Evaluator(model_list, train, all_data, n_repeat=n_repeat)
evaluated = evaluator.evaluate_model_n_times()

print("WAY 4", evaluated)