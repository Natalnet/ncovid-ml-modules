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
end = "2022-02-20"

data_constructor = data_manner.DataConstructor()
data_repo = data_constructor.collect_dataframe(path, repo, feature, begin, end)

# --------- BUILDING TRAIN AND TEST AND EVALUATE DATA
train, test = data_constructor.build_train_test(data_repo)

data_constructor.is_predicting = True
all_data = data_constructor.build_test(data_repo) 

# Evaluating N models N times


# where 'path' is required to load the model_manner from the configure.json

# the params list can be like:
# params_list_varitaion = {
#         "nodes":       {"percent_variation": 0.5, "times": 2, "direction": 'p'}, 
#         "epochs":      {"percent_variation": 0.5, "times": 2, "direction": 'b'},
#         "batch_size":  {"percent_variation": 0.5, "times": 1, "direction": 'n'}
#     }

# or can be:
n_repeat = 5
params_list_varitaion = {"nodes": [250, 300, 350],
                         "epochs": [250, 300],
                         "dropout": [0.1, 0.15],
                        }

# and both:
# params_list_varitaion = {
#         "nodes":       {"percent_variation": 0.5, "times": 2, "direction": 'p'}, 
#         "epochs":       [100, 200, 250],
#         "batch_size":  {"percent_variation": 0.5, "times": 1, "direction": 'n'}
#     }

# this can lead a huge time to finish, depending on the n_repeat and the number of variations

evaluator = evaluator_manner.Evaluator(path, train, all_data, n_repeat=n_repeat)
evaluated = evaluator.evaluate_n_models_n_times(params_list_varitaion, save=True)
