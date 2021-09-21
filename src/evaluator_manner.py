from math import sqrt

import numpy as np
from sklearn.metrics import mean_squared_error

import itertools as it

from sklearn.model_selection import GridSearchCV

import model_manner
import data_manner


def evaluate_model(model, data):
    yhat = model.predict(data.x, verbose=0)

    rmse_scores = list()
    for i in range(len(yhat)):
        mse = mean_squared_error(data.y[i], yhat[i])
        rmse = sqrt(mse)
        rmse_scores.append(rmse)

    return yhat, rmse_scores


def evaluate_forecast(model, train, test):
    # walk-forward validation over each week
    history = train
    predictions = list()
    rmses = list()
    for i in range(len(test.x)):
        # predict the week
        yhat, rmse = evaluate_model(model, history)
        # store the predictions
        predictions.append(yhat)
        rmses.append(rmse)
        # get real observation and add to history for predicting the next week
        history.x = np.vstack((history.x, test.x[i:i + 1:, ]))
        history.y = np.vstack((history.y, test.y[i:i + 1:, ]))
    # evaluate predictions days for each week
    # predictions = np.array(predictions)
    return predictions, rmses


def evaluate_model_n_repeat(n_repeat, model_config, train, epochs=100, batch_size=32, verbose=0):
    regressor_list = list()
    y_hat_list = list()
    rmse_list = list()
    for i in range(n_repeat):
        regressor_list.append(model_manner.build_model(model_config))
        regressor_list[i].fit_model(train, epochs, batch_size, verbose)
        y_hat, rmse = evaluate_model(regressor_list[i].model, train)
        y_hat_list.append(y_hat)
        rmse_list.append(rmse)

    return list(zip(regressor_list, y_hat_list, rmse_list))

def grid_search_params(n_repeat, model_config_dict, data, verbose=0):

    my_dict_model = model_config_dict

    # mount all model conficguration in the dictionary
    combinations = it.product(*(my_dict_model[param_name] for param_name in my_dict_model))
    combinations_configs = list(combinations)

    grid_search_result = list()

    # iterate around those combinations
    for config in combinations_configs:
        # split the model configuratioon and the evaluate configuration
        current_model_config = config[:-2]  
        # epochs and batch size are differentially treaties
        epoch, bs = config[-2:]

        # building the data according with the current model configuration
        week_size = current_model_config[0]
        train, test = data_manner.build_data(data, step_size=week_size, size_data_test=21)

        # unique config evaluation for 'n' times
        grid_repeat_unique_config = evaluate_model_n_repeat(n_repeat, current_model_config, train, epochs=epoch, batch_size=bs, verbose=verbose)

        evaluate_n_repeat_config = list()
        # iterate in the unique repeat configuration and summing the evaluation metric for each 'n' time.
        for uniq_config in grid_repeat_unique_config:
            evaluate_n_repeat_config.append(np.round(sum(uniq_config[2]), 3))
        
        # creating a dict for each configuration and result
        config_results_dict = {'config': config, 'n_rmse_distribution': evaluate_n_repeat_config}
        
        grid_search_result.append(config_results_dict)
    return grid_search_result