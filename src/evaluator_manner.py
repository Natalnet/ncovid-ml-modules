from math import sqrt

import numpy as np
from sklearn.metrics import mean_squared_error

import model_manner


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
