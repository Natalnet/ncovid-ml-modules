import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
import data_manner
import evaluator_manner
import model_manner


def read_csv_file(path, column_date, last_day, first_date=None):
    df = pd.read_csv(path)
    if first_date is not None:
        return df[(df[column_date] < last_day) & (df[column_date] >= first_date)]
    return df[(df[column_date] < last_day)][1:]


db_folder = '../dbs/'
data_limite = '2021-03-21'
df_araraquara = read_csv_file(db_folder + 'df_araraquara.csv', 'date', data_limite, None)
df_araraquara.confirmed = df_araraquara.confirmed.diff(7).dropna()
df_araraquara.deaths = df_araraquara.deaths.diff(7).dropna()
df_araraquara = df_araraquara.dropna()

# ---------------------

data = [df_araraquara.deaths.values,
        df_araraquara.confirmed.values]

week_size = 7
train, test = data_manner.build_data(data, step_size=week_size, size_data_test=21)

# model_config: [n_input, n_lstm_nodes, dropout, n_features]
model_config = [week_size, 200, 0.0, train.x.shape[2]]
regressor = model_manner.build_model(model_config)


regressor.fit_model(train, epochs=1, batch_size=16, verbose=0)
yhat_train, rmse_scores_train = evaluator_manner.evaluate_model(regressor.model, train)
yhat_test, rmse_scores_test = evaluator_manner.evaluate_model(regressor.model, test)

print(sum(rmse_scores_test))

plt.rcParams["figure.figsize"] = (20, 5)
plt.plot(train.x_label.reshape(train.x_label.shape[0], train.x_label.shape[1])[:, :1], label='original')
plt.plot(yhat_train.reshape(yhat_train.shape[0], yhat_train.shape[1])[:, :1], label='predicted')
plt.show()
#print(yhat_test[0].reshape(-1, 1))
#plt.plot(yhat_test[0].reshape(-1, 1), label='predicted')
