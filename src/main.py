import pandas as pd
from enums import feature_enum
import matplotlib.pyplot as plt

import data_manner
import evaluator_manner
import plot_manner
import glossary_manner

df_araraquara = pd.read_csv('../dbs/df_araraquara.csv')
print(df_araraquara.columns)

###### glossary test code
glossary_1 = glossary_manner.Glossary()
glossary_1.find_column(df_araraquara.columns, type_feat=feature_enum.Feature.DEATHS)
print(glossary_1.features_dict)
# {'deaths': 'deaths'}

glossary_2 = glossary_manner.create_glossary(df_araraquara.columns)
glossary_2.find_column(df_araraquara.columns, type_feat=feature_enum.Feature.DEATHS)
glossary_2.find_column(df_araraquara.columns, type_feat=feature_enum.Feature.RECOVERED)
print(glossary_2.features_dict)
# {'cases': 'confirmed'}

glossary_3 = glossary_manner.create_glossary(df_araraquara.columns,
                                             feat_preset=feature_enum.BaseCollecting.EPIDEMIOLOGICAL)
print(glossary_3.features_dict)
# {'cases': 'confirmed', 'recovered': None, 'deaths': 'deaths'}

glossary_4 = glossary_manner.create_glossary(df_araraquara.columns,
                                             feat_preset=feature_enum.BaseCollecting.BASE)
print(glossary_4.features_dict)
# {'cases': 'confirmed', 'deaths': 'deaths'}


###### training and data_manner test code
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

# # ---------- Sample 1 - Without griding-search
# regressor = model_manner.build_model(model_config)
#
# regressor.fit_model(train, epochs=1, batch_size=16, verbose=0)
#
# train.y_hat, train.rmse = evaluator_manner.evaluate_model(regressor.model, train)
# test.y_hat, test.rmse = evaluator_manner.evaluate_model(regressor.model, test)
#
# print(sum(test.rmse))
#
# plt.rcParams["figure.figsize"] = (20, 5)
# plt.plot(train.x_label.reshape(train.x_label.shape[0], train.x_label.shape[1])[:, :1], label='original')
# plt.plot(train.y_hat.reshape(train.y_hat.shape[0], train.y_hat.shape[1])[:, :1], label='predicted')
# plt.show()

# ----------- Sample 2 - griding-search for a single configuration model based over the train data

n_repeat = 2
#grid_result_list = evaluator_manner.evaluate_model_n_repeat(n_repeat, model_config, train, epochs=10, batch_size=32, verbose=0)

model_config_dict = {'n_input': [7, 14],
                    'n_lstm_nodes': [100, 200],
                    'dropout': [0.0],
                    'n_features': [train.x.shape[2]],
                    'epochs': [5, 10],
                    'batch_size': [32]
                    }
# grid_search_params (n_repeat, model_config_dict, data, verbose=0)
gridsearch_result_list = evaluator_manner.grid_search_params(n_repeat, model_config_dict, data, verbose=0)

for i in gridsearch_result_list:
    print(i)
    
    
# (gridsearch_result_list, n_repeat, min_value, max_value)
# parameters min_value and max_value adjust the boxplot y-axis range
plot_manner.boxplot_experiments(gridsearch_result_list, n_repeat)

# grid_result_list[i][0]: LSTMRegressor object
# grid_result_list[i][1]: y_hat list of the previous LSTMRegressor object
# grid_result_list[i][2]: rmse list of the previous LSTMRegressor object
  
# for i in range(len(grid_result_list)):
#     print(sum(grid_result_list[i][2]))

# plt.rcParams["figure.figsize"] = (20, 5)
# plt.plot(train.y.reshape(train.y.shape[0], train.y.shape[1])[:, :1], label='original')
# plt.plot(grid_result_list[0][1].reshape(grid_result_list[0][1].shape[0], grid_result_list[0][1].shape[1])[:, :1], label='predicted')
# plt.show()
