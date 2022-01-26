import pandas as pd
import matplotlib.pyplot as plt

import data_manner

db_folder = '../dbs/'
last_date = '2021-03-21'
df_araraquara = data_manner.DataConstructor.read_csv_file(db_folder + 'df_araraquara.csv', 'date', last_date, None)

# --------- PLOTTING DATA
data_plot = list()
# acumulado
data_plot.append(df_araraquara.deaths.values.copy())
# média móvel 7 dias
data_plot.append(df_araraquara.deaths.diff(periods=7).copy())
# média móvel 7 dias
data_plot.append(df_araraquara.deaths.rolling(14).sum())

for data_series in data_plot:
    plt.plot(data_series)
plt.show()


# --------- MANIPULATING DATA
df_araraquara.confirmed = df_araraquara.confirmed.diff(7).dropna()
df_araraquara.deaths = df_araraquara.deaths.diff(7).dropna()
df_araraquara = df_araraquara.dropna()
data = [df_araraquara.deaths.values, df_araraquara.confirmed.values]

# FORMA 1
step_size = 7
test_size = step_size * 3
construtor_dados_1 = data_manner.DataConstructor(is_training=True)
train, test = construtor_dados_1.build_train_test(data)
print(train.x[-1], train.x.shape, train.y.shape)
print("----------")
# FORMA 2
construtor_dados_2 = data_manner.DataConstructor()
train = construtor_dados_2.build_train(data)
test = construtor_dados_2.build_test(data)
print(train.x[-1], train.x.shape, train.y.shape)