import sys

sys.path.insert(0, '../src')

import pandas as pd

df_araraquara = pd.read_csv('../dbs/df_araraquara.csv')
df_araraquara.confirmed = df_araraquara.confirmed.diff(7).dropna()
df_araraquara.deaths = df_araraquara.deaths.diff(7).dropna()
df_araraquara = df_araraquara.dropna()

data = [df_araraquara.deaths.values, df_araraquara.confirmed.values]

# ----------------------- PREPARANDO OS DADOS
import data_manner

# FORMA 1
step_size = 7
construtor_dados = data_manner.DataConstructor(step_size=step_size, is_training=True)
train, test = construtor_dados.build_train_test(data, size_data_test=step_size * 3)

# FORMA 2
# train = construtor_dados.build_train(data)
# test = construtor_dados.build_test(data)

# ----------------------- CRIANDO UM MODELO, TREINANDO E FAZENDO PREVISOES COM DADOS DE TESTE
import models.lstm_manner as model_manner

print("TREINANDO UM MODELO E PREDICOES- 1\n")
modelo = model_manner.ModelLSTM(n_inputs=step_size, n_features=train.x.shape[2], n_nodes=200, dropout=0.0)
print(modelo)
if modelo.fit_model(train.x, train.y):
    y_hat, rmse = modelo.make_predictions(test)
    print(sum(rmse))
modelo.save_model(locale='Araraquara')

# ----------------------- AVALIANDO UM UNICO MODELO
import evaluator_manner

print("criando um avaliador de UM MODELO (TREINO + TESTES) - 1\n")
# FORMA 1
avaliador_de_um_modelo_1 = evaluator_manner.Evaluator(model=modelo, data_train=train, data_test=test)
print(avaliador_de_um_modelo_1)

print("criando um avaliador de  UM MODELO (TREINO + TESTES) - 2\n")
# FORMA 2
avaliador_de_um_modelo_2 = evaluator_manner.Evaluator()
avaliador_de_um_modelo_2.model = modelo
avaliador_de_um_modelo_2.data_train = train
avaliador_de_um_modelo_2.data_test = test
print(avaliador_de_um_modelo_2)

print("AVALIANDO OS (UM) MODELO (TREINO + TESTES) - 1 e 2\n")
predictions_1, rmses_1 = avaliador_de_um_modelo_1.evaluate_model()
predictions_2, rmses_2 = avaliador_de_um_modelo_2.evaluate_model()
if rmses_1[0][0] == rmses_2[0][0]:
    print(rmses_1[0][0], rmses_2[0][0])
    print("equals\n")

# ----------------------- AVALIANDO UM UNICO MODELO MUITAS VEZES
print("Avaliando UM MODELO n vezes ( 2 repeticoes (OU 1, padrao)- 1\n")
# FORMA 1
# setando quantidade de repeticoes
modelo_avaliado_n_vezes_1 = avaliador_de_um_modelo_1.evaluate_model_n_times(n_repeat=2)
print(len(modelo_avaliado_n_vezes_1))

print("CRIANDO avaliador de UM MODELO n vezes (setando quantidade de repeticoes = 1) - 2\n")
# FORMA 2
avaliador_de_um_modelo_2.n_repeat = 3
modelo_avaliado_n_vezes_2 = avaliador_de_um_modelo_2.evaluate_model_n_times()
print(len(modelo_avaliado_n_vezes_2))

# ----------------------- AVALIANDO MUITOS MODELOS
print("CRIANDO avaliador de MUITOS MODELOS n vezes\n")
modelo_1 = model_manner.ModelLSTM(n_inputs=step_size, n_features=train.x.shape[2], n_nodes=200, dropout=0.0)
print(modelo_1)
modelo_2 = model_manner.ModelLSTM(n_inputs=step_size, n_features=train.x.shape[2], n_nodes=100, dropout=0.0)
print(modelo_2)
modelo_3 = model_manner.ModelLSTM(n_inputs=step_size, n_features=train.x.shape[2], n_nodes=50, dropout=0.3)
print(modelo_3)
modelos = [modelo_1, modelo_2, modelo_3]

print("criando avaliador de MUITOS MODELOS n vezes\n")
avaliador_de_varios_modelos = evaluator_manner.Evaluator(data_train=train, data_test=test)
avaliador_de_varios_modelos.models = modelos
print(avaliador_de_varios_modelos)

print("AVALIANDO de MUITOS MODELOS n vezes\n")
n_modelos_avaliados_n_vezes = avaliador_de_varios_modelos.evaluate_n_models_n_times()
print(n_modelos_avaliados_n_vezes)