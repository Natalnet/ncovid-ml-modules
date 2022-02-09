import sys
sys.path.append("../src")

import data_manner
from models.artificial import lstm_manner

# --------- MANIPULATING DATA
repo = "p971074907"
path = "brl:rn"
feature = "date:deaths:newCases:"
begin = "2020-05-01"
end = "2021-07-01"

construtor_dados = data_manner.DataConstructor()
data_araraquara = construtor_dados.collect_dataframe(path, repo, feature, begin, end)
train, test = construtor_dados.build_train_test(data_araraquara)
print(train.x.shape, train.y.shape)
print(test.x.shape, test.y.shape)

# ----------------------- CRIANDO UM MODELO, TREINANDO E FAZENDO PREVISOES COM DADOS DE TESTE
import models.artificial.lstm_manner

print("TREINANDO UM MODELO E PREDICOES- 1\n")
modelo = lstm_manner.ModelLSTM("Araraquara")
print(modelo)
if modelo.fiting(train.x, train.y):
    y_hat, rmse = modelo.predicting(test)
    print(sum(rmse))
modelo.saving()

# ----------------------- AVALIANDO UM UNICO MODELO
import evaluator_manner

print("criando um avaliador de UM MODELO (TREINO + TESTES) - 1\n")
# FORMA 1
avaliador_de_um_modelo_1 = evaluator_manner.Evaluator(
    model=modelo, data_train=train, data_test=test
)
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

print(
    "CRIANDO avaliador de UM MODELO n vezes (setando quantidade de repeticoes = 1) - 2\n"
)
# FORMA 2
avaliador_de_um_modelo_2.n_repeat = 3
modelo_avaliado_n_vezes_2 = avaliador_de_um_modelo_2.evaluate_model_n_times()
print(len(modelo_avaliado_n_vezes_2))

# ----------------------- AVALIANDO MUITOS MODELOS
print("CRIANDO avaliador de MUITOS MODELOS n vezes\n")
modelo_1 = lstm_manner.ModelLSTM("Araraquara", nodes=200, dropout=0.0)
print(modelo_1)
modelo_2 = lstm_manner.ModelLSTM("Araraquara", nodes=100, dropout=0.0)
print(modelo_2)
modelo_3 = lstm_manner.ModelLSTM("Araraquara", nodes=50, dropout=0.3)
print(modelo_3)
modelos = [modelo_1, modelo_2, modelo_3]

print("criando avaliador de MUITOS MODELOS n vezes\n")
avaliador_de_varios_modelos = evaluator_manner.Evaluator(
    data_train=train, data_test=test
)
avaliador_de_varios_modelos.models = modelos
print(avaliador_de_varios_modelos)

print("AVALIANDO de MUITOS MODELOS n vezes\n")
n_modelos_avaliados_n_vezes = avaliador_de_varios_modelos.evaluate_n_models_n_times()
print(n_modelos_avaliados_n_vezes)

[modelo_i.saving() for modelo_i in modelos]
