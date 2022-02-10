import matplotlib.pyplot as plt
from statistics import mean

import data_manner
import evaluator_manner
from models.artificial import lstm_manner

# --------- EXTRACT DATA
repo = "p971074907"
path = "brl:rn"
feature = "date:newDeaths:newCases:"
begin = "2020-05-01"
end = "2021-07-01"

construtor_dados = data_manner.DataConstructor()
data_araraquara = construtor_dados.collect_dataframe(path, repo, feature, begin, end)

# --------- BUILDING TRAIN AND TEST
train, test = construtor_dados.build_train_test(data_araraquara)
print(train.x.shape, train.y.shape)
print(test.x.shape, test.y.shape)

# --------- MODEL: CREATE NEW / LOAD LOCALY / LOAD REMOTELY - TRAIN - SAVE

# lstm_model_web = lstm_manner.ModelLSTM("brl:to")
# lstm_model_web.loading()

lstm_model_local_2 = lstm_manner.ModelLSTM(path)
lstm_model_local_2.loading()

avaliador_modelo = evaluator_manner.Evaluator(lstm_model_local_2, train, test)
ys, y_hats, rmses = avaliador_modelo.evaluate_model()
print(mean(rmses))

plt.plot(ys, label="real", linewidth=1)
plt.plot(y_hats, label="pred", linewidth=1)
plt.legend(loc="best")
plt.show()
