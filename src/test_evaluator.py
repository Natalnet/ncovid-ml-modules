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
history_eval = avaliador_modelo.evaluate_model()

for in_list in history_eval.rmse:
    print(mean(in_list))


colors = ["r", "g", "b"]
points = [".", "_", "o"]
i = 0
plt.legend(loc="best")
for in_list in history_eval.y_hat:
    plt.plot(
        in_list.reshape(in_list.shape[0], in_list.shape[1])[:, :1],
        marker=points[i],
        color=colors[i],
    )
    i += 1
plt.show()

plt.plot(ys, label="real", linewidth=1)
plt.plot(y_hats, label="pred", linewidth=1)
plt.legend(loc="best")
plt.show()
print()
