import matplotlib.pyplot as plt
from statistics import mean

import logger
import data_manner
from models.artificial import lstm_manner
from predictor_manner import PredictorConstructor

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

# --------- MODEL: CREATE NEW - TRAIN - SAVE
lstm_model_local_2 = lstm_manner.ModelLSTM(path)
lstm_model_local_2.creating()
lstm_model_local_2.fiting(train.x, train.y, verbose=0)
lstm_model_local_2.saving()
test.y_hat = lstm_model_local_2.predicting(test)
test.rmse = lstm_model_local_2.calculate_rmse(test.y, test.y_hat)
print(mean(test.rmse))

plt.plot(test.y.reshape(-1), label="real")
plt.plot(test.y_hat.reshape(-1), label="pred")
plt.show()

# --------- USING PREDICTOR WEBSITE NCOVID
predictor_rn = PredictorConstructor(path, repo, feature, begin, end)
y_hat_predictor = predictor_rn.predict()

plt.plot(y_hat_predictor.input_data.y.reshape(-1), label="real")
plt.plot(y_hat_predictor, label="predictor")
plt.show()
y_hat_predictor = predictor_rn.predictions_to_weboutput(y_hat_predictor)

logger.debug_log("MAIN", " - ", "FINISH")
