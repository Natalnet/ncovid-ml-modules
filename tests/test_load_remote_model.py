import sys
sys.path.append("../src")
import data_manner
from models.artificial import lstm_manner

# --------- MANIPULATING DATA
repo = "p971074907"
path = "brl:rn"
feature = "date:deaths:newCases:"
begin = "2020-05-01"
end = "2020-07-28"

# using the same data from "brl:rn" for both cases just for example
construtor_dados = data_manner.DataConstructor()
data_araraquara = construtor_dados.collect_dataframe(path, repo, feature, begin, end)
train, test = construtor_dados.build_train_test(data_araraquara)

#local model (exist in ../dbs/fitted_model)
local_model = lstm_manner.ModelLSTM("brl:rn")
local_model.loading()

y, rmse = local_model.predicting(test)
print("Local model predict", y, rmse)

#remote model (does not exist in ../dbs/fitted_model)
remote_model = lstm_manner.ModelLSTM("brl:pb")
remote_model.loading()

y, rmse = remote_model.predicting(test)
print("Remote model predict", y, rmse)