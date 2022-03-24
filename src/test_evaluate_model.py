import data_manner
import evaluator_manner
from models.artificial import lstm_manner

# --------- EXTRACT DATA
repo = "p971074907"
path = "brl:rn"
feature = "date:newDeaths:newCases:"
begin = "2020-03-14"
end = "2021-06-22"

data_constructor = data_manner.DataConstructor()
data_repo = data_constructor.collect_dataframe(path, repo, feature, begin, end)
data_constructor.is_predicting = True
all_data = data_constructor.build_test(data_repo)

# --------- MODEL: CREATE NEW / ->{LOAD LOCALY}<- / LOAD REMOTELY - TRAIN - SAVE

lstm1 = lstm_manner.ModelLSTM(path)
lstm1.creating()

lstm2 = lstm_manner.ModelLSTM(path)
lstm2.creating()

# metrics = ['rmse', 'mse']
evaluator = evaluator_manner.Evaluator(model=lstm1)

evaluator = evaluator_manner.Evaluator()
evaluator.model = lstm1

model_evals = evaluator.evaluate_model(data=all_data)

print(model_evals)

print("----------FINISH")
