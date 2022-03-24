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

# --------- BUILDING TRAIN AND TEST AND EVALUATE DATA
train, test = data_constructor.build_train_test(data_repo)

data_constructor.is_predicting = True
all_data = data_constructor.build_test(data_repo)

# Evaluating Model N times

# Do this mode
n_repeat = 2
model_list = [lstm_manner.ModelLSTM(path) for _ in range(n_repeat)]

print("WAY 1 --- init")
evaluator = evaluator_manner.Evaluator()
evaluated = evaluator.evaluate_n_models_n_times(
    models=model_list, data_train=train, data=all_data, n_repeat=n_repeat
)
print("WAY 1", evaluated)
print("FINISH---------\n")
