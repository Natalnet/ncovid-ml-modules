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
model_list[0].nodes = 150
model_list[0].batch_size = 32
model_list[0].epochs = 50

print("WAY 1 --- init")
evaluator = evaluator_manner.Evaluator()
models_evals = {}

for idx, model in enumerate(model_list):
    models_evals[idx] = evaluator.evaluate_models_autoconfig_n_times(
        model=model,
        data_train=train,
        data=all_data,
        n_repeat=n_repeat,
        params_variations={
            "nodes": {"times": 3, "percent": 0.2, "direction": "p"},
            "batch_size": {"times": 2, "percent": 0.15, "direction": "b"},
        },
    )

evaluator.export_to_json(models_evals, "evals_1")

print("WAY 1", models_evals)
print("\n")
