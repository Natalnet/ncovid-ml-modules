import pandas as pd
import matplotlib.pyplot as plt
import json
import numpy as np

import data_manner
import evaluator_manner
# I thing this import will be not necessary in future
import model_manner

def predict_for_rquest(responsed_data):
    
    data_copy = responsed_data.copy()
    data_copy.deaths = responsed_data.deaths.diff(7).dropna()
    data_copy.newCases = responsed_data.newCases.diff(7).dropna()
    data_copy = data_copy.dropna()
 
    data = [data_copy.deaths.values, data_copy.newCases.values]

    week_size = 7
    # this method just split the data by the week_size
    data_to_predict = data_manner.build_data_prediction(data[1], week_size)

    #Train data
    train, _ = data_manner.build_data(data, step_size=week_size, size_data_test=7)

    # passing the data in load_model just for an example, this method do not have argouments and
    # returns a model already trained
    model_loaded = load_model(train)

    prediction = model_loaded.model.predict(data_to_predict, verbose=0)

    return prediction.reshape(-1)

def load_model(train):
    # Here I will train a model just for example
    # model_config: [n_input, n_lstm_nodes, dropout, n_features]
    week_size = 7
    preseted_model_config = [week_size, 200, 0.0, train.x.shape[2]]

    regressor = model_manner.build_model(preseted_model_config)

    regressor.fit_model(train, epochs=20, batch_size=16, verbose=0)

    return regressor

def convert_output_to_json(output_of_prediction, rqt_data):
    returned_dictionaty = []
    for date, value in zip(rqt_data.index[-len(output_of_prediction):], output_of_prediction):
        returned_dictionaty.append({"date": date, "deaths": str(value)})

    returned_json = json.dumps(str(returned_dictionaty), indent=3, separators=(",", ":"))
    return returned_json

def predict(repo, path, feature, begin, end):
    requested_data = pd.read_csv(f"http://ncovid.natalnet.br/datamanager/repo/{repo}/path/{path}/feature/{feature}/begin/{begin}/end/{end}/as-csv", index_col='date')
    
    predicted_values = predict_for_rquest(requested_data)

    predictied_json = convert_output_to_json(predicted_values, requested_data)

    return predictied_json

# como deve ser o retorno JSON
# [
#     {
#         "date": "2020-03-13",
#         "deaths": "0"
#     },
#     {
#         "date": "2020-03-14",
#         "deaths": "3"
#     }
# ]