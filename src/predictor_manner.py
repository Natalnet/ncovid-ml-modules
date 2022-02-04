import data_manner
import datetime
import pandas as pd
import configs_manner
exec(f'from models.{configs_manner.model_type.lower()} import {configs_manner.model_subtype}_manner as model_manner')

class PredictorConstructor():
    def __init__(self, path, repo=None, feature=None, begin=None, end=None):
        self.path = path
        self.repo = repo
        self.feature = feature
        self.begin = begin 
        self.end = end
        self.input_data = self.data_collector()
        self.model = self.model_assemble()

    def model_assemble(self):
        model = 'Model' + str(configs_manner.model_subtype.upper())
        model_obj = getattr(model_manner, model)(self.path)
        model_obj.loading()
        return model_obj

    def data_collector(self):
        self.data_obj = data_manner.DataConstructor()
        self.data_obj.is_predicting = True
        self.data_collected = self.data_obj.collect_to_predict(self.path, self.repo, self.feature, self.begin, self.end)
    
        return self.data_obj.build_test(self.data_collected)

    def predict(self):
        predicted, self.rmse = self.model.predicting(self.input_data)
        self.predicted = predicted.reshape(-1) 
        return self.convert_to_string_format()

    def convert_to_string_format(self):
        time_interval = pd.date_range(self.begin, self.end)

        returned_dictionaty = []
        for date, value in zip(time_interval, self.predicted):
            str_value = str(value)
            str_date = datetime.datetime.strftime(date, "%Y-%m-%d")
            returned_dictionaty.append({"date": str_date, "deaths": str_value})

        return str(returned_dictionaty)