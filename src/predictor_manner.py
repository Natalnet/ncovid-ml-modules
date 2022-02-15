import datetime
import pandas as pd

import logger
import data_manner
import configs_manner

exec(
    f"from models.{configs_manner.model_type.lower()} import {configs_manner.model_subtype}_manner as model_manner"
)


class PredictorConstructor:
    def __init__(self, path, repo=None, feature=None, begin=None, end=None):
        self.path = path
        self.repo = repo
        self.feature = feature
        self.begin = begin
        self.end = end
        try:
            self.input_data = self.__data_collector(path, repo, feature, begin, end)
            self.model = self.__model_assemble(path)
            logger.debug_log(
                self.__class__.__name__,
                self.__init__.__name__,
                "Predictor: All requirements have been met: Data and model found.",
            )
        except Exception as e:
            logger.error_log(
                self.__class__.__name__, self.__init__.__name__, f"Error: {e}."
            )

    def __model_assemble(self, path):
        model = "Model" + str(configs_manner.model_subtype.upper())
        model_obj = getattr(model_manner, model)(path)
        model_obj.loading()
        return model_obj

    def __data_collector(self, path, repo=None, feature=None, begin=None, end=None):
        data_constructor = data_manner.DataConstructor(is_predicting=True)
        data_collected = data_constructor.collect_dataframe(
            path, repo, feature, begin, end
        )
        return data_constructor.build_test(data_collected)

    def predict(self, data_to_predict=None):
        data = data_to_predict if data_to_predict is not None else self.input_data
        try:
            y_hat = self.model.predicting(data)
            return y_hat.reshape(-1)
        except Exception as e:
            logger.error_log(
                self.__class__.__name__, self.__init__.__name__, f"Error: {e}."
            )
            raise

    def predictions_to_weboutput(self, y_hat):
        period = pd.date_range(self.begin, self.end)
        returned_dictionaty = list()
        for date, value in zip(period, y_hat):
            returned_dictionaty.append(
                {
                    "date": datetime.datetime.strftime(date, "%Y-%m-%d"),
                    "prediction": str(value),
                }
            )
        return str(returned_dictionaty)
