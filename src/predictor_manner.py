import datetime
import pandas as pd

import logger
import data_manner
import configs_manner

exec(
    f"from models.{configs_manner.type_used.lower()} import {configs_manner.model}_manner as model_manner"
)


class PredictorConstructor:
    def __init__(self, model_id, path, repo=None, feature=None, begin=None, end=None):
        """Predictor designed to forecast values through trained models.
        Args:
            path (string): [description]
            repo (string, optional): The key number of the repository to data acquisition. Defaults to None.
            feature (string, optional): Columns names to be used as model input. Defaults to None.
            begin (string, optional): Date to start the forecasting. Defaults to None.
            end (string, optional): Date to finish the forecasting. Defaults to None.
        """
        self.model_id = model_id
        self.path = path
        self.repo = repo
        self.feature = feature
        self.begin = begin
        self.end = end
        self.raw_y_hat = None
        try:
            self.data_to_train_model = self.__data_collector(path, repo, feature, begin, end)
            self.data_X = self.data_to_train_model.x
            self.model = self.__model_assemble(model_id)
            logger.debug_log(
                self.__class__.__name__,
                self.__init__.__name__,
                "Predictor: All requirements have been met: Data and model found.",
            )
        except Exception as e:
            logger.error_log(
                self.__class__.__name__, self.__init__.__name__, f"Error: {e}."
            )

    def __get_model_obj(self, model_id):
        model = "Model" + str(configs_manner.model.upper())
        return getattr(model_manner, model)(model_id)

    def __model_assemble(self, model_id):
        model_obj = self.__get_model_obj(model_id)
        model_obj.loading(model_id)
        return model_obj

    def __data_collector(self, path, repo=None, feature=None, begin=None, end=None):
        data_constructor = data_manner.DataConstructor(is_predicting=True)
        data_collected = data_constructor.collect_dataframe(
            path, repo, feature, begin, end
        )
        self.__get_periods_from_data_constructor(data_constructor)
        return data_constructor.build_test(data_collected)

    def __get_periods_from_data_constructor(self, data_constructor):
        # TODO better way to get this variables from data_constructor
        self.begin_raw = str(data_constructor.begin_raw.date())
        self.end_raw = str(data_constructor.end_raw.date())
        self.begin_forecast = str(data_constructor.begin_forecast.date())
        self.end_forecast = str(data_constructor.end_forecast.date())
        self.number_of_days_requested = (datetime.datetime.strptime(self.end, "%Y-%m-%d") - datetime.datetime.strptime(self.begin, "%Y-%m-%d")).days + 1
        self.number_of_days_available = (datetime.datetime.strptime(self.end_raw, "%Y-%m-%d") - datetime.datetime.strptime(self.begin_raw, "%Y-%m-%d")).days + 1

    def predict(self, data_X=None):
        """This method forecast deaths values to data in the constructor object from begin to end date.
        Args:
            data_X (data.x, optional): data.x variable to predict. Defaults to None.
        Returns:
            string: A string containing the forecasting values and them respective date. 
        """
        data_X = data_X if data_X is not None else self.data_X
        try:
            y_hat = self.model.predicting(data_X)
            self.raw_y_hat = y_hat.reshape(-1)
            
            return y_hat.reshape(-1)[-self.number_of_days_requested:]
        except Exception as e:
            logger.error_log(
                self.__class__.__name__, self.__init__.__name__, f"Error: {e}."
            )
            raise

    def predictions_to_weboutput(self, y_hat):
        period = pd.date_range(self.begin, self.end)
        returned_dictionary = list()
        for date, value in zip(period, y_hat):
            returned_dictionary.append(
                {
                    "date": datetime.datetime.strftime(date, "%Y-%m-%d"),
                    "prediction": str(value),
                }
            )

        return returned_dictionary