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
        """Predictor designed to forecast values through trained models.

        Args:
            path (string): [description]
            repo (string, optional): The key number of the repository to data acquisition. Defaults to None.
            feature (string, optional): Columns names to be used as model input. Defaults to None.
            begin (string, optional): Date to start the forecasting. Defaults to None.
            end (string, optional): Date to finish the forecasting. Defaults to None.
        """
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

    def __get_model_obj(self, path):
        model = "Model" + str(configs_manner.model_subtype.upper())
        return getattr(model_manner, model)(path)

    def __model_assemble(self, path):
        model_obj = self.__get_model_obj(path)
        model_obj.loading()
        return model_obj

    def __data_collector(self, path, repo=None, feature=None, begin=None, end=None):
        data_constructor = data_manner.DataConstructor(is_predicting=True)
        data_collected = data_constructor.collect_to_predict(
            path, repo, feature, begin, end
        )
        return data_constructor.build_test(data_collected)

    def predict(self, data_to_predict=None):
        """This method forecast deaths values to data in the constructor object from begin to end date.

        Args:
            data_to_predict (object, optional): Data object containing the data.x and a data.y variables. Defaults to None.

        Returns:
            string: A string containing the forecasting values and them respective date. 
        """
        data = data_to_predict if data_to_predict is not None else self.input_data
        try:
            y_hat = self.model.predicting(data)
            return self.__list_to_string(y_hat[0].reshape(-1))
        except Exception as e:
            logger.error_log(
                self.__class__.__name__, self.__init__.__name__, f"Error: {e}."
            )
            raise

    def __list_to_string(self, y_hat):
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
