import numpy as np
from data_manner import Train, Test, Data
from models.model_interface import ModelInterface
import configs_manner
from statistics import mean
from math import sqrt
import logger

exec(
    f"from models.{configs_manner.model_type.lower()} import {configs_manner.model_subtype}_manner as model_manner"
)

class Evaluator:
    def __init__(
        self,
        model: "ModelInterface" or list or str = None,
        data_train: "Train" = None,
        all_data: "Data" = None,
        n_repeat: int = 1,
        save: bool = False,
    ):
        self._data_train = data_train
        self._all_data = all_data
        self._n_repeat = n_repeat
        self._model = model
        self._models = list()
        self._save = save

    @property
    def data_train(self):
        return self._data_train

    @data_train.setter
    def data_train(self, train):
        self._data_train = train

    @property
    def data_test(self):
        return self._data_test

    @data_test.setter
    def data_test(self, test):
        self._data_test = test

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, new_model):
        self._model = new_model
        self._models.append(new_model)

    @property
    def models(self):
        return self._models

    @models.setter
    def models(self, new_models):
        self._models = new_models

    def clean_models(self):
        self._models = list()

    # def evaluate_model(
    #     self, model=None, data_train: list = None, data_test: list = None
    # ):
    #     """Evaluate model over train and test

    #     Args:
    #         model (model, optional): model trained. Defaults to None.
    #         data_train (Train, optional): Data train to be trained. Defaults to None.
    #         data_test (Test, optional): Data test to be evaluated. Defaults to None.

    #     Returns:
    #         y_hats, rmses: predictions and rmses
    #     """
    #     from copy import copy

    #     if data_train is None:
    #         data_train = self.data_train
    #     if data_test is None:
    #         data_test = self.data_test
    #     if model is None:
    #         model = self._model

    #     # walk-forward validation over each week
    #     history = copy(data_train)
    #     history.y_hat = list()
    #     history.rmse = list()
    #     for idx in range(len(data_test.x) + 1):
    #         yhat = model.predicting(history)
    #         rmse = model.calculate_rmse(history.y, yhat)
    #         history.y_hat.append(yhat)
    #         history.rmse.append(rmse)

    #         # get real observation and add to history for predicting the next week
    #         history.x = np.vstack((history.x, data_test.x[idx : idx + 1 :,]))
    #         history.y = np.vstack((history.y, data_test.y[idx : idx + 1 :,]))

    #     return history
        
    def _evaluate_model_rmse(self, model, data: "Data") -> dict:
        test_period = int(configs_manner.model_infos['data_test_size_in_days']/configs_manner.model_infos['data_window_size'])
        
        yhat = model.predicting(data.x) 
        rmse = model.calculate_rmse(data.y, yhat)
                
        rmse_dict = {"rmse_total": sqrt(mean([r**2 for r in rmse])),
                "rmse_train": sqrt(mean([r**2 for r in rmse[:test_period]])), 
                "rmse_test": sqrt(mean([r**2 for r in rmse[:-test_period]]))
                }
        
        return rmse_dict
    
    def _evaluate_model_mse(self, model, data: "Data") -> dict:
        test_period = int(configs_manner.model_infos['data_test_size_in_days']/configs_manner.model_infos['data_window_size'])
        
        yhat = model.predicting(data.x) 
        mse = model.calculate_mse(data.y, yhat)
                
        mse_dict = {"mse_total": mean(mse),
                "mse_train": mean(mse[:test_period]), 
                "mse_test": mean(mse[:-test_period])
                }
        
        return mse_dict
    
    def evaluate_model(
        self, 
        model: "ModelInterface" = None, 
        all_data: "Data" = None, 
        metrics: list = None
        )-> dict:
        """Evaluate model over train and test

         Args:
             model (ModelInterface, optional): model trained. Defaults to None.
             data (Data, optional): Data to be evaluated. Defaults to None.
             metrics (list, optional): list of metrics to be returned

         Returns:
             evaluated_dict: returned metrics
         """
        from copy import copy

        if all_data is None:
            all_data = self._all_data
        if model is None:
            model = self._model

        data = copy(all_data)
        evaluated_dict = {}
        local_metrics = ['rmse', 'mse']
        if metrics is not None:
            local_metrics = metrics

        for metric in local_metrics:
            evaluated_dict[metric] = getattr(self, f"_evaluate_model_{metric}")(model, data)
            
        return evaluated_dict
            
    def evaluate_model_n_times(
        self,
        model_list: list or str = None,
        train: "Train" = None,
        all_data: "Data" = None,
        n_repeat: int = 1,
        verbose=0,
    ) -> dict:
        """
        Fit and Evaluate a single model over train and test multiple times
        :param model: Specify model to training and evaluate
        :param train: Specify train temporal series to evaluate or use the train temporal series inserted in class
        :param test: Specify test temporal series to evaluate or use the test temporal series inserted in class
        :param n_repeat:
        :param verbose: Specify training should be verbose or silent
        :return: regressor_list with predictions and rmses for unique model
        """
        local_model_list = model_list
        if (local_model_list is None) and (type(self._model) is str or list):
            local_model_list = self._model
            
        if type(local_model_list) is str:
            path = local_model_list
            model = "Model" + str(configs_manner.model_subtype.upper())
            local_model_list = [getattr(model_manner, model)(path) for _ in range(n_repeat)]   
        elif type(local_model_list) is list:
            local_model_list = local_model_list
        else:
            logger.error_log(
                    self.__class__.__name__,
                    self.evaluate_model_n_times.__name__,
                    "None list or path was given in: {}".format(self.__str__),
                )
            raise Exception("None list or path was given")
        
        from copy import copy

        if train is None:
            train = self._data_train
        if all_data is None:
            all_data = self._all_data
        if n_repeat is None:
            n_repeat = self._n_repeat
        
        train_data = copy(train)
        model_dict ={}
        
        for idx, model in enumerate(local_model_list):
            model.creating()
            model.fiting(train_data.x, train_data.y, verbose=verbose)
            model_dict["n_time_" + str(idx)] = self.evaluate_model(model, all_data)
            
        return model_dict
        
    def evaluate_n_models_n_times(
        self,
        list_models: list = None,
        train: list = None,
        test: list = None,
        n_repeat: int = 1,
        verbose=0,
    ) -> dict:
        """
        Fit and Evaluate multiple models over train and test multiple times
        :param list_models: Specify model list to training and evaluate
        :param train: Specify train temporal series to evaluate or use the train temporal series inserted in class
        :param test: Specify test temporal series to evaluate or use the test temporal series inserted in class
        :param verbose: Specify training should be verbose or silent
        :return: regressors_list with predictions and rmses for any model from list models
        """

        if list_models is None:
            list_models = self._models

        regressors_list = list()

        for model in list_models:
            regressors_list.append(
                self.evaluate_model_n_times(model, train, test, n_repeat, verbose)
            )
            self._model = model

        return regressors_list

    def __str__(self):
        if type(self._data_train) is Train and type(self._data_test) is Test:
            return (
                f"\nQuantity Models: {len(self._models)}"
                f"\nLast Model added and settled: {self._model}"
                f"\nData train: {self._data_train.x.shape}"
                f"\nData test: {self._data_test.x.shape}"
                f"\nRepetitions: {self.n_repeat}\n"
            )

        return (
            f"\nQuantity Models: {len(self._models)}"
            f"\nLast Model added and settled: {self._model}"
            f"\nData train: {self._data_train}"
            f"\nData test: {self._data_test}"
            f"\nRepetitions: {self.n_repeat}\n"
        )
