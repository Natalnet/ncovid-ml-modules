import numpy as np
from data_manner import Train, Test, Data
from models.model_interface import ModelInterface
import configs_manner
from statistics import mean
from math import sqrt, ceil
import logger
import json
from datetime import date
import itertools as it

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
    ):
        self._data_train = data_train
        self._all_data = all_data
        self._n_repeat = n_repeat
        self._model = model
        self._models = list()

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
        
    def _evaluate_model_rmse(
        self, 
        model: "ModelInterface" = None, 
        data: "Data" = None,
    ) -> dict:
        
        if data is None and self._all_data is None:
            logger.error_log(
                    self.__class__.__name__,
                    self.evaluate_model_n_times.__name__,
                    "None data was given in: {}".format(self.__str__),
                )
            raise Exception("None data was given")
        local_data = self._all_data
        if data is not None:
            local_data = data
            
        if model is None and self._model is None:
            logger.error_log(
                    self.__class__.__name__,
                    self.evaluate_model_n_times.__name__,
                    "None model was given in: {}".format(self.__str__),
                )
            raise Exception("None model was given")
        local_model = self._model
        if model is not None:
            local_model = model
            
        test_period = int(configs_manner.model_infos['data_test_size_in_days']/configs_manner.model_infos['data_window_size'])
        
        yhat = local_model.predicting(local_data.x) 
        rmse = local_model.calculate_rmse(local_data.y, yhat)
                
        rmse_dict = {"rmse_total": sqrt(mean([r**2 for r in rmse])),
                "rmse_train": sqrt(mean([r**2 for r in rmse[:test_period]])), 
                "rmse_test": sqrt(mean([r**2 for r in rmse[:-test_period]]))
                }
        
        return rmse_dict
    
    def _evaluate_model_mse(
        self, 
        model: "ModelInterface", 
        data: "Data"
    ) -> dict:
        
        if data is None and self._all_data is None:
            logger.error_log(
                    self.__class__.__name__,
                    self.evaluate_model_n_times.__name__,
                    "None data was given in: {}".format(self.__str__),
                )
            raise Exception("None data was given")
        local_data = self._all_data
        if data is not None:
            local_data = data
            
        if model is None and self._model is None:
            logger.error_log(
                    self.__class__.__name__,
                    self.evaluate_model_n_times.__name__,
                    "None model was given in: {}".format(self.__str__),
                )
            raise Exception("None model was given")
        local_model = self._model
        if model is not None:
            local_model = model
            
        test_period = int(configs_manner.model_infos['data_test_size_in_days']/configs_manner.model_infos['data_window_size'])
        
        yhat = local_model.predicting(local_data.x) 
        mse = local_model.calculate_mse(local_data.y, yhat)
                
        mse_dict = {"mse_total": mean(mse),
                "mse_train": mean(mse[:test_period]), 
                "mse_test": mean(mse[:-test_period])
                }
        
        return mse_dict
    
    def evaluate_model(
        self, 
        model: "ModelInterface" = None, 
        all_data: "Data" = None, 
        metrics: list = None,
        save: bool = False,
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
        
        if save:
            self.save_as_json(evaluated_dict)
            
        return evaluated_dict
            
    def evaluate_model_n_times(
        self,
        model_list: list or str = None,
        train: "Train" = None,
        all_data: "Data" = None,
        n_repeat: int = 1,
        verbose=0,
        save: bool = False,
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
        
        if train is None:
            train = self._data_train
        if all_data is None:
            all_data = self._all_data
        if n_repeat is None:
            n_repeat = self._n_repeat
        
           
        local_model_list = model_list
        
        if (local_model_list is None) and (type(self._model) is str or list):
            local_model_list = self._model
            
        if type(local_model_list) is str:
            local_model_list = self.__n_times_generate(local_model_list, n_repeat=n_repeat)

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

        train_data = copy(train)

        reuturned_params_list = ["nodes", "epochs", "dropout", "batch_size"]
        params_dict = { param: self.__get_model_param_value(param) for param in reuturned_params_list}
        
        model_dict = {}
        model_dict["params"] = params_dict
        
        each_model_evaluate = []
        for idx, model in enumerate(local_model_list):
            model.creating()
            model.fiting(train_data.x, train_data.y, verbose=verbose)
            each_model_evaluate.append({"model": idx, "metrics" : self.evaluate_model(model, all_data)})
            # model_dict["metrics_model_" + str(idx) ] =  self.evaluate_model(model, all_data)
        
        model_dict["models"] = each_model_evaluate
        
        if save:
            self.save_as_json(model_dict)
            
        return model_dict
        
    def evaluate_n_models_n_times(
        self,
        list_models,
        train: list = None,
        test: list = None,
        n_repeat: int = 1,
        verbose=0,
        save: bool = False,
    ) -> dict:
        """
        Fit and Evaluate multiple models over train and test multiple times
        :param list_models: Specify model list to training and evaluate
        :param train: Specify train temporal series to evaluate or use the train temporal series inserted in class
        :param test: Specify test temporal series to evaluate or use the test temporal series inserted in class
        :param verbose: Specify training should be verbose or silent
        :return: regressors_list with predictions and rmses for any model from list models
        """
        if type(list_models) is dict:
            params_list_varitaion = list_models
    
            param_names, combinations = self.__all_models_combination_generate(params_list_varitaion=params_list_varitaion)
            
            evaluated_models = {}
            for idx, combination in enumerate(combinations):
                self.__set_model_params_by_combination(param_names, combination)
                evaluated_models[str(idx)] = self.evaluate_model_n_times()
        else:
            # implement if it's given a model list (thing about it)
            pass
                
        if save:
            self.save_as_json(evaluated_models)
        
        return evaluated_models
        
        
    def __n_times_generate(
        self,
        path_model: str = None,
        n_repeat: int = 1,
    ) -> list:
        
        path = path_model
        if path is None:
            logger.error_log(
                    self.__class__.__name__,
                    self.evaluate_model_n_times.__name__,
                    "None path was given in: {}".format(self.__str__),
                )
            raise Exception("None path was given")
        
        if n_repeat == 1:
            n_repeat = self._n_repeat
    
        model = "Model" + str(configs_manner.model_subtype.upper())
        return [getattr(model_manner, model)(path) for _ in range(n_repeat)]   
        
        
    def __all_models_combination_generate(
        self,
        params_list_varitaion: dict = {"nodes": {"times": 2}, "epochs": {"times": 2}},
    ) ->list:
        
        param_variation = {}
        
        for param in params_list_varitaion: 
            if type(params_list_varitaion[param]) is dict: 
                param_variation[param] = self.__get_model_param_variation(
                    param, 
                    percent_variation=params_list_varitaion[param].get("percent_variation")  , 
                    times= params_list_varitaion[param].get("times") , 
                    direction=params_list_varitaion[param].get("direction")
                )
            else:
                param_variation[param] = params_list_varitaion[param]
        
        all_params = sorted(param_variation)
        all_params.reverse()
        
        return all_params, list(it.product(*(param_variation[param] for param in all_params))) 
    
    def __get_model_param_value(
        self,
        param_name: str
    ):
        return configs_manner.model_infos['model_' + param_name]
    
    def __set_model_params_by_combination(
        self,
        params_list: list,
        combination: list
    ):
        for param_name, value in zip(params_list, combination):
            configs_manner.model_infos['model_' + param_name] = value
                            
    def __get_model_param_variation(
        self,
        param_name: str,
        **kwargs,
    ) -> list:
        
        if kwargs['direction']:direction = kwargs['direction'] 
        else:direction = "bilateral"
        
        if kwargs['percent_variation']:percent_variation = kwargs['percent_variation'] 
        else:percent_variation = 0.1
        
        if kwargs['times']: times = kwargs['times']
        else:times = 2
        
        param_value = self.__get_model_param_value(param_name)
        
        if direction == 'bilateral' or direction == 'b':
            return self.__bilateral_grow(param_value, percent_variation, times)
        if direction == 'positive' or direction == 'p':
            return self.__positive_grow(param_value, percent_variation, times)
        if direction == 'negative' or direction == 'n': 
            return self.__negative_grow(param_value, percent_variation, times)
        
        
    def __bilateral_grow(
        self,
        param_value: int,
        percent_variation,
        times: int,
    ) -> list:
        
        positve_grow_times = ceil(times/2)
        negative_grow_times = times - positve_grow_times
        
        positive_grow_variation = self.__positive_grow(param_value, percent_variation, positve_grow_times)
        negative_grow_variation = self.__negative_grow(param_value, percent_variation, negative_grow_times)
        
        bilateral_grow_variation = negative_grow_variation[:-1] + positive_grow_variation
        
        return bilateral_grow_variation
    
    
    def __positive_grow(
        self,
        param_value: int,
        percent_variation: float,
        times: int,
    ) -> list:
        
        grow_value = ceil(param_value*percent_variation)
        variation_param_values = [param_value]
        for t in range(times):
            param_value = param_value + grow_value
            variation_param_values.append(param_value) 
            grow_value = ceil(param_value*percent_variation)
        
        return variation_param_values
    
    def __negative_grow(
        self,
        param_value: int,
        percent_variation: float,
        times: int,
    ) -> list:
        
        grow_value = ceil(param_value*percent_variation)
        variation_param_values = [param_value]
        for t in range(times):
            param_value = param_value - grow_value
            if param_value <= 0 and type(param_value) == int: param_value = 1
            if param_value <= 0 and type(param_value) == float: param_value = 0.01
            variation_param_values.append(param_value) 
            grow_value = ceil(param_value*percent_variation)
        variation_param_values.reverse()
        
        return variation_param_values
    
    def save_as_json(
        self,
        dictionary_to_save: dict,
        file_name: str = None,
    ):
        if file_name is None:
            file_name = date.today()
            
        with open('evaluated/' + str(file_name) + 'evalation.json', 'w') as fp:
            json.dump(dictionary_to_save, fp,  indent=4)
        
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