import logger, configs_manner
from data_manner import Train, Test, Data
from models.model_interface import ModelInterface

import statistics, math
import json, typing, copy
from datetime import date


exec(
    f"from models.{configs_manner.model_type.lower()} import {configs_manner.model_subtype}_manner as model_manner"
)


class Evaluator:
    def __init__(
        self,
        model: "ModelInterface" = None,
        models: typing.List["ModelInterface"] = None,
        metrics: typing.List[str] = ["mse", "rmse"],
    ):
        self.__model = model
        if models is None:
            self.__models = list()
        else:
            self.__models = models
        self.__metrics = metrics

    @property
    def model(self):
        return self.__model

    @model.setter
    def model(self, model: "ModelInterface"):
        self.__model = model
        self.__models.append(model)

    @property
    def models(self):
        return self.__models

    @models.setter
    def models(self, models: typing.List["ModelInterface"]):
        self.__models = models

    @property
    def metrics(self):
        return self.__metrics

    @metrics.setter
    def metrics(self, metrics: typing.List[str]):
        self.__metrics = metrics

    def clean_models(self):
        self._models = list()

    def evaluate_model(self, data: "Data", model: "ModelInterface" = None) -> dict:
        model = copy.copy(self.__model) if model is None else model
        assert model is not None, logger.error_log(
            self.__class__.__name__,
            self.evaluate_model_n_times.__name__,
            "Empty model",
        )

        eval_result = dict()
        for metric in self.__metrics:
            eval_result[metric] = getattr(Evaluator, f"_extracting_{metric}")(
                model=model, data=data
            )

        return eval_result

    def evaluate_model_n_times(
        self,
        data: "Data",
        data_train: "Train",
        model: "ModelInterface" = None,
        n_repeat: int = 2,
        verbose=0,
    ) -> dict:
        model = copy.copy(self.__model) if model is None else model
        assert model is not None, logger.error_log(
            self.__class__.__name__,
            self.evaluate_model_n_times.__name__,
            "Empty model",
        )

        params = ["nodes", "epochs", "dropout", "batch_size"]
        model_params = {param: model._extract_param_value(param) for param in params}

        model_evals = {}
        model_evals["params"] = model_params

        evaluations = []
        for idx in range(n_repeat):
            model.creating()
            model.fiting(data_train.x, data_train.y, verbose=verbose)
            evaluations.append(
                {"model": idx, "metrics": self.evaluate_model(model=model, data=data)}
            )

        model_evals["eval"] = evaluations

        return model_evals

    def evaluate_n_models_n_times(
        self,
        data: "Data",
        data_train: "Train",
        models: typing.List["ModelInterface"] = None,
        n_repeat: int = 2,
        verbose=0,
    ) -> dict:

        models = copy.copy(self.__models) if models is None else models
        assert models is not None, logger.error_log(
            self.__class__.__name__,
            self.evaluate_n_models_n_times.__name__,
            "Empty model",
        )

        models_evals = {}
        for idx, model in enumerate(models):
            models_evals[str(idx)] = self.evaluate_model_n_times(
                model=model,
                data=data,
                data_train=data_train,
                n_repeat=n_repeat,
                verbose=verbose,
            )

        return models_evals

    def evaluate_models_autoconfig_n_times(
        self,
        data: "Data",
        data_train: "Train",
        model: "ModelInterface" or typing.List["ModelInterface"],
        params_variations: dict = {
            "nodes": {"times": 2, "percent": 5, "direction": "b"},
            "epochs": {"times": 2, "percent": 5, "direction": "b"},
        },
        n_repeat: int = 2,
        verbose=0,
    ) -> dict:

        model_configs = model._params_self_modify(params_variations=params_variations)

        evals = {}
        for idx, param in enumerate(model_configs):
            for idx_value, new_config in enumerate(model_configs[param]):
                configs_manner.model_infos["model_" + param] = new_config
                model.creating()
                evals[str(idx) + "." + str(idx_value)] = self.evaluate_model_n_times(
                    model=model,
                    data_train=data_train,
                    data=data,
                    n_repeat=n_repeat,
                    verbose=verbose,
                )

        return evals

    def _extracting_mse(model: "ModelInterface", data: "Data") -> dict:
        test_period = (
            configs_manner.model_infos["data_test_size_in_days"]
            // configs_manner.model_infos["data_window_size"]
        )

        yhat = model.predicting(data.x)
        mse = model.calculate_mse(data.y, yhat)

        mse_dict = {
            "mse_total": statistics.mean(mse),
            "mse_train": statistics.mean(mse[:test_period]),
            "mse_test": statistics.mean(mse[:-test_period]),
        }

        return mse_dict

    def _extracting_rmse(model: "ModelInterface", data: "Data") -> dict:
        test_period = (
            configs_manner.model_infos["data_test_size_in_days"]
            // configs_manner.model_infos["data_window_size"]
        )

        yhat = model.predicting(data.x)
        rmse = model.calculate_rmse(data.y, yhat)

        rmse_dict = {
            "rmse_total": math.sqrt(statistics.mean([r ** 2 for r in rmse])),
            "rmse_train": math.sqrt(
                statistics.mean([r ** 2 for r in rmse[:test_period]])
            ),
            "rmse_test": math.sqrt(
                statistics.mean([r ** 2 for r in rmse[:-test_period]])
            ),
        }

        return rmse_dict

    def export_to_json(
        self, dictionary_to_save: dict, file_name: str = None,
    ):
        if file_name is None:
            file_name = date.today()

        with open("evaluated/" + str(file_name) + "_evaluation.json", "w") as fp:
            json.dump(dictionary_to_save, fp, indent=4)

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
