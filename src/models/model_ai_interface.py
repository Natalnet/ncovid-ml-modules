import io, requests, h5py, uuid, json
from typing import Tuple

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from math import sqrt
from sklearn.metrics import mean_squared_error

import logger
import configs_manner
from data_manner import Test
from models.model_interface import ModelInterface


class ModelArtificalInterface(ModelInterface):
    def __init__(self, locale: str):
        super().__init__(locale)
        self.nodes = configs_manner.model_infos["model_nodes"]
        self.epochs = configs_manner.model_infos["model_epochs"]
        self.dropout = configs_manner.model_infos["model_dropout"]
        self.batch_size = configs_manner.model_infos["model_batch_size"]
        self.is_output_in_input = configs_manner.model_infos["model_is_output_in_input"]
        self.is_predicting = configs_manner.model_is_predicting
        self.data_window_size = configs_manner.model_infos["data_window_size"]
        self.earlystop = EarlyStopping(
            monitor="loss",
            mode="min",
            verbose=0,
            patience=configs_manner.model_infos["model_earlystop"],
        )
        self.n_features = configs_manner.model_infos["data_n_features"]

    def _resolve_model_name(self, model_id, is_remote=False):
        return (
            str(self.model_path_remote if is_remote else self.model_path)
            + model_id
            + ".h5"
        )

    def __model_id_generate(self):
        return str(uuid.uuid1())

    def __saving_metadata_file(self, model_id):
        with open(configs_manner.doc_folder + "configure.json") as json_file:
            data = json.load(json_file)
            metadata = {}
            metadata["folder_configs"] = {
                "model_remot_path": configs_manner.model_path_remote
            }
            metadata["model_configs"] = {
                "model_id": model_id,
                "type_used": configs_manner.model_type,
                "is_predicting": configs_manner.model_is_predicting,
                configs_manner.model_type: data["model_configs"][
                    configs_manner.model_type
                ],
            }

            with open(configs_manner.doc_folder + "metadata.json", "w") as json_to_save:
                json.dump(metadata, json_to_save, indent=4)

    def saving(self):
        model_id_to_save = self.__model_id_generate()
        self.model.save(self._resolve_model_name(model_id_to_save))
        self.__saving_metadata_file(model_id_to_save)
        logger.debug_log(self.__class__.__name__, self.saving.__name__, "Model Saved")

    def loading(self, model_name: str = None):
        """Load model locally and remotely. For remote option, is necessary to fill `configure.json/model_path_remote`.
        Args:
            model_name (str, optional): The known `path + model` name. Defaults to None.
        Raises:
            ose: Exception OSError if model not found locally or remotely
        """
        try:
            self.model = (
                tf.keras.models.load_model(model_name)
                if model_name
                else tf.keras.models.load_model(
                    self._resolve_model_name(configs_manner.model_infos["model_id"])
                )
            )
        except OSError:
            try:
                model_web_content = requests.get(
                    self._resolve_model_name(
                        configs_manner.model_infos["model_id"], True
                    )
                ).content
                model_bin = io.BytesIO(model_web_content)
                model_obj = h5py.File(model_bin, "r")
                self.model = tf.keras.models.load_model(model_obj)

            except OSError as ose:
                logger.error_log(
                    self.__class__.__name__,
                    self.loading.__name__,
                    "Model not found - {}".format(ose.__str__),
                )
                raise ose("Model not found")
        else:
            logger.debug_log(
                self.__class__.__name__, self.loading.__name__, "Model loaded"
            )

    def fiting(self, x: list, y: list, verbose: int = 0) -> bool:
        """Fit model based on Train data

        Args:
            x (Train.x): Data used as input to the model
            y (Train.y): Data used as outcome to the model
            verbose (int, optional): Lied to observe the fit model in run time. Defaults to 0.

        Returns:
            bool: True if everything finish well 
        """
        try:
            self.model.fit(
                x=x,
                y=y,
                epochs=self.epochs,
                batch_size=self.batch_size,
                callbacks=[self.earlystop],
                verbose=verbose,
            )
            logger.debug_log(
                self.__class__.__name__, self.fiting.__name__, "Model fitted"
            )
            return True
        except Exception as e:
            logger.error_log(
                self.__class__.__name__, self.fiting.__name__, f"Error: {e}."
            )
            raise

    def predicting(self, data: Test) -> list:
        """Make predictions (often test data)

        Args:
            data (Test): data to make predictions

        Returns:
            Test.y_hat and Test.rmse: predictions and its rmse
        """
        yhat = self.model.predict(data, verbose=0)

        logger.debug_log(
            self.__class__.__name__, self.predicting.__name__, "Data predicted"
        )

        return yhat

    def calculate_rmse(
        self, y_orig: list or Tuple[list, list], y_hat: list or Tuple[list, list]
    ) -> list or Tuple[list, list]:
        if len(y_orig) != len(y_hat):
            logger.error_log(
                self.__class__.__name__,
                self.calculate_rmse.__name__,
                "The list must to have same size",
            )
            return None

        rmse = [
            sqrt(mean_squared_error(y_hat[idx], y_orig[idx]))
            for idx, _ in enumerate(y_orig)
        ]

        logger.debug_log(
            self.__class__.__name__, self.predicting.__name__, "RMSE calculated"
        )

        return rmse

    def calculate_mse(
        self, y_orig: list or Tuple[list, list], y_hat: list or Tuple[list, list]
    ) -> list or Tuple[list, list]:
        if len(y_orig) != len(y_hat):
            logger.error_log(
                self.__class__.__name__,
                self.calculate_rmse.__name__,
                "The list must to have same size",
            )
            return None

        mse = [
            mean_squared_error(y_hat[idx], y_orig[idx]) for idx, _ in enumerate(y_orig)
        ]

        logger.debug_log(
            self.__class__.__name__, self.predicting.__name__, "MSE calculated"
        )
        return mse

    def calculate_mae():
        pass

    def calculate_R2():
        pass

    def calculate_rse():
        pass

    def calculate_nrsme():
        pass

    def param_value(self, param_name: str):
        return configs_manner.model_infos["model_" + param_name]

    def _extract_param_value(self, param_name: str):
        if "model_" + param_name in configs_manner.model_infos:
            return configs_manner.model_infos["model_" + param_name]
        return None

    def _params_self_modify(
        self,
        params_variations: dict = {
            "nodes": {"times": 2, "percent": 0.1, "direction": "b"},
            "epochs": {"times": 2, "percent": 0.1, "direction": "b"},
        },
    ) -> list:
        import itertools as it

        def model_param_variation_generator(
            param_name: str,
            n_variations=2,
            percent_variation=0.1,
            direction_variation="bilateral",
        ) -> list:
            import math

            param_value = self.param_value(param_name)

            def pos_grow(param_value, percent_variation, n_variations) -> list:
                grow_value = math.ceil(param_value * percent_variation)
                variation_param_values = [param_value]
                for _ in range(n_variations):
                    param_value = param_value + grow_value
                    variation_param_values.append(param_value)
                    grow_value = math.ceil(param_value * percent_variation)
                return variation_param_values

            def neg_grow(param_value, percent_variation, n_variations) -> list:
                grow_value = math.ceil(param_value * percent_variation)
                variation_param_values = [param_value]
                for _ in range(n_variations):
                    param_value = param_value - grow_value
                    if param_value <= 0:
                        if type(param_value) is int:
                            param_value = 1
                        if type(param_value) is float:
                            param_value = 0.01
                    variation_param_values.append(param_value)
                    grow_value = math.ceil(param_value * percent_variation)
                variation_param_values.reverse()
                return variation_param_values

            def bilateral_grow(param_value, percent_variation, n_variations) -> list:
                positve_grow_times = math.ceil(n_variations / 2)
                negative_grow_times = n_variations - positve_grow_times
                positive_grow_variation = pos_grow(
                    param_value, percent_variation, positve_grow_times
                )
                negative_grow_variation = neg_grow(
                    param_value, percent_variation, negative_grow_times
                )
                bilateral_grow_variation = (
                    negative_grow_variation[:-1] + positive_grow_variation
                )
                return bilateral_grow_variation

            options = {
                "b" or "bilateral": bilateral_grow,
                "p" or "positive": pos_grow,
                "n" or "negative": neg_grow,
            }
            return options[direction_variation](
                param_value, percent_variation, n_variations
            )

        new_configs = dict()
        for param in params_variations:
            new_configs[param] = model_param_variation_generator(
                param,
                n_variations=params_variations[param].get("times"),
                percent_variation=params_variations[param].get("percent"),
                direction_variation=params_variations[param].get("direction"),
            )

        return new_configs
