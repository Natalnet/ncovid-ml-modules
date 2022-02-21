import io, requests, h5py
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

    def _resolve_model_name(self, is_remote=False):
        return (
            str(self.model_path_remote if is_remote else self.model_path)
            + self.locale
            + "_"
            + self.model_subtype
            + "_"
            + str(self.data_window_size)
            + "_"
            + str(self.n_features)
            + "_"
            + str(self.nodes)
            + "_"
            + str(int(self.dropout * 100))
            + ".h5"
        )

    def saving(self):
        self.model.save(self._resolve_model_name())
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
                else tf.keras.models.load_model(self._resolve_model_name())
            )
        except OSError:
            try:
                model_web_content = requests.get(self._resolve_model_name(True)).content
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
        yhat = self.model.predict(data.x, verbose=0)

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
