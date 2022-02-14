import io, requests, h5py
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from math import sqrt
from sklearn.metrics import mean_squared_error

import logger
import configs_manner
from models.model_interface import ModelInterface


class ModelArtificalInterface(ModelInterface):
    def __init__(self, locale):
        super().__init__(locale)
        self.nodes = configs_manner.model_infos["model_nodes"]
        self.epochs = configs_manner.model_infos["model_epochs"]
        self.dropout = configs_manner.model_infos["model_dropout"]
        self.batch_size = configs_manner.model_infos["model_batch_size"]
        self.is_output_in_input = configs_manner.model_infos["model_is_output_in_input"]
        self.is_predicting = configs_manner.model_is_predicting
        self.data_window_size = configs_manner.model_infos["data_window_size"]
        self.data_test_size_in_days = configs_manner.model_infos["data_test_size_in_days"]
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

    def loading(self, model_name=None):
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

    def fiting(self, x, y, verbose=0):
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

    def predicting(self, data):
        """Make predictions (often test data)

        Args:
            data (Test): data to make predictions

        Returns:
            Test.y_hat and Test.rmse: predictions and its rmse
        """
        yhat = self.model.predict(data.x, verbose=0)
        rmse_scores = list()
        score = list()
        for idx, _ in enumerate(yhat):
            mse = mean_squared_error(data.y[idx], yhat[idx])
            rmse = sqrt(mse)
            rmse_scores.append(rmse)

        # calculate overall RMSE
        s = 0
        s_test = 0
        s_train = 0
        n_test = self.data_test_size_in_days
        n_input = self.data_window_size
        for row in range(data.y.shape[0]):
            for col in range(data.y.shape[1]):
                # geral (treino + test)
                s += (data.y[row, col] - yhat[row, col])**2
                # só teste
                if row > data.y.shape[0]-(n_test/n_input):
                    s_test += (data.y[row, col] - yhat[row, col])**2
                # só treino
                if row < data.y.shape[0]-(n_test/n_input):
                    s_train += (data.y[row, col] - yhat[row, col])**2
        # rmse geral (treino + teste)
        score = sqrt(s / data.y.shape[1])
        # rmse para dados de teste (test)
        score_test = sqrt(s_test / data.y.shape[1])
        # rmse para dados de treino (train)
        score_train = sqrt(s_train / data.y.shape[1])

        logger.debug_log(
            self.__class__.__name__, self.predicting.__name__, "Model Predicted"
        )

        return yhat, rmse_scores, score, score_test, score_train
