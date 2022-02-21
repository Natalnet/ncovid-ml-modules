import numpy as np
import logger
from models.model_ai_interface import ModelArtificalInterface
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.layers import LSTM, RepeatVector, TimeDistributed
from tensorflow.keras.models import Model


class ModelLSTM(ModelArtificalInterface):
    def __init__(
        self,
        locale: str,
        model: Model = None,
        data_window_size: int = None,
        nodes: int = None,
        dropout: float = None,
    ):
        """Manager model LSTM.

        Args:
            locale (str): Local to predict(eg. `brl:rn` or `brl`)
            model (Keras, optional): Pre-existed keras model. Defaults to None.
            data_window_size (int, optional): Window size applied to the model. Defaults to None.
            nodes (int, optional): Number of nodes in model. Defaults to None.
            dropout (float, optional): Percent dropout applied in model. Defaults to None.
        """

        super().__init__(locale)

        self.model = model if model else None

        if data_window_size or nodes or dropout:
            if data_window_size:
                self.data_window_size = data_window_size
            if nodes:
                self.nodes = nodes
            if dropout:
                self.dropout = dropout
            # self.model = self.__model_architecture()

    def _model_architecture(self):
        try:
            input_model = Input(shape=(self.data_window_size, self.n_features))
            lstm_1 = LSTM(self.nodes, activation="relu")(input_model)
            repeat_vect = RepeatVector(self.data_window_size)(lstm_1)
            lstm_2 = LSTM(self.nodes, activation="relu", return_sequences=True)(
                repeat_vect
            )
            if self.dropout != 0.0 and self.dropout is not None:
                lstm_2 = Dropout(self.dropout)(lstm_2)
            time_dist_layer = TimeDistributed(
                Dense(np.floor(self.nodes / 2), activation="relu")
            )(lstm_2)
            output_model = TimeDistributed(Dense(1))(time_dist_layer)
            model = Model(inputs=input_model, outputs=output_model)
            model.compile(loss="mse", optimizer="adam")
            logger.debug_log(
                self.__class__.__name__,
                self._model_architecture.__name__,
                "Model created",
            )
            return model
        except Exception:
            logger.error_log(
                self.__class__.__name__,
                self._model_architecture.__name__,
                "Failling in create model",
            )

    def __str__(self):
        return (
            f"\nLSTM Model"
            f"\n\tinput, output and timesteps: {self.data_window_size}"
            f"\n\tlstm nodes: {self.nodes}"
            f"\n\tfeatures: { self.n_features }"
            f"\n\tdropout: {self.dropout}"
        )

