import numpy as np
import os
from models.model_ai_interface import ModelArtificalInterface
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.layers import LSTM, RepeatVector, TimeDistributed
from tensorflow.keras.models import Model


class ModelLSTM(ModelArtificalInterface):
    def __init__(self, locale, model=None):
        """[summary]

        Args:
            locale (str): Location name that is submitted the model (country, state, country)
        """
        super().__init__(locale)
        self.model_name = self._resolve_model_name()

        if model is not None:
            self.model = model
        elif os.path.isfile(self.model_name):
            self.model = self.loading(self.model_name)
        else:
            self.model = self.__model_architecture()

    def __model_architecture(self):
        input_model = Input(shape=(self.data_window_size, self.n_features))
        lstm_1 = LSTM(self.nodes, activation="relu")(input_model)
        repeat_vect = RepeatVector(self.data_window_size)(lstm_1)
        lstm_2 = LSTM(self.nodes, activation="relu", return_sequences=True)(repeat_vect)
        if self.dropout != 0.0 and self.dropout is not None:
            lstm_2 = Dropout(self.dropout)(lstm_2)
        time_dist_layer = TimeDistributed(
            Dense(np.floor(self.nodes / 2), activation="relu")
        )(lstm_2)
        output_model = TimeDistributed(Dense(1))(time_dist_layer)
        model = Model(inputs=input_model, outputs=output_model)
        model.compile(loss="mse", optimizer="adam")
        return model

    def __str__(self):
        return (
            f"\nLSTM Model"
            f"\n\tinput, output and timesteps: {self.data_window_size}"
            f"\n\tlstm nodes: {self.nodes}"
            f"\n\tfeatures: { self.n_features }"
            f"\n\tdropout: {self.dropout}"
        )

