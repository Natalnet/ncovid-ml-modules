import sys

import numpy as np
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.layers import LSTM, RepeatVector, TimeDistributed
from tensorflow.keras.models import Model

sys.path.append('../')
from models.model_interface import ModelInterface


class ModelLSTM(ModelInterface):

    def __init__(self, n_inputs=7, n_nodes=128, n_features=1, dropout=0.0):
        super().__init__(n_inputs, n_nodes, n_features, dropout)
        self.timesteps = self.n_inputs
        self.model = self.__model_architecture()

    def __model_architecture(self):
        input_model = Input(shape=(self.timesteps, self.n_features))
        lstm_1 = LSTM(self.n_nodes, activation='relu')(input_model)
        repeat_vect = RepeatVector(self.n_outputs)(lstm_1)
        lstm_2 = LSTM(self.n_nodes, activation='relu', return_sequences=True)(repeat_vect)
        if self.dropout != 0.0 and self.dropout is not None:
            lstm_2 = Dropout(self.dropout)(lstm_2)
        time_dist_layer = TimeDistributed(Dense(np.floor(self.n_nodes / 2), activation='relu'))(lstm_2)
        output_model = TimeDistributed(Dense(1))(time_dist_layer)
        model = Model(inputs=input_model, outputs=output_model)
        model.compile(loss='mse', optimizer='adam')
        return model

    def __str__(self):
        return f'\nLSTM Model' \
               f'\n\tinput, output and timesteps: {self.n_inputs}' \
               f'\n\tlstm nodes: {self.n_nodes}' \
               f'\n\tfeatures: {self.n_features}' \
               f'\n\tdropout: {self.dropout}'
