import numpy as np
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.layers import LSTM, RepeatVector, TimeDistributed
from tensorflow.keras.models import Model

import configuration as pipeline_configs


class LSTMRegressor:

    def __init__(self, n_input, n_lstm_nodes, dropout, n_timesteps, n_features, n_outputs):
        self.n_input = n_input
        self.n_lstm_nodes = n_lstm_nodes
        self.dropout = dropout
        self.n_timesteps = n_timesteps
        self.n_features = n_features
        self.n_outputs = n_outputs
        self.model = self.__model_architecture()
        self.stop_training = EarlyStopping(monitor='loss', mode='min', verbose=0, patience=50)

    def __model_architecture(self):
        inputs = Input(shape=(self.n_timesteps, self.n_features))
        lstm_1 = LSTM(self.n_lstm_nodes, activation='relu')(inputs)
        repeat_vect = RepeatVector(self.n_outputs)(lstm_1)
        lstm_2 = LSTM(self.n_lstm_nodes, activation='relu', return_sequences=True)(repeat_vect)
        if self.dropout != 0.0 and self.dropout is not None:
            lstm_2 = Dropout(self.dropout)(lstm_2)
        time_dist_layer = TimeDistributed(Dense(np.floor(self.n_lstm_nodes / 2), activation='relu'))(lstm_2)
        outputs = TimeDistributed(Dense(1))(time_dist_layer)
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(loss='mse', optimizer='adam')
        return model

    def fit_model(self, data, verbose=0):
        self.model.fit(x=data.x,
                       y=data.y,
                       epochs=pipeline_configs.model_train_epochs,
                       batch_size=pipeline_configs.model_batch_size,
                       verbose=verbose,
                       callbacks=[self.stop_training])


def build_model(model_config):
    if model_config:
        n_input, n_lstm_nodes, dropout, n_features = model_config
        n_timesteps = n_outputs = n_input
        return LSTMRegressor(n_input, n_lstm_nodes, dropout, n_timesteps, n_features, n_outputs)
    return None
