import numpy as np
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.layers import LSTM, RepeatVector, TimeDistributed
from tensorflow.keras.models import Model

import configs_manner


class LSTMRegressor:

    def __init__(self, n_input, n_lstm_nodes, dropout, n_timesteps, n_features, n_outputs):
        self.n_input = n_input
        self.n_lstm_nodes = n_lstm_nodes
        self.dropout = dropout
        self.n_timesteps = n_timesteps
        self.n_features = n_features
        self.n_outputs = n_outputs
        self.model = self.__model_architecture()
        self.stop_training = EarlyStopping(monitor='loss', mode='min', verbose=0,
                                           patience=configs_manner.model_patience_earlystop)

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

    def __fitting(self, x, y, epochs, batch_size, verbose):
        self.model.fit(x=x,
                       y=y,
                       epochs=epochs,
                       batch_size=batch_size,
                       verbose=verbose,
                       callbacks=[self.stop_training])

    def fit_model(self, data, epochs=None, batch_size=None, verbose=0):
        if epochs and batch_size:
            self.__fitting(data.x, data.y, epochs, batch_size, verbose)
        elif epochs and not batch_size:
            self.__fitting(data.x, data.y, epochs, configs_manner.model_batch_size, verbose)
        elif batch_size and not epochs:
            self.__fitting(data.x, data.y, configs_manner.model_train_epochs, batch_size, verbose)
        else:
            self.__fitting(data.x, data.y,
                           configs_manner.model_train_epochs,
                           configs_manner.model_batch_size,
                           verbose)


def build_model(model_config):
    if model_config:
        n_input, n_lstm_nodes, dropout, n_features = model_config
        n_timesteps = n_outputs = n_input
        return LSTMRegressor(n_input, n_lstm_nodes, dropout, n_timesteps, n_features, n_outputs)
    return None