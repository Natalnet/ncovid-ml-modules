from math import sqrt

from sklearn.metrics import mean_squared_error
from tensorflow.keras.callbacks import EarlyStopping

import configs_manner


class ModelInterface:
    def __init__(self, n_inputs, n_nodes, n_features, dropout, n_outputs=None):
        self.n_inputs = n_inputs
        self.n_nodes = n_nodes
        self.n_features = n_features
        self.dropout = dropout
        self.n_outputs = None
        self.predictions = None
        self.stop_training = EarlyStopping(monitor='loss', mode='min', verbose=0,
                                           patience=configs_manner.model_patience_earlystop)
        if n_outputs is None:
            self.n_outputs = n_inputs
        self.model = None

    def save_model(self, locale):
        self.model.save(
            "../dbs/fitted_model/" +
            locale + "_" +
            self.__class__.__name__ + "_" +
            str(self.n_inputs) + "_" +
            str(self.n_features) + "_" +
            str(self.n_nodes) +
            ".h5")

    def fit_model(self, x, y, verbose=0):
        self.model.fit(x=x,
                       y=y,
                       epochs=configs_manner.model_train_epochs,
                       batch_size=configs_manner.model_batch_size,
                       verbose=verbose,
                       callbacks=[self.stop_training])
        return True

    def make_predictions(self, data):
        """
        Make predictions (often test data)
        :param data: data to make predictions
        :return: prediction and prediction's rmse
        """
        yhat = self.model.predict(data.x, verbose=0)

        rmse_scores = list()
        for i in range(len(yhat)):
            mse = mean_squared_error(data.y[i], yhat[i])
            rmse = sqrt(mse)
            rmse_scores.append(rmse)

        return yhat, rmse_scores
