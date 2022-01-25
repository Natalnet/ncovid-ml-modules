from math import sqrt

import numpy as np
from sklearn.metrics import mean_squared_error


class DataConstructor:
    def __init__(self, step_size, is_training=False, type_norm=None):
        self.step_size = step_size
        self.is_training = is_training
        self.type_norm = type_norm

    def build_train(self, data):
        if not self.is_training:
            data = self.reshape_data(data)
        data = self.windowing_data(data)
        return Train(data, self.step_size, self.type_norm)

    def build_test(self, data):
        if not self.is_training:
            data = self.reshape_data(data)
        data = self.windowing_data(data)
        return Test(data, self.step_size, self.type_norm)

    def build_train_test(self, data, size_data_test):
        data = self.reshape_data(data)
        data_train, data_test = self.split_data_train_test(data, size_data_test)
        train = self.build_train(data_train)
        test = self.build_test(data_test)
        return train, test

    def split_data_train_test(self, data, n_test):
        # makes dataset multiple of n_days
        data = data[data.shape[0] % self.step_size:]
        # make test set multiple of n_days
        n_test -= n_test % self.step_size
        # split into standard weeks
        train, test = data[:-n_test], data[-n_test:]
        return train, test

    def reshape_data(self, data):
        return np.array(data).T

    def windowing_data(self, data):
        return np.array(np.split(data, len(data) / self.step_size))


class Data:
    def __init__(self, step_size, type_norm=None):
        self.x = None
        self.y = None
        self._y_hat = None
        self._rmse = None
        self.type_norm = type_norm
        self.step_size = step_size

    @property
    def y_hat(self):
        return self._y_hat

    @y_hat.setter
    def y_hat(self, y_hat_list):
        self._y_hat = y_hat_list
        self._rmse = list()
        for i in range(len(self.y_hat)):
            mse_i = mean_squared_error(self.y[i], self.y_hat[i])
            rmse_i = sqrt(mse_i)
            self._rmse.append(rmse_i)

    @property
    def rmse(self):
        return self._rmse

    @rmse.setter
    def rmse(self, rmse_list):
        self._rmse = rmse_list


class Train(Data):
    def __init__(self, data, step_size, type_norm=None):
        super().__init__(step_size, type_norm)
        x, y = Train.walk_forward(data, step_size)
        self.x_labeled = x[:, :, : 1]
        self.x = x[:, :, 1:]  # self.x = x
        self.y = y.reshape((y.shape[0], y.shape[1], 1))

    @staticmethod
    def walk_forward(data_, step_size):
        # flatten data
        data = data_.reshape((data_.shape[0] * data_.shape[1], data_.shape[2]))
        x, y = list(), list()
        in_start = 0
        # step over the entire history one time step at a time
        for _ in range(len(data)):
            # define the end of the input sequence
            in_end = in_start + step_size
            out_end = in_end + step_size
            # ensure we have enough data for this instance
            if out_end < len(data):
                x.append(data[in_start:in_end, :])
                y.append(data[in_end:out_end, 0])
            # move along one time step
            in_start += 1
        return np.array(x), np.array(y)


class Test(Data):
    def __init__(self, data, step_size, type_norm=None):
        super().__init__(step_size, type_norm)
        self.x = data[:, :, 1:]  # self.x = x
        self.y = data[:, :, :1]
        self.y = self.y.reshape((self.y.shape[0], self.y.shape[1], 1))
