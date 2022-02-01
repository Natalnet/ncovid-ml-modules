import numpy as np

import configs_manner
from enums import model_enum


class DataConstructor:
    def __init__(self, is_predicting=False):
        """Class that constructs the data correctly to the model.
        For editing more details in your data, use doc/configure.json file.
        
        Args:
            is_predicting (bool, optional): Flag that descripts if data is for testing. Defaults to False.
        """
        if configs_manner.model_type == model_enum.Model.ARTIFICIAL.value:
            self.window_size = configs_manner.model_infos["data_window_size"]
            self.test_size_in_days = configs_manner.model_infos[
                "data_test_size_in_days"
            ]
            self.type_norm = configs_manner.model_infos["data_type_norm"]

            self.is_predicting = configs_manner.model_infos[
                "model_is_predicting"
            ]  # setado para treinamento
            if is_predicting:
                self.is_predicting = is_predicting

    def build_train(self, data):
        if self.is_predicting:
            data = self.__reshape_data(data)
        data = self.__windowing_data(data)
        return Train(data, self.window_size, self.type_norm)

    def build_test(self, data):
        if self.is_predicting:
            data = self.__reshape_data(data)
        data = self.__windowing_data(data)
        return Test(data, self.window_size, self.type_norm)

    def build_train_test(self, data):
        data = self.__reshape_data(data)
        data_train, data_test = self.__split_data_train_test(data)
        train = self.build_train(data_train)
        test = self.build_test(data_test)
        return train, test

    def __split_data_train_test(self, data):
        # makes dataset multiple of n_days
        data = data[data.shape[0] % self.window_size :]
        # make test set multiple of n_days
        n_test = self.test_size_in_days
        n_test -= self.test_size_in_days % self.window_size
        # split into standard weeks
        train, test = data[:-n_test], data[-n_test:]
        return train, test

    def __reshape_data(self, data):
        return np.array(data).T

    def __windowing_data(self, data):
        check_size = data.shape[0] // self.window_size
        if check_size * self.window_size != data.shape[0]:
            data = data[: check_size * self.window_size]
        return np.array(np.split(data, len(data) // self.window_size))

    def collect_dataframe(self, path, repo=None, feature=None, begin=None, end=None):
        """[summary]

        Args:
            path ([type]): [description]
            repo ([type], optional): [description]. Defaults to None.
            feature ([type], optional): [description]. Defaults to None.
            begin ([type], optional): [description]. Defaults to None.
            end ([type], optional): [description]. Defaults to None.

        Returns:
            [Array of Arrays]: returns the dataframme in array format (not as pandas dataframe)
        """
        import pandas as pd

        if repo and feature and begin and end is not None:
            dataframe = pd.read_csv(
                f"http://ncovid.natalnet.br/datamanager/repo/{repo}/path/{path}/feature/{feature}/begin/{begin}/end/{end}/as-csv",
                parse_dates=["date"],
                index_col="date",
            )
        else:
            dataframe = pd.read_csv(path, parse_dates=["date"], index_col="date",)

        return self.__clean_dataframe(dataframe)

    def __clean_dataframe(self, dataframe):
        if configs_manner.model_infos["data_is_accumulated_values"]:
            for column in dataframe.columns:
                dataframe[column] = (
                    dataframe[column]
                    .diff(configs_manner.model_infos["data_window_size"])
                    .dropna()
                )

        dataframe = dataframe.dropna()

        def is_different_values(s):
            a = s.to_numpy()  # s.values (pandas<0.24)
            return (a[0] == a).all()

        column_with_values = [
            not (is_different_values(dataframe[column])) for column in dataframe.columns
        ]
        dataframe = dataframe.loc[:, column_with_values]

        return [dataframe[col].values for col in dataframe.columns]


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
        from sklearn.metrics import mean_squared_error
        from math import sqrt

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
        x, y = self.__walk_forward(data, step_size)

        if configs_manner.model_infos["model_is_output_in_input"]:
            self.x = x
        else:
            self.x = x[:, :, 1:]

        configs_manner.model_infos["data_n_features"] = self.x.shape[-1]
        self.y = y.reshape((y.shape[0], y.shape[1], 1))

    def __walk_forward(self, data_, step_size):
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

        if configs_manner.model_infos["model_is_output_in_input"]:
            self.x = data[:-1, :, :]
        else:
            self.x = data[:-1, :, 1:]

        self.y = data[1:, :, :1]
        self.y = self.y.reshape((self.y.shape[0], self.y.shape[1], 1))
