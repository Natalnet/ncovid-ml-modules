import numpy as np

import configs_manner


class DataConstructor:
    def __init__(self, is_predicting=False):
        """Data manager designed to collect and prepare data for the project.
        More details look up at doc/configure.json file.
        
        Args:
            is_predicting (bool, optional): Flag that descripts if data is for testing. Defaults extracted from configure.json.
        """
        self.is_predicting = configs_manner.model_is_predicting
        if is_predicting:
            self.is_predicting = is_predicting

        eval("self._constructor_" + configs_manner.model_type + "()")

    def _constructor_Autoregressive(self):
        self.test_size_in_days = configs_manner.model_infos["data_test_size_in_days"]

    def _constructor_Epidemiological(self):
        # TO DO
        pass

    def _constructor_Artificial(self):
        self.window_size = configs_manner.model_infos["data_window_size"]
        self.test_size_in_days = configs_manner.model_infos["data_test_size_in_days"]
        self.type_norm = configs_manner.model_infos["data_type_norm"]

    def build_train_test(self, data):
        """To build train and test data for training and predicting.

        Args:
            data (list[list]): bi-dimensional data numpy array that needs to be prepared for the model.
            The first dimension represents the number of features. 
            The second dimension represents the time-serie of each dimension. 

        Returns:
            Train, Test: Train and Test data types
        """
        assert data is not None, "Empty data"
        data = self.__transpose_data(data)
        data_train, data_test = self.split_data_train_test(data)
        train = self.build_train(data_train)
        test = self.build_test(data_test)
        return train, test

    def build_train(self, data):
        """To build train data for training.

        Args:
            data (list[list]): bi-dimensional data numpy array that needs to be prepared for the model.
            The first dimension represents the number of features. 
            The second dimension represents the time-serie of each dimension. 

        Returns:
            Test: Test data type 
        """
        assert type(data) is not np.array, "Format data is not np.array"
        return eval("self._build_data_" + configs_manner.model_type + "(Train, data)")

    def build_test(self, data):
        """To build test data for predicting.

        Args:
            data (list[list]): bi-dimensional data numpy array that needs to be prepared for the model.
            The first dimension represents the number of features. 
            The second dimension represents the time-serie of each dimension. 

        Returns:
            Train: Train data type 
        """
        assert type(data) is not np.array, "Format data is not np.array"
        return eval("self._build_data_" + configs_manner.model_type + "(Test, data)")

    def _build_data_Autoregressive(self, data_type, data):
        return data_type(data, self.window_size, self.type_norm)

    def _build_data_Epidemiological(self, data_type, data):
        return data_type(data, self.window_size, self.type_norm)

    def _build_data_Artificial(self, data_type, data):
        def __windowing_data(data):
            check_size = data.shape[0] // self.window_size
            if check_size * self.window_size != data.shape[0]:
                data = data[: check_size * self.window_size]
            return np.array(np.split(data, len(data) // self.window_size))

        if self.is_predicting:
            data = self.__transpose_data(data)
        data = __windowing_data(data)
        return data_type(data, self.window_size, self.type_norm)

    def __transpose_data(self, data):
        return np.array(data).T

    def split_data_train_test(self, data):
        """Split a numpy bi-dimensional array in train and test.

        Args:
            data (list[list]): bi-dimensional data numpy array that needs to be prepared for the model.
            The first dimension represents the number of features. 
            The second dimension represents the time-serie of each dimension. 

        Returns:
            train and test (list[list]): bi-dimensional data numpy array that needs to be prepared for the model.
        """
        # makes dataset multiple of n_days
        data = data[data.shape[0] % self.window_size :]
        # make test set multiple of n_days
        n_test = self.test_size_in_days
        n_test -= self.test_size_in_days % self.window_size
        # split into standard weeks
        train, test = data[:-n_test], data[-n_test:]
        return train, test

    def collect_dataframe(self, path, repo=None, feature=None, begin=None, end=None):
        """Collect a dataframe from the repository or web
        
        Args:
            path (str): [description]
            repo (str, optional): [description]. Defaults to None.
            feature (str, optional): [description]. Defaults to None.
            begin (str, optional): [description]. Defaults to None.
            end (str, optional): [description]. Defaults to None.

        Returns:
            dataframe: Pandas dataframe
        """
        import pandas as pd

        if repo and feature and begin and end is not None:
            if self.is_predicting:
                begin, end = self.__add_period(begin, end)

            dataframe = pd.read_csv(
                (
                    "http://ncovid.natalnet.br/datamanager/"
                    + "repo/{}/"
                    + "path/{}/"
                    + "feature/{}/"
                    + "begin/{}/"
                    + "end/{}/as-csv"
                ).format(repo, path, feature, begin, end),
                parse_dates=["date"],
                index_col="date",
            )
        else:
            dataframe = pd.read_csv(path, parse_dates=["date"], index_col="date",)

        preprocessor = self.Preprocessing()
        return preprocessor.pipeline(dataframe)

    def __add_period(self, begin, end):
        import datetime

        DATE_FORMAT = "%Y-%m-%d"

        begin = datetime.datetime.strptime(begin, DATE_FORMAT)
        end = datetime.datetime.strptime(end, DATE_FORMAT)

        period_to_add = (end - begin + datetime.timedelta(days=1)).days
        offset_days = period_to_add + (
            self.window_size - period_to_add % self.window_size
        )

        new_first_day = begin - datetime.timedelta(days=self.window_size)
        new_last_day = end + datetime.timedelta(
            days=offset_days - period_to_add + self.window_size
        )

        return new_first_day.strftime(DATE_FORMAT), new_last_day.strftime(DATE_FORMAT)

    class Preprocessing:
        def __init__(self):
            # TO DO
            pass

        def pipeline(self, dataframe):
            assert dataframe is not None, "Empty data"

            dataframe_not_cumulative = self.remove_na_values(
                self.solve_cumulative(dataframe)
            )
            dataframe_columns_elegible = self.select_columns_with_values(
                dataframe_not_cumulative
            )
            dataframe_as_list = self.convert_dataframe_to_list(
                dataframe_columns_elegible
            )
            return dataframe_as_list

        def solve_cumulative(self, dataframe):
            if configs_manner.model_infos["data_is_accumulated_values"]:
                for column in dataframe.columns:
                    dataframe[column] = (
                        dataframe[column]
                        .diff(configs_manner.model_infos["data_window_size"])
                        .dropna()
                    )
            return dataframe

        def select_columns_with_values(self, dataframe):
            def is_different_values(s):
                a = s.to_numpy()  # s.values (pandas<0.24)
                return (a[0] == a).all()

            column_with_values = [
                not (is_different_values(dataframe[column]))
                for column in dataframe.columns
            ]
            return dataframe.loc[:, column_with_values]

        def convert_dataframe_to_list(self, dataframe):
            return [dataframe[col].values for col in dataframe.columns]

        def remove_na_values(self, dataframe):
            return dataframe.dropna()


class Data:
    def __init__(self, step_size=None, type_norm=None):
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
    def __init__(self, data, step_size=None, type_norm=None):
        super().__init__(step_size, type_norm)
        self.x, self.y = eval(
            "self._builder_train_" + configs_manner.model_type + "(data)"
        )

    def _builder_train_Autoregressive(self, data):
        # TO DO
        pass

    def _builder_train_Epidemiological(self, data):
        # TO DO
        pass

    def _builder_train_Artificial(self, data):
        def __walk_forward(data_):
            # flatten data
            data = data_.reshape((data_.shape[0] * data_.shape[1], data_.shape[2]))
            x, y = list(), list()
            in_start = 0
            # step over the entire history one time step at a time
            for _ in range(len(data)):
                # define the end of the input sequence
                in_end = in_start + self.step_size
                out_end = in_end + self.step_size
                # ensure we have enough data for this instance
                if out_end < len(data):
                    x.append(data[in_start:in_end, :])
                    y.append(data[in_end:out_end, 0])
                # move along one time step
                in_start += 1
            return np.array(x), np.array(y)

        x, y = __walk_forward(data)

        if configs_manner.model_infos["model_is_output_in_input"]:
            x = x
        else:
            x = x[:, :, 1:]

        configs_manner.model_infos["data_n_features"] = x.shape[-1]
        y = y.reshape((y.shape[0], y.shape[1], 1))

        return x, y


class Test(Data):
    def __init__(self, data, step_size=None, type_norm=None):
        super().__init__(step_size, type_norm)
        self.x, self.y = eval(
            "self._builder_test_" + configs_manner.model_type + "(data)"
        )

    def _builder_test_Autoregressive(self, data):
        # TO DO
        pass

    def _builder_test_Epidemiological(self, data):
        # TO DO
        pass

    def _builder_test_Artificial(self, data):
        if configs_manner.model_infos["model_is_output_in_input"]:
            x = data[:-1, :, :]
        else:
            x = data[:-1, :, 1:]

        y = data[1:, :, :1]
        y = y.reshape((y.shape[0], y.shape[1], 1))

        return x, y
