from typing import Tuple

import numpy as np
import pandas as pd

import logger
import configs_manner


class DataConstructor:
    def __init__(self, is_predicting: bool = False) -> "DataConstructor":
        """Data manager designed to collect and prepare data for the project.
        More details look up at doc/configure.json file.
        
        Args:
            is_predicting (bool, optional): Flag that descripts if data is for testing. 
            Defaults extracted from configure.json.
        """

        self.is_predicting = (
            is_predicting if is_predicting else configs_manner.model_is_predicting
        )

        try:
            getattr(self, f"_constructor_{configs_manner.model_type}")()
        except Exception as e:
            logger.error_log(
                self.__class__.__name__, self.__init__.__name__, f"Error: {e}."
            )
            raise

    def _constructor_Autoregressive(self):
        self.test_size_in_days = configs_manner.model_infos["data_test_size_in_days"]

    def _constructor_Epidemiological(self):
        # TO DO
        pass

    def _constructor_Artificial(self):
        self.window_size = configs_manner.model_infos["data_window_size"]
        self.test_size_in_days = configs_manner.model_infos["data_test_size_in_days"]
        self.type_norm = configs_manner.model_infos["data_type_norm"]

    def build_train_test(self, data: np.array | list) -> Tuple["Train", "Test"]:
        """To build train and test data for training and predicting.

        Args:
            data (list[list]): bi-dimensional data numpy array that needs to be prepared for the model.
            The first dimension represents the number of features. 
            The second dimension represents the time-serie of each dimension. 

        Returns:
            Train, Test: Train and Test data types
        """
        assert type(data) == list, logger.error_log(
            self.__class__.__name__, self.build_test.__name__, "Format data"
        )

        data_t = self.__transpose_data(data)
        data_train, data_test = self.split_data_train_test(data_t)
        return self.build_train(data_train), self.build_test(data_test)

    def build_train(self, data: np.array | list) -> "Train":
        """To build train data for training.

        Args:
            data (list[list]): bi-dimensional data numpy array that needs to be prepared for the model.
            The first dimension represents the number of features. 
            The second dimension represents the time-serie of each dimension. 

        Returns:
            Train: Train data type 
        """
        assert type(data) == np.ndarray or type(data) == list, logger.error_log(
            self.__class__.__name__, self.build_train.__name__, "Format data",
        )
        try:
            return getattr(self, f"_build_data_{configs_manner.model_type}")(
                Train, data
            )
        except Exception as e:
            logger.error_log(
                self.__class__.__name__, self.build_train.__name__, f"Error: {e}."
            )
            raise

    def build_test(self, data: np.array | list) -> "Test":
        """To build test data for predicting.

        Args:
            data (list[list]): bi-dimensional data numpy array that needs to be prepared for the model.
            The first dimension represents the number of features. 
            The second dimension represents the time-serie of each dimension. 

        Returns:
            Test: Test data type 
        """
        assert type(data) == np.ndarray or type(data) == list, logger.error_log(
            self.__class__.__name__, self.build_test.__name__, "Format data",
        )
        try:
            return getattr(self, f"_build_data_{configs_manner.model_type}")(Test, data)
        except Exception as e:
            logger.error_log(
                self.__class__.__name__, self.build_test.__name__, f"Error: {e}."
            )
            raise

    def _build_data_Autoregressive(self, data_type, data):
        # TO DO
        pass

    def _build_data_Epidemiological(self, data_type, data):
        # TO DO
        pass

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

    def split_data_train_test(self, data: list) -> Tuple[list, list]:
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

    def collect_dataframe(
        self,
        path: str,
        repo: str = None,
        feature: str = None,
        begin: str = None,
        end: str = None,
    ) -> np.array | list:
        """Collect a dataframe from the repository or web
        
        Args:
            path (str): a raw web link or db locale to be predicted (eg. `brl:rn`)
            repo (str, optional): DB repository that contains the dataframe. Defaults to None.
            feature (str, optional): Features separated by `:`, presented in the dataframe(eg. `date:deaths:newCases:`). 
                Defaults to None.
            begin (str, optional): First day of the temporal time series `YYYY-MM-DD`. Defaults to None.
            end (str, optional): Last day of the temporal time series `YYYY-MM-DD`. Defaults to None.

        Returns:
            dataframe: Pandas dataframe
        """

        def read_file(file_name):
            import pandas as pd

            return pd.read_csv(file_name, parse_dates=["date"], index_col="date")

        if repo and feature and begin and end is not None:
            if self.is_predicting:
                begin, end = self.__add_period(begin, end)

            dataframe = read_file(
                "".join(
                    f"http://ncovid.natalnet.br/datamanager/"
                    f"repo/{repo}/"
                    f"path/{path}/"
                    f"feature/{feature}/"
                    f"begin/{begin}/"
                    f"end/{end}/as-csv"
                )
            )
        else:
            dataframe = read_file(path)

        preprocessor = self.Preprocessing()
        return preprocessor.pipeline(dataframe)

    def __add_period(self, begin: str, end: str):
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

        return (
            new_first_day.strftime(DATE_FORMAT),
            new_last_day.strftime(DATE_FORMAT),
        )

    class Preprocessing:
        def __init__(self):
            """
            Preprocess that sould be applied the data for training and predicting steps
            """
            # TO DO
            pass

        def pipeline(self, dataframe: pd.DataFrame) -> list:
            """Auto and basic pipeline applied to the data
            1- Resolve cumulativa columns
            2- Remove columns with any nan values
            3- Remove columns with unique values
            4- Apply moving average
            5- Convert dataframe to list

            Args:
                dataframe (pandas): Temporal time series

            Returns:
                list: Preprocessed dataframe as list
            """
            logger.debug_log(
                self.__class__.__name__,
                self.pipeline.__name__,
                "Preprocessing pipeline",
            )
            assert dataframe is not None, logger.error_log(
                self.__class__.__name__, self.pipeline.__name__, "Data is empty"
            )

            dataframe_not_cumulative = self.remove_na_values(
                self.solve_cumulative(dataframe)
            )

            dataframe_columns_elegible = self.select_columns_with_values(
                dataframe_not_cumulative
            )

            dataframe_rolled = self.moving_average(dataframe_columns_elegible)

            dataframe_as_list = self.convert_dataframe_to_list(dataframe_rolled)

            logger.debug_log(
                self.__class__.__name__, self.pipeline.__name__, "Pipeline finished",
            )
            return dataframe_as_list

        def solve_cumulative(self, dataframe: pd.DataFrame) -> pd.DataFrame:
            if configs_manner.model_infos["data_is_accumulated_values"]:
                return dataframe.diff(
                    configs_manner.model_infos["data_window_size"]
                ).dropna()
            return dataframe

        def moving_average(self, dataframe: pd.DataFrame) -> pd.DataFrame:
            if configs_manner.model_infos["data_is_apply_moving_average"]:
                return (
                    dataframe.rolling(configs_manner.model_infos["data_window_size"])
                    .mean()
                    .fillna(method="bfill")
                    .fillna(method="ffill")
                )
            return dataframe

        def select_columns_with_values(self, dataframe: pd.DataFrame) -> pd.DataFrame:
            def is_different_values(s):
                a = s.to_numpy()  # s.values (pandas<0.24)
                return (a[0] == a).all()

            column_with_values = [
                not (is_different_values(dataframe[column]))
                for column in dataframe.columns
            ]
            return dataframe.loc[:, column_with_values]

        def remove_na_values(self, dataframe: pd.DataFrame) -> pd.DataFrame:
            return dataframe.dropna()

        def convert_dataframe_to_list(self, dataframe: pd.DataFrame) -> list:
            return [dataframe[col].values for col in dataframe.columns]


class Data:
    def __init__(self, step_size: int = None, type_norm: str = None):
        self.x = None
        self.y = None
        self._y_hat = None
        self._rmse = None
        self.type_norm = type_norm
        self.step_size = step_size

        logger.debug_log(
            self.__class__.__name__,
            self.__init__.__name__,
            f"Data -> type_norm: {type_norm} - step_size: {step_size}",
        )

    @property
    def y_hat(self):
        return self._y_hat

    @y_hat.setter
    def y_hat(self, y_hat_list):
        self._y_hat = y_hat_list

    @property
    def rmse(self):
        return self._rmse

    @rmse.setter
    def rmse(self, rmse_list):
        self._rmse = rmse_list


class Train(Data):
    def __init__(self, data: np.array, step_size: int = None, type_norm: str = None):
        """Data Train Object. Use the attribute `model_type` in `docs/configure.json` to determine the type of data should be expected.

        Args:
            data (np.array): Temporal serie used as train data
            step_size (int, optional): Indicates the size of each data sample. Defaults to None.
            type_norm (str, optional): Describes the normalization method applied to the data. Defaults to None.
        """
        try:
            super().__init__(step_size, type_norm)
            self.x, self.y = getattr(
                self, f"_builder_train_{configs_manner.model_type}"
            )(data)
            logger.debug_log(
                self.__class__.__name__, self.__init__.__name__, "Data Train Created"
            )
        except Exception as e:
            logger.error_log(
                self.__class__.__name__, self.__init__.__name__, f"Error: {e}."
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
            for _ in enumerate(data):
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

        x = x if configs_manner.model_infos["model_is_output_in_input"] else x[:, :, 1:]

        configs_manner.model_infos["data_n_features"] = x.shape[-1]
        y = y.reshape((y.shape[0], y.shape[1], 1))

        return x, y


class Test(Data):
    def __init__(self, data, step_size: int = None, type_norm: str = None):
        """Data Test Object. Use the attribute `model_type` in `docs/configure.json` to determine the type of data should be expected

        Args:
            data (np.array): Temporal serie used as test data
            step_size (int, optional): Indicates the size of each data sample. Defaults to None.
            type_norm (str, optional): Describes the normalization method applied to the data. Defaults to None.
        """
        try:
            super().__init__(step_size, type_norm)
            self.x, self.y = getattr(
                self, f"_builder_test_{configs_manner.model_type}"
            )(data)
            logger.debug_log(
                self.__class__.__name__, self.__init__.__name__, "Data Test Created"
            )
        except Exception as e:
            logger.error_log(
                self.__class__.__name__, self.__init__.__name__, f"Error: {e}."
            )

    def _builder_test_Autoregressive(self, data):
        # TO DO
        pass

    def _builder_test_Epidemiological(self, data):
        # TO DO
        pass

    def _builder_test_Artificial(self, data):

        x = (
            data[:-1, :, :]
            if configs_manner.model_infos["model_is_output_in_input"]
            else data[:-1, :, 1:]
        )

        configs_manner.model_infos["data_n_features"] = x.shape[-1]

        y = data[1:, :, :1]
        y = y.reshape((y.shape[0], y.shape[1], 1))

        return x, y
