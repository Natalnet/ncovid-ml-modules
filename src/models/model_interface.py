from abc import abstractmethod

import configs_manner


class ModelInterface:
    def __init__(self, locale):
        self.locale = locale
        self.model = None
        self.model_path = configs_manner.model_path
        self.model_path_remote = configs_manner.model_path_remote
        self.model_type = str(configs_manner.model_type)
        self.model_subtype = str(configs_manner.model_subtype)

    @abstractmethod
    def _resolve_model_name(self):
        pass

    @abstractmethod
    def _model_architecture(self):
        pass

    @abstractmethod
    def creating(self):
        self.model = self._model_architecture()

    @abstractmethod
    def loading(self):
        pass

    @abstractmethod
    def fiting(self, x, y):
        pass

    @abstractmethod
    def predicting(self, data):
        pass

