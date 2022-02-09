import configs_manner


class ModelInterface:
    def __init__(self, locale):
        self.locale = locale
        self.model = None
        self.model_path = configs_manner.model_path
        self.model_path_remote = configs_manner.model_path_remote
        self.model_type = str(configs_manner.model_type)
        self.model_subtype = str(configs_manner.model_subtype)

