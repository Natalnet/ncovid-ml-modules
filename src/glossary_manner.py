import json

from enums import feature_enum


class Glossary:

    def __init__(self, vocabulary=None):
        if vocabulary:
            self.__vocabulary = vocabulary
        else:
            self.fill_vocabulary_from_file()

        self.features_dict = dict()

    @property
    def vocabulary(self):
        return self.__vocabulary

    @vocabulary.setter
    def vocabulary(self, vocabulary_list):
        if type(vocabulary_list) is list:
            self.__vocabulary = vocabulary_list

    def fill_vocabulary_from_file(self):
        """
        Extract all features that may is contained in json glossary file
        :return: glossary list
        """
        import configuration as pipeline_configs

        file = open(pipeline_configs.doc_folder + pipeline_configs.glossary)
        data = json.load(file)
        self.__vocabulary = data['wordlist']

    @staticmethod
    def find_feature_glossary(all_columns, type_feat, glossary_list):
        """
        :param all_columns: df columns list
        :param type_feat: feature_enum.Feature type
        :param glossary_list: glossary list
        :return: string column name or None
        """
        for g in glossary_list:
            if g['name'] == type_feat:
                for column_name in all_columns:
                    if Glossary.in_glossary(column_name, g):
                        return column_name
                return None

    @staticmethod
    def in_glossary(column_name, glossary_list):
        if column_name.upper() in [each_synonym.upper()
                                   for each_synonym
                                   in glossary_list['synonyms']]:
            return True
        return False

    def find_column(self, df_columns, type_feat=feature_enum.Feature.CASES):
        """
        :param df_columns: df columns list
        :param type_feat: feature_enum.Feature type
        """
        type_feat_name = type_feat.value
        self.features_dict[type_feat_name] = self.find_feature_glossary(df_columns, type_feat_name, self.__vocabulary)

    def find_epidemiological_columns(self, df_columns):
        """
        :param df_columns: df columns list
        """
        infected = feature_enum.Feature.CASES.value
        removed = feature_enum.Feature.RECOVERED.value
        deceased = feature_enum.Feature.DEATHS.value
        self.features_dict[infected] = self.find_feature_glossary(df_columns, infected, self.__vocabulary)
        self.features_dict[removed] = self.find_feature_glossary(df_columns, removed, self.__vocabulary)
        self.features_dict[deceased] = self.find_feature_glossary(df_columns, deceased, self.__vocabulary)


def create_glossary(df_columns,
                    feat_preset=feature_enum.BaseCollecting.ONE,
                    type_feat=feature_enum.Feature.CASES):
    """
    :param df_columns: df columns list
    :param feat_preset: feature_enum.BaseCollecting type
    :param type_feat: feature_enum.Feature type
    :return:
    """
    glossary = Glossary()

    if feat_preset is feature_enum.BaseCollecting.EPIDEMIOLOGICAL:
        glossary.find_epidemiological_columns(df_columns)
    else:
        if type_feat is not None:
            glossary.find_column(df_columns, type_feat)
        else:
            if feat_preset is feature_enum.BaseCollecting.ONE:
                glossary.find_column(df_columns)

    return glossary
