import glossary_manner
from enums import feature_enum


def is_feature(feature, feature_type):
    """
    Check if column is a feature contained in the glossary
    :param feature: dataframe column
    :param feature_type: Enum type of Feature (enums/feature_enum)
    :return: bool
    """
    if feature.upper() in glossary_manner.features[feature_type.value]:
        return True
    return False


def find_features(df):
    """
    Method to find features in dataframe
    :param df: pandas dataframe
    :return: features dict()
    """
    features_dict = dict()

    for df_column in df.columns:
        if is_feature(df_column, feature_enum.Feature.CASES):
            features_dict[feature_enum.Feature.CASES.value] = df_column
        if is_feature(df_column, feature_enum.Feature.DEATHS):
            features_dict[feature_enum.Feature.DEATHS.value] = df_column

    return features_dict
