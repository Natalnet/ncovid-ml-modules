#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 15:58:53 2021

@author: davi
"""

import pandas as pd
from enums import feature_enum
import matplotlib.pyplot as plt

import data_manner
import evaluator_manner
import plot_manner
import glossary_manner

df_araraquara = pd.read_csv('../dbs/df_araraquara.csv')
print(df_araraquara.columns)

###### glossary test code
glossary_1 = glossary_manner.Glossary()
glossary_1.find_column(df_araraquara.columns, type_feat=feature_enum.Feature.DEATHS)
print(glossary_1.features_dict)
# {'deaths': 'deaths'}

glossary_2 = glossary_manner.create_glossary(df_araraquara.columns)
glossary_2.find_column(df_araraquara.columns, type_feat=feature_enum.Feature.DEATHS)
glossary_2.find_column(df_araraquara.columns, type_feat=feature_enum.Feature.RECOVERED)
print(glossary_2.features_dict)
# {'cases': 'confirmed'}

glossary_3 = glossary_manner.create_glossary(df_araraquara.columns,
                                             feat_preset=feature_enum.BaseCollecting.EPIDEMIOLOGICAL)
print(glossary_3.features_dict)
# {'cases': 'confirmed', 'recovered': None, 'deaths': 'deaths'}

glossary_4 = glossary_manner.create_glossary(df_araraquara.columns,
                                             feat_preset=feature_enum.BaseCollecting.BASE)
print(glossary_4.features_dict)
# {'cases': 'confirmed', 'deaths': 'deaths'}


###### training and data_manner test code
def read_csv_file(path, column_date, last_day, first_date=None):
    df = pd.read_csv(path)
    if first_date is not None:
        return df[(df[column_date] < last_day) & (df[column_date] >= first_date)]
    return df[(df[column_date] < last_day)][1:]