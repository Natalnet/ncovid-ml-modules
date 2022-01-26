from enums import feature_enum

import glossary_manner
import data_manner

db_folder = '../dbs/'
last_date = '2021-03-21'
df_araraquara = data_manner.DataConstructor.read_csv_file(db_folder + 'df_araraquara.csv', 'date', last_date, None)
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
