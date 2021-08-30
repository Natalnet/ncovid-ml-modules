import pandas as pd
import feature_manner

df_araraquara = pd.read_csv('../dbs/df_araraquara.csv')

features = feature_manner.find_features(df_araraquara)

print(features)
print(features['cases'])