# model local id to be loaded.
model_id = "cee94ec2-ac6e-11ec-84ad-48a47252b4f8"
# specif code to the remote repository data.
repo = "p971074907"
# coutry and state acronym splited by a ":"
path = "brl:rn"
# columns (or features) to be extracted from the database, each one splited by a ":"
feature = "date:newDeaths:newCases:"
# start date for the data request.
begin = "2021-05-01"
# finish date for the data request.
end = "2021-06-01"

import predictor_manner

predictor = predictor_manner.PredictorConstructor(model_id, "../dbs/df_araraquara.csv")

print(predictor.predict())
