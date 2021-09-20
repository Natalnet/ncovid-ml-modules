import pandas as pd
import numpy as np

import data_manner
import model_manner


week_size = 7
data = pd.read_csv(f"http://ncovid.natalnet.br/datamanager/repo/p971074907/path/brl:rn/feature/date:newCases:deaths:/begin/2020-05-01/end/2021-05-01/as-csv", index_col='date')

for column in data.columns:
        data[column] = data[column].diff(7).dropna()
    
data_processed = data.dropna()

data_final = [data_processed[col].values for col in data_processed.columns]

train, test = data_manner.build_data(data_final, step_size=week_size, size_data_test=7)

model_config = [week_size, 250, 0.0, train.x.shape[2]]

regressor = model_manner.build_model(model_config)

regressor.fit_model(train, epochs=100, batch_size=32, verbose=0)

regressor.model.save('models/model_test_to_load')