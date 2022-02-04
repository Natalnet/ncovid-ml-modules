import predictor_manner
import data_manner
from models import model_interface
#from models.artificial import lstm_manner
import configs_manner

# --------- MANIPULATING DATA
repo = "p971074907"
path = "brl:rn"
feature = "date:deaths:newCases:"
begin = "2020-05-01"
end = "2020-05-28"


predictor = predictor_manner.PredictorConstructor(path, repo, feature, begin, end)

print(predictor.predict())

