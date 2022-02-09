import sys
sys.path.append("../src")
import predictor_manner

# --------- MANIPULATING DATA
repo = "p971074907"
path = "brl:rn"
feature = "date:deaths:newCases:"
begin = "2020-05-01"
end = "2020-07-28"

predictor = predictor_manner.PredictorConstructor(path, repo, feature, begin, end)

print(predictor.predict())