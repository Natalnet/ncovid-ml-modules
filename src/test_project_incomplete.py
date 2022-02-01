import data_manner
from models.artificial import lstm_manner

# --------- MANIPULATING DATA
repo = "p971074907"
path = "brl:rn"
feature = "date:deaths:newCases:"
begin = "2020-05-01"
end = "2021-07-01"

construtor_dados = data_manner.DataConstructor()
data_araraquara = construtor_dados.collect_dataframe(path, repo, feature, begin, end)
train, test = construtor_dados.build_train_test(data_araraquara)
print(train.x.shape, train.y.shape)
print(test.x.shape, test.y.shape)

# Creating model, training, and predicting
lstm_model = lstm_manner.ModelLSTM("Araraquara")
if lstm_model.fiting(train.x, train.y, verbose=0):
    y_hat, rmse = lstm_model.predicting(test)
    print(sum(rmse))

lstm_model.saving()

# FORMA 2
construtor_dados_2 = data_manner.DataConstructor()
construtor_dados_2.is_predicting = True
test_2 = construtor_dados_2.build_test(data_araraquara)
print(test_2.x.shape, test_2.y.shape)

# Loading model, and predicting
lstm_model_2 = lstm_manner.ModelLSTM("Araraquara")
y_hat, rmse = lstm_model_2.predicting(test)
