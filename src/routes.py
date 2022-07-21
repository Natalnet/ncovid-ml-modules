from flask import Flask, jsonify, request

app = Flask(__name__)

import json

import configs_manner
import predictor_manner
import data_manner
from models.artificial.lstm_manner import ModelLSTM


@app.route(
    "/api/v1/modelType/<modelType>/train/repo/<repo>/path/<path>/",
    methods=["POST"],
)
def train_recurrent(modelType, repo, path):

    metadata_to_train = json.loads(request.form.get("metadata"))

    configs_manner.overwrite(metadata_to_train)

    data_constructor = data_manner.DataConstructor()
    collected_data = data_constructor.collect_dataframe(
        path,
        repo,
        metadata_to_train["inputFeatures"],
        metadata_to_train["begin"],
        metadata_to_train["end"],
    )

    train, test = data_constructor.build_train_test(collected_data)

    lstm_model = ModelLSTM(path)
    lstm_model.creating()

    lstm_model.fiting(train.x, train.y, verbose=0)
    ytrue = test.y
    yhat = lstm_model.predicting(test.x)
    score, stest, strain = lstm_model.calculate_score(ytrue, yhat)
    lstm_model.score = score

    lstm_model.saving()
    model_metadata = lstm_model.metadata
    return jsonify(model_metadata)


@app.route(
    "/api/v1/modelCategory/<modelCategory>/predict/modelInstance/<modelInstance>/begin/<begin>/end/<end>/",
    methods=["GET"],
)
def predict_recurrent(modelCategory, modelInstance, begin, end):

    predictor_obj = predictor_manner.PredictorConstructor(
        model_id=modelInstance, begin=begin, end=end
    )
    response = predictor_obj.predict()

    yhat_json = predictor_obj.predictions_to_weboutput(response)

    response_json = jsonify(yhat_json)

    return response_json


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
