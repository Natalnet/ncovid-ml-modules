from flask import Flask, jsonify, request

app = Flask(__name__)

import json
import predictor_manner
import configs_manner


@app.route(
    "/lstm/repo/<repo>/path/<path>/feature/<feature>/begin/<begin>/end/<end>/",
    methods=["POST"],
)
def lstm(repo, path, feature, begin, end):
    info_json = json.loads(request.form.get("metadata"))
    configs_manner.overwrite(dict(info_json))
    predictor_obj = predictor_manner.PredictorConstructor(
        configs_manner.model_id, path, repo, feature, begin, end
    )
    response = predictor_obj.predict()
    response_json = jsonify(predictor_obj.predictions_to_weboutput(response))
    return response_json


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
