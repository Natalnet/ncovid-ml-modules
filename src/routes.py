from flask import Flask, jsonify

app = Flask(__name__)

import predictor_manner


@app.route("/lstm/repo/<repo>/path/<path>/feature/<feature>/begin/<begin>/end/<end>/", methods=["GET"])
def lstm(repo, path, feature, begin, end):

    predictor_obj = predictor_manner.PredictorConstructor(path, repo, feature, begin, end)
    response = predictor_obj.predict()
    response_json = jsonify(response)
    return response_json

# @app.route(
#     "/rnn/repo/<repo>/path/<path>/feature/<feature>/begin/<begin>/end/<end>/",
#     methods=["GET"],
# )
# def rnn(repo, path, feature, begin, end):
#     response = predictor_manner.predict(repo, path, feature, begin, end)
#     return response


# @app.route('/arima/repo/<repo>/path/<path>', methods=['GET'])

# def arima(variables):
# resposta = datamanager.predict(variables)
# return resposta

if __name__ == "__main__":
    app.run(debug=True)
