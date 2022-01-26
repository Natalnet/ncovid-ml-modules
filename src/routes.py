from flask import Flask, request

app = Flask(__name__)

import predictor_manner


@app.route('/lstm/repo/<repo>/path/<path>/feature/<feature>/begin/<begin>/end/<end>/', methods=['GET'])
def lstm(repo, path, feature, begin, end):
    response = predictor_manner.predict(repo, path, feature, begin, end)
    return response


# @app.route('/arima/repo/<repo>/path/<path>', methods=['GET'])

# def arima(variables):
# resposta = datamanager.predict(variables)
# return resposta

if __name__ == "__main__":
    app.run(debug=True)
