from flask import Flask, request

app = Flask(__name__)

import predictor_manner

@app.route('/lstm/repo/<repo>/path/<path>/feature/<feature>/begin/<begin>/end/<end>/', methods=['GET'])
def lstm(repo, path, feature, begin, end):
<<<<<<< HEAD
    resposta = predictor_manner.predict(repo, path, feature, begin, end)
    return resposta

#@app.route('/arima/repo/<repo>/path/<path>', methods=['GET'])

# def ARIMA(variables):
=======
    response = predictor_manner.predict(repo, path, feature, begin, end)
    return response

#@app.route('/arima/repo/<repo>/path/<path>', methods=['GET'])

# def arima(variables):
>>>>>>> 53002a1cad3cbe644d7ac9e14d4468b88b75f3bf
    # resposta = datamanager.predict(variables)
    # return resposta

if __name__ == "__main__":
  app.run(debug=True)
