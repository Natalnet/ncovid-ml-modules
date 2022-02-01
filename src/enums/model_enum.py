from enum import Enum


class Model(Enum):
    ARTIFICIAL = "Artificial"
    EPIDEMIOLOGICAL = "Epidemiological"
    AUTOREGRESSIVE = "Autoregressive"

    class Artificial(Enum):
        LSTM = "lstm"
        AUTO_ENCODE = "autoencode"

    class Epidemiological(Enum):
        SIR = "sir"
        SEIR = "seir"
        SEIRD = "seird"

    class Autoregressive(Enum):
        AR = "ar"
        MA = "ma"
        ARIMA = "arima"

