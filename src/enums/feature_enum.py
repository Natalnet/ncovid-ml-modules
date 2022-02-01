from enum import Enum


class Feature(Enum):
    CASES = "cases"
    DEATHS = "deaths"
    INFECTED = "infected"
    RECOVERED = "recovered"


class BaseCollecting(Enum):
    ONE = 0
    BASE = 1
    EPIDEMIOLOGICAL = 2
