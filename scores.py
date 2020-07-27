import numpy as np
from abc import ABC, abstractmethod


class Score:

    @abstractmethod
    def calculate(self, values, **statistics):
        pass


class ZScore(Score):

    def calculate(self, values, **statistics):
        mean, std = (statistics.get('mean', None), statistics.get('std', None))
        assert type(values == np.array), "Only numpy arrays are eligible"
        return (values - mean) / std
