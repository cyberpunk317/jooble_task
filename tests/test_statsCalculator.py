import sys
import unittest2
import pandas as pd
import time
import numpy as np
import random

sys.path.insert(0, '..')
from utils import StatsCalculator, Preprocessor

FEATURES = 256


class TestStatsCalculator(unittest2.TestCase):

    def test1(self):
        stats_calc = StatsCalculator()
        preprocessor = Preprocessor()
        df = pd.read_csv('data/train.tsv', sep='\t')
        col_names = [f'feature_{i}' for i in range(FEATURES)]
        df = preprocessor.split_features(df)
        df = preprocessor.f_to_int(df)
        col = random.choice(col_names)

        res = stats_calc.calc_mean(df, col)
        valid_res = np.mean(df[col])

        self.assertEqual(res, valid_res, "Wrong mean calculation")

    def test_speed(self):
        stats_calc = StatsCalculator()
        preprocessor = Preprocessor()
        df = pd.read_csv('data/train.tsv', sep='\t')
        col_names = [f'feature_{i}' for i in range(FEATURES)]
        df = preprocessor.split_features(df)
        df = preprocessor.f_to_int(df)
        col = random.choice(col_names)

        def wrapper(func):
            def inner(df, col, multiproc=False):
                start = time.time()
                res = func(df, col)
                end = time.time()
                print(f'\nResult of calculation: {res}')
                if multiproc:
                    print(f'Timing of calc in parallel: {end - start}')
                else:
                    print(f'Timing of sequential calc: {end - start}')
                return res

            return inner

        seq_calc = wrapper(stats_calc.calc_mean)
        res = seq_calc(df, col)

        parallel_calc = wrapper(stats_calc.calc_mean)
        parallel_calc(df, col, multiproc=True)

        true_value = np.mean(df[col])
        self.assertEqual(res, true_value, "Wrong mean calculation")
