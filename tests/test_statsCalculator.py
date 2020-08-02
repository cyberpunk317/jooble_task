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

    def setUp(self):
        self.stats_calc = StatsCalculator()
        self.preprocessor = Preprocessor()
        self.col_names = [f'feature_{i}' for i in range(FEATURES)]

    def tearDown(self):
        self.stats_calc = None
        self.preprocessor = None

    def _get_df(self):
        df = pd.read_csv('data/train.tsv', sep='\t')
        df = self.preprocessor.split_features(df)
        df = self.preprocessor.f_to_int(df)
        return df

    def test_mean_calc(self):
        df = self._get_df()
        col = random.choice(self.col_names)

        res = self.stats_calc.calc_mean(df, col)
        valid_res = np.mean(df[col])

        self.assertEqual(res, valid_res, "Wrong mean calculation")

    def test_std_calc(self):
        df = self._get_df()
        col = random.choice(self.col_names)

        res = self.stats_calc.calc_std(df, col)
        valid_res = np.std(df[col])

        self.assertEqual(res, valid_res, "Wrong std calculation")

    def test_speed(self):
        """
        Test parallelized mean calculation
        """
        df = self._get_df()
        col = random.choice(self.col_names)

        def wrapper(func):
            def inner(df, col, multiproc=False):
                start = time.time()
                result = func(df, col)
                end = time.time()
                print(f'\nResult of calculation: {result}')
                if multiproc:
                    print(f'Timing of calc in parallel: {end - start}')
                else:
                    print(f'Timing of sequential calc: {end - start}')
                return result
            return inner

        seq_calc = wrapper(self.stats_calc.calc_mean)
        res = seq_calc(df, col)
        parallel_calc = wrapper(self.stats_calc.calc_mean)
        parallel_calc(df, col, multiproc=True)

        true_value = np.mean(df[col])
        self.assertEqual(res, true_value, "Wrong mean calculation")
