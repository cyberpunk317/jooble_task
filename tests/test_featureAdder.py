import sys

import numpy as np
import pandas as pd
import unittest2

sys.path.insert(0, '..')
from utils import StatsCalculator, Preprocessor, FeatureAdder, find_train_stats, find_test_max_values

FEATURES = 256


class TestFeatureAdder(unittest2.TestCase):

    def setUp(self):
        self.stats_calc = StatsCalculator()
        self.preprocessor = Preprocessor()
        self.feature_adder = FeatureAdder()
        self.col_names = [f'feature_{i}' for i in range(FEATURES)]

    def tearDown(self):
        self.stats_calc = None
        self.preprocessor = None
        self.feature_adder = None

    def _get_df(self):
        df = pd.read_csv('data/train.tsv', sep='\t')
        df = self.preprocessor.split_features(df)
        df = self.preprocessor.f_to_int(df)
        return df

    def test_max_index_feature(self):
        """
        Test that new feature 'max_feature_2_index' lies in proper range and has dtype 'int64'
        """
        df = self._get_df()
        new_feature = 'max_feature_2_index'

        df = self.feature_adder.max_index_feature(df)
        valid_range, valid_dtype = (0, 255), 'int64'

        assert df[new_feature].between(*valid_range).all() and df[new_feature].dtype == valid_dtype, \
            "max_feature_2_index feature not in range OR has wrong dtype"

    def test_abs_mean_diff_feature(self):
        """
        Test that new feature 'max_feature_2_abs_mean_diff' is valid
        """
        df = self._get_df()
        df = self.feature_adder.max_index_feature(df)
        new_feature = 'max_feature_2_abs_mean_diff'
        cols = np.array(self.col_names)[df['max_feature_2_index'].values]
        train_stats = find_train_stats('data/train.tsv', chunksize=10000)
        test_stats = find_test_max_values('data/test.tsv', chunksize=10000)
        df = self.feature_adder.abs_mean_diff_feature(df, train_stats, test_stats)
        results = []

        for i, col in enumerate(cols):
            # keep in mind outliers in test data
            lower_bound, upper_bound = 0, max(train_stats[col]['std'],
                                              test_stats[col]['max'] - train_stats[col]['mean'])
            results.append(lower_bound <= df[new_feature][i] <= upper_bound)

        self.assertTrue(np.all(results),
                        "max_feature_2_index feature not in range OR has wrong dtype")
