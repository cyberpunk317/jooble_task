import random
import sys

import pandas as pd
import unittest2

sys.path.insert(0, '..')
from utils import Preprocessor

FEATURES = 256


class TestPreprocessor(unittest2.TestCase):

    def setUp(self):
        self.preprocessor = Preprocessor()
        self.col_names = [f'feature_{i}' for i in range(FEATURES)]

    def tearDown(self):
        self.preprocessor = None

    def _get_df(self):
        df = pd.read_csv('data/train.tsv', sep='\t')
        return df

    def test_split_features(self):
        """
        Test that single 'features' column splits into ['f0', 'f1', ..., 'f255']
        """
        df = self._get_df()

        len_df_cols = len(self.preprocessor.split_features(df).columns[1:])
        valid_res = FEATURES

        self.assertEqual(valid_res, len_df_cols, "Wrong feature splitting")

    def test_f_to_int(self):
        """
        Test that dataframe columns are properly converted to 'int64' dtype
        """
        df = self._get_df()
        df = self.preprocessor.split_features(df)
        col = random.choice(self.col_names)

        res = self.preprocessor.f_to_int(df)
        valid_res = pd.Series(df[col], dtype='int64')

        assert ((res[col].reset_index(drop=True) == valid_res.reset_index(drop=True)).all(),
                "Wrong feature parsing")
