import unittest2
from utils import Preprocessor

import sys
sys.path.insert(0, '..')


class TestPreprocessor(unittest2.TestCase):

    def test1(self):
        stats_calc = Preprocessor()

        res = None

        assert not res, 'Failed'


