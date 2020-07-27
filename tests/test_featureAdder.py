import unittest2
from utils import FeatureAdder

import sys
sys.path.insert(0, '..')


class TestFeatureAdder(unittest2.TestCase):

    def test1(self):
        stats_calc = FeatureAdder()

        res = None

        assert not res, 'Failed'


