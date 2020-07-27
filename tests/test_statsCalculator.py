import sys
import unittest2
from utils import StatsCalculator

sys.path.insert(0, '..')


class TestStatsCalculator(unittest2.TestCase):

    def test1(self):
        stats_calc = StatsCalculator()

        res = None

        assert not res, 'Failed'
