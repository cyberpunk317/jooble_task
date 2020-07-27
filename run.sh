#!/usr/bin/bash
echo "Testing StatsCalculator..." 
python -m unittest2 tests/test_statsCalculator.py
echo "Testing other classes..."
python -m unittest2 discover ./tests
python main.py --factor 2 --norm Z-score
