#!/usr/bin/bash
python -m unittest discover ./tests
python main.py --factor 2 --norm Z-score
