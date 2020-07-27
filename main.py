import argparse
from collections import defaultdict

import pandas as pd

from scores import ZScore
from utils import FeatureAdder, Preprocessor, StatsCalculator

TRAIN = 'data/train.tsv'
TEST = 'data/test.tsv'
SUB = 'data/test_proc.tsv'
FEATURES = 256
COL_NAMES = [f'feature_{i}' for i in range(FEATURES)]


def standardize(df, feature_stats):
    zscorer = ZScore()
    for col in COL_NAMES:
        df[col] = zscorer.calculate(df[col].values, **feature_stats[col])
    return df


def main():
    train = pd.read_csv(TRAIN, sep='\t')
    test = pd.read_csv(TEST, sep='\t')

    preprocessor = Preprocessor()
    train = preprocessor.split_features(train)
    train = preprocessor.f_to_int(train)
    test = preprocessor.split_features(test)
    test = preprocessor.f_to_int(test)

    stats_calc = StatsCalculator()
    feature_stats = defaultdict(defaultdict)

    for col in COL_NAMES:
        feature_stats[col]['mean'] = stats_calc.calc_mean(train, col)
        feature_stats[col]['std'] = stats_calc.calc_std(train, col)
        feature_stats[col]['max'] = train[col].max()

    feature_adder = FeatureAdder()

    train = feature_adder.max_index_feature(train)
    test = feature_adder.max_index_feature(test)

    train = feature_adder.abs_mean_diff_feature(train, feature_stats)
    test = feature_adder.abs_mean_diff_feature(test, feature_stats)
    test = standardize(test, feature_stats)

    test.rename(columns={f"feature_{i}": f"feature_2_stand_{i}" for i in range(FEATURES)}, inplace=True)
    test.to_csv(SUB, sep='\t', index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some tables.')
    parser.add_argument('--factor',
                        help='no. of batch of features to process')
    parser.add_argument('--norm',
                        default="Z-score",
                        help='normalization technique to use in processing')

    args = parser.parse_args()
    f, norm = int(args.factor), args.norm
    assert norm == 'Z-score', "Only \'Z-score\' is supported"
    assert f == 2, "Only 2-nd set of features is supported"
    main()
    print('test_proc.tsv created successfully')
