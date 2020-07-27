import argparse
from collections import defaultdict
import pandas as pd
import numpy as np
from utils import Preprocessor, FeatureAdder, StatsCalculator

TRAIN = 'data/train.tsv'
TEST = 'data/test.tsv'
SUB = 'data/test_proc.tsv'
FEATURES = 256

def main():
    train = pd.read_csv(TRAIN, sep='\t')
    test = pd.read_csv(TEST, sep='\t')

    preprocessor = Preprocessor()
    train = preprocessor.split_features(train)
    train = preprocessor.f_to_int(train)
    test = preprocessor.split_features(test)
    test = preprocessor.f_to_int(test)

    stats_calc = StatsCalculator()
    features_mean_std = defaultdict(defaultdict)

    col_names = [f'feature_{i}' for i in range(FEATURES)]
    for col in col_names:
        features_mean_std[col]['mean'] = stats_calc.calc_mean(train, col)
        features_mean_std[col]['std'] = stats_calc.calc_std(train, col)
        features_mean_std[col]['max'] = train[col].max()

    feature_adder = FeatureAdder()
    # map features to their indices
    map_feat_idx = {f: i for i, f in enumerate(col_names)}

    train = feature_adder.max_index_feature(train)
    test = feature_adder.max_index_feature(test)

    train = feature_adder.abs_mean_diff_feature(train, features_mean_std)
    test = feature_adder.abs_mean_diff_feature(test, features_mean_std)

    test.drop(col_names, axis=1, inplace=True)
    sub = pd.read_csv(TEST, sep='\t').merge(test, on='id_job')

    sub.to_csv(SUB, sep='\t', index=False)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process some tables.')
    parser.add_argument('--factor',
                    help='no. of batch of features to process')
    parser.add_argument('--norm',
                    default="Z-score",
                    help='normalization technique to use in processing')

    args = parser.parse_args()
    f, norm = int(args.factor), args.norm
    assert norm == 'Z-score', "Not this time"
    assert f == 2, "Not this time as well"
    main()
    print('Success')
