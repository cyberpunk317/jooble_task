import argparse
import glob
from collections import defaultdict

import pandas as pd

from scores import ZScore
from utils import FeatureAdder, Preprocessor, StatsCalculator
from utils import get_data

FEATURES = 256
COL_NAMES = [f'feature_{i}' for i in range(FEATURES)]
NORM_MAPPER = {'Z-score': ZScore()}


def standardize(df, feature_stats):
    standardizer = NORM_MAPPER.get(*NORM, "")
    for col in COL_NAMES:
        df[col] = standardizer.calculate(df[col].values, **feature_stats[col])
    return df


def main():
    train_data, test_data = get_data()

    for f_batch, (train_path, test_path) in zip(FEATURE_BATCH, zip(train_data, test_data)):
        train = pd.read_csv(train_path, sep='\t')
        test = pd.read_csv(test_path, sep='\t')

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

        test = feature_adder.max_index_feature(test)

        test = feature_adder.abs_mean_diff_feature(test, feature_stats)
        test = standardize(test, feature_stats)

        test.rename(columns={f"feature_{i}": f"feature_{f_batch}_stand_{i}" for i in range(FEATURES)},
                    inplace=True)
        import os

        if not os.path.isdir('output'):
            os.mkdir('output')

        if f_batch == 2:
            test.to_csv('output/test_proc.tsv', sep='\t', index=False)
        else:
            test.to_csv(f'output/test_proc_batch{f_batch}.tsv', sep='\t', index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process tables with features.')
    parser.add_argument('-fb', '--feature_batches', nargs='+',
                        default='2',
                        help='feature batches to process')
    parser.add_argument('-n', '--normalization', action='append',
                        default=[],
                        help='normalization techniques to use in processing')
    args = parser.parse_args()
    fb, norm = [int(x) for x in args.feature_batches], args.normalization

    assert norm == ['Z-score'], "Only \'Z-score\' is supported"
    assert fb == [2], "Only 2nd set of features is supported"
    assert len(norm) == 1, "Only one normalization tech. for all features"
    FEATURE_BATCH = fb
    NORM = norm
    main()
    print('test_proc.tsv created successfully')
