import argparse

from utils import *

NORM = []
SUB = 'output/test_proc.tsv'


def main():
    train_data, test_data = get_data()
    chunksize = 10000
    for f_batch, (train_path, test_path) in zip(FEATURE_BATCH, zip(train_data, test_data)):
        train_stats = find_train_stats(train_path, chunksize)

        submit_results(train_stats, test_path, chunksize, NORM)

        merge_results(SUB)


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
