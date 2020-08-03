import os
import warnings
import glob
from functools import reduce
from multiprocessing import Pool
from collections import defaultdict

import numpy as np
import pandas as pd

from scores import ZScore

warnings.filterwarnings('ignore')

FEATURES = 256
COL_NAMES = [f'feature_{i}' for i in range(FEATURES)]
MAP_FEAT_IDX = {f: i for i, f in enumerate(COL_NAMES)}
N_CORES = 8
NORM_MAPPER = {'Z-score': ZScore()}
LEN_DF = len(pd.read_csv('data/train.tsv', sep='\t', usecols=['id_job']))

feature_stats = defaultdict(lambda: defaultdict(list))
test_feature_stats = defaultdict(lambda: defaultdict(list))


class Preprocessor:

    def split_features(self, df):
        df['features'] = df['features'] \
            .transform(lambda x: x.split(','), axis=0) \
            .apply(lambda x: x[1:])

        df = self.make_cols(df, 'features')
        return df

    @staticmethod
    def make_cols(df, list_col):
        df[COL_NAMES] = pd.DataFrame(df.features.tolist(), index=df.index)
        df.drop(list_col, axis=1, inplace=True)
        return df

    @staticmethod
    def f_to_int(df):
        for col in COL_NAMES:
            df[col] = df[col].astype('int')
        return df


class StatsCalculator:

    @staticmethod
    def process(x, _=None):
        return x.sum()

    def calc_mean(self, df, col, multiproc=False):
        if multiproc:
            BATCH = len(df) // N_CORES
            pool = Pool(N_CORES)
            results = [pool.apply_async(self.process, (df.loc[i * BATCH:i * BATCH + BATCH, col].values,))
                       for i in range(N_CORES)]

            results = [x.get() for x in results]
            results = reduce(lambda a, b: a + b, results)
            results /= len(df)

            return results
        else:
            return df[col].values.mean()

    @staticmethod
    def calc_std(df, col):
        return df[col].values.std()


class FeatureAdder:

    @staticmethod
    def max_index_feature(df, new_feature='max_feature_2_index'):
        df[new_feature] = df.loc[:, COL_NAMES].idxmax(axis=1).values
        df[new_feature] = df[new_feature].map(lambda x: MAP_FEAT_IDX[x])
        return df

    @staticmethod
    def abs_mean_diff_feature(df, train_feature_stats,
                              new_feature='max_feature_2_abs_mean_diff'):
        cols = np.array(COL_NAMES)[df['max_feature_2_index'].values]
        mean_values = [train_feature_stats[c]['mean'] for c in cols]
        df[new_feature] = abs(df.apply(lambda x: max(x), axis=1) - mean_values)
        return df


def get_data():
    train_data, test_data = (sorted(glob.glob('data/train*')),
                             sorted(glob.glob('data/test*')))
    assert len(train_data) == len(test_data), "Unequal train/test sizes"
    return train_data, test_data


def standardize(df, NORM):
    standardizer = NORM_MAPPER.get(*NORM, "")
    for col in COL_NAMES:
        df[col] = standardizer.calculate(df[col].values, **feature_stats[col])
    return df


def process_data_chunk(x, test=False):
    preprocessor = Preprocessor()
    x = preprocessor.split_features(x)
    x = preprocessor.f_to_int(x)

    if not test:
        for col in COL_NAMES:
            feature_stats[col]['mean'].append(float(reduce(lambda a, b: a + b, x[col].values)))
    return x


def add_test_features(x):
    feature_adder = FeatureAdder()

    x = feature_adder.max_index_feature(x)

    x = feature_adder.abs_mean_diff_feature(x, feature_stats)
    return x


def find_mean_values(x):
    for col in COL_NAMES:
        feature_stats[col]['mean'].append(x[col].sum())


def submit_test_chunk(x, chunk, f_batch=2):
    x.rename(columns={f"feature_{i}": f"feature_{f_batch}_stand_{i}" for i in range(FEATURES)},
             inplace=True)
    import os

    if not os.path.isdir('output'):
        os.mkdir('output')

    if f_batch == 2:
        x.to_csv(f'output/test_proc_{chunk}.tsv', sep='\t', index=False)
    else:
        x.to_csv(f'output/test_proc_fbatch{f_batch}_{chunk}.tsv', sep='\t', index=False)


def calculate_mean():
    for col in COL_NAMES:
        feature_stats[col]['mean'] = float(reduce(lambda a, b: a if a > b else b, feature_stats[col]['mean']))
        feature_stats[col]['mean'] = feature_stats[col]['mean'] / LEN_DF


def find_std_values(x):
    for col in COL_NAMES:
        mean = feature_stats[col]['mean']
        feature_stats[col]['std'].append(reduce(lambda a, b: a + b, (x[col].values - mean) ** 2))


def find_train_stats(train_path, chunksize):
    for train_batch in pd.read_csv(train_path, sep='\t', chunksize=chunksize):
        x = process_data_chunk(train_batch)
        find_mean_values(x)

    calculate_mean()

    for train in pd.read_csv(train_path, sep='\t', chunksize=chunksize):
        x = process_data_chunk(train, test=True)
        find_std_values(x)

    for col in COL_NAMES:
        feature_stats[col]['std'] = float(reduce(lambda a, b: a + b, feature_stats[col]['std']))
        feature_stats[col]['std'] = feature_stats[col]['std'] / LEN_DF
        feature_stats[col]['std'] = np.sqrt(feature_stats[col]['std'])

    return feature_stats


def submit_results(test_path, chunksize, norm):
    for chunk, test in enumerate(pd.read_csv(test_path, sep='\t', chunksize=chunksize)):
        x = process_data_chunk(test, test=True)
        x = add_test_features(x)
        x = standardize(x, norm)
        submit_test_chunk(x, chunk)


def merge_results(sub):
    for i, df_path in enumerate(glob.glob('output/test_proc_*')):
        df = pd.read_csv(df_path, sep='\t')
        if i == 0:
            header = True
        else:
            header = False
        df.to_csv(sub, mode='a', sep='\t', header=header, index=False)
        os.remove(df_path)
