import warnings
from functools import reduce
from multiprocessing import Pool

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

FEATURES = 256
COL_NAMES = [f'feature_{i}' for i in range(FEATURES)]
MAP_FEAT_IDX = {f: i for i, f in enumerate(COL_NAMES)}
N_CORES = 8


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


def process(x, _=None):
    return x.sum()


class StatsCalculator:

    @staticmethod
    def calc_mean(df, col, multiproc=False):
        if multiproc:
            BATCH = len(df) // N_CORES
            pool = Pool(N_CORES)
            results = [pool.apply_async(process, (df.loc[i*BATCH:i*BATCH+BATCH, col].values,))
                       for i in range(N_CORES)]

            results = [x.get() for x in results]
            results = reduce(lambda a, b: a+b, results)
            results /= len(df)

            return results
        else:
            return df[col].values.mean()

    @staticmethod
    def calc_std(df, col):
        return df[col].values.std()


class FeatureAdder:

    def max_index_feature(self, df, new_feature='max_feature_2_index'):
        df[new_feature] = df.loc[:, COL_NAMES].idxmax(axis=1).values
        df[new_feature] = [MAP_FEAT_IDX[x] for x in df[new_feature]]
        return df

    def abs_mean_diff_feature(self, df, feature_stats, new_feature='max_feature_2_abs_mean_diff'):
        df[new_feature] = None
        cols = np.array(COL_NAMES)[df['max_feature_2_index'].values]
        for i, c in enumerate(cols):
            df[new_feature][i] = abs(feature_stats[c]['max'] - feature_stats[c]['mean'])
        return df
