import warnings
from functools import reduce
from multiprocessing import Pool

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

FEATURES = 256
COL_NAMES = [f'feature_{i}' for i in range(FEATURES)]
MAP_FEAT_IDX = {f: i for i, f in enumerate(COL_NAMES)}


class Preprocessor:

    def split_features(self, df):
        df['features'] = df['features'] \
            .transform(lambda x: x.split(','), axis=0) \
            .apply(lambda x: x[1:])

        df = self.make_cols(df, 'features')
        return df

    def make_cols(self, df, list_col):
        COL_NAMES = [f'feature_{i}' for i in range(FEATURES)]
        df[COL_NAMES] = pd.DataFrame(df.features.tolist(), index=df.index)
        df.drop(list_col, axis=1, inplace=True)
        return df

    def f_to_int(self, df):
        COL_NAMES = [f'feature_{i}' for i in range(FEATURES)]
        for col in COL_NAMES:
            df[col] = df[col].astype('int')
        return df


def process(x, _=None):
    return x.sum()


class StatsCalculator:

    def calc_mean(self, df, col, multiproc=False):
        if multiproc:
            N_CORES = 8
            DF_LENGTH = len(df)
            BATCH = DF_LENGTH // N_CORES

            pool = Pool(N_CORES)
            results = [pool.apply_async(process, (df.loc[i*BATCH:i*BATCH+BATCH, col].values,))
                       for i in range(N_CORES)]

            results = [x.get() for x in results]
            results = reduce(lambda a, b: a+b, results)
            results /= DF_LENGTH

            return results
        else:
            return df[col].values.mean()

    def calc_std(self, df, col):
        return df[col].values.std()


class FeatureAdder:

    def max_index_feature(self, df, new_feature='max_feature_2_index'):
        df[new_feature] = df.loc[:, COL_NAMES].idxmax(axis=1).values
        df[new_feature] = [MAP_FEAT_IDX[x] for x in df[new_feature]]
        return df

    def abs_mean_diff_feature(self, df, features_mean_std, new_feature='max_feature_2_abs_mean_diff'):
        df[new_feature] = None
        cols = np.array(COL_NAMES)[df['max_feature_2_index'].values]
        for i, c in enumerate(cols):
            df[new_feature][i] = abs(features_mean_std[c]['max'] - features_mean_std[c]['mean'])
        return df
