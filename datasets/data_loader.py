import pandas as pd
import numpy as np
from constants import PROCESSED_DIR


def load_train_set(project, datasets=['roadmap', 'eigen', 'regbase'], make_new=True):
    proj_loc = PROCESSED_DIR / project

    try:
        if make_new:
            raise Exception('Compiling train matrix.')
        train = pd.read_csv(proj_loc / f'matrix_train.csv')

    except Exception as e:
        train = pd.read_csv(proj_loc / 'train_label.csv')

        for ds in datasets:
            df = pd.read_csv(proj_loc / f'train_{ds}.csv')
            if 'ref' in df.columns:
                df.drop('ref', axis=1, inplace=True)
            train = pd.merge(train, df, on=['chr', 'pos'], suffixes=('', '__y'))
            train.drop(list(train.filter(regex='__y$')), axis=1, inplace=True)
        
        train.drop_duplicates(['chr', 'pos'], inplace=True)
        train.to_csv(proj_loc / 'matrix_train.csv', index=False)
    return train


def load_test_set(project, datasets=['roadmap', 'eigen', 'regbase'], make_new=True):
    proj_loc = PROCESSED_DIR / project

    try:
        if make_new:
            raise Exception('Compiling test matrix.')
        test = pd.read_csv(proj_loc / f'matrix_test.csv')

    except Exception as e:
        test = pd.read_csv(proj_loc / 'test_label.csv')

        for ds in datasets:
            df = pd.read_csv(proj_loc / f'test_{ds}.csv')
            if 'ref' in df.columns:
                df.drop('ref', axis=1, inplace=True)
            test = pd.merge(test, df, on=['chr', 'pos'], suffixes=('', '__y'))
            test.drop(list(test.filter(regex='__y$')), axis=1, inplace=True)

        test.drop_duplicates(['chr', 'pos'], inplace=True)
        test.to_csv(proj_loc / 'matrix_test.csv', index=False)
    return test


def load_train_neighbors(project, n_neigh=40, sample_res=25):
    proj_loc = PROCESSED_DIR / project
    fname = proj_loc / 'neighbors' / f'train_{n_neigh}_{sample_res}_E116.npy'
    return np.load(fname)
