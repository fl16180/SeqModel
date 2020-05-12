import pandas as pd
import numpy as np
from constants import PROCESSED_DIR


def load_data_set(project, split='all',
                  datasets=['roadmap', 'eigen', 'regbase'],
                  make_new=True):
    ''' Combines processed data sources to a compiled matrix.
    split: ('train', 'test', 'all') matching the saved label names
    '''
    proj_loc = PROCESSED_DIR / project

    try:
        if make_new:
            raise Exception(f'Compiling {split} matrix.')
        dat = pd.read_csv(proj_loc / f'matrix_{split}.csv')

    except Exception as e:
        dat = pd.read_csv(proj_loc / f'{split}_label.csv')

        for ds in datasets:
            df = pd.read_csv(proj_loc / f'{split}_{ds}.csv')
            if 'ref' in df.columns:
                df.drop('ref', axis=1, inplace=True)
            dat = pd.merge(dat, df,
                           on=['chr', 'pos'], suffixes=('', '__y'))
            dat.drop(list(dat.filter(regex='__y$')), axis=1, inplace=True)
        
        dat.drop_duplicates(['chr', 'pos'], inplace=True)
        dat.to_csv(proj_loc / f'matrix_{split}.csv', index=False)
    return dat


def load_neighbors_set(project, split='all',
                       n_neigh=40, sample_res=25, tissue='E116'):
    proj_loc = PROCESSED_DIR / project / 'neighbors'
    fname = proj_loc / f'{split}_{n_neigh}_{sample_res}_E116.npy'
    return np.load(fname)
