import os
import pandas as pd
import numpy as np
import io, subprocess
from tqdm import tqdm
# from sklearn.utils import resample

from utils.rollmean_utils import pull_features
from constants import *


def _read_bed(x, **kwargs):
    """ Helper function to parse output from a tabix query """
    return pd.read_csv(x, sep=r'\s+', header=None, index_col=False, **kwargs)


def extract_eigen(bedfile, outpath):
    if os.path.exists(outpath):
        os.remove(outpath)

    head_file = EIGEN_DIR / 'header_noncoding.txt'
    with open(head_file, 'r') as f:
        header = f.read()
    header = header.strip('\n').split('  ')

    command = f'tabix {EIGEN_DIR}/{EIGEN_BASE} '

    with open(outpath, 'w') as fp:
        fp.write('\t'.join(header) + '\n')

        for row in tqdm(bedfile.itertuples()):
            args = f'{row.chr}:{row.pos}-{row.pos}'
            full_cmd = command.replace('XX', str(row.chr)) + args

            p = subprocess.Popen(full_cmd, shell=True, stdout=subprocess.PIPE)
            stdout, _ = p.communicate()

            if len(stdout) > 0:
                features = _read_bed(io.StringIO(stdout.decode('utf-8')))
                features = features.astype(str).values.tolist()
                for row in features:
                    fp.write('\t'.join(row) + '\n')


def extract_regbase(bedfile, outpath):
    """ Extract rows from regBase file corresponding to variants in
    input bedfile. Command line piping adapted from
    https://github.com/mulinlab/regBase/blob/master/script/regBase_predict.py

    """
    if os.path.exists(outpath):
        os.remove(outpath)

    head_file = REGBASE_DIR / REGBASE
    header = pd.read_csv(head_file, sep=r'\s+', nrows=3).columns
    header = header.values.tolist()

    command = f'tabix {REGBASE_DIR}/{REGBASE} '

    with open(outpath, 'w') as fp:
        fp.write('\t'.join(header) + '\n')

        for row in tqdm(bedfile.itertuples()):
            args = f'{row.chr}:{row.pos}-{row.pos}'
            full_cmd = command + args

            p = subprocess.Popen(full_cmd, shell=True, stdout=subprocess.PIPE)
            stdout, _ = p.communicate()

            if len(stdout) > 0:
                features = _read_bed(io.StringIO(stdout.decode('utf-8')))
                features = features.astype(str).values.tolist()
                for row in features:
                    fp.write('\t'.join(row) + '\n')


def extract_roadmap(bedfile_loc, outpath, project):
    if os.path.exists(outpath):
        os.remove(outpath)

    if project in STANDARD_MPRA:
        data = load_mpra_data(project)
        data.drop(['rs', 'Label'], axis=1, inplace=True)
        data.to_csv(outpath, sep='\t', index=False)

    elif project == 'mpra_deseq2':
        pull_features(bedfile_loc)


def clean_eigen_data(filename):
    eigen = pd.read_csv(filename, sep='\t', na_values='.')

    # manually convert scores to float due to NaN processing
    eigen.iloc[:, 4:] = eigen.iloc[:, 4:].astype(float)

    # average over variant substitutions
    eigen = eigen.rename(columns={'chr': 'chr', 'position': 'pos'}) \
                    .drop('alt', axis=1) \
                    .groupby(['chr', 'pos', 'ref'], as_index=False) \
                    .mean()
    return eigen


def clean_regbase_data(filename):
    regbase = pd.read_csv(filename, sep='\t', na_values='.')

    # manually convert scores to float due to NaN processing
    regbase.iloc[:, 5:] = regbase.iloc[:, 5:].astype(float)

    # average over variant substitutions
    regbase = regbase.rename(columns={'#Chrom': 'chr', 'Pos_end': 'pos', 'Ref': 'ref'}) \
                        .drop(['Pos_start', 'Alts'], axis=1) \
                        .groupby(['chr', 'pos', 'ref'], as_index=False) \
                        .mean()
    return regbase


def load_mpra_data(dataset, benchmarks=False):
    ''' processes raw MPRA data files and optionally benchmark files '''
    mpra_files = MPRA_TABLE[dataset]
    data = pd.read_csv(MPRA_DIR / mpra_files[0], sep='\t')

    if dataset == 'mpra_deseq2':
        data.rename(columns={'chrom': 'chr'}, inplace=True)
        data['rs'] = data[['chr', 'pos']].apply(lambda x: ':'.join(x), axis=1)

    ### setup mpra/epigenetic data ###
    data_prepared = (data.assign(chr=data['chr'].apply( lambda x: int(x[3:]) ))
                         .sort_values(['chr', 'pos'])
                         .reset_index(drop=True))

    if benchmarks:
        if not mpra_files[1]:
            return data_prepared, None

        bench = pd.read_csv(MPRA_DIR / mpra_files[1], sep='\t')

        ### setup benchmark data ###
        # modify index column to extract chr and pos information
        chr_pos = (bench.reset_index()
                        .loc[:, 'index']
                        .str
                        .split('-', expand=True)
                        .astype(int))

        # update benchmark data with chr and pos columns
        bench_prepared = (bench.reset_index()
                            .assign(chr=chr_pos[0])
                            .assign(pos=chr_pos[1])
                            .drop('index', axis=1)
                            .sort_values(['chr', 'pos'])
                            .reset_index(drop=True))

        # put chr and pos columns in front for readability
        reordered_columns = ['chr', 'pos'] + bench_prepared.columns.values.tolist()[:-2]
        bench_prepared = bench_prepared[reordered_columns]
        return data_prepared, bench_prepared

    return data_prepared


def get_tissue_scores(data, tissue='E116'):
    ids = ['chr', 'pos', 'rs', 'Label']
    feats = [x for x in data.columns if x not in ids and tissue in x]

    df_select = data[ids + feats]
    return df_select


def split_train_test(data, test_frac=0.2, seed=None):
    if seed:
        np.random.seed(seed)

    m  = data.shape[0]
    test_size = int(test_frac * m)
    perm = np.random.permutation(m)

    test = data.iloc[perm[:test_size], :]
    train = data.iloc[perm[test_size:], :]

    return train, test


def split_train_dev_test(data, dev_frac, test_frac, seed=None):
    if seed:
        np.random.seed(seed)

    m  = data.shape[0]
    dev_size = int(dev_frac * m)
    test_size = int(test_frac * m)
    perm = np.random.permutation(m)

    dev = data.iloc[perm[:dev_size], :]
    test = data.iloc[perm[dev_size:dev_size + test_size], :]
    train = data.iloc[perm[dev_size + test_size:], :]

    return train, dev, test


def rearrange_by_epigenetic_marker(df):

    marks = ['DNase','H3K27ac','H3K27me3','H3K36me3','H3K4me1','H3K4me3','H3K9ac','H3K9me3']
    nums = [x for x in range(1, 130) if x not in [60, 64]]

    cols = [x + '-E{:03d}'.format(y) for x in marks for y in nums]

    return df.loc[:, cols]


def downsample_negatives(train, p):

    train_negs = train[train.Label == 0]
    train_pos = train[train.Label == 1]

    train_downsample = resample(train_negs,
                                replace=False,
                                n_samples=int(p * train_negs.shape[0]),
                                random_state=111)

    train_balanced = pd.concat([train_downsample, train_pos])
    return train_balanced.sample(frac=1, axis=0)


def upsample_positives(train, scale):

    train_negs = train[train.Label == 0]
    train_pos = train[train.Label == 1]

    train_upsample = resample(train_pos,
                              replace=True,
                              n_samples=scale * train_pos.shape[0],
                              random_state=111)

    train_balanced = pd.concat([train_negs, train_upsample])
    return train_balanced.sample(frac=1, axis=0)
