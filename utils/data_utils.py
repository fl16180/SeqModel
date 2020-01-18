import os
import pandas as pd
import numpy as np
from sklearn.utils import resample
from subprocess import check_call

from constants import *


# chr.index<-1

# header.eigen<-read.table("/home/skardia_lab/zihuai/FST/Score/header_noncoding.txt",header=F)

# Eigen_hg19_coding_annot_04092016.tab.bgz

# dir<-paste0('tabix /oak/stanford/groups/zihuai/FST/Score/Eigen_hg19_noncoding_annot_chr',chr.index,'.tab.bgz ',chr.index,':',
#             1000000,'-',1005000,' >temp',chr.index,'.txt')

# dir<-paste0('tabix /oak/stanford/groups/zihuai/FST/Score/Eigen_hg19_coding_annot_04092016.tab.bgz ',chr.index,':',
#             1000000,'-',1020000,' >temp',chr.index,'.txt')

# system(dir)
# temp.data<-try(as.matrix(read.table(paste0('temp',chr.index,'.txt'))),silent = T)
# if(class(temp.data) == "try-error") {next}
# temp.value<-mean(as.numeric(temp.data[,which(header.eigen=='Eigen-raw')]))
# temp.result<-c(temp.result,temp.value)

def extract_eigen_data(bedfile):
    groups = bedfile.groupby('chr')

    head_file = EIGEN_DIR / 'header_noncoding.txt'
    with open(head_file, 'r') as f:
        header = f.read()
    header = header.strip('\n').split('  ')

    command = f'tabix {EIGEN_DIR}/Eigen_hg19_noncoding_annot_chr1.tab.bgz '




def load_mpra_data(dataset, benchmarks=False):
    ''' processes raw MPRA data files and optionally benchmark files '''
    mpra_files = MPRA_TABLE[dataset]
    data = pd.read_csv(MPRA_DIR / mpra_files[0]), delimiter="\t")

    ### setup mpra/epigenetic data ###
    data_prepared = (data.assign(chr=data['chr'].apply( lambda x: int(x[3:]) ))
                         .sort_values(['chr','pos'])
                         .reset_index(drop=True))

    if benchmarks:
        bench = pd.read_csv(MPRA_DIR / mpra_files[1]), delimiter="\t")

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
                            .sort_values(['chr','pos'])
                            .reset_index(drop=True))

        # put chr and pos columns in front for readability
        reordered_columns = ['chr','pos'] + bench_prepared.columns.values.tolist()[:-2]
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
