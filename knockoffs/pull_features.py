import argparse
import os
import pandas as pd

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parent.parent))

from constants import PROCESSED_DIR
from utils.bed_utils import get_bed_from_mpra, load_bed_file, save_bed_file
from utils.data_utils import extract_roadmap
from kf_utils import *

proj_dir = 'knockoffs'
bedfiles = ['pos_causal_hg19', 'pos_correl_hg19', 'controls_hg19']


def pull_data(args):
    if args.roadmap or args.regbase or args.eigen:
        bed = load_bed_file(proj_dir, args.bed, chrXX=True)

    if args.roadmap:
        print('Extracting Roadmap: ')
        fname = PROCESSED_DIR / proj_dir / f'roadmap_extract_{args.bed}.tsv'
        extract_roadmap(bed, fname, args.bed, keep_rs_col=True)

    if args.regbase:
        print('Extracting regBase: ')
        fname = PROCESSED_DIR / proj_dir / f'regBase_extract_{args.bed}.tsv'
        extract_regbase(bed, fname)

    if args.eigen:
        print('Extracting Eigen: ')
        fname = PROCESSED_DIR / proj_dir / f'eigen_extract_{args.bed}.tsv'
        extract_eigen(bed, fname)


def process_regbase(version):
    df = pd.read_csv(PROCESSED_DIR / proj_dir / f'regBase_extract_{version}.tsv',
                     sep='\t')
    df = df.rename(columns={'#Chrom': 'chr',
                            'Pos_start': 'pos_start',
                            'Pos_end': 'pos_end',
                            'rs': 'variant'}) \
           .drop(['Ref', 'Alts'], axis=1)
    df[['chr', 'pos_start', 'pos_end']] = \
        df[['chr', 'pos_start', 'pos_end']].astype(int)
    return df


def process_eigen(version):
    df = pd.read_csv(PROCESSED_DIR / proj_dir / f'eigen_extract_{version}.tsv',
                     sep='\t')
    df = df.rename(columns={'position': 'pos_start',
                            'rs': 'variant'}) \
           .drop(['ref', 'alt'], axis=1)
    df[['chr', 'pos_start', 'pos_end']] = \
        df[['chr', 'pos_start', 'pos_end']].astype(int)
    return df


def process_roadmap(version):
    df = pd.read_csv(PROCESSED_DIR / proj_dir / f'roadmap_extract_{version}.tsv',
                     sep='\t')
    df['chr'] = df['chr'].map(lambda x: x[3:])
    df = df.rename(columns={'pos': 'pos_start', 'rs': 'variant'})
    df[['chr', 'pos_start']] = df[['chr', 'pos_start']].astype(int)
    return df


def merge_data(args):
    if not args.merge:
        return

    rbs = pd.concat([process_regbase(x) for x in bedfiles], axis=0)
    egs = pd.concat([process_eigen(x) for x in bedfiles], axis=0)

    df = pd.merge(rbs, egs, on='variant', suffixes=('', '__y'))
    assert all(df['pos_start'] == df['pos_start__y'])
    df.drop(list(df.filter(regex='__y$')), axis=1, inplace=True)
    print(df.shape)

    ros = pd.concat([process_roadmap(x) for x in bedfiles], axis=0)
    df = pd.merge(df, ros, on='variant', suffixes=('', '__y'))
    assert all(df['pos_start'] == df['pos_start__y'])
    df.drop(list(df.filter(regex='__y$')), axis=1, inplace=True)
    print(df.shape)

    def var_class(x):
        if 'causal' in x:
            return 'W_KS'
        if 'correl' in x:
            return 'P_KS'
        if 'neg' in x:
            return 'control'
    
    df['label'] = df['variant'].map(lambda x: var_class(x))
    
    preds = [x for x in df.columns if x not in ['chr', 'pos_start', 'pos_end', 'label', 'variant']]
    df = df.loc[:, ['chr', 'pos_start', 'pos_end', 'label', 'variant'] + preds]
    df.to_csv(PROCESSED_DIR / proj_dir / 'knockoffs_predictors.tsv', sep='\t', index=False)
    print(df.shape)
    # a = pd.read_csv(PROCESSED_DIR / proj_dir / 'roadmap_extract_pos_correl_hg19.tsv', sep='\t')
    # a['rs'] = a['rs'].map(lambda x: x.replace('causal', 'correl'))
    # a.to_csv(PROCESSED_DIR / proj_dir / 'roadmap_extract_pos_correl_hg19.tsv', sep='\t', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bed', '-b', required=True, choices=bedfiles,
                        help='bedfile to extract')
    parser.add_argument('--roadmap', '-r', default=False, action='store_true',
                        help='extract Roadmap data')
    parser.add_argument('--regbase', '-rb', default=False, action='store_true',
                        help='extract regBase data')
    parser.add_argument('--eigen', '-e', default=False, action='store_true',
                        help='extract Eigen data')
    parser.add_argument('--merge', '-m', default=False, action='store_true',
                        help='merge datasets together after extraction')
    args = parser.parse_args()

    pull_data(args)
    merge_data(args)
