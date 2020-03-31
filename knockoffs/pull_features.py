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
bedfiles = ['top_W_matched_hg19', 'top_P_matched_hg19']


def pull_data(args):
    if args.genonet or args.roadmap or args.regbase or args.eigen:
        bed = load_bed_file(proj_dir, args.bed, chrXX=True)

    if args.genonet:
        print('Extracting GenoNet: ')
        fname = PROCESSED_DIR / proj_dir / f'GenoNet_{args.bed}.tsv'
        extract_genonet(bed.copy(), fname, summarize=args.stat)

    if args.regbase:
        print('Extracting regBase: ')
        fname = PROCESSED_DIR / proj_dir / f'regBase_{args.bed}.tsv'
        extract_regbase(bed.copy(), fname, summarize=args.stat)

    if args.eigen:
        print('Extracting Eigen: ')
        fname = PROCESSED_DIR / proj_dir / f'eigen_{args.bed}.tsv'
        extract_eigen(bed.copy(), fname, summarize=args.stat)

    if args.roadmap:
        print('Extracting Roadmap: ')
        fname = PROCESSED_DIR / proj_dir / f'roadmap_{args.bed}.tsv'
        extract_roadmap(bed.copy(), fname, args.bed,
                        keep_rs_col=True, summarize=args.stat)


def process_genonet(version):
    df = pd.read_csv(PROCESSED_DIR / proj_dir / f'GenoNet_{version}.tsv',
                     sep='\t')
    df['chr'] = df['chr'].map(lambda x: x[3:])
    df = df.rename(columns={'rs': 'variant'}).drop('reg_id', axis=1)
    df[['chr', 'pos_start', 'pos_end']] = \
        df[['chr', 'pos_start', 'pos_end']].astype(int)

    return df


def process_regbase(version):
    df = pd.read_csv(PROCESSED_DIR / proj_dir / f'regBase_{version}.tsv',
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
    df = pd.read_csv(PROCESSED_DIR / proj_dir / f'eigen_{version}.tsv',
                     sep='\t')
    df = df.rename(columns={'position': 'pos_start',
                            'rs': 'variant'}) \
           .drop(['ref', 'alt'], axis=1)
    df[['chr', 'pos_start', 'pos_end']] = \
        df[['chr', 'pos_start', 'pos_end']].astype(int)
    return df


def process_roadmap(version):
    df = pd.read_csv(PROCESSED_DIR / proj_dir / f'roadmap_{version}.tsv',
                     sep='\t')
    df['chr'] = df['chr'].map(lambda x: x[3:])
    df = df.rename(columns={'pos': 'pos_start', 'rs': 'variant'})
    df[['chr', 'pos_start']] = df[['chr', 'pos_start']].astype(int)
    return df


def merge_data(args):
    if not args.merge:
        return

    rb = process_regbase(args.bed)
    eg = process_eigen(args.bed)
    gn = process_genonet(args.bed)
    ro = process_roadmap(args.bed)

    df = pd.merge(rb, eg, on='variant', suffixes=('', '__y'))
    assert all(df['pos_start'] == df['pos_start__y'])
    df.drop(list(df.filter(regex='__y$')), axis=1, inplace=True)
    print(df.shape)

    df = pd.merge(df, gn, on='variant', suffixes=('', '__y'))
    assert all(df['pos_start'] == df['pos_start__y'])
    df.drop(list(df.filter(regex='__y$')), axis=1, inplace=True)
    print(df.shape)

    df = pd.merge(df, ro, on='variant', suffixes=('', '__y'))
    assert all(df['pos_start'] == df['pos_start__y'])
    df.drop(list(df.filter(regex='__y$')), axis=1, inplace=True)
    print(df.shape)

    def var_class(x):
        if 'topW' in x:
            return 'W_KS'
        if 'topP' in x:
            return 'P_KS'
        if 'neg' in x:
            return 'control'
    
    df['label'] = df['variant'].map(lambda x: var_class(x))
    
    preds = [x for x in df.columns if x not in ['chr', 'pos_start', 'pos_end', 'label', 'variant']]
    
    hg38_bed = args.bed.split('_hg19')[0]
    ref = load_bed_file(proj_dir, hg38_bed, chrXX=True,
                        extra_cols=['W_KS', 'P_KS'])
    ref = ref.rename(columns={'pos': 'pos_start_hg38',
                              'pos_end': 'pos_end_hg38',
                              'rs': 'variant'})
    print(df.head())
    print(ref.head())
    df = pd.merge(df, ref, on=['chr', 'variant'])
    
    df = df.loc[:, ['chr', 'pos_start', 'pos_end', 'pos_start_hg38',
                    'pos_end_hg38', 'label', 'variant', 'W_KS', 'P_KS'] + preds]
    df = df.rename(columns={'pos_start': 'pos_start_hg19', 'pos_end': 'pos_end_hg19'})
    df.to_csv(PROCESSED_DIR / proj_dir / f'all_{args.stat}_{args.bed}.tsv', sep='\t', index=False)
    print(df.head())
    print(df.shape)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bed', '-b', required=True, choices=bedfiles,
                        help='bedfile to extract')
    parser.add_argument('--genonet', '-g', default=False, action='store_true',
                        help='extract GenoNet data')
    parser.add_argument('--roadmap', '-r', default=False, action='store_true',
                        help='extract Roadmap data')
    parser.add_argument('--regbase', '-rb', default=False, action='store_true',
                        help='extract regBase data')
    parser.add_argument('--eigen', '-e', default=False, action='store_true',
                        help='extract Eigen data')
    parser.add_argument('--stat', '-s', default='mean', type=str,
                        help='summary statistic (mean or max) to extract')
    parser.add_argument('--merge', '-m', default=False, action='store_true',
                        help='merge datasets together after extraction')
    args = parser.parse_args()

    pull_data(args)
    merge_data(args)
