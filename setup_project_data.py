import argparse
import os
import pandas as pd
from utils.bed_utils import get_bed_from_mpra, load_bed_file, save_bed_file
from utils.data_utils import extract_roadmap, extract_eigen, extract_regbase, split_train_test, load_mpra_data
from constants import PROCESSED_DIR


PROJ_CHOICES = ['mpra_e116']


def setup(args):

    project_dir = PROCESSED_DIR / args.project

    # load or create variants bedfile
    try:
        if args.bed:
            raise Exception('Overwriting bedfile')

        bedfile = load_bed_file(args.project)
        print(f'Loaded bedfile from {args.project}')
    except Exception as e:
        bedfile = get_bed_from_mpra(args.project)
        save_bed_file(bedfile, args.project)
        print(f'Generated new bedfile in {args.project}')

    if args.roadmap:
        print('Extracting Roadmap: ')
        fname = project_dir / 'roadmap_extract.tsv'
        extract_roadmap(bedfile, fname, args.project)

    if args.regbase:
        print('Extracting regBase: ')
        fname = project_dir / 'regBase_extract.tsv'
        extract_regbase(bedfile, fname)

    if args.eigen:
        print('Extracting Eigen: ')
        fname = project_dir / 'eigen_extract.tsv'
        extract_eigen(bedfile, fname)

    # split data into train/test sets
    if args.split:
        bed_train, bed_test = split_train_test(bedfile, test_frac=0.2, seed=args.seed)

        # generate labels
        mpra = load_mpra_data(args.project)
        y_train = pd.merge(mpra, bed_train, on=['chr', 'pos']).loc[:, ['chr', 'pos', 'Label']]
        y_train.to_csv(project_dir / 'train_label.csv', sep=',', index=False)
        y_test = pd.merge(mpra, bed_test, on=['chr', 'pos']).loc[:, ['chr', 'pos', 'Label']]
        y_test.to_csv(project_dir / 'test_label.csv', sep=',', index=False)

        if os.path.exists(project_dir / 'roadmap_extract.tsv'):
            roadmap = pd.read_csv(project_dir / 'roadmap_extract.tsv', sep='\t')
            r_train = pd.merge(roadmap, bed_train[['chr', 'pos']], on=['chr', 'pos'])
            r_train.to_csv(project_dir / 'train_roadmap.csv', sep=',', index=False)
            r_test = pd.merge(roadmap, bed_test[['chr', 'pos']], on=['chr', 'pos'])
            r_test.to_csv(project_dir / 'test_roadmap.csv', sep=',', index=False)

        if os.path.exists(project_dir / 'regBase_extract.tsv'):
            regbase = pd.read_csv(project_dir / 'regBase_extract.tsv', sep='\t', na_values='.')

            # manually convert scores to float due to NaN processing
            regbase.iloc[:, 5:] = regbase.iloc[:, 5:].astype(float)

            # average over variant substitutions
            regbase = regbase.rename(columns={'#Chrom': 'chr', 'Pos_end': 'pos', 'Ref': 'ref'}) \
                             .drop(['Pos_start', 'Alts'], axis=1) \
                             .groupby(['chr', 'pos', 'ref'], as_index=False) \
                             .mean()

            r_train = pd.merge(regbase, bed_train[['chr', 'pos']])
            r_train.to_csv(project_dir / 'train_regbase.csv', index=False)
            r_test = pd.merge(regbase, bed_test[['chr', 'pos']])
            r_test.to_csv(project_dir / 'test_regbase.csv', index=False)

        if os.path.exists(project_dir / 'eigen_extract.tsv'):
            eigen = pd.read_csv(project_dir / 'eigen_extract.tsv', sep='\t', na_values='.')

            # manually convert scores to float due to NaN processing
            eigen.iloc[:, 4:] = eigen.iloc[:, 4:].astype(float)

            # average over variant substitutions
            eigen = eigen.rename(columns={'chr': 'chr', 'position': 'pos'}) \
                         .drop('alt', axis=1) \
                         .groupby(['chr', 'pos', 'ref'], as_index=False) \
                         .mean()

            e_train = pd.merge(eigen, bed_train[['chr', 'pos']])
            e_train.to_csv(project_dir / 'train_eigen.csv', index=False)
            e_test = pd.merge(eigen, bed_test[['chr', 'pos']])
            e_test.to_csv(project_dir / 'test_eigen.csv', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', '-p', choices=PROJ_CHOICES, required=True)
    parser.add_argument('--bed', '-b', default=False, action='store_true',
                        help='(re-)extract variant bed from target data')
    parser.add_argument('--roadmap', '-r', default=False, action='store_true',
                        help='extract Roadmap data')
    parser.add_argument('--regbase', '-rb', default=False, action='store_true',
                        help='extract regBase data')
    parser.add_argument('--eigen', '-e', default=False, action='store_true',
                        help='extract Eigen data')
    parser.add_argument('--split', default=False, action='store_true',
                        help='split all data into train/test sets')
    parser.add_argument('--seed', default=9999, help='train/test random seed')
    args = parser.parse_args()

    setup(args)

