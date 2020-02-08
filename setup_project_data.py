import argparse
import os
import pandas as pd
from utils.bed_utils import get_bed_from_mpra, load_bed_file, save_bed_file
from utils.data_utils import *
from constants import PROCESSED_DIR


PROJ_CHOICES = ['mpra_e116', 'mpra_e118', 'mpra_e123', 'mpra_nova']
SPLIT_CHOICES = [None, 'train-test', 'test']


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

    # further process data and split into train/test sets
    if args.split == 'train-test':
        bed_train, bed_test = split_train_test(bedfile,
                                               test_frac=0.2,
                                               seed=args.seed)

        process_datasets(args, bed_train, split='train')
        process_datasets(args, bed_test, split='test')

    elif args.split == 'test':
        process_datasets(args, bedfile, split='test')


def process_datasets(args, bedfile, split='test'):
    project_dir = PROCESSED_DIR / args.project
    mpra = load_mpra_data(args.project)

    if args.project == 'mpra_nova':
        mpra['Label'] = mpra['pvalue_expr'].apply(lambda x: x < 1e-5).astype(int)

    y_split = pd.merge(mpra, bedfile, on=['chr', 'pos']).loc[:, ['chr', 'pos', 'Label']]
    y_split.to_csv(project_dir / f'{split}_label.csv', sep=',', index=False)

    if os.path.exists(project_dir / 'roadmap_extract.tsv'):
        roadmap = pd.read_csv(project_dir / 'roadmap_extract.tsv', sep='\t')
        r_split = pd.merge(roadmap, bedfile[['chr', 'pos']], on=['chr', 'pos'])
        r_split.to_csv(project_dir / f'{split}_roadmap.csv', sep=',', index=False)

    if os.path.exists(project_dir / 'regBase_extract.tsv'):
        regbase = clean_regbase_data(project_dir / 'regBase_extract.tsv')
        r_split = pd.merge(regbase, bedfile[['chr', 'pos']])
        r_split.to_csv(project_dir / f'{split}_regbase.csv', index=False)

    if os.path.exists(project_dir / 'eigen_extract.tsv'):
        eigen = clean_eigen_data(project_dir / 'eigen_extract.tsv')
        e_split = pd.merge(eigen, bedfile[['chr', 'pos']])
        e_split.to_csv(project_dir / f'{split}_eigen.csv', index=False)


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
    parser.add_argument('--split', default=None, choices=SPLIT_CHOICES,
                        help='split data into train/test sets or just test')
    parser.add_argument('--seed', default=9999, help='train/test random seed')
    args = parser.parse_args()

    setup(args)
