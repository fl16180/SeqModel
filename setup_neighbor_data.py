import argparse

from constants import *
from utils.bed_utils import load_bed_file
from utils.data_utils import get_roadmap_col_order
from utils.bigwig_utils import pull_roadmap_features
from utils.neighbor_utils import add_bed_neighbors, process_roadmap_neighbors
from datasets.data_loader import *


PROJ_CHOICES = ['mpra_e116', 'mpra_e118', 'mpra_e123', 'mpra_nova']
SPLIT_CHOICES = [None, 'train-test', 'test']


def setup(args):
    n_neigh, sample_res = map(int, args.neighbor_param.split(','))

    project_dir = PROCESSED_DIR / args.project
    bedfile = load_bed_file(args.project)

    # assign new variant name for data merging later
    bedfile['rs'] = bedfile['chr'].map(str) + '-' + bedfile['pos'].map(str)
    bedfile.drop_duplicates('rs', inplace=True)

    # extract roadmap data from bigwig files
    if args.extract:
        neighbor_bed = add_bed_neighbors(bedfile, n_neigh, sample_res)
        print(neighbor_bed.head())
        pull_roadmap_features(neighbor_bed)

    # roadmap feature ordering (useful for organizing CNN)
    if args.tissue == 'all':
        features = get_roadmap_col_order(order='marker')
    else:
        features = [f'{x}-{args.tissue}' for x in ROADMAP_MARKERS]

    # prepare roadmap data into train and test sets corresponding to train/test
    # matrices generated from other datasets
    if args.split == 'train_test':
        train_df = load_train_set(args.project,
                                  datasets=['eigen', 'regbase'],
                                  make_new=True)
        outpath = project_dir / f'train_neighbor_{n_neigh}_{sample_res}'
        process_roadmap_neighbors(train_df, outpath, features)

        test_df = load_test_set(args.project,
                                datasets=['eigen', 'regbase'],
                                make_new=True)
        outpath = project_dir / f'test_neighbor_{n_neigh}_{sample_res}'
        process_roadmap_neighbors(test_df, outpath, features)

    elif args.split == 'test':
        test_df = load_test_set(args.project,
                                datasets=['eigen', 'regbase'],
                                make_new=True)
        outpath = project_dir / f'test_neighbor_{n_neigh}_{sample_res}'
        process_roadmap_neighbors(test_df, outpath, features)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', '-p', choices=PROJ_CHOICES, required=True)
    parser.add_argument('--extract', '-e', default=False, action='store_true',
                        help='extract neighboring Roadmap data')
    parser.add_argument('--tissue', '-t', default='all', type=str,
                        help='get neighbor data for specific tissue e.g. E116')
    parser.add_argument('--split', '-s', default=None, choices=SPLIT_CHOICES,
                        help='split data into train/test sets or just test')
    parser.add_argument('--neighbor_param', '-npr', default='0,0', type=str,
                        help='Roadmap neighbor params: (n_neigh,sample_res)')
    args = parser.parse_args()

    setup(args)
