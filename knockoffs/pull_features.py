import argparse
import os
import pandas as pd

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parent.parent))

from constants import PROCESSED_DIR
from utils.bed_utils import get_bed_from_mpra, load_bed_file, save_bed_file
from utils.data_utils import *
from utils.neighbor_utils import pull_roadmap_with_neighbors, roadmap_neighbors_to_mat


proj_dir = 'knockoffs'
bedfiles = ['controls_hg19', 'pos_causal_hg19', 'pos_correl_hg19']


def pull_data(args):
    bed = load_bed_file(proj_dir, args.bed)

    if args.roadmap:
        print('Extracting Roadmap: ')
        fname = PROCESSED_DIR / proj_dir / 'roadmap_extract.tsv'
        extract_roadmap(bed, fname, args.project)

    if args.regbase:
        print('Extracting regBase: ')
        fname = PROCESSED_DIR / proj_dir / 'regBase_extract.tsv'
        extract_regbase(bed, fname)

    if args.eigen:
        print('Extracting Eigen: ')
        fname = PROCESSED_DIR / proj_dir / 'eigen_extract.tsv'
        extract_eigen(bed, fname)


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
    parser.add_argument('--split', '-s', default=None, choices=SPLIT_CHOICES,
                        help='split data into train/test sets or just test')
    parser.add_argument('--seed', default=9999, help='train/test random seed')
    args = parser.parse_args()

    setup(args)
