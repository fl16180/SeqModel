''' Functions for extracting epigenetic features from bigwig files given input
bed file with genomic locations. '''

from subprocess import call
import glob
import os
# import multiprocessing as mp
import pandas as pd
from tqdm import tqdm

import sys
sys.path.append('../')
from constants import *


MARKERS = ['DNase', 'H3K27ac', 'H3K27me3', 'H3K36me3',
           'H3K4me1', 'H3K4me3', 'H3K9ac', 'H3K9me3']
TAIL = '.imputed.pval.signal.bigwig'
TMP_DIR = PROCESSED_DIR / 'tmp'


def pull_features(bedfile):
    """ For each ROADMAP marker, execute bigwig pulls for each tissue
    """
    for marker in MARKERS:
        for i in range(1, 130):
            pull_command(marker, i, bedfile)


def pull_command(marker, i, bedfile):
    """ Extraction function to query a bedfile against a ROADMAP marker.

    Inputs:
        marker (str): ROADMAP marker (e.g. 'DNase')
        i (int): Tissue number, for example 58 will be formatted into E058
        bedfile (path): path to bedfile of variant locations to query

    Outputs are saved in a temporary storage directory to be aggregated later.
    """
    # for example: [...]/E058-DNase.imputed.pval.signal.bigwig
    filestr = ROADMAP_DIR / marker / f'E{i:03d}-{marker}{TAIL}'
    output = TMP_DIR / f'{marker}-E{i:03d}'

    command = f'{BIGWIG_UTIL} {filestr} {bedfile} {output}'
    call(command, shell=True)


def features_to_csv(outpath, feature_dir=TMP_DIR):
    """ Post-extraction, combine temporary outputs into a DataFrame and save
    to csv.
    """
    os.chdir(feature_dir)
    features = glob.glob('*')

    df = pd.DataFrame()
    for fn in tqdm(features):
        tmp = pd.read_csv(f'./{fn}', sep='\t',
                          names=['variant', 'c1', 'c2', 'v1', 'v2', f'{fn}'])
        tmp = tmp.drop(['c1','c2','v1','v2'], axis=1)
        df = pd.concat([df, tmp], axis=1)
        df = df.loc[:, ~df.columns.duplicated()]

    df.to_csv(outpath, index=False)


if __name__ == '__main__':

    def load_bed_file(project):
        path = PROCESSED_DIR / f'{project}/{project}.bed'
        bed = pd.read_csv(path, sep='\t', header=None,
                        names=['chr', 'pos', 'pos_end', 'rs'])
        return bed, path

    outpath = PROCESSED_DIR / 'roadmap_extract.csv'
    # features_to_csv(outpath)

    roadmap = pd.read_csv(outpath)

    # merge chr and pos back in
    bedfile = load_bed_file('mpra_deseq2')[0]

    # format chr column as integer
    bedfile = bedfile.assign(chr=bedfile['chr'].apply(lambda x: int(x[3:]))) \
                     .sort_values(['chr', 'pos']) \
                     .reset_index(drop=True)

    roadmap = pd.merge(bedfile[['chr', 'pos', 'rs']], roadmap, left_on='rs', right_on='variant')
    roadmap.drop(['rs', 'variant'], axis=1, inplace=True)

    cols = pd.read_csv(PROCESSED_DIR / 'mpra_e116' / 'roadmap_extract.tsv', sep='\t')
    roadmap = roadmap.loc[:, cols.columns]
    roadmap.to_csv(PROCESSED_DIR / 'mpra_deseq2' / 'roadmap_extract.tsv', sep='\t', index=False)
