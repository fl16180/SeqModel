''' Functions for extracting epigenetic features from bigwig files given input
bed file with genomic locations. '''

from subprocess import call
import glob
import os
import pandas as pd
from tqdm import tqdm
import re

from constants import *


def pull_roadmap_features(bedfile, outpath, col_order, feature_dir=TMP_DIR):
    """ For each ROADMAP marker, execute bigwig pulls for each tissue
    """
    # clear tmp directory
    for f in os.listdir(feature_dir):
        os.remove(feature_dir / f)

    # save temporary bedfile with correct input format
    bedfile['chr'] = bedfile['chr'].map(lambda x: f'chr{x}')
    bedfile.to_csv(TMP_DIR / 'tmp.bed', sep='\t', header=False, index=False)

    # query each feature from bigwig
    for marker in ROADMAP_MARKERS:
        for i in range(1, 130):
            pull_command(marker, i, TMP_DIR / 'tmp.bed')

    # stack features into dataframe
    compiled = features_to_csv(feature_dir)
    compiled['chr'] = compiled['variant'].map(lambda x: x.split(':')[0]) \
                                         .map(lambda x: x[3:]) \
                                         .astype(int)

    compiled['pos'] = compiled['variant'].map(lambda x: x.split(':')[1]) \
                                         .astype(int)

    compiled.drop('variant', axis=1, inplace=True)

    compiled = compiled.loc[:, col_order]
    compiled.to_csv(outpath, sep='\t', index=False)


def pull_command(marker, i, bedfile):
    """ Extraction function to query a bedfile against a ROADMAP marker.

    Inputs:
        marker (str): ROADMAP marker (e.g. 'DNase')
        i (int): Tissue number, for example 58 will be formatted into E058
        bedfile (path): path to bedfile of variant locations to query

    Outputs are saved in a temporary storage directory to be aggregated later.
    """
    # for example: [...]/E058-DNase.imputed.pval.signal.bigwig
    filestr = ROADMAP_DIR / marker / f'E{i:03d}-{marker}{BIGWIG_TAIL}'
    output = TMP_DIR / f'{marker}-E{i:03d}'

    command = f'{BIGWIG_UTIL} {filestr} {bedfile} {output}'
    call(command, shell=True)


def features_to_csv(feature_dir=TMP_DIR):
    """ Post-extraction, combine temporary outputs into a DataFrame and save
    to csv.
    """
    features = [x for x in os.listdir(feature_dir) if re.search('E\d{3}$', x)]

    cols = []
    for fn in tqdm(features):
        tmp = pd.read_csv(feature_dir / f'./{fn}', sep='\t',
                          names=['variant', 'c1', 'c2', 'v1', 'v2', f'{fn}'])
        tmp = tmp.drop(['c1','c2','v1','v2'], axis=1)
        cols.append(tmp)

    df = pd.concat(cols, axis=1)
    df = df.loc[:, ~df.columns.duplicated()]
    return df
