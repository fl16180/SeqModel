''' Functions for extracting epigenetic features from bigwig files given input
bed file with genomic locations. '''

from subprocess import call
import glob
import os
import multiprocessing as mp
import pandas as pd

from constants import ROADMAP_DIR, BIGWIG_UTIL, PROCESSED_DIR


MARKERS = ['DNase', 'H3K27ac', 'H3K27me3', 'H3K36me3',
           'H3K4me1', 'H3K4me3', 'H3K9ac', 'H3K9me3']
TAIL = '.imputed.pval.signal.bigwig'
TMP_DIR = PROCESSED_DIR / 'tmp'


def pull_features(bedfile):
    """ For each ROADMAP marker, execute parallel bigwig pulls for each tissue
    """
    for marker in MARKERS:
        pool = mp.Pool(processes=4)
        results = [pool.apply_async(pull_command, args=(marker, i, bedfile))
                   for i in range(1, 130)]


def pull_command(marker, i, bedfile):
    """ Extraction function to query a bedfile against a ROADMAP marker.

    Inputs:
        marker (str): ROADMAP marker (e.g. 'DNase')
        i (int): Tissue number, for example 58 will be formatted into E058
        bedfile (path): path to bedfile of variant locations to query

    Outputs are saved in a temporary storage directory to be aggregated later.
    """
    # for example: [...]/E058-DNase.imputed.pval.signal.bigwig
    filestr = ROADMAP_DIR / marker / f'E{i:03d}-{marker}{tail}'
    output = TMP_DIR / f'{marker}-E{i:03d}'

    command = f'{BIGWIG_UTIL} {filestr} {bedfile} {output}'
    call(command, shell=True)


def features_to_csv(feature_dir, outpath):
    """ Post-extraction, combine temporary outputs into a DataFrame and save
    to csv.
    """
    os.chdir(feature_dir)
    features = glob.glob('*')

    df = pd.DataFrame()
    for fn in features:
        print(fn)
        tmp = pd.read_csv(f'./{fn}', sep='\t',
                          names=['variant', 'c1', 'c2', 'v1', 'v2', '{fn}'])
        tmp = tmp.drop(['c1','c2','v1','v2'], axis=1)
        df = pd.concat([df, tmp], axis=1)
        df = df.loc[:, ~df.columns.duplicated()]

    df.to_csv(outpath, index=False)


if __name__ == '__main__':

    # bedfile = '/home/users/fredlu/E116.bed'
    # pull_features(bedfile)

    feature_dir = 'C:/Users/fredl/Documents/datasets/functional_variants/bigwig/tmp/'
    features_to_csv(feature_dir)