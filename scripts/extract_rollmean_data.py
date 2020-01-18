''' Functions for extracting epigenetic features from bigwig files given input bed file with
genomic locations. '''

from subprocess import call
import glob
import os
import multiprocessing as mp


def pull_features(bedfile):
    markers = ['DNase','H3K27ac','H3K27me3','H3K36me3','H3K4me1','H3K4me3','H3K9ac','H3K9me3']
    tail = '.imputed.pval.signal.bigwig'

    for mark in markers:
        pool = mp.Pool(processes=4)
        results = [pool.apply_async(pull_command, args=(mark, i, bedfile)) for i in range(1, 130)]

        # for i in range(1, 130):
            # pull_command(mark, i, bedfile)
            # filedir = '/scratch/groups/zihuai/Epigenetics_Features/rollmean/{0}'.format(mark)
            # filestr = filedir + '/E{:03d}-'.format(i) + mark + tail
            # output_name = '/home/users/fredlu/output/' + mark + '-E{:03d}'.format(i)
            #
            # command = '/home/users/fredlu/opt/bigWigAverageOverBed {0} {1} {2}'.format(filestr, bedfile, output_name)
            #
            # call(command, shell=True)


def pull_command(mark, i, bedfile):

    filedir = '/scratch/groups/zihuai/Epigenetics_Features/rollmean/{0}'.format(mark)
    filestr = filedir + '/E{:03d}-'.format(i) + mark + tail
    output_name = '/home/users/fredlu/output/' + mark + '-E{:03d}'.format(i)

    command = '/home/users/fredlu/opt/bigWigAverageOverBed {0} {1} {2}'.format(filestr, bedfile, output_name)

    call(command, shell=True)


def features_to_csv(feature_dir):
    import pandas as pd

    os.chdir(feature_dir)
    features = glob.glob('*')

    df = pd.DataFrame()
    for fn in features:
        print(fn)
        tmp = pd.read_csv('./{0}'.format(fn), sep='\t', names=['variant','c1','c2','v1','v2','{0}'.format(fn)])
        tmp = tmp.drop(['c1','c2','v1','v2'], axis=1)
        df = pd.concat([df, tmp], axis=1)
        df = df.loc[:, ~df.columns.duplicated()]

    df.to_csv('./concat.csv', index=False)


if __name__ == '__main__':

    # bedfile = '/home/users/fredlu/E116.bed'
    # pull_features(bedfile)

    feature_dir = 'C:/Users/fredl/Documents/datasets/functional_variants/bigwig/tmp/'
    features_to_csv(feature_dir)
