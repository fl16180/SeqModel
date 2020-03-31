import os
import pandas as pd
import numpy as np
import io, subprocess
from tqdm import tqdm

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parent.parent))
from constants import *



def _read_bed(x, **kwargs):
    """ Helper function to parse output from a tabix query """
    return pd.read_csv(x, sep=r'\s+', header=None, index_col=False, **kwargs)


def extract_genonet(bedfile, outpath, summarize='max'):
    """ Extract rows from GenoNet file corresponding to variants in
    input bedfile.
    """
    if os.path.exists(outpath):
        os.remove(outpath)

    cols = ['GenoNet-E{:03d}'.format(x) for x in range(1, 130) if x not in [60, 64]]
    header = ['chr', 'pos_start', 'pos_end', 'reg_id'] + cols

    command = f'tabix {GENONET_DIR}/{GENONET_BASE} '

    with open(outpath, 'w') as fp:
        full_header = header + ['rs']
        fp.write('\t'.join(full_header) + '\n')

        for row in tqdm(bedfile.itertuples()):
            args = f'chr{row.chr}:{row.pos}-{row.pos_end}'
            full_cmd = command.replace('XX', str(row.chr)) + args

            p = subprocess.Popen(full_cmd, shell=True, stdout=subprocess.PIPE)
            stdout, _ = p.communicate()

            if len(stdout) > 0:
                features = _read_bed(io.StringIO(stdout.decode('utf-8')),
                                     na_values='.')
                features.columns = header
                features['reg_id'] = 0.0

                if summarize == 'mean':
                    features = features.groupby('chr', as_index=False).mean()
                elif summarize == 'max':
                    features = features.groupby('chr', as_index=False).max()

                features['pos_start'] = row.pos
                features['pos_end'] = row.pos_end
                features['rs'] = row.rs

                features = features.astype(str).values.tolist()
                for row in features:
                    fp.write('\t'.join(row) + '\n')


def extract_eigen(bedfile, outpath, summarize='max'):
    """ Extract rows from Eigen file corresponding to variants in
    input bedfile. Command line piping adapted from
    https://github.com/mulinlab/regBase/blob/master/script/regBase_predict.py

    """
    if os.path.exists(outpath):
        os.remove(outpath)

    head_file = EIGEN_DIR / 'header_noncoding.txt'
    with open(head_file, 'r') as f:
        header = f.read()
    header = header.strip('\n').split('  ')

    command = f'tabix {EIGEN_DIR}/{EIGEN_BASE} '

    with open(outpath, 'w') as fp:
        full_header = header + ['pos_end', 'rs']
        fp.write('\t'.join(full_header) + '\n')

        for row in tqdm(bedfile.itertuples()):
            args = f'{row.chr}:{row.pos}-{row.pos_end}'
            full_cmd = command.replace('XX', str(row.chr)) + args

            p = subprocess.Popen(full_cmd, shell=True, stdout=subprocess.PIPE)
            stdout, _ = p.communicate()

            if len(stdout) > 0:
                features = _read_bed(io.StringIO(stdout.decode('utf-8')),
                                     na_values='.')
                features.columns = header
                features[['ref', 'alt']] = 0.0
                
                features = features.groupby(['chr', 'position'], as_index=False).mean()
                if summarize == 'mean':
                    features = features.groupby('chr', as_index=False).mean()
                elif summarize == 'max':
                    features = features.groupby('chr', as_index=False).max()

                features['position'] = row.pos
                features['pos_end'] = row.pos_end
                features['rs'] = row.rs

                features = features.astype(str).values.tolist()
                for row in features:
                    fp.write('\t'.join(row) + '\n')


def extract_regbase(bedfile, outpath, summarize='max'):
    """ Extract rows from regBase file corresponding to variants in
    input bedfile.
    """
    if os.path.exists(outpath):
        os.remove(outpath)

    head_file = REGBASE_DIR / REGBASE
    header = pd.read_csv(head_file, sep=r'\s+', nrows=3).columns
    header = header.values.tolist()

    command = f'tabix {REGBASE_DIR}/{REGBASE} '

    with open(outpath, 'w') as fp:
        full_header = header + ['rs']
        fp.write('\t'.join(full_header) + '\n')

        for row in tqdm(bedfile.itertuples()):
            args = f'{row.chr}:{row.pos}-{row.pos_end}'
            full_cmd = command + args

            p = subprocess.Popen(full_cmd, shell=True, stdout=subprocess.PIPE)
            stdout, _ = p.communicate()

            if len(stdout) > 0:
                features = _read_bed(io.StringIO(stdout.decode('utf-8')),
                                     na_values='.')
                features.columns = header
                features[['Ref', 'Alts']] = 0.0

                features = features.groupby(['#Chrom', 'Pos_start'], as_index=False).mean()
                if summarize == 'mean':
                    features = features.groupby('#Chrom', as_index=False).mean()
                elif summarize == 'max':
                    features = features.groupby('#Chrom', as_index=False).max()

                features['Pos_start'] = row.pos
                features['Pos_end'] = row.pos_end
                features['rs'] = row.rs

                features = features.astype(str).values.tolist()
                for row in features:
                    fp.write('\t'.join(row) + '\n')
