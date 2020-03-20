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


def extract_eigen(bedfile, outpath):
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
                
                pos_mean = features.groupby(['chr', 'position'], as_index=False).mean()
                reg_mean = pos_mean.groupby('chr', as_index=False).mean()
                reg_mean['position'] = row.pos
                reg_mean['pos_end'] = row.pos_end
                reg_mean['rs'] = row.rs

                reg_mean = reg_mean.astype(str).values.tolist()
                for row in reg_mean:
                    fp.write('\t'.join(row) + '\n')


def extract_regbase(bedfile, outpath):
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

                pos_mean = features.groupby(['#Chrom', 'Pos_start'], as_index=False).mean()
                reg_mean = pos_mean.groupby('#Chrom', as_index=False).mean()
                reg_mean['Pos_start'] = row.pos
                reg_mean['Pos_end'] = row.pos_end
                reg_mean['rs'] = row.rs

                reg_mean = reg_mean.astype(str).values.tolist()
                for row in reg_mean:
                    fp.write('\t'.join(row) + '\n')
