import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import subprocess

CONTROL_PVAL = 0.9

HOME_DIR = Path('/oak/stanford/groups/zihuai')
PROJ_DIR = HOME_DIR / 'fredlu/processed/knockoffs'
KS_FILE = HOME_DIR / 'WGS_knockoff/Analysis_Results/ADSP_Analysis_KS_FA_all.txt'


def match_controls_on_length(selected, pool, n_match):
    np.random.seed(1000)
    select_len = selected['end'] - selected['start']
    pool = pool.copy().reset_index(drop=True)
    pool['len'] = pool['end'] - pool['start']

    controls = []
    for l in select_len:
        match = np.random.choice(pool[np.abs(l - pool['len']) <= l / 100].index,
                                 n_match,
                                 replace=False)
        controls.append(pool.loc[match, :])
        pool = pool.drop(match)

    return pd.concat(controls, axis=0)


def extract_regions(n_sig, n_match):
    # extract regions from analysis file
    print('Loading')
    kf = pd.read_csv(KS_FILE, sep='\t')
    kf = kf[['chr', 'start', 'end', 'P_KS', 'W_KS']]
    kf[['start', 'end']] = kf[['start', 'end']].astype(int)
    kf['chr'] = kf['chr'].map(lambda x: f'chr{x}')
    kf['end'] = kf['end'].astype(int) + 1

    # extract top W_KS and match controls
    print('W_KS')
    pos_causal = kf.sort_values('W_KS', ascending=False).iloc[:n_sig, :].copy()
    pos_causal['name'] = np.arange(pos_causal.shape[0])
    pos_causal['name'] = pos_causal['name'].map(lambda x: f'topW{x}')
    pos_causal = pos_causal[['chr', 'start', 'end', 'name', 'W_KS', 'P_KS']]

    controls = kf.loc[kf['P_KS'] > CONTROL_PVAL, :].copy()
    controls = match_controls_on_length(pos_causal, controls, n_match=n_match)
    controls = controls.sort_values('P_KS', ascending=False)
    controls['name'] = np.arange(controls.shape[0])
    controls['name'] = controls['name'].map(lambda x: f'neg{x}')
    controls = controls[['chr', 'start', 'end', 'name', 'W_KS', 'P_KS']]

    pd.concat([pos_causal, controls], axis=0).to_csv(
        PROJ_DIR / f'top_W_matched_{n_sig}.bed',
        sep='\t', index=False, header=False)

    # extract top P_KS and match controls
    print('P_KS')
    pos_correl = kf.sort_values('P_KS', ascending=True).iloc[:n_sig, :].copy()
    pos_correl['name'] = np.arange(pos_correl.shape[0])
    pos_correl['name'] = pos_correl['name'].map(lambda x: f'topP{x}')
    pos_correl = pos_correl[['chr', 'start', 'end', 'name', 'W_KS', 'P_KS']]

    controls = kf.loc[kf['P_KS'] > CONTROL_PVAL, :].copy()
    controls = match_controls_on_length(pos_correl, controls, n_match=n_match)
    controls = controls.sort_values('P_KS', ascending=False)
    controls['name'] = np.arange(controls.shape[0])
    controls['name'] = controls['name'].map(lambda x: f'neg{x}')
    controls = controls[['chr', 'start', 'end', 'name', 'W_KS', 'P_KS']]

    pd.concat([pos_correl, controls], axis=0).to_csv(
        PROJ_DIR / f'top_P_matched_{n_sig}.bed',
        sep='\t', index=False, header=False)


def convert_regions(n_sig):
    # convert files from hg38 to hg19
    cmd = f"liftOver {PROJ_DIR / f'top_W_matched_{n_sig}.bed'} {PROJ_DIR / 'hg38ToHg19.over.chain.gz'} {PROJ_DIR / f'top_W_matched_hg19_{n_sig}.bed'} unMapped.out"
    subprocess.call(cmd, shell=True)

    cmd = f"liftOver {PROJ_DIR / f'top_P_matched_{n_sig}.bed'} {PROJ_DIR / 'hg38ToHg19.over.chain.gz'} {PROJ_DIR / f'top_P_matched_hg19_{n_sig}.bed'} unMapped.out"
    subprocess.call(cmd, shell=True)

    # take out score columns in hg19 version
    df = pd.read_csv(PROJ_DIR / f'top_W_matched_hg19_{n_sig}.bed', sep='\t', header=None,
                     names=['chr', 'pos', 'pos_end', 'rs', 'w', 'p'])
    df = df.loc[:, ['chr', 'pos', 'pos_end', 'rs']]
    df.to_csv(PROJ_DIR / f'top_W_matched_hg19_{n_sig}.bed', sep='\t', header=None, index=False)

    df = pd.read_csv(PROJ_DIR / f'top_P_matched_hg19_{n_sig}.bed', sep='\t', header=None,
                     names=['chr', 'pos', 'pos_end', 'rs', 'w', 'p'])
    df = df.loc[:, ['chr', 'pos', 'pos_end', 'rs']]
    df.to_csv(PROJ_DIR / f'top_P_matched_hg19_{n_sig}.bed', sep='\t', header=None, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--nsig', type=int, default=100,
                        help='No. significant variants')
    parser.add_argument('--nmatch', type=int, default=5,
                        help='No. matches')
    args = parser.parse_args()

    # extract_regions(args.nsig, args.nmatch)
    convert_regions(args.nsig)
