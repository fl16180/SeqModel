from pathlib import Path
import pandas as pd
import numpy as np
import subprocess

N_W_STAT = 1000
N_P_STAT = 1000
CONTROL_PVAL = 0.8
N_MATCH = 5

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


def extract_regions():
    # extract regions from analysis file
    print('Loading')
    kf = pd.read_csv(KS_FILE, sep='\t')
    kf = kf[['chr', 'start', 'end', 'P_KS', 'W_KS']]
    kf[['start', 'end']] = kf[['start', 'end']].astype(int)
    kf['chr'] = kf['chr'].map(lambda x: f'chr{x}')
    kf['end'] = kf['end'].astype(int) + 1

    # extract top W_KS and match controls
    print('W_KS')
    pos_causal = kf.sort_values('W_KS', ascending=False).iloc[:N_W_STAT, :].copy()
    pos_causal['name'] = np.arange(pos_causal.shape[0])
    pos_causal['name'] = pos_causal['name'].map(lambda x: f'topW{x}')
    pos_causal = pos_causal[['chr', 'start', 'end', 'name', 'W_KS', 'P_KS']]

    controls = kf.loc[kf['P_KS'] > CONTROL_PVAL, :].copy()
    controls = match_controls_on_length(pos_causal, controls, n_match=N_MATCH)
    controls = controls.sort_values('P_KS', ascending=False)
    controls['name'] = np.arange(controls.shape[0])
    controls['name'] = controls['name'].map(lambda x: f'neg{x}')
    controls = controls[['chr', 'start', 'end', 'name', 'W_KS', 'P_KS']]

    pd.concat([pos_causal, controls], axis=0).to_csv(
        PROJ_DIR / 'top_W_matched.bed', sep='\t', index=False, header=False)

    # extract top P_KS and match controls
    print('P_KS')
    pos_correl = kf.sort_values('P_KS', ascending=True).iloc[:N_P_STAT, :].copy()
    pos_correl['name'] = np.arange(pos_correl.shape[0])
    pos_correl['name'] = pos_correl['name'].map(lambda x: f'topP{x}')
    pos_correl = pos_correl[['chr', 'start', 'end', 'name', 'W_KS', 'P_KS']]

    controls = kf.loc[kf['P_KS'] > CONTROL_PVAL, :].copy()
    controls = match_controls_on_length(pos_correl, controls, n_match=N_MATCH)
    controls = controls.sort_values('P_KS', ascending=False)
    controls['name'] = np.arange(controls.shape[0])
    controls['name'] = controls['name'].map(lambda x: f'neg{x}')
    controls = controls[['chr', 'start', 'end', 'name', 'W_KS', 'P_KS']]

    pd.concat([pos_correl, controls], axis=0).to_csv(
        PROJ_DIR / 'top_P_matched.bed', sep='\t', index=False, header=False)


def convert_regions():
    # convert files from hg38 to hg19
    cmd = f"liftOver {PROJ_DIR / 'top_W_matched.bed'} {PROJ_DIR / 'hg38ToHg19.over.chain.gz'} {PROJ_DIR / 'top_W_matched_hg19.bed'} unMapped.out"
    subprocess.call(cmd, shell=True)

    cmd = f"liftOver {PROJ_DIR / 'top_P_matched.bed'} {PROJ_DIR / 'hg38ToHg19.over.chain.gz'} {PROJ_DIR / 'top_P_matched_hg19.bed'} unMapped.out"
    subprocess.call(cmd, shell=True)


if __name__ == '__main__':
    extract_regions()
    # convert_regions()
