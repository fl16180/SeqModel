from pathlib import Path
import pandas as pd
import numpy as np
import subprocess

np.random.seed(1000)

N_W_STAT = 1000
N_P_STAT = 1000
N_CONTROL = 7000

PROJ_DIR = Path('/oak/stanford/groups/zihuai/fredlu/processed/knockoffs')
# KS_FILE = Path('/oak/stanford/groups/zihuai/WGS_knockoff/Analysis_Results/ADSP_Analysis_KS.txt')
KS_FILE = Path('/oak/stanford/groups/zihuai/WGS_knockoff/Analysis_Results/ADSP_Analysis_KS_FA_all.txt')

kf = pd.read_csv(KS_FILE, sep='\t')
kf = kf[['chr', 'start', 'end', 'P_KS', 'W_KS']]
kf[['start', 'end']] = kf[['start', 'end']].astype(int)
kf['chr'] = kf['chr'].map(lambda x: f'chr{x}')
kf['end'] = kf['end'].astype(int) + 1


# controls = kf[kf.P_KS >= np.percentile(kf.P_KS, 90)]
# cont_sel = np.random.choice(controls.shape[0], N_CONTROL, replace=False)
# controls = controls.iloc[cont_sel, :]
controls = kf.sort_values('P_KS', ascending=False).iloc[:N_CONTROL, :].copy()
controls['name'] = np.arange(controls.shape[0])
controls['name'] = controls['name'].map(lambda x: f'neg{x}')
controls = controls[['chr', 'start', 'end', 'name']]
controls.to_csv(PROJ_DIR / 'controls.bed', sep='\t', index=False, header=False)

pos_causal = kf.sort_values('W_KS', ascending=False).iloc[:N_W_STAT, :].copy()
pos_causal['name'] = np.arange(pos_causal.shape[0])
pos_causal['name'] = pos_causal['name'].map(lambda x: f'causal{x}')
pos_causal = pos_causal[['chr', 'start', 'end', 'name']]
pos_causal.to_csv(PROJ_DIR / 'pos_causal.bed', sep='\t', index=False, header=False)

pos_correl = kf.sort_values('P_KS', ascending=True).iloc[:N_P_STAT, :].copy()
pos_correl['name'] = np.arange(pos_correl.shape[0])
pos_correl['name'] = pos_correl['name'].map(lambda x: f'correl{x}')
pos_correl = pos_correl[['chr', 'start', 'end', 'name']]
pos_correl.to_csv(PROJ_DIR / 'pos_correl.bed', sep='\t', index=False, header=False)

# convert files from hg38 to hg19

cmd = f"liftOver {PROJ_DIR / 'controls.bed'} {PROJ_DIR / 'hg38ToHg19.over.chain.gz'} {PROJ_DIR / 'controls_hg19.bed'} log.out"
subprocess.call(cmd, shell=True)

cmd = f"liftOver {PROJ_DIR / 'pos_causal.bed'} {PROJ_DIR / 'hg38ToHg19.over.chain.gz'} {PROJ_DIR / 'pos_causal_hg19.bed'} log.out"
subprocess.call(cmd, shell=True)

cmd = f"liftOver {PROJ_DIR / 'pos_correl.bed'} {PROJ_DIR / 'hg38ToHg19.over.chain.gz'} {PROJ_DIR / 'pos_correl_hg19.bed'} log.out"
subprocess.call(cmd, shell=True)
