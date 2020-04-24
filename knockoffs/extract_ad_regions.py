from pathlib import Path
import pandas as pd
import numpy as np
import subprocess

from extract_knockoff_regions import match_controls_on_length

HOME_DIR = Path('/oak/stanford/groups/zihuai')
PROJ_DIR = HOME_DIR / 'fredlu/processed/knockoffs'
AD_DIR = HOME_DIR / 'fredlu/AD_GWAS'
AD_FILE = 'AD_sumstats_Jansenetal_2019sept.txt'

# # pre-sort the file to save time later
# ad = pd.read_csv(AD_DIR / AD_FILE, sep='\t')
# ad = ad.sort_values('P', ascending=True)
# ad.to_csv(AD_DIR / 'AD_sumstats_sorted.csv', sep='\t', index=False)
# ad = ad.iloc[::-1, :]
# ad.to_csv(AD_DIR / 'AD_sumstats_sorted_rev.csv', sep='\t', index=False)

ad = pd.read_csv(AD_DIR / 'AD_sumstats_sorted.csv', sep='\t',
                 nrows=100000)
ad_sig = ad[ad['P'] < 5e-8].copy()

ad_sig['BP_end'] = ad_sig['BP'] + 1
ad_sig = ad_sig.loc[:, ['CHR', 'BP', 'BP_end', 'SNP']]
ad_sig['CHR'] = ad_sig['CHR'].map(lambda x: f'chr{x}')
ad_sig.to_csv(PROJ_DIR / 'AD_sig.bed', sep='\t', index=False, header=False)


ad_ctrl = pd.read_csv(AD_DIR / 'AD_sumstats_sorted_rev.csv', sep='\t',
                 nrows=5000)
ad_ctrl['BP_end'] = ad_ctrl['BP'] + 1
ad_ctrl = ad_ctrl.loc[:, ['CHR', 'BP', 'BP_end', 'SNP']]
ad_ctrl['CHR'] = ad_ctrl['CHR'].map(lambda x: f'chr{x}')
ad_ctrl.to_csv(PROJ_DIR / 'AD_ctrl.bed', sep='\t', index=False, header=False)
