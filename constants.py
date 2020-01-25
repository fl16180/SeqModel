from pathlib import Path


HOME_DIR = Path('/oak/stanford/groups/zihuai/fredlu')
EIGEN_DIR = Path('/oak/stanford/groups/zihuai/FST/Score')
ROADMAP_DIR = ''
BBANK_DIR = HOME_DIR / 'BioBank'
REGBASE_DIR = HOME_DIR / 'regBase' / 'V1.1'
MPRA_DIR = HOME_DIR / 'MPRA'
CODE_DIR = HOME_DIR / 'SeqModel'
PROCESSED_DIR = HOME_DIR / 'processed'

REGBASE = 'regBase_V1.1.gz'
EIGEN_BASE = 'Eigen_hg19_noncoding_annot_chrXX.tab.bgz'

MPRA_TABLE = {
    'mpra_e116': ('LabelData_CellPaperE116.txt', 'TestData_MPRA_E116_unbalanced.txt'),
    'mpra_e118': ('LabelData_KellisE118.txt', 'TestData_MPRA_E118.txt'),
    'mpra_e123': ('LabelData_KellisE123.txt', 'TestData_MPRA_E123.txt'),
    'mpra_deseq2': ('1KG_bartender_novaSeq_DESeq2_pvals.txt', '')
}

ROADMAP_TABLE = {
    'mpra_e116': 'LabelData_CellPaperE116.txt',
    'mpra_e118': 'LabelData_KellisE118.txt',
    'mpra_e123': 'LabelData_KellisE123.txt',
    'mpra_deseq2': None
}
