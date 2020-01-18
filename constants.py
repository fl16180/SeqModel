from pathlib import Path


HOME_DIR = Path('/oak/stanford/groups/zihuai/fredlu')
EIGEN_DIR = Path('/oak/stanford/groups/zihuai/FST/Score')
ROADMAP_DIR = ''
BBANK_DIR = HOME_DIR / 'BioBank'
REGBASE_DIR = HOME_DIR / 'regBase' / 'V1.1'
MPRA_DIR = HOME_DIR / 'MPRA'
CODE_DIR = HOME_DIR / 'SeqModel'
PROCESSED_DIR = HOME_DIR / 'processed'

MPRA_TABLE = {
    'E116': ('LabelData_CellPaperE116.txt', 'TestData_MPRA_E116_unbalanced.txt'),
    'E118': ('LabelData_KellisE118.txt', 'TestData_MPRA_E118.txt'),
    'E123': ('LabelData_KellisE123.txt', 'TestData_MPRA_E123.txt'),
    'DESeq2': ('1KG_bartender_novaSeq_DESeq2_pvals.txt', '')
}

ROADMAP_TABLE = {
    'E116': 'LabelData_CellPaperE116.txt',
    'E118': 'LabelData_KellisE118.txt',
    'E123': 'LabelData_KellisE123.txt',
    'DESeq2': None
}

EIGEN_TABLE = {}

BBANK_TABLE = {}

REGBASE_TABLE = {}

