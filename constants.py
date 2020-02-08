from pathlib import Path


HOME_DIR = Path('/oak/stanford/groups/zihuai/fredlu')
EIGEN_DIR = Path('/oak/stanford/groups/zihuai/FST/Score')
ROADMAP_DIR = Path('/oak/stanford/groups/zihuai/SemiSupervise/bigwig/rollmean')
BBANK_DIR = HOME_DIR / 'BioBank'
REGBASE_DIR = HOME_DIR / 'regBase' / 'V1.1'
MPRA_DIR = HOME_DIR / 'MPRA'
CODE_DIR = HOME_DIR / 'SeqModel'
PROCESSED_DIR = HOME_DIR / 'processed'
TMP_DIR = HOME_DIR / 'processed' / 'tmp'

REGBASE = 'regBase_V1.1.gz'
EIGEN_BASE = 'Eigen_hg19_noncoding_annot_chrXX.tab.bgz'

MPRA_TABLE = {
    'mpra_e116': ('LabelData_CellPaperE116.txt', 'TestData_MPRA_E116_unbalanced.txt'),
    'mpra_e118': ('LabelData_KellisE118.txt', 'TestData_MPRA_E118.txt'),
    'mpra_e123': ('LabelData_KellisE123.txt', 'TestData_MPRA_E123.txt'),
    'mpra_nova': ('1KG_bartender_novaSeq_DESeq2_pvals.txt', '')
}

# experiments with ROADMAP data included
STANDARD_MPRA = ('mpra_e116', 'mpra_e118', 'mpra_e123')

BIGWIG_UTIL = '/home/users/fredlu/opt/bigWigAverageOverBed'
BIGWIG_TAIL = '.imputed.pval.signal.bigwig'
ROADMAP_MARKERS = ['DNase', 'H3K27ac', 'H3K27me3', 'H3K36me3',
                   'H3K4me1', 'H3K4me3', 'H3K9ac', 'H3K9me3']
ROADMAP_COL_ORDER_REF = 'mpra_e116'
