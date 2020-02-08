import numpy as np
import pandas as pd
from utils.data_utils import load_mpra_data
from constants import *


# very rough chromosome lengths for sampling random basepairs
GenRange = {1: 248e6, 2: 242e6, 3: 198e6, 4: 190e6,
            5: 181e6, 6: 170e6, 7: 159e6, 8: 145e6,
            9: 138e6, 10: 133e6, 11: 135e6, 12: 133e6,
            13: 114e6, 14: 107e6, 15: 101e6, 16: 90e6,
            17: 83e6, 18: 80e6, 19: 58e6, 20: 64e6,
            21: 46e6, 22: 50e6
}


def load_bed_file(project):
    path = PROCESSED_DIR / f'{project}/{project}.bed'
    bed = pd.read_csv(path, sep='\t', header=None,
                      names=['chr', 'pos', 'pos_end', 'rs'])
    return bed


def save_bed_file(bedfile, project):
    path = PROCESSED_DIR / f'{project}/{project}.bed'
    bedfile.to_csv(path, sep='\t', index=False, header=False)


def get_bed_from_mpra(dataset):
    """ The raw MPRA data files contain non-coding variant locations already
    merged with ROADMAP epigenetic data. This function extracts the variant
    locations and converts to bedfile format.
    """
    data = load_mpra_data(dataset)
    bed = data[['chr', 'pos', 'rs']].copy()

    bed['pos_end'] = bed['pos'] + 1
    bed = bed[['chr', 'pos', 'pos_end', 'rs']]
    return bed


def get_random_bed(data_dir, n_samples=200000):
    """ Generate random bedfile corresponding to n_samples random locations
    in the genome. Used for picking locations to extract unlabeled
    data for semi-supervised learning.
    """
    total_range = sum([x for x in GenRange.values()])

    chrs = []
    samples = []
    for chr in range(1, 23):
        select = int(n_samples * GenRange[chr] / total_range)
        bps = np.random.randint(low=0, high=GenRange[chr], size=select)
        samples.append(bps)
        chrs.extend([chr] * select)

    samples = np.concatenate(samples)
    chrs = np.array(chrs)

    bed = pd.DataFrame({'chr': chrs,
                        'pos': samples,
                        'pos1': samples+1,
                        'rs': ['ul{0}'.format(x+1) for x in range(len(samples))]})
    bed['chr'] = bed['chr'].map(lambda x: 'chr{0}'.format(x))
    return bed

