import numpy as np
import pandas as pd

from constants import *
from utils.bigwig_utils import pull_roadmap_features
from utils.data_utils import get_roadmap_col_order


def add_bed_neighbors(bed, n_neighbor=40, sample_res=25):
    """ Takes a dataframe bedfile and adds entries for neighboring
    variants for each variant in the dataset.
    """
    nrange = n_neighbor * sample_res
    neighbors = np.arange(-nrange, nrange + 1, sample_res)
    rs_tag = np.array([f';N{x}' for x in neighbors])

    chr_n = np.repeat(bed['chr'].values[:, None], len(neighbors), axis=1)
    pos_n = bed['pos'].values[:, None] + neighbors[None, :]
    pos_end_n = pos_n + 1
    rs_n = bed['rs'].values[:, None] + rs_tag[None, :]

    neigh_bed = pd.DataFrame({'chr': chr_n.flatten(),
                              'pos': pos_n.flatten(),
                              'pos_end': pos_end_n.flatten(),
                              'rs': rs_n.flatten()})
    return neigh_bed


def process_roadmap_neighbors():
    return

def pull_roadmap_neighbors():
    success = pull_roadmap_features(neighbor_bed)


    # figure out column ordering
    col_order = ['chr', 'pos'] + col_order
    if keep_rs_col:
        compiled.drop(['variant', 'pos_end'], axis=1, inplace=True)
        col_order = col_order + ['rs']
    else:
        compiled.drop(['rs', 'variant', 'pos_end'], axis=1, inplace=True)

    compiled = compiled.loc[:, col_order]
    compiled.to_csv(outpath, sep='\t', index=False)

def pull_roadmap_with_neighbors(bedfile, outpath,
                                n_neighbor=40, sample_res=25):

    # col_order = get_roadmap_col_order(order='marker')
    col_order = [f'{x}-E116' for x in ROADMAP_MARKERS]

    pull_roadmap_features(neighbor_bed, outpath, col_order, keep_rs_col=True)



def roadmap_neighbors_to_mat(ref_bed, neighbors, outpath):

    # current pos is neighbor position instead of variant position
    neighbors.drop(['chr', 'pos'], axis=1, inplace=True)

    # get variant chr/pos from reference bedfile
    neighbors[['rs', 'N']] = neighbors['rs'].str.split(';N', n=1, expand=True)
    neighbors['N'] = neighbors['N'].astype(int)
    neighbors = pd.merge(ref_bed[['chr', 'pos', 'rs']], neighbors, on='rs')

    # pivot dataframe on neighbors to get wide form
    #features = get_roadmap_col_order(order='marker')
    features = [f'{x}-E116' for x in ROADMAP_MARKERS]

    tmp = pd.pivot(neighbors, index='rs', values=features, columns='N')
    tmp.fillna(tmp.mean(), inplace=True)

    # reorder index and columns to match ref_bed variants
    ordered_nbrs = np.sort(neighbors['N'].unique())
    col_order = [(feat, nbr) for feat in features for nbr in ordered_nbrs]
    tmp = tmp.reindex(ref_bed['rs'])
    tmp = tmp.loc[:, col_order]

    n_neighbors = len(neighbors['N'].unique())
    feat_mat = tmp.values.reshape(tmp.shape[0], len(features), n_neighbors)

    # store feature matrix and chr/pos reference array
    np.save(outpath, feat_mat)
    np.save(outpath.with_suffix('.ref'), ref_bed[['chr', 'pos']].values)


if __name__ == '__main__':

    test_bed = {'chr': [1, 1, 3, 4],
                'pos': [12000, 13000, 4000, 1015],
                'pos_end': [12001, 13001, 4001, 1016],
                'rs': ['v1', 'v2', 'v3', 'v4']}

    bed = pd.DataFrame(test_bed)

    neigh_bed = add_bed_neighbors(bed, n_neighbor=2, sample_res=10)

    neigh_bed['feat1'] = np.arange(10, 10 + neigh_bed.shape[0])
    neigh_bed['feat2'] = np.arange(2.1, 2.1 + neigh_bed.shape[0])
    neigh_bed.drop(['pos_end'], axis=1, inplace=True)
