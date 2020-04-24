from pathlib import Path
import pandas as pd
import numpy as np
import pickle

# hide sklearn deprecation message triggered within skorch
from warnings import simplefilter
simplefilter('ignore', category=FutureWarning)

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score
from glmnet import LogitNet

import torch
from torch import nn
import torch.nn.functional as F
from skorch import NeuralNetClassifier
from skorch.callbacks import EpochScoring, LRScheduler

import sys
sys.path.append(str(Path(__file__).absolute().parent.parent))
from datasets import Processor

from nn_params import *
from fit_models import FCNet


REF_COL = ['chr', 'pos_start_hg19', 'pos_end_hg19',
           'pos_start_hg38', 'pos_end_hg38', 'label',
           'variant', 'W_KS', 'P_KS']
VAL_REF_COL = ['chr', 'pos_start_hg19', 'pos_end_hg19', 'variant']

DATA_DIR = Path('/oak/stanford/groups/zihuai/fredlu/processed/knockoffs')
MODEL_DIR = DATA_DIR / 'models'


def load_model(dataset, model):
    with open(MODEL_DIR / f'{model}_{dataset}.pkl', 'rb') as f:
        mod = pickle.load(f)
    return mod


def setup_val_data(dataset):
    # load original and fit preprocessor
    train_data = pd.read_csv(DATA_DIR / f'{dataset}.tsv', sep='\t')

    train_data.iloc[:, 203:] = np.log(train_data.iloc[:, 203:])
    proc = Processor(dataset)
    train_data = proc.fit_transform(train_data, non_feats=REF_COL)
    train_cols = train_data.drop(REF_COL, axis=1).columns.tolist()

    # load val data and repeat preprocessing steps
    sig_data = pd.read_csv(DATA_DIR / 'all_mean_AD_sig.tsv', sep='\t')
    ctrl_data = pd.read_csv(DATA_DIR / 'all_mean_AD_ctrl.tsv', sep='\t')
    val_data = pd.concat([sig_data, ctrl_data], axis=0)

    val_data.iloc[:, 198:] = np.log(val_data.iloc[:, 198:])
    val_data = proc.transform(val_data)

    ref = val_data[VAL_REF_COL]
    X = val_data.drop(VAL_REF_COL, axis=1)
    feats = X.columns.tolist()
    y = np.concatenate((np.ones(sig_data.shape[0]),
                        np.zeros(ctrl_data.shape[0])))
    y = y.astype(np.int64)

    print(set(feats) - set(train_cols))
    print(set(train_cols) - set(feats))
    assert feats == train_data.drop(REF_COL, axis=1).columns.tolist()

    X = X.values.astype(np.float32)
    return X, y, ref, feats
    

def internal_validate(dataset):
    print('Internal: ')

    scores = pd.read_csv(MODEL_DIR / f'preds_{dataset}.tsv', sep='\t')
    y = scores['label'].map(lambda x: x != 'control').astype(np.int64)

    print('nn: ')
    nn_score = scores['NN_score']
    AUC = roc_auc_score(y, nn_score)
    APR = average_precision_score(y, nn_score)
    nn_metric = (AUC, APR)
    print(nn_metric)

    glm_metric = None
    if 'GLM_score' in scores.columns:
        print('glm: ')
        glm_score = scores['GLM_score']
        AUC = roc_auc_score(y, glm_score)
        APR = average_precision_score(y, glm_score)
        glm_metric = (AUC, APR)
        print(glm_metric)


def external_validate(dataset):
    print('External: ')

    # load AD GWAS data, preprocess to match original dataset
    X, y, ref, feats = setup_val_data(dataset)

    # predict on AD GWAS and score
    print('nn: ')
    nn = load_model(dataset, 'nn')
    scores = nn.predict(X)

    AUC = roc_auc_score(y, scores)
    APR = average_precision_score(y, scores)
    print('AUC/APR: ', AUC, APR)

    pcs = np.percentile(scores, [5, 50, 95])
    print(pcs)
    print(np.mean(scores[y == 1]))
    print(np.mean(scores[y == 0]))

    try:
        print('glm: ')
        glm = load_model(dataset, 'glm')
        scores = glm.predict(X)

        AUC = roc_auc_score(y, scores)
        APR = average_precision_score(y, scores)
        print('AUC/APR: ', AUC, APR)

        pcs = np.percentile(scores, [5, 50, 95])
        print(pcs)
        print(np.mean(scores[y == 1]))
        print(np.mean(scores[y == 0]))
    except Exception as e:
        print(e)


if __name__ == '__main__':

    datasets = [
                'all_mean_top_w_matched_hg19_100',
                'all_mean_top_W_matched_hg19_200',
                'all_mean_top_W_matched_hg19_300',
                'all_mean_top_W_matched_hg19_500',
                'all_mean_top_W_matched_hg19_1000']

    for ds in datasets:
        print(ds)
        internal_validate(ds)

    for ds in datasets:
        print(ds)
        external_validate(ds)
