import argparse
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

REF_COL = ['chr', 'pos_start_hg19', 'pos_end_hg19',
           'pos_start_hg38', 'pos_end_hg38', 'label',
           'variant', 'W_KS', 'P_KS']
DATA_DIR = Path('/oak/stanford/groups/zihuai/fredlu/processed/knockoffs')
MODEL_DIR = DATA_DIR / 'models'
DATASETS = ['all_mean_top_W_matched_hg19_100',
            'all_mean_top_W_matched_hg19_200',
            'all_mean_top_W_matched_hg19_300',
            'all_mean_top_W_matched_hg19_500',
            'all_mean_top_W_matched_hg19_1000',
            'all_mean_top_P_matched_hg19_100',
            'all_mean_top_P_matched_hg19_200',
            'all_mean_top_P_matched_hg19_300',
            'all_mean_top_P_matched_hg19_500',
            'all_mean_top_P_matched_hg19_1000',
            'all_max_top_W_matched_hg19_100',
            'all_max_top_W_matched_hg19_200',
            'all_max_top_W_matched_hg19_300',
            'all_max_top_W_matched_hg19_500',
            'all_max_top_W_matched_hg19_1000',
            'all_max_top_P_matched_hg19_100',
            'all_max_top_P_matched_hg19_200',
            'all_max_top_P_matched_hg19_300',
            'all_max_top_P_matched_hg19_500',
            'all_max_top_P_matched_hg19_1000']


def load_and_preprocess(tsv):
    data = pd.read_csv(DATA_DIR / tsv, sep='\t')
    
    # transform roadmap to make features approximately more normal
    data.iloc[:, 203:] = np.log(data.iloc[:, 203:])
    
    proc = Processor(tsv)
    data = proc.fit_transform(data, non_feats=REF_COL)

    ref = data[REF_COL]
    X = data.drop(REF_COL, axis=1)
    y = data.label.map(lambda x: x != 'control')
    feats = X.columns.tolist()

    X = X.values.astype(np.float32)
    y = y.values.astype(np.int64)
    return X, y, ref, feats


class FCNet(nn.Module):
    def __init__(self, n_input, n_units, nonlin=torch.sigmoid, dropout=0.2):
        super(FCNet, self).__init__()
        self.nonlin = nonlin
        self.dropout = nn.Dropout(0.2)
        self.dense1 = nn.Linear(n_input, n_units[0])
        self.dense2 = nn.Linear(n_units[0], n_units[1])
        if len(n_units) == 3:
            self.dense3 = nn.Linear(n_units[1], n_units[2])
            self.final = nn.Linear(n_units[2], 2)
        else:
            self.final = nn.Linear(n_units[1], 2)
        self.nlayer = len(n_units)

    def forward(self, X, **kwargs):
        X = self.nonlin(self.dense1(X))
        X = self.dropout(X)
        X = self.nonlin(self.dense2(X))
        X = self.dropout(X)
        if self.nlayer == 3:
            X = self.nonlin(self.dense3(X))
        X = F.softmax(self.final(X), dim=-1)
        return X


def fit_nn(args, X, y):

    print('Neural net:')
    auc = EpochScoring(scoring='roc_auc', lower_is_better=False)
    apr = EpochScoring(scoring='average_precision', lower_is_better=False)
    # lrs = LRScheduler(policy='StepLR', step_size=10, gamma=0.7)

    params = param_lookup[args.dataset]

    net = NeuralNetClassifier(
        FCNet,
        criterion=torch.nn.NLLLoss,
        optimizer=torch.optim.Adam,
        iterator_train__shuffle=True,
        module__n_input=1206,        
        callbacks=[auc, apr],
        train_split=None,
        verbose=0,
        **params
    )

    # fit on full dataset and save model
    torch.manual_seed(1000)
    net.fit(X, y)

    # net.save_params(f_params=MODEL_DIR / f'nn_{args.dataset}.pkl')
    with open(MODEL_DIR / f'nn_{args.dataset}.pkl', 'wb') as f:
        pickle.dump(net, f)

    # generate in-dataset CV predictions
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1111)
    torch.manual_seed(1000)
    cv_scores = cross_val_predict(net, X, y, cv=kf,
                                  method='predict_proba', n_jobs=-1)
    AUC = roc_auc_score(y, cv_scores[:, 1])
    APR = average_precision_score(y, cv_scores[:, 1])
    print('\tAUC ', np.round(AUC, 4))
    print('\tAPR ', np.round(APR, 4))

    np.save(MODEL_DIR / f'nn_{args.dataset}.npy', cv_scores[:, 1])


def fit_glm(args, X, y):
    print('GLM')

    # fit on full dataset and save model
    np.random.seed(1000)
    glm = LogitNet(alpha=0.5, n_lambda=20, n_jobs=5)
    glm.fit(X, y)

    with open(MODEL_DIR / f'glm_{args.dataset}.pkl', 'wb') as f:
        pickle.dump(glm, f)

    print('In-sample: ')
    tmp = glm.predict_proba(X)
    AUC = roc_auc_score(y, tmp[:, 1])
    APR = average_precision_score(y, tmp[:, 1])
    print('\tAUC ', np.round(AUC, 4))
    print('\tAPR ', np.round(APR, 4))

    print('Out-of-sample: ')
    print(glm.lambda_best_)

    # generate in-dataset CV predictions
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1111)
    np.random.seed(1000)
    glm = LogitNet(alpha=0.5, n_lambda=1, lambda_path=[glm.lambda_best_])
    cv_scores = cross_val_predict(glm, X, y, cv=kf,
                                  method='predict_proba', n_jobs=-1)
    AUC = roc_auc_score(y, cv_scores[:, 1])
    APR = average_precision_score(y, cv_scores[:, 1])
    print('\tAUC ', np.round(AUC, 4))
    print('\tAPR ', np.round(APR, 4))

    np.save(MODEL_DIR / f'glm_{args.dataset}.npy', cv_scores[:, 1])


def add_preds_to_data(args):
    dataset = pd.read_csv(DATA_DIR / f'{args.dataset}.tsv', sep='\t')

    try:
        preds = np.load(MODEL_DIR / f'nn_{args.dataset}.npy')
        print(preds.shape, dataset.shape)

        cols = dataset.columns.tolist()
        dataset['NN_score'] = preds
        newcols = cols[:9] + ['NN_score'] + cols[9:]
        dataset = dataset.loc[:, newcols]
    except Exception as e:
        print(e)

    try:
        preds = np.load(MODEL_DIR / f'glm_{args.dataset}.npy')
        print(preds.shape, dataset.shape)

        cols = dataset.columns.tolist()
        dataset['GLM_score'] = preds
        newcols = cols[:9] + ['GLM_score'] + cols[9:]
        dataset = dataset.loc[:, newcols]
    except Exception as e:
        print(e)

    dataset.to_csv(MODEL_DIR / f'preds_{args.dataset}.tsv', sep='\t', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-d', type=str, choices=DATASETS,
                        help='dataset file')
    parser.add_argument('--nn', '-nn', action='store_true', default=False,
                        help='fit neural net')
    parser.add_argument('--glm', '-glm', action='store_true', default=False,
                        help='fit GLM')
    parser.add_argument('--merge', '-m', action='store_true', default=False,
                        help='add predictions to datasets')
    args = parser.parse_args()

    # setup data
    print(f'Running models on {args.dataset}')
    X, y, ref, feats = load_and_preprocess(args.dataset + '.tsv')

    if args.nn:
        fit_nn(args, X, y)

    if args.glm:
        fit_glm(args, X, y)

    if args.merge:
        add_preds_to_data(args)
