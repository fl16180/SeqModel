import argparse
from pathlib import Path
import math
import pandas as pd
import numpy as np
import optuna

import torch
from torch import nn
import torch.nn.functional as F

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, cross_val_predict, StratifiedKFold
from sklearn.metrics import plot_roc_curve, roc_auc_score
from sklearn.metrics import plot_precision_recall_curve, average_precision_score
from sklearn.linear_model import LogisticRegression

from skorch import NeuralNetClassifier
from skorch.callbacks import EpochScoring, LRScheduler

import sys
sys.path.append(str(Path(__file__).absolute().parent.parent))
from datasets import Processor

REF_COL = ['chr', 'pos_start_hg19', 'pos_end_hg19',
           'pos_start_hg38', 'pos_end_hg38', 'label',
           'variant', 'W_KS', 'P_KS']
DATA_DIR = Path('/oak/stanford/groups/zihuai/fredlu/processed/knockoffs')


def load_and_preprocess(tsv):
    data = pd.read_csv(DATA_DIR / tsv, sep='\t')
    
    # pre-shuffle data
    data = data.sample(frac=1, random_state=1).reset_index(drop=True)
    
    # transform roadmap to make features approximately more normal
    data.iloc[:, 203:] = np.log(data.iloc[:, 203:])

    # standardize data, deal with missing values, etc.
    proc = Processor(tsv)
    data = proc.fit_transform(data, non_feats=REF_COL)
    
    ref = data[REF_COL]
    X = data.drop(REF_COL, axis=1)
    y = data.label.map(lambda x: x != 'control')
    feats = X.columns.tolist()

    X = X.values.astype(np.float32)
    y = y.values.astype(np.int64)
    return X, y, ref, feats


def main(args):
    print('Loading data')
    X, y, ref, feats = load_and_preprocess(args.dataset)

    print('Initializing neural net')
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


    class SNNet(nn.Module):
        def __init__(self, n_input, n_units, dropout=0.1):
            super(SNNet, self).__init__()
            self.dense1 = nn.Linear(n_input, n_units[0], bias=False)
            self.dense2 = nn.Linear(n_units[0], n_units[1], bias=False)
            self.dense3 = nn.Linear(n_units[1], n_units[2], bias=False)
            self.final = nn.Linear(n_units[2], 2, bias=True)
            
            self.selu1 = nn.SELU()
            self.selu2 = nn.SELU()
            self.selu3 = nn.SELU()
            self.drop1 = nn.AlphaDropout(p=dropout)
            self.drop2 = nn.AlphaDropout(p=dropout)
            self.drop3 = nn.AlphaDropout(p=dropout)

        def forward(self, X, **kwargs):
            X = self.drop1(self.selu1(self.dense1(X)))
            X = self.drop2(self.selu2(self.dense2(X)))
            X = self.drop3(self.selu3(self.dense3(X)))
            X = F.softmax(self.final(X), dim=-1)
            return X

    print('Setting up trials')

    def objective(trial):
        nl = trial.suggest_categorical('n_layer', [2, 3])
        bs = trial.suggest_categorical('batch_size', [256])
        l2 = trial.suggest_uniform('l2', 1e-8, 1e-3)
        lr = trial.suggest_uniform('lr', 5e-5, 5e-3)
        eps = trial.suggest_categorical('epochs', [30, 40, 50])
        drop = trial.suggest_uniform('dropout', 0, 0.2)
        nodes1 = trial.suggest_categorical('nodes1', [200, 300, 400, 500, 600])
        nodes2 = trial.suggest_categorical('nodes2', [200, 300, 400, 500, 600])
        nodes3 = trial.suggest_categorical('nodes3', [200, 300, 400, 500, 600])
        
        auc = EpochScoring(scoring='roc_auc', lower_is_better=False)
        apr = EpochScoring(scoring='average_precision', lower_is_better=False)
        # lrs = LRScheduler(policy='StepLR', step_size=10, gamma=0.75)
        
        net = NeuralNetClassifier(
            FCNet,
            batch_size=bs,
            criterion=torch.nn.NLLLoss,
            optimizer=torch.optim.Adam,
            optimizer__weight_decay=l2,

            lr=lr,
            max_epochs=eps,

            iterator_train__shuffle=True,

            module__n_input=1206,
            module__n_units=(nodes1, nodes2, nodes3) if nl == 3 else (nodes1, nodes2),
            module__dropout=drop,

            callbacks=[auc, apr],

            train_split=None,
            verbose=0
        )
        
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1111)
        torch.manual_seed(1000)
        cv_scores = cross_val_predict(net, X, y, cv=kf, method='predict_proba', n_jobs=-1)
        return roc_auc_score(y, cv_scores[:, 1])

    print('Starting trials')
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=args.iter)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-d', type=str,
                        help='e.g. all_max_top_P_matched_hg19_100.tsv')
    parser.add_argument('--iter', '-i', type=int,
                        help='Number of search iterations')
    args = parser.parse_args()
    main(args)
