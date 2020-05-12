import argparse
import numpy as np
import pandas as pd

# hide sklearn deprecation message triggered within skorch
from warnings import simplefilter
simplefilter('ignore', category=FutureWarning)

import torch
from skorch import NeuralNetClassifier
from skorch.callbacks import EpochScoring, LRScheduler
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score
from glmnet import LogitNet

from constants import *
from datasets import load_data_set, load_neighbors_set, Processor
import models
from utils.model_utils import save_model
from utils.data_utils import get_roadmap_col_order

N_NEIGH = 40
SAMPLE_RES = 25

MODEL_CHOICES = ['glm', 'standard', 'neighbors']

MODEL_CFG = {
    'mpra': None,
    'mpra+scores': None,
    'neighbor': None,
    'neighbor+scores': None
}


def save_scores(args, scores, y_check):
    proj_dir = PROCESSED_DIR / args.project

    try:
        df = pd.read_csv(proj_dir / 'output' / f'nn_preds_{args.project}.csv',
                         sep=',')
    except FileNotFoundError:
        df = pd.read_csv(proj_dir / 'matrix_all.csv', sep=',')
        cols = ['chr', 'pos', 'Label', f'NN_{args.model}']
        df = df.loc[:, cols]
    
    assert np.all(y_check == df.Label)
    
    df[f'NN_{args.model}'] = scores
    df.to_csv(proj_dir / 'output' / f'nn_preds_{args.project}.csv',
              sep=',', index=False)


def load_and_preprocess(args):
    print(f'Loading data for {args.model}:')
    project = args.project

    if args.model in ['glm', 'standard']:
        df = load_data_set(project, split='all', make_new=False)
        roadmap_cols = get_roadmap_col_order(order='marker')
        df[roadmap_cols] = np.log(df[roadmap_cols])

        proc = Processor(project)
        df = proc.fit_transform(df, na_thresh=0.05)
        proc.save(args.model)

        X = df.drop(['chr', 'pos', 'Label'], axis=1) \
              .values \
              .astype(np.float32)
        y = df['Label'].values.astype(np.int64)

    elif args.model == 'neighbors':
        X_neighbor = load_neighbors_set(project, split='all',
                                        n_neigh=N_NEIGH,
                                        sample_res=SAMPLE_RES)
        X_neighbor = np.log(X_neighbor.astype(np.float32))

        df = load_data_set(project, split='all',
                           make_new=False)
        roadmap_cols = get_roadmap_col_order(order='marker')
        df[roadmap_cols] = np.log(df[roadmap_cols])

        proc = Processor(project)
        df = proc.fit_transform(df, na_thresh=0.05)
        proc.save(args.model)

        rm_cols = [f'{x}-E116' for x in ROADMAP_MARKERS]
        # rm_cols = get_roadmap_col_order(order='marker')
        X_score = df.drop(['chr', 'pos', 'Label'] + rm_cols, axis=1) \
                    .values \
                    .astype(np.float32)
        y = df['Label'].values.astype(np.int64)
        assert X_neighbor.shape[0] == y.shape[0]

        X_neighbor = X_neighbor.reshape(
            X_neighbor.shape[0], X_neighbor.shape[1] * X_neighbor.shape[2])
        X = np.hstack((X_score, X_neighbor))

    print('X.shape: ', X.shape)
    print('y.shape: ', y.shape)
    return X, y


def fit_model(args, X, y):
    print(f'Fitting model for {args.model}:')

    auc = EpochScoring(scoring='roc_auc', lower_is_better=False)
    apr = EpochScoring(scoring='average_precision', lower_is_better=False)
    lrs = LRScheduler(policy='StepLR', step_size=10, gamma=0.5)

    if args.model == 'glm':
        glm = LogitNet(alpha=0.5, n_lambda=50, n_jobs=-1)
        glm.fit(X, y)

        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1111)
        net = LogitNet(alpha=0.5, n_lambda=1, lambda_path=[glm.lambda_best_])

    if args.model == 'standard':

        net = NeuralNetClassifier(
            models.MpraDense,
            
            batch_size=256,
            optimizer=torch.optim.Adam,
            optimizer__weight_decay=2e-6,
            lr=1e-4,
            max_epochs=20,
            module__n_input=1079,
            module__n_units=(400, 250),
            module__dropout=0.3,
            
            callbacks=[auc, apr],
            iterator_train__shuffle=True,
            train_split=None
        )

    elif args.model == 'neighbors':

        net = NeuralNetClassifier(
            models.MpraFullCNN,

            batch_size=256,
            optimizer=torch.optim.Adam,
            optimizer__weight_decay=1e-2,
            lr=5e-5,
            max_epochs=20,

            callbacks=[auc, apr],
            iterator_train__shuffle=True,
            train_split=None      
        )

    # generate CV predictions
    np.random.seed(1000)
    torch.manual_seed(1000)
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1000)
    cv_scores = cross_val_predict(net, X, y, cv=kf,
                                  method='predict_proba', n_jobs=-1)
    AUC = roc_auc_score(y, cv_scores[:, 1])
    APR = average_precision_score(y, cv_scores[:, 1])
    print('\tAUC ', np.round(AUC, 4))
    print('\tAPR ', np.round(APR, 4))

    save_scores(args, cv_scores[:, 1], y)

    # refit and store model on all data
    net.fit(X, y)
    save_model(net, args.project, args.model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', '-p', choices=PROJ_CHOICES, required=True)
    parser.add_argument('--model', '-m', default='standard', choices=MODEL_CHOICES,
                        help='Which data/model to train on')
    args = parser.parse_args()

    X, y = load_and_preprocess(args)
    fit_model(args, X, y)
