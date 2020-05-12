import argparse
import numpy as np
import pandas as pd

# hide sklearn deprecation message triggered within skorch
from warnings import simplefilter
simplefilter('ignore', category=FutureWarning)

import torch
from skorch import NeuralNetClassifier
from skorch.callbacks import EpochScoring, LRScheduler
from sklearn.model_selection import cross_val_predict, StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
from glmnet import LogitNet

from constants import *
from datasets import load_data_set, load_neighbors_set, Processor
import models
from models.evaluator import Evaluator

from utils.model_utils import save_model
from utils.data_utils import get_roadmap_col_order

from fit_cv_models import load_and_preprocess
from evaluate_scores import merge_with_validation_info

MODEL_CHOICES = ['standard', 'neighbors']

from pdb import set_trace as st


def sim_fit_model(args, X, y):

    auc = EpochScoring(scoring='roc_auc', lower_is_better=False)
    apr = EpochScoring(scoring='average_precision', lower_is_better=False)
    lrs = LRScheduler(policy='StepLR', step_size=10, gamma=0.5)

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

    # generate predictions
    torch.manual_seed(1000)
    net.fit(X, y)
    return net


def prioritize_variants(args, net, X_eval, y_eval, progress, itr=0):

    val_scores = net.predict_proba(X_eval)[:, 1]
    AUC = roc_auc_score(y_eval, val_scores)
    APR = average_precision_score(y_eval, val_scores)
    print('\tRemaining AUC ', np.round(AUC, 4))
    print('\tRemaining APR ', np.round(APR, 4))

    if args.strat == 'score':
        # select high-score variants
        thresh_score = sorted(val_scores, reverse=True)[args.n_select]
        selected = val_scores > thresh_score
    elif args.strat == 'random':
        selected = np.zeros_like(val_scores, dtype=bool)
        idx = np.random.choice(len(val_scores), args.n_select, replace=False)
        selected[idx] = True

    progress[itr][f'remain_AUC'] = np.round(AUC, 4)
    progress[itr][f'remain_APR'] = np.round(APR, 4)
    progress[itr][f'n_discovered'] = np.sum(y_eval[selected])
    progress[itr][f'n_remaining'] = np.sum(y_eval[~selected])
    return selected


def rebalance_new_datasets(args, X_train, y_train, X_eval, y_eval, selected):

    X_new = X_eval[selected, :]
    y_new = y_eval[selected]

    X_eval_remain = X_eval[~selected, :]
    y_eval_remain = y_eval[~selected]

    if args.no_retrain:
        X_train_new = X_train
        y_train_new = y_train
    else:
        X_train_new = np.vstack([X_train, X_new])
        y_train_new = np.concatenate([y_train, y_new])

    return X_train_new, y_train_new, X_eval_remain, y_eval_remain


def score_holdout(args, net, X_holdout, y_holdout, progress, itr=0):

    test_scores = net.predict_proba(X_holdout)[:, 1]
    AUC = roc_auc_score(y_holdout, test_scores)
    APR = average_precision_score(y_holdout, test_scores)
    print('\tHoldout AUC ', np.round(AUC, 4))
    print('\tHoldout APR ', np.round(APR, 4))

    progress[itr]['test_AUC'] = np.round(AUC, 4)
    progress[itr]['test_APR'] = np.round(APR, 4)


def run_simulation(args, X, y, X_eval, y_eval, X_holdout, y_holdout):
    progress_scores = {}
    X_train = X
    y_train = y

    for i in range(args.n_iter):
        print(f'Iter {i + 1}: ')
        progress_scores[i] = {'n_train': len(y_train)}

        net = sim_fit_model(args, X_train, y_train)
        score_holdout(args, net, X_holdout, y_holdout,
                      progress_scores, i)
        selected = prioritize_variants(args, net, X_eval, y_eval,
                                       progress_scores, i)
        X_train, y_train, X_eval, y_eval = \
            rebalance_new_datasets(args, X_train, y_train,
                                   X_eval, y_eval, selected)
        print(progress_scores)
    return progress_scores


if __name__ == '__main__':
    np.random.seed(1234)

    parser = argparse.ArgumentParser()
    parser.add_argument('--project', '-p', default='mpra_e116', choices=PROJ_CHOICES)
    parser.add_argument('--model', '-m', default='standard', choices=MODEL_CHOICES,
                        help='Which data/model to train on')
    parser.add_argument('--n_iter', '-n', default=5, type=int,
                        help='Number of validation iterations')
    parser.add_argument('--n_select', '-ns', default=1000,
                        help='Number of prioritized variants to validate each round')
    parser.add_argument('--strat', '-s', choices=['score', 'random'], default='score',
                        help='Strategy to prioritize variants')
    parser.add_argument('--no_retrain', '-nr', action='store_true',
                        help='Disable retraining each round with validated variants')
    parser.add_argument('--pval', default='bonf_01')
    parser.add_argument('--init', default=False, action='store_true',
                        help='Run first-time data setup')
    args = parser.parse_args()

    # load initial training data (mpra_e116)
    X, y = load_and_preprocess(args)
    print('Initial training data: ', args.project)

    if args.init:
        # load initial candidate pool (mpra_nova)
        evlt = Evaluator(trained_data=args.project, eval_data='mpra_nova')
        evlt.setup_data(args.model, split='all')
        X_eval, y_eval, ref = evlt.X, evlt.y, evlt.ref

        # need to re-merge with original to get label columns
        df = pd.concat([ref, pd.DataFrame(X_eval)], axis=1)
        df = merge_with_validation_info(df, 'mpra_nova')
        df.to_csv(PROCESSED_DIR / 'mpra_nova' / 'merged_Xy_with_dup.csv', index=False)
    else:
        df = pd.read_csv(PROCESSED_DIR / 'mpra_nova' / 'merged_Xy_with_dup.csv')

    bonf_01 = 0.01 / 29685
    bonf_05 = 0.05 / 29685

    ref = df[['chr', 'pos', 'Pool', 'pvalue_expr', 'padj_expr',
              'pvalue_allele', 'padj_allele', 'Label']].copy()
    df['Label'] = ref['pvalue_expr'] < bonf_05
    df.drop_duplicates(subset=['chr', 'pos', 'Label'], inplace=True)
    
    y_eval = df['Label'].values.astype(np.int64)
    X_eval = df.drop(ref.columns, axis=1).values.astype(np.float32)

    # prepare holdout validation set
    X_eval, X_holdout, y_eval, y_holdout = train_test_split(
        X_eval, y_eval, test_size=0.2, stratify=y_eval
    )

    print('Total candidates: ', len(y_eval))
    print('Total significant to be discovered: ', np.sum(y_eval))

    print('\n--- starting simulation ---')
    summary = run_simulation(args, X, y, X_eval, y_eval, X_holdout, y_holdout)

    table = pd.DataFrame.from_dict(summary, orient='index')
    method = args.strat
    retrain = 'retrain' if not args.no_retrain else 'no_retrain'
    n_select = args.n_select
    table['method'] = method
    table['retrain'] = retrain
    table['n_select'] = n_select
    table['iter'] = np.arange(table.shape[0])

    table.to_csv(PROCESSED_DIR / f'mpra_nova/output/online/{method}_{n_select}_{retrain}.csv',
                 index=False)
