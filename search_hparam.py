import argparse
import numpy as np

# hide sklearn deprecation message triggered within skorch
from warnings import simplefilter
simplefilter('ignore', category=FutureWarning)

import torch
from skorch import NeuralNetClassifier
from skorch.callbacks import LRScheduler
from skorch.callbacks import EpochScoring
from sklearn.metrics import plot_roc_curve, roc_auc_score
from sklearn.metrics import plot_precision_recall_curve, average_precision_score

from constants import *
from datasets import *
import models
from utils.model_utils import *
from utils.data_utils import get_roadmap_col_order


DATA_CHOICES = ['mpra', 'mpra+scores', 'neighbor', 'neighbor+scores']

MODEL_CFG = {
    'mpra': None,
    'mpra+scores': None,
    'neighbor': None,
    'neighbor+scores': None
}


def fit_model(args):
    torch.manual_seed(1000)
    print(f'Fitting model for {args.data}:')
    project = args.project

    auc = EpochScoring(scoring='roc_auc', lower_is_better=False)
    apr = EpochScoring(scoring='average_precision', lower_is_better=False)

    if args.data == 'mpra': 
        train_df = load_train_set(project, datasets=['roadmap'])
        proc = Processor(project)
        train_df = proc.fit_transform(train_df, na_thresh=0.05)
        proc.save(args.data)

        X_train = train_df.drop(['chr', 'pos', 'Label'], axis=1) \
                          .values \
                          .astype(np.float32)
        y_train = train_df['Label'].values.astype(np.int64)

        net = NeuralNetClassifier(
            models.MpraDense,
            
            batch_size=256,
            optimizer=torch.optim.Adam,
            optimizer__weight_decay=2e-6,
            lr=1e-4,
            max_epochs=20,
            module__n_input=1016,
            module__n_units=(400, 250),
            module__dropout=0.3,
            
            callbacks=[auc, apr],
            iterator_train__shuffle=True,
            train_split=None
        )

    elif args.data == 'mpra+scores':
        train_df = load_train_set(project, datasets=['roadmap', 'eigen', 'regbase'])
        proc = Processor(project)
        train_df = proc.fit_transform(train_df, na_thresh=0.05)
        proc.save(args.data)

        X_train = train_df.drop(['chr', 'pos', 'Label'], axis=1) \
                          .values \
                          .astype(np.float32)
        y_train = train_df['Label'].values.astype(np.int64)

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

    elif args.data == 'neighbor':
        X_train = load_train_neighbors(project).astype(np.float32)
        
        tmp = load_train_set(project, datasets=['roadmap', 'eigen', 'regbase'],
                             make_new=False)
        y_train = tmp['Label'].values.astype(np.int64)
        assert X_train.shape[0] == y_train.shape[0]

        net = NeuralNetClassifier(
            models.MpraCNN,

            batch_size=256,
            optimizer=torch.optim.Adam,
            optimizer__weight_decay=1e-4,
            lr=5e-4,
            max_epochs=20,

            callbacks=[auc, apr],
            iterator_train__shuffle=True          
        )

    elif args.data == 'neighbor+scores':
        print('\tLoading neighbors')
        X_neighbor = load_train_neighbors(project).astype(np.float32)
        
        print('\tLoading scores')
        train_df = load_train_set(project, datasets=['roadmap', 'eigen', 'regbase'])
        proc = Processor(project)
        train_df = proc.fit_transform(train_df, na_thresh=0.05)
        proc.save(args.data)

        print('\tArranging data')
        rm_cols = [f'{x}-E116' for x in ROADMAP_MARKERS]
        # rm_cols = [x for x in get_roadmap_col_order(order='marker') if 'E116' in x]
        X_score = train_df.drop(['chr', 'pos', 'Label'] + rm_cols, axis=1) \
                          .values \
                          .astype(np.float32)
        y_train = train_df['Label'].values.astype(np.int64)
        X_train = (X_neighbor, X_score)

        net = NeuralNetClassifier(
            models.MpraFullCNN,

            batch_size=256,
            optimizer=torch.optim.Adam,
            optimizer__weight_decay=0,
            lr=5e-4,
            max_epochs=20,

            callbacks=[auc, apr],
            iterator_train__shuffle=True          
        )

        # import sys; sys.exit()

    net.fit(X_train, y_train)
    class_pred = net.predict(X_train)
    score_pred = net.predict_proba(X_train)

    print('\tAUROC: ', roc_auc_score(y_train, score_pred[:, 1]))
    print('\tAUPR: ', average_precision_score(y_train, score_pred[:, 1]))

    save_model(net, project, args.data)


def evaluate_model(args):
    print(f"Evaluating model for {args.data}:")
    project = args.project
    net = load_model(project, args.data)

    X_test = load_test_neighbors(project)
    X_test = X_test.astype(np.float32)

    tmp = load_test_set(project, datasets=['roadmap', 'eigen', 'regbase'])
    y_test = tmp['Label'].values.astype(np.int64)

    # test_df = load_test_set(project, datasets=['roadmap', 'eigen', 'roadmap'])
    # proc = Processor(project)
    # proc.load(args.data)
    # test_df = proc.transform(test_df)

    # X_test = test_df.drop(['chr', 'pos', 'Label'], axis=1) \
    #                 .values \
    #                 .astype(np.float32)
    # y_test = test_df['Label'].values.astype(np.int64)

    class_pred = net.predict(X_test)
    score_pred = net.predict_proba(X_test)

    print('\tAUROC: ', roc_auc_score(y_test, score_pred[:, 1]))
    print('\tAUPR: ', average_precision_score(y_test, score_pred[:, 1]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', '-p', choices=PROJ_CHOICES, required=True)
    parser.add_argument('--data', '-d', default='mpra+scores', choices=DATA_CHOICES,
                        help='Which data/model to train on')
    parser.add_argument('--full', default=False,
                        help='Fit all models (overrides --data)')
    parser.add_argument('--evaluate', '-e', action='store_true', default=False,
                        help='Evaluate model on test set after fitting')
    args = parser.parse_args()

    fit_model(args)
    if args.evaluate:
        evaluate_model(args)