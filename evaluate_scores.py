import argparse
import numpy as np

from sklearn.metrics import plot_roc_curve, roc_auc_score
from sklearn.metrics import plot_precision_recall_curve, average_precision_score

from constants import *
from datasets import *
from utils.model_utils import *
from utils.data_utils import get_roadmap_col_order

# from sklearn.linear_model import LogisticRegressionCV
from glmnet import LogitNet

project = 'mpra_e116'
def baseline1():
    print('BASELINE 1')
    train_df = load_train_set(project, datasets=['roadmap', 'eigen', 'regbase'], make_new=False)
    proc = Processor(project)
    train_df = proc.fit_transform(train_df, na_thresh=0.05)
    # proc.save(args.data)

    X_train = train_df.drop(['chr', 'pos', 'Label'], axis=1) \
                        .values \
                        .astype(np.float32)
    y_train = train_df['Label'].values.astype(np.int64)

    # lr = LogisticRegressionCV

    # from sklearn.datasets import load_breast_cancer
    # X, y = load_breast_cancer(True)
    # X_train = X[:500, :]
    # y_train = y[:500]

    # X_test = X[500:, :]
    # y_test = y[500:]

    print('fitting')
    lr = LogitNet(alpha=0.5, n_lambda=20, n_jobs=6, verbose=True)
    lr.fit(X_train, y_train)

    print('Train set: ')
    class_pred = lr.predict(X_train)
    score_pred = lr.predict_proba(X_train)
    print('\tAUROC: ', roc_auc_score(y_train, score_pred[:, 1]))
    print('\tAUPR: ', average_precision_score(y_train, score_pred[:, 1]))

  
    test_df = load_test_set(project, datasets=['roadmap', 'eigen', 'roadmap'], make_new=False)
    proc = Processor(project)
    proc.load('mpra+scores')
    test_df = proc.transform(test_df)

    X_test = test_df.drop(['chr', 'pos', 'Label'], axis=1) \
                    .values \
                    .astype(np.float32)
    y_test = test_df['Label'].values.astype(np.int64)
    
    print('Test set: ')
    class_pred = lr.predict(X_test)
    score_pred = lr.predict_proba(X_test)
    print('\tAUROC: ', roc_auc_score(y_test, score_pred[:, 1]))
    print('\tAUPR: ', average_precision_score(y_test, score_pred[:, 1]))


def baseline2():
    print('BASELINE 2')
    X_neighbor = load_train_neighbors(project).astype(np.float32)
    
    train_df = load_train_set(project, datasets=['roadmap', 'eigen', 'regbase'],
                                make_new=False)
    proc = Processor(project)
    train_df = proc.fit_transform(train_df, na_thresh=0.05)
    # proc.save(args.data)

    rm_cols = [f'{x}-E116' for x in ROADMAP_MARKERS]
    X_score = train_df.drop(['chr', 'pos', 'Label'] + rm_cols, axis=1) \
                        .values \
                        .astype(np.float32)
    y_train = train_df['Label'].values.astype(np.int64)
    assert X_neighbor.shape[0] == y_train.shape[0]
    X_neighbor = X_neighbor.reshape(X_neighbor.shape[0], X_neighbor.shape[1] * X_neighbor.shape[2])
    X_train = np.hstack((X_neighbor, X_score))

    lr = LogitNet(alpha=0.5, n_lambda=20, n_jobs=6)
    lr.fit(X_train, y_train)

    print('Train set: ')
    class_pred = lr.predict(X_train)
    score_pred = lr.predict_proba(X_train)
    print('\tAUROC: ', roc_auc_score(y_train, score_pred[:, 1]))
    print('\tAUPR: ', average_precision_score(y_train, score_pred[:, 1]))

    X_neighbor = load_test_neighbors(project).astype(np.float32)
    test_df = load_test_set(project, datasets=['roadmap', 'eigen', 'regbase'],
                                make_new=False)
    proc = Processor(project)
    proc.load('neighbor+scores')
    test_df = proc.transform(test_df)

    rm_cols = [f'{x}-E116' for x in ROADMAP_MARKERS]
    X_score = test_df.drop(['chr', 'pos', 'Label'] + rm_cols, axis=1) \
                        .values \
                        .astype(np.float32)
    y_test = test_df['Label'].values.astype(np.int64)

    X_neighbor = X_neighbor.reshape(X_neighbor.shape[0], X_neighbor.shape[1] * X_neighbor.shape[2])
    X_test = np.hstack((X_neighbor, X_score))

    print('Test set: ')
    class_pred = lr.predict(X_test)
    score_pred = lr.predict_proba(X_test)
    print('\tAUROC: ', roc_auc_score(y_test, score_pred[:, 1]))
    print('\tAUPR: ', average_precision_score(y_test, score_pred[:, 1]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', '-p', required=True)
    parser.add_argument('--baseline', '-b', type=int, default=0)
    parser.add_argument('--data', '-d', default='mpra+scores')
    args = parser.parse_args()

    if args.baseline == 1:
        baseline1()
    elif args.baseline == 2:
        baseline2()
