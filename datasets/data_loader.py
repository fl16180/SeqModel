import pandas as pd
from constants import PROCESSED_DIR


def load_train_set(project, datasets=['roadmap', 'eigen', 'regbase']):
    proj_loc = PROCESSED_DIR / project

    train = pd.read_csv(proj_loc / 'train_label.csv')

    for ds in datasets:
        df = pd.read_csv(proj_loc / f'train_{ds}.csv')
        if 'ref' in df.columns:
            df.drop('ref', axis=1, inplace=True)
        train = pd.merge(train, df, on=['chr', 'pos'], suffixes=('', '__y'))
        train.drop(list(train.filter(regex='__y$')), axis=1, inplace=True)

    y_train = train['Label']
    x_train = train.drop('Label', axis=1)
    return x_train, y_train


def load_test_set(project, datasets=['roadmap', 'eigen', 'regbase']):
    proj_loc = PROCESSED_DIR / project

    test = pd.read_csv(proj_loc / 'test_label.csv')

    for ds in datasets:
        df = pd.read_csv(proj_loc / f'test_{ds}.csv')
        if 'ref' in df.columns:
            df.drop('ref', axis=1, inplace=True)
        test = pd.merge(test, df, on=['chr', 'pos'], suffixes=('', '__y'))
        test.drop(list(test.filter(regex='__y$')), axis=1, inplace=True)

    y_test = test['Label']
    x_test = test.drop('Label', axis=1)
    return x_test, y_test
