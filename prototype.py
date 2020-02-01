import torch
from torch import nn
import torch.nn.functional as F

from sklearn.model_selection import GridSearchCV
from skorch import NeuralNetClassifier
from skorch.callbacks import EpochScoring

torch.manual_seed(9999)

from constants import *
from datasets.data_loader import *

from sklearn.preprocessing import StandardScaler


class MpraNet(nn.Module):
    def __init__(self, n_input, n_units, nonlin=F.sigmoid, l2=2e-6, dropout=0.4):
        super(MpraNet, self).__init__()
        self.num_units = num_units
        self.nonlin = nonlin
        self.dropout = dropout

        self.dense1 = nn.Linear(n_input, n_units[0])
        self.dropout = nn.Dropout(dropout)
        self.dense2 = nn.Linear(n_units[0], n_units[1])
        self.final = nn.Linear(n_units[1], 2)

    def forward(self, X, **kwargs):
        X = self.nonlin(self.dense1(X))
        X = self.dropout(X)
        X = self.nonlin(self.dense2(X))
        X = self.dropout(X)
        X = F.softmax(self.final(X), dim=-1)
        return X


project = 'mpra_e116'
project_dir = PROCESSED_DIR / project


train_df = load_train_set(project, make_new=False)

na_thresh = 0.05
na_filt = (train_df.isna().sum() > na_thresh * len(train_df))
omit_cols = train_df.columns[na_filt].tolist()
omit_cols += [x + '_PHRED' for x in omit_cols]

train_df.drop(omit_cols, axis=1, inplace=True)
mean_cols = train_df.mean()
train_df.fillna(mean_cols, inplace=True)



np.save(project_dir / 'train_means.npy', mean_cols)
# train_df.to_csv('train_dat.csv', index=False)

y_train = train_df['Label']
X_train = train_df.drop('Label', axis=1)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

net = NeuralNetClassifier(
    MpraNet,
    max_epochs=30,
    lr=1e-4,
    optimizer__n_input=1076,
    optimizer__units=(450, 250),
    optimizer__l2=1e-6,
    optimizer__dropout=0.4
    callbacks=[auc],
)

params = {
    'lr': [0.05, 0.1],
    'module__num_units': [10, 20],
    'module__dropout': [0, 0.5],
    'optimizer__nesterov': [False, True],
}
gs = GridSearchCV(net, params, refit=False, cv=3, scoring='accuracy', verbose=2)


net.fit(X_train, y_train)

y_pred = net.predict(X[:5])
y_pred

y_proba = net.predict_proba(X[:5])
y_proba

auc = EpochScoring(scoring='roc_auc', lower_is_better=False)

# test_df = load_test_set(project, make_new=False)

# test_df.drop(omit_cols, axis=1, inplace=True)
# test_df.fillna(mean_cols, inplace=True)

# test_df.to_csv('test_dat.csv', index=False)


