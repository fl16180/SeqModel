
import numpy as np

# from sklearn.model_selection import GridSearchCV
# from skorch import NeuralNetClassifier
# from skorch.callbacks import EpochScoring

# torch.manual_seed(9999)

from constants import *
from datasets.data_loader import *
from datasets.processor import Processor


project = 'mpra_e116'
project_dir = PROCESSED_DIR / project

train_df = load_train_set(project, make_new=True)
test_df = load_test_set(project, make_new=True)

test_df = load_test_set('mpra_deseq2', make_new=True)

# proc = Processor(project)
# train_df = proc.fit_transform(train_df, na_thresh=0.05)

# X_train = train_df.drop(['chr', 'pos', 'Label'], axis=1).astype(np.float32)
# y_train = train_df['Label'].astype(np.int64)



# test_df = load_test_set(project, make_new=True)
# test_df = load_test_set('mpra_deseq2', make_new=True)



# test_df = proc.transform(test_df)

# X_test = test_df.drop(['chr', 'pos', 'Label'], axis=1).astype(np.float32)
# y_test = test_df['Label'].astype(np.int64)



# from sklearn.linear_model import LogisticRegression
# # lr = LogisticRegression(max_iter=10000, C=10.0)
# lr = RandomForestRegressor()
# lr.fit(X_train, y_train)
# # p_test = lr.predict_proba(X_test)
# p_test = lr.predict(X_test)

# from scipy.stats import pearsonr
# print(pearsonr(p_test, y_test))

# thresh = 1e-4
# y_score = np.zeros(len(y_test))
# y_score[y_test < thresh] = 1

# # y_pred = np.zeros(len(y_test))
# # p_test[p_test < thresh] = 1


# from sklearn.metrics import plot_roc_curve, roc_auc_score
# from sklearn.metrics import plot_precision_recall_curve, average_precision_score
# print(roc_auc_score(y_score, p_test[:, 1]))
# print(average_precision_score(y_score, p_test[:, 1]))



# train_df.to_csv('e116_train.csv', index=False)


# class MpraNet(nn.Module):
#     def __init__(self, n_input, n_units, nonlin=torch.sigmoid, dropout=0.4):
#         super(MpraNet, self).__init__()
#         self.nonlin = nonlin

#         self.dense1 = nn.Linear(n_input, n_units[0])
#         self.dropout = nn.Dropout(dropout)
#         self.dense2 = nn.Linear(n_units[0], n_units[1])
#         self.final = nn.Linear(n_units[1], 2)

#     def forward(self, X, **kwargs):
#         X = self.nonlin(self.dense1(X))
#         X = self.dropout(X)
#         X = self.nonlin(self.dense2(X))
#         X = self.dropout(X)
#         X = F.softmax(self.final(X), dim=-1)
#         return X



# net = NeuralNetClassifier(
#     MpraNet,
#     batch_size=256,

#     criterion=CostSensitiveLoss,

#     optimizer=torch.optim.Adam,
#     optimizer__weight_decay=2e-6,

#     lr=1e-4,
#     max_epochs=20,

#     iterator_train__shuffle=True,

#     module__n_input=1079,
#     module__n_units=(400, 250),

#     module__dropout=0.3,
#     callbacks=[auc, apr],
# )

# torch.manual_seed(1000)

# net.fit(x_train, y_train)

# c_test = net.predict(x_test)
# p_test = net.predict_proba(x_test)







# test_df.to_csv('e116_test.csv', index=False)


# net = NeuralNetClassifier(
#     MpraNet,
#     max_epochs=30,
#     lr=1e-4,
#     optimizer__n_input=1076,
#     optimizer__units=(450, 250),
#     optimizer__l2=1e-6,
#     optimizer__dropout=0.4
#     callbacks=[auc],
# )

# params = {
#     'lr': [0.05, 0.1],
#     'module__num_units': [10, 20],
#     'module__dropout': [0, 0.5],
#     'optimizer__nesterov': [False, True],
# }
# gs = GridSearchCV(net, params, refit=False, cv=3, scoring='accuracy', verbose=2)


# net.fit(X_train, y_train)

# y_pred = net.predict(X[:5])
# y_pred

# y_proba = net.predict_proba(X[:5])
# y_proba

# auc = EpochScoring(scoring='roc_auc', lower_is_better=False)



