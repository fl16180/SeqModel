import torch
from torch import nn
import torch.nn.functional as F


class MpraDense(nn.Module):
    def __init__(self, n_input, n_units, nonlin=torch.sigmoid, dropout=0.3):
        super(MpraDense, self).__init__()
        self.nonlin = nonlin
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


class MpraCNN(nn.Module):
    def __init__(self, nonlin=torch.relu, dropout=0.0):
        super(MpraCNN, self).__init__()
        self.nonlin = nonlin
        
        # input (N, 8, 81)
        self.conv1 = nn.Conv1d(8, 16, kernel_size=4)
        self.conv2 = nn.Conv1d(16, 16, kernel_size=7)

        # self.conv1_drop = nn.Dropout
        self.dense1 = nn.Linear(16 * 16, 100)
        self.dense2 = nn.Linear(100, 2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X, **kwargs):
        X = self.nonlin(F.avg_pool1d(self.conv1(X), 2))
        X = self.nonlin(F.avg_pool1d(self.conv2(X), 2))

        X = X.view(-1, X.size(1) * X.size(2))

        X = self.nonlin(self.dense1(X))
        # X = self.dropout(X)
        X = F.softmax(self.dense2(X), dim=-1)
        return X


class MpraFullCNN(nn.Module):
    def __init__(self, nonlin=torch.relu, dropout=0.0):
        super(MpraFullCNN, self).__init__()
        self.nonlin = nonlin
        
        # input (N, 8, 81)
        self.conv1 = nn.Conv1d(8, 16, kernel_size=4)
        self.conv2 = nn.Conv1d(16, 16, kernel_size=7)

        self.dense_sc = nn.Linear(1071, 512)
        self.dense1 = nn.Linear(16 * 16 + 512, 256)

        # self.conv1_drop = nn.Dropout
        # self.dense1 = nn.Linear(16 * 16 + 1071, 200)
        self.dense2 = nn.Linear(256, 2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X, **kwargs):
        X_neigh, X_score = X

        # CNN layers on neighbor sequence
        X_neigh = self.nonlin(F.avg_pool1d(self.conv1(X_neigh), 2))
        X_neigh = self.nonlin(F.avg_pool1d(self.conv2(X_neigh), 2))
        X_neigh = X_neigh.view(-1, X_neigh.size(1) * X_neigh.size(2))

        X_score = self.nonlin(self.dense_sc(X_score))
        X_score = self.dropout(X_score)

        X = torch.cat([X_neigh, X_score], dim=1)
        X = self.nonlin(self.dense1(X))
        X = self.dropout(X)
        X = F.softmax(self.dense2(X), dim=-1)
        return X
