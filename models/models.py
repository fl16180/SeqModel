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
    def __init__(self, n_input):
        super(MpraCNN, self).__init__()
        self.conv1 = nn.Conv1d(8, 4, kernel_size=5)
        # self.conv1_drop = nn.Dropout
        self.dense1 = nn.Linear()




class Cnn(nn.Module):
    def __init__(self, dropout=0.5):
        super(Cnn, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv2_drop = nn.Dropout2d(p=dropout)
        self.fc1 = nn.Linear(1600, 100) # 1600 = number channels * width * height
        self.fc2 = nn.Linear(100, 10)
        self.fc1_drop = nn.Dropout(p=dropout)

    def forward(self, x):
        x = torch.relu(F.max_pool2d(self.conv1(x), 2))
        x = torch.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        
        # flatten over channel, height and width = 1600
        x = x.view(-1, x.size(1) * x.size(2) * x.size(3))
        
        x = torch.relu(self.fc1_drop(self.fc1(x)))
        x = torch.softmax(self.fc2(x), dim=-1)
        return x