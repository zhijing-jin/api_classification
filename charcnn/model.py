from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F


class CharCNN(nn.Module):
    def __init__(self, n_class=4, n_char=70, max_seq_len=1014,
                 kernel_sizes=[7, 7, 3, 3, 3, 3],
                 channel_size=256, pool_size=3, fc_size=1024, dropout=0.5):
        super(CharCNN, self).__init__()
        self.n_class = n_class
        self.n_char = n_char
        self.max_seq_len = max_seq_len
        self.kernel_sizes = kernel_sizes
        self.channel_size = channel_size
        self.pool_size = pool_size
        self.dropout = dropout
        self.final_linear_len = (max_seq_len - 96) // 27

        self.conv1 = nn.Sequential(
            nn.Conv1d(n_char, channel_size, kernel_size=kernel_sizes[0], stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=pool_size, stride=3)
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(channel_size, channel_size, kernel_size=kernel_sizes[1], stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=pool_size, stride=3)
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(channel_size, channel_size, kernel_size=kernel_sizes[2], stride=1),
            nn.ReLU()
        )

        self.conv4 = nn.Sequential(
            nn.Conv1d(channel_size, channel_size, kernel_size=kernel_sizes[3], stride=1),
            nn.ReLU()
        )

        self.conv5 = nn.Sequential(
            nn.Conv1d(channel_size, channel_size, kernel_size=kernel_sizes[4], stride=1),
            nn.ReLU()
        )

        self.conv6 = nn.Sequential(
            nn.Conv1d(channel_size, channel_size, kernel_size=kernel_sizes[5], stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=pool_size, stride=3)
        )

        self.fc1 = nn.Sequential(
            nn.Linear(channel_size * self.final_linear_len, fc_size),
            nn.ReLU(),
            nn.Dropout(p=dropout)
        )

        self.fc2 = nn.Sequential(
            nn.Linear(fc_size, fc_size),
            nn.ReLU(),
            nn.Dropout(p=dropout)
        )

        self.fc3 = nn.Linear(fc_size, n_class)
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)

        # collapse
        x = x.view(x.size(0), -1)
        # linear layer
        x = self.fc1(x)
        # linear layer
        x = self.fc2(x)
        # linear layer
        x = self.fc3(x)
        # output layer
        x = self.log_softmax(x)

        return x
