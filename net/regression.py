import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision.datasets as dsets

import os
import numpy as np
# import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
# import matplotlib.pyplot as plt
# from skimage import io, transform, util
# from scipy.fftpack import fft2
# from sklearn.preprocessing import RobustScaler, MinMaxScaler
import argparse
import logging
import time

from regression_data_set import Problem_Dataset
from architectures import PlaNet
import net_config

#device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
config = net_config.config()

prob_data = Problem_Dataset(config.data_csv_path, config)
train_size = int(0.6 * len(prob_data))
valid_size = int(0.2 * len(prob_data))
test_size = len(prob_data) - train_size - valid_size

test_dataset = len(prob_data) - train_size - valid_size
prob_data = Problem_Dataset(config.data_csv_path, config)
train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(prob_data,
                                                                               [train_size, valid_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        outputs = self.linear(x)
        return outputs

batch_size = 100
n_iters = 3000
epochs = n_iters / (len(train_dataset) / batch_size)
input_dim = 16384
output_dim = 10
lr_rate = 0.001

model = LogisticRegression(input_dim, output_dim)

criterion = torch.nn.CrossEntropyLoss() # computes softmax and then the cross entropy
optimizer = torch.optim.SGD(model.parameters(), lr=lr_rate)

if __name__ == '__main__':

    iter = 0
    for epoch in range(int(epochs)):
        for i, (images, labels) in enumerate(train_loader):
            images = Variable(images.view(-1, 16384))
            labels = Variable(labels)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            iter+=1
            if iter%500==0:
                # calculate Accuracy
                correct = 0
                total = 0
                for images, labels in test_loader:
                    images = Variable(images.view(-1, 16384))
                    outputs = model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total+= labels.size(0)
                    # for gpu, bring the predicted and labels back to cpu fro python operations to work
                    correct+= (predicted == labels).sum()
                accuracy = 100 * correct/total
                print("Iteration: {}. Loss: {}. Accuracy: {}.".format(iter, loss.item(), accuracy))
