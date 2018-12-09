from __future__ import print_function, division
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torch.nn.functional as F

import numpy as np

import torchvision
from torchvision import datasets, models, transforms

import matplotlib.pyplot as plt
import time
import os

from utils.folder import SimImagePreloader

use_gpu = torch.cuda.is_available()
DEVICE = 0

#Data statistics
num_classes = 9

#Training parameters
lr = 0.01
momentum = 0.9
num_epochs = 100
batch_size = 512
step_size = 100
gamma = 1
num_workers = 4

#dataset_name = 'intelligent_bfs'
dataset_name = 'dummies_bfs'

data_dir = 'dataset/' + dataset_name + '/'
weights_dir = 'weights/'
train_csv = 'dataset/' + dataset_name + '/train.csv'
test_csv = 'dataset/' + dataset_name + '/test.csv'

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.conv1 = nn.Conv2d(8, 64, kernel_size=7)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3)
        self.conv3_drop = nn.Dropout2d()
        self.fc2 = nn.Linear(256, 9)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv3_drop(self.conv3(x)), 2))
        #print(x.size())
        x = x.view(-1, 256)
        x = F.relu(self.fc2(x))
        return x


def train_model(model, criterion, optimizer, scheduler, num_epochs=2000):
    """
        Training model with given criterion, optimizer for num_epochs.
    """
    since = time.time()

    best_model_wts = model.state_dict()
    best_acc = 0.0

    train_loss = []
    train_acc = []
    test_acc = []
    test_loss = []

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train', 'test']:
            if phase == 'train':
                scheduler.step()
                model.train(True)
            else:
                model.train(False)

            running_loss = 0.0
            running_corrects = 0

            for data in dataloaders[phase]:
                inputs, labels = data
                if use_gpu:
                    inputs = Variable(inputs.type(torch.FloatTensor).cuda(DEVICE))
                    labels = Variable(labels.type(torch.LongTensor).cuda(DEVICE))
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                optimizer.zero_grad()
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)

                loss = criterion(outputs, labels)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                running_loss += loss.item()
                running_corrects += torch.sum(preds == labels.data).item()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            if phase=='train':
                train_loss.append(epoch_loss)
                train_acc.append(epoch_acc)
                print ('##############################################################')
                print ("{} loss = {}, acc = {},".format(phase, epoch_loss, epoch_acc))
            else:
                test_loss.append(epoch_loss)
                test_acc.append(epoch_acc)
                print (" {} loss = {}, acc = {},".format(phase, epoch_loss, epoch_acc))
                print ('##############################################################')


            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model, weights_dir + dataset_name + '_sim.pt')


    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best test Acc: {:4f}'.format(best_acc))

    model = torch.load( weights_dir + dataset_name + '_sim.pt')

    return model


#Dataload and generator initialization
image_datasets = {'train': SimImagePreloader(data_dir + 'train/', train_csv),
                    'test': SimImagePreloader(data_dir + 'test/', test_csv)}

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=num_workers) for x in ['train', 'test']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
file_name = __file__.split('/')[-1].split('.')[0]

#Create model and initialize/freeze weights
model_conv = SimpleNet()

if use_gpu:
    model_conv = model_conv.cuda(DEVICE)

#Initialize optimizer and loss function
criterion = nn.CrossEntropyLoss()
optimizer_conv = optim.SGD(model_conv.parameters(), lr=lr, momentum=momentum)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=step_size, gamma=gamma)

#Train model
model_conv = train_model(model_conv, criterion, optimizer_conv, exp_lr_scheduler, num_epochs=num_epochs)
