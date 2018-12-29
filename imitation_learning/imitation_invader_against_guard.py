import sys
sys.path.append('../utils')
from dataloader import SimRunPreloader

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

use_gpu = torch.cuda.is_available()
DEVICE = 0

#Data statistics
num_actions = 9
history_length = 1
num_channels = 3

#Training parameters
lr = 0.01
momentum = 0.9
num_epochs = 15
batch_size = 2048
step_size = 2
gamma = 1
num_workers = 12

dataset_name = 'intelligent_bfs'
agent = 'invader'
# dataset_name = 'dummies_bfs'

data_dir = '../dataset/' + dataset_name + '/'
weights_dir = '../weights/'

class SimpleNet(nn.Module):
    def __init__(self, num_channels, history_length, num_actions):
        super(SimpleNet, self).__init__()
        self.conv1 = nn.Conv2d(num_channels*history_length, 256, kernel_size=5, stride=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=5, stride=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=5, stride=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.fc1 = nn.Linear(256*20*20, 2048)
        self.fc2 = nn.Linear(2048, num_actions)
        
    def forward(self, x):
        x = self.bn1(F.relu(self.conv1(x)))
        x = self.bn2(F.relu(self.conv2(x)))
        x = self.bn3(F.relu(self.conv3(x)))
#         print (x.size())
        x = x.view(x.size()[0], -1)
        x = F.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)


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
                inputs, _, labels = data
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
                torch.save(model, weights_dir + dataset_name + '_' + agent + '_sim.pt')



#Dataload and generator initialization
image_datasets = {'train': SimRunPreloader(data_dir, 'train', history_length),
                    'test': SimRunPreloader(data_dir, 'test', history_length)}

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=num_workers) for x in ['train', 'test']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
print (dataset_sizes)
file_name = __file__.split('/')[-1].split('.')[0]

#Create model and initialize/freeze weights
model_conv = SimpleNet(num_channels, history_length, num_actions)

if use_gpu:
    model_conv = model_conv.cuda(DEVICE)

#Initialize optimizer and loss function
criterion = nn.CrossEntropyLoss()
optimizer_conv = optim.SGD(model_conv.parameters(), lr=lr, momentum=momentum)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=step_size, gamma=gamma)

#Train model
model_conv = train_model(model_conv, criterion, optimizer_conv, exp_lr_scheduler, num_epochs=num_epochs)
