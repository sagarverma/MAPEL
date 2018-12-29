import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, num_channels, history_length, num_actions):
        super(DQN, self).__init__()
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