import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module, Sequential, Conv2d, ReLU,LeakyReLU
class conv_net1(nn.Module):
    def __init__(self):
        super(conv_net1, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, 3, 1)
        self.conv2 = nn.Conv2d(4, 8, 3, 1)
        self.conv3 = nn.Conv2d(8, 16, 3, 1)
        self.conv4 = nn.Conv2d(16, 32, 3, 1)
        self.conv5 = nn.Conv2d(32, 64, 2, 1)
        self.conv6 = nn.Conv2d(64, 128, 2, 1)
        self.dropout0 = nn.Dropout2d(0.1)          
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        #self.fc1 = nn.Linear(2304, 128)        
        #self.fc1 = nn.Linear(256, 64)
        self.fc1 = nn.Linear(64, 32)
        #self.fc1 = nn.Linear(246016, 128)
        self.fc2 = nn.Linear(32, 2)
        #self.fc2 = nn.Linear(128, 3)
        self.lrelu = nn.LeakyReLU(0.1)
         

    def forward(self, x): 
        x = self.dropout0(self.lrelu(self.conv1(x))) # ouput 30x30
        x = self.dropout0(self.lrelu(self.conv2(x))) # ouput 28x28
        x = F.max_pool2d(x, 2) # ouput 14x14
        x = self.dropout0(self.lrelu(self.conv3(x))) # ouput 12x12
        x = F.max_pool2d(x, 2) # ouput 6x6
        x = self.dropout0(self.lrelu(self.conv4(x))) # ouput 4x4
        #x = F.max_pool2d(x, 2)
        #x = self.dropout0(self.lrelu(self.conv5(x)))
        
        x = F.max_pool2d(x, 2) # ouput 12x12
        
        x = self.dropout1(self.lrelu(self.conv5(x)))# ouput 1x1x32
        #print(f'x.shape = {x.shape}')        
        #x = torch.flatten(x, 1)
        x = x.view(x.size(0), -1)
        #print(f'x.shape = {x.shape}')
        x = self.dropout2(self.lrelu(self.fc1(x)))
        x= F.log_softmax(self.fc2(x), dim=1)
        return x

class conv_net2(nn.Module):
    def __init__(self):
        super(conv_net2, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, 3, 1)
        self.conv2 = nn.Conv2d(4, 8, 3, 1)
        self.conv3 = nn.Conv2d(8, 16, 3, 1)
        self.conv4 = nn.Conv2d(16, 32, 3, 1)
        self.conv5 = nn.Conv2d(32, 64, 2, 1)
        self.conv6 = nn.Conv2d(64, 128, 2, 1)
        self.dropout0 = nn.Dropout2d(0.1)          
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        #self.fc1 = nn.Linear(2304, 128)        
        #self.fc1 = nn.Linear(256, 64)
        self.fc1 = nn.Linear(64, 32)
        #self.fc1 = nn.Linear(246016, 128)
        self.fc2 = nn.Linear(32, 2)
        #self.fc2 = nn.Linear(128, 3)
        self.lrelu = nn.LeakyReLU(0.1)
         

    def forward(self, x): 
        x = self.dropout0(self.lrelu(self.conv1(x)))
        x = self.dropout0(self.lrelu(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = self.dropout0(self.lrelu(self.conv3(x)))
        x = F.max_pool2d(x, 2)
        x = self.dropout0(self.lrelu(self.conv4(x)))
        #x = F.max_pool2d(x, 2)
        #x = self.dropout0(self.lrelu(self.conv5(x)))
        
        x = F.max_pool2d(x, 2)
        
        x = self.dropout1(self.lrelu(self.conv5(x)))
        #print(f'x.shape = {x.shape}')        
        #x = torch.flatten(x, 1)
        x = x.view(x.size(0), -1)
        #print(f'x.shape = {x.shape}')
        x = self.dropout2(self.lrelu(self.fc1(x)))
        x= self.fc2(x)
        return x