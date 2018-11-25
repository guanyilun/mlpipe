import torch
from torch import nn
import torch.nn.functional as F

from mlpipe import Model


class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=5)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=5)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=5)
        self.fc = nn.Linear(128, 2)
    
    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(out)
        
        out = self.conv2(out)
        out = F.relu(out)
        out = F.avg_pool1d(out, kernel_size=4, stride=2)
        
        out = self.conv3(out)
        out = F.relu(out)
        out = F.avg_pool1d(out, kernel_size=4, stride=2)
        
        out = self.conv4(out)
        out = F.relu(out)
        out = F.avg_pool1d(out, kernel_size=out.shape[2:])
        
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out


class CNNModel(Model):

    name = 'CNNModel'

    def __init__(self):
        self.device = None
        self.model = None
        self.optimizer = None
        self.criterion = None

    def setup(self, device):
        self.device = device
        self.model = NeuralNet().to(device)        

        # Loss and optimizer
        learning_rate = 0.001
        self.criterion = nn.CrossEntropyLoss()

        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=learning_rate)

    def train(self, data, labels, metadata):
        gpu = self.device

        data = torch.from_numpy(data[:,None,::10]).type(torch.FloatTensor)
        labels = torch.from_numpy(labels)
        data, labels = data.to(gpu), labels.to(gpu)

        # Model computations
        outputs = self.model(data)
        loss = self.criterion(outputs, labels)
    
        # Backward and optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
    def validate(self, data, labels, metedata):
        gpu = self.device
        data = torch.from_numpy(data[:,None,:]).type(torch.FloatTensor)
        labels = torch.from_numpy(labels)
        data, labels = data.to(gpu), labels.to(gpu)
        outputs = self.model(data)
        _, predicted = torch.max(outputs, 1)
        return predicted.cpu().numpy()


    def save(self, filename):
        torch.save(self.model, filename)
