import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from mlpipe import Model


class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()

        self.tdata_model = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3),
            nn.ReLU(),
            nn.Conv1d(16, 16, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3),
            nn.Conv1d(16, 32, kernel_size=3),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3),
            nn.Conv1d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3),
        )

        self.fdata_model = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3),
            nn.ReLU(),
            nn.Conv1d(16, 16, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3),
            nn.Conv1d(16, 32, kernel_size=3),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3),
            nn.Conv1d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3),
        )

        self.fc = nn.Sequential(
            nn.Linear(4490, 2245),
            nn.ReLU(),
            nn.Linear(2245, 2),
        )

    def forward(self, tdata, fdata, features):
        tout = self.tdata_model(tdata)
        fout = self.fdata_model(fdata)

        tout = tout.reshape(tout.size(0), -1)
        fout = fout.reshape(fout.size(0), -1)

        out = torch.cat((tout,fout,features), dim=1)
        out = self.fc(out)

        return out


class CNNModel(Model):

    name = 'DeepCNN'

    def __init__(self):
        self.device = None
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.features = ['corrLive', 'rmsLive', 'kurtLive', 'DELive',
                         'MFELive', 'skewLive', 'normLive', 'darkRatioLive',
                         'jumpLive', 'gainLive']

    def setup(self, device):
        self.device = device
        self.model = NeuralNet().to(device)

        # Loss and optimizer
        learning_rate = 0.05
        self.criterion = nn.CrossEntropyLoss()

        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=learning_rate)

    def train(self, data, labels, metadata):
        gpu = self.device

        # Get input data and transform to torch tensor
        tdata = torch.from_numpy(data[:, 0, None, ...]).type(torch.FloatTensor)
        fdata = torch.from_numpy(data[:, 1, None, ...]).type(torch.FloatTensor)
        labels = torch.from_numpy(labels)

        features = np.hstack([metadata[key] for key in self.features])
        features = torch.from_numpy(features).type(torch.FloatTensor)

        # Transfer input data into device (gpu / cpu)
        tdata, fdata = tdata.to(gpu), fdata.to(gpu)
        labels, features = labels.to(gpu), features.to(gpu)

        # Model computations
        outputs = self.model(tdata, fdata, features)
        loss = self.criterion(outputs, labels)

        # Backward and optimize
        print("Loss: %f" % float(loss.cpu()))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


    def validate(self, data, labels, metadata):
        gpu = self.device

        # Get input data and transform to torch tensor
        tdata = torch.from_numpy(data[:, 0, None, ...]).type(torch.FloatTensor)
        fdata = torch.from_numpy(data[:, 1, None, ...]).type(torch.FloatTensor)
        labels = torch.from_numpy(labels)

        features = np.hstack([metadata[key] for key in self.features])
        features = torch.from_numpy(features).type(torch.FloatTensor)

        # Transfer input data into device (gpu / cpu)
        tdata, fdata = tdata.to(gpu), fdata.to(gpu)
        labels, features = labels.to(gpu), features.to(gpu)

        outputs = self.model(tdata, fdata, features)
        _, predicted = torch.max(outputs, 1)
        return predicted.cpu().numpy(), None


    def save(self, filename):
        torch.save(self.model, filename)

    def load(self, filename):
        self.model = torch.load(filename)
        self.model.eval()
        print("Saved model loaded!")
    
