import torch
from torch.utils import data
from main import Dataset
from torch import nn
import torch.nn.functional as F

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

# Parameters
params = {'batch_size': 1,
          'shuffle': True,
          'num_workers': 1,
          }
max_epochs = 100


# Generators
training_set = Dataset(src='data/dataset.h5', label='train')
training_generator = data.DataLoader(training_set, **params)

# Fully connected neural network with one hidden layer
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


# Initialize model
model = NeuralNet().to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
learning_rate = 0.01
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# Loop over epochs
for epoch in range(max_epochs):
    # Training
    for i, (local_batch, local_labels) in enumerate(training_generator):
        # Transfer to GPU
        local_batch, local_labels = local_batch.to(device), local_labels.to(device)

        # Model computations
        outputs = model(local_batch)
        loss = criterion(outputs, local_labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print 'Epoch: {}, Loss: {}'.format(epoch, loss.item())
