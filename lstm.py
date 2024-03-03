import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader   
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib as plt
from graphics import *
import pandas as pd
import numpy as np

# Set Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Hyper parameters
input_size = 1
hidden_size = 256
num_stacked_layers = 10
batch_size = 64
num_epochs = 10


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_stacked_layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_stacked_layers = num_stacked_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_stacked_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size,1)


    def forward(self,x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(device)
        c0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(device)
        out, _ =  self.lstm(x, (h0,c0))
        out = self.fc(out[:,-1,:])
        return out

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]

def train_one_epoch():
    model.train(True)
    print(f'Epoch: {epoch + 1}')
    running_loss = 0.0

    for batch_index, batch in enumerate(train_loader):
        x_batch, y_batch = batch[0].to(device), batch[1].to(device)
        output = model(x_batch)
        loss = loss_function(output, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_index % 100 == 99:
            avg_loss_across_batches = running_loss / 100
            print('Batch {0}, Loss: {1,.3f}'.format(batch_index+1,avg_loss_across_batches))

            running_loss = 0.0
        
    print()


def validate_one_epoch():
    model.train(False)
    running_loss = 0.0
    
    for batch_index, batch in enumerate(test_loader):
        x_batch, y_batch = batch[0].to(device), batch[1].to(device)

        with torch.no_grad():
            output = model(x_batch)
            loss = loss_function(output, y_batch)
            running_loss += loss
    
    avg_loss_across_batches = running_loss / len(test_loader)

    print('Val Loss: {0:.3f}'.format(avg_loss_across_batches))
    print('*****************************************************')
    print()


#play around with the lr, batch size, model architecture
learning_rate = 0.001  
loss_function = nn.MSELoss()
model = LSTM(1,4,1)
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
num_epochs = 10

for epoch in range(num_epochs):
    train_one_epoch()
    validate_one_epoch()


# with torch.no_grad():
#     predicted = model(X_train.to(device)).to('cpu').numpy()

# #plt.plot(y_train, label='Actual val')
# plt.plot(predicted, label = 'Predicted val')
# plt.xlabel('Day')
# plt.ylabel('Close')
# plt.legend()
# plt.show()