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
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from copy import deepcopy as dc



# Set Device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Hyper parameters
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




# Load Data here


df_icu = pd.read_csv('./Preprocessing/preprocessed_data/ICUSTAYS.csv')
df_icu['INTIME'] = pd.to_datetime(df_icu['INTIME'])
df_icu['OUTTIME'] = pd.to_datetime(df_icu['OUTTIME'])


data = df_icu
# Normalize the data
scaler = MinMaxScaler(feature_range=(-1,1))
data = scaler.fit_transform(data)


X = data[:,1:]
y = data[:,0]
#X = dc(np.flip(X,axis=1))

# Split the data into 95% and 5% for training and testing
split_index = int(len(X) * 0.95)

X_train = X[:split_index]
X_test = X[split_index:]

y_train = y[:split_index]
y_test = y[split_index:]

# Lookback 4 hours of data before to predict the next hour
lookback = 4

X_train = X_train.reshape((-1, lookback, 1))
X_test = X_test.reshape((-1, lookback, 1))

y_train = y_train.reshape((-1, 1))
y_test = y_test.reshape((-1, 1))


# Convert into torch type
X_train = torch.tensor(X_train).float()
y_train = torch.tensor(y_train).float()
X_test = torch.tensor(X_test).float()
y_test = torch.tensor(y_test).float()

# Write in timeseries
train_dataset = TimeSeriesDataset(X_train, y_train)
test_dataset = TimeSeriesDataset(X_test, y_test)

# Load Data for model
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

for _, batch in enumerate(train_loader):
    x_batch, y_batch = batch[0].to(device), batch[1].to(device)
    print(x_batch.shape, y_batch.shape)
    break


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


with torch.no_grad():
     predicted = model(X_train.to(device)).to('cpu').numpy()

# plt.plot(y_train, label='Actual val')
# plt.plot(predicted, label = 'Predicted val')
# plt.xlabel('Day')
# plt.ylabel('Close')
# plt.legend()
# plt.show()
    


train_predictions = predicted.flatten()

dummies = np.zeros((X_train.shape[0], lookback+1))
dummies[:,0] = train_predictions
dummies = scaler.inverse_transform(dummies)

train_predictions = dc(dummies[:, 0])

dummies = np.zeros((X_train.shape[0], lookback+1))
dummies[:,0] = y_train.flatten()
dummies = scaler.inverse_transform(dummies)

new_y_train = dc(dummies[:, 0])

# plt.plot(new_y_train, label='Actual val')
# plt.plot(train_predictions, label = 'Predicted val')
# plt.xlabel('Day')
# plt.ylabel('Close')
# plt.legend()
# plt.show()




test_predictions = model(X_test.to(device)).detach().cpu().numpy().flatten()

dummies = np.zeros((X_test.shape[0], lookback+1))
dummies[:,0] = test_predictions
dummies = scaler.inverse_transform(dummies)

test_predictions = dc(dummies[:,0])

dummies = np.zeros((X_test.shape[0], lookback+1))
dummies[:,0] = y_test.flatten()
dummies = scaler.inverse_transform(dummies)

new_y_test = dc(dummies[:,0])

# plt.plot(new_y_test, label='Actual val')
# plt.plot(train_predictions, label = 'Predicted val')
# plt.xlabel('Day')
# plt.ylabel('Close')
# plt.legend()
# plt.show()
