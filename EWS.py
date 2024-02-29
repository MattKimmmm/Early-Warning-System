import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader   
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from graphics import *
import pandas as pd
import numpy as np

# Set Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Hyperparameters
input_size = 1
sequence_length = 4
num_layers = 2
hidden_size = 256
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 2


#Create a RNN 
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size*sequence_length, num_classes)


    def forward(self,x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)


        out, _ = self.rnn(x, h0)
        out = out.reshape(out.shape[0], -1)
        out = self.fc(out)

        return out


   
class CustomDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        # Assuming images are in a column named 'image' that need to be reshaped
        # and a column named 'label' for labels.
        image = self.dataframe.iloc[idx, :-1].values.astype(np.uint8).reshape(28, 28)  # Reshape to 28x28 if flat
        label = self.dataframe.iloc[idx, -1]
        image = image[:, :, None]  # Add channel dimension if needed

        if self.transform:
            image = self.transform(image)

        return image, label




#Load data
path = "/Users/yurockheo/Desktop/Early-Warning-System/data/"
df1 = pd.read_csv(path+"DIAGNOSES_ICD.csv")
#df2 = pd.read_csv(path+"ICUSTAYS.csv")


transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Adjust as necessary
])


#custom_dataset = CustomDataset(df1, transform=transform)
#train_loader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=True)
#test_loader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=True)


#Extraction
subjectIds = df1.loc[df1['icd9_code'] == icd9_code].loc[:,'subject_id'].tolist
subjectIds = subjectIds.unique()
print(subjectIds)

#intimes = pd.to_datetime(df2['intime'])
#outtimes = intimes+pd.Timedelta(hours=48)


#Change icd9_code: 51881 => Acute repiratry failure
icd9_code = "51881"





# Initialize network
model = RNN(input_size, hidden_size, num_layers, num_classes).to(device)

#Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(train_loader):
        # Get data to cuda if possible
        data = data.to(device=device)
        targets = targets.to(device=device)

        #Get to correct shapee
        data = data.reshape(daat.shape[0], -1)

        #forward
        scores = model(data)
        loss = crierion(scores, targets)

        #backward
        optimizer.zero_grad()
        loss.backward()

        #gradient descentt or adam step
        optimizer.step()

# Check accuracy on training & test to see how good our model
def check_accuracy(loader, model):
    if loader.dataset.train:
        print("Checking accuracy on training data")
    else:
        print("Checking accuracy on test data")

    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device).squeeze(1)
            y = y.to(device=device)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(f'Got {num_correct} / {num_samples} with accuracy \
            {float(num_correct)/float(num_samples)*100:.2f}')

    model.train()
    

check_accuracy(train_loader, model)
check_accuracy(test_loader, model)
    


def main():
    win = GraphWin("Paitents", 500,500)
    title = Text(Point(win.getWidth()/2, 200), "Early Warning System")
    title.draw(win)

    win.getMouse()
    win.close
