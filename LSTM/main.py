import torch

import torch.nn as nn
import torch.optim as optim

from utils import load_tensors
from models import LSTM_base
from train import training
from evaluation import evaluate_model

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"device: {DEVICE}")

INPUT_SIZE = 1693 # length of the multivariate features per timestep
HIDDEN_DIM = 256 # (LSTM) hidden layer size
NUM_LAYERS= 10 # Number of LSTM layers
NUM_EPOCHS = 300 # The maximum training epochs
LR = 0.0001 # For training
BATCH_SIZE = 16

MODEL = LSTM_base(INPUT_SIZE, HIDDEN_DIM, DEVICE, NUM_LAYERS)
LOSS = nn.BCELoss(reduction='none')
OPTIMIZER = optim.Adam(MODEL.parameters(), lr=LR)

# load dataloaders
train_loader, test_loader = load_tensors("cardiac arrest")

training(MODEL, BATCH_SIZE, NUM_EPOCHS, train_loader, DEVICE, OPTIMIZER, LOSS)

# Define the path to your saved model file
saved_model_path = '../data/models/output_lab_avg.pth'

# Load the saved state_dict
model_state_dict = torch.load(saved_model_path)

# Apply the loaded state_dict to your model
MODEL.load_state_dict(model_state_dict)

evaluate_model(MODEL, test_loader, DEVICE)

