import torch

import torch.nn as nn
import torch.optim as optim

from utils import load_tensors, load_tensors_bin
from models import LSTM_base
from train import training, training_bin
from evaluation import evaluate_model
from train_test import train_test_bin, train_test_bins

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"device: {DEVICE}")

INPUT_SIZE = 1664 # length of the multivariate features per timestep
HIDDEN_DIM = 256 # (LSTM) hidden layer size
NUM_LAYERS= 10 # Number of LSTM layers
NUM_EPOCHS = 300 # The maximum training epochs
LR = 0.0001 # For training
BATCH_SIZE = 16
NUMS_BINS = [2, 5, 10, 15, 20, 25]

train_test_bin(HIDDEN_DIM, DEVICE, NUM_LAYERS, "cardiac arrest", 10, BATCH_SIZE, NUM_EPOCHS, LR)
# train_test_bins(HIDDEN_DIM, DEVICE, NUM_LAYERS, "cardiac arrest", NUMS_BINS)