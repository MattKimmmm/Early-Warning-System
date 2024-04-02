import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, TensorDataset
from torch import Tensor
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time

class lmLSTM(nn.Module):
    def __init__(self,  input_size, hidden_dim, device, num_layers=1):
        """
        Args:
            input_size = # features per time window
        """
        super(lmLSTM, self).__init__()
        # Your code goes here
        self.device = device
        self.input_size = input_size
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        # initialize parameters
        self.lstm = nn.LSTM(input_size, hidden_dim, num_layers, batch_first=True)

        # a linear layer to map outputs[hidden_dim] to vocab size[vocab_size]
        self.linear = nn.Linear(hidden_dim, input_size)

    def init_state(self, batch_size):
        
        # Your code goes here
        hidden_state = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(self.device)
        cell_state = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(self.device)
        return (hidden_state, cell_state)

    def detach_states(self, states):
        
        return (states[0].detach(), states[1].detach())

    def forward(self, inputs, states):

        logits, states = self.lstm(inputs, states)
        logits = self.linear(logits)
        # print(f"At forward logit shape: {logits.shape}")

        return logits, states