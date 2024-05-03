import torch

import torch.nn as nn
import torch.optim as optim

from utils import load_tensors, load_tensors_bin
from models import LSTM_base
from train import training, training_bin
from evaluation import evaluate_model


def train_test_bin(hidden_dim, device, num_layers, keyword, num_bins, batch_size, num_epochs, learning_rate):

    # load dataloaders
    train_loader, test_loader = load_tensors_bin(keyword, num_bins)
    
    # Load one batch of data
    data_iter = iter(train_loader)
    samples, labels = next(data_iter)

    # Check the shape of the samples to find the third dimension
    if len(samples.shape) > 2:  # Ensuring that there are at least three dimensions
        input_size = samples.shape[2]
    else:
        print("The data does not have a third dimension.")

    # print("Input size (third dimension):", input_size)

    model = LSTM_base(input_size, hidden_dim, device, num_layers)
    loss = nn.BCELoss(reduction='none')
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    training_bin(model, batch_size, num_epochs, train_loader, device, optimizer, loss, num_bins, keyword)

    # Define the path to your saved model file
    saved_model_path = f'../data/models/output_lab_{keyword}_{num_bins}bin.pth'

    # Load the saved state_dict
    model_state_dict = torch.load(saved_model_path)

    # Apply the loaded state_dict to your model
    model.load_state_dict(model_state_dict)

    print(f"Evaluation starting for {keyword} with {num_bins} bins.")
    evaluate_model(model, test_loader, device)

def train_test_bins(hidden_dim, device, num_layers, keyword, nums_bins):
    for num_bins in nums_bins:
        train_test_bin(hidden_dim, device, num_layers, keyword, num_bins)