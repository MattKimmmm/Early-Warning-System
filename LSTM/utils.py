import torch
import os

from torch.utils.data import Dataset, DataLoader, TensorDataset

# load tensors and return dataloaders
def load_tensors(keyword):
    script_dir = os.path.dirname(__file__)
    file_path = os.path.join(script_dir, "../data", "tensors")

    # Load datasets
    data_train_tensor = torch.load(os.path.join(file_path, f'data_train_tensor_{keyword}.pt'))
    labels_train_tensor = torch.load(os.path.join(file_path, f'labels_train_tensor_{keyword}.pt'))
    data_test_tensor = torch.load(os.path.join(file_path, f'data_test_tensor_{keyword}.pt'))
    labels_test_tensor = torch.load(os.path.join(file_path, f'labels_test_tensor_{keyword}.pt'))

    # Load DataLoader parameters
    dataloader_params = torch.load(os.path.join(file_path, f'dataloader_params_{keyword}.pt'))

    # Recreate TensorDatasets and DataLoaders
    train_dataset = TensorDataset(data_train_tensor, labels_train_tensor)
    test_dataset = TensorDataset(data_test_tensor, labels_test_tensor)

    print(f"Training Data Shape: {train_dataset.tensors[0].shape}")
    print(f"Training Labels Shape: {train_dataset.tensors[1].shape}")
    print(f"Test Data Shape: {test_dataset.tensors[0].shape}")
    print(f"Test Labels Shape: {test_dataset.tensors[1].shape}")

    train_loader = DataLoader(train_dataset, **dataloader_params)
    test_loader = DataLoader(test_dataset, **{**dataloader_params, 'shuffle': False})

    return train_loader, test_loader

# load tensors and return dataloaders
def load_tensors_bin(keyword, num_bins):
    script_dir = os.path.dirname(__file__)
    file_path = os.path.join(script_dir, "../data", "tensors")

    # Load datasets
    data_train_tensor = torch.load(os.path.join(file_path, f'data_train_tensor_{keyword}_{num_bins}bin.pt'))
    labels_train_tensor = torch.load(os.path.join(file_path, f'labels_train_tensor_{keyword}_{num_bins}bin.pt'))
    data_test_tensor = torch.load(os.path.join(file_path, f'data_test_tensor_{keyword}_{num_bins}bin.pt'))
    labels_test_tensor = torch.load(os.path.join(file_path, f'labels_test_tensor_{keyword}_{num_bins}bin.pt'))

    # Load DataLoader parameters
    dataloader_params = torch.load(os.path.join(file_path, f'dataloader_params_{keyword}.pt'))

    # Recreate TensorDatasets and DataLoaders
    train_dataset = TensorDataset(data_train_tensor, labels_train_tensor)
    test_dataset = TensorDataset(data_test_tensor, labels_test_tensor)

    print(f"Training Data Shape: {train_dataset.tensors[0].shape}")
    print(f"Training Labels Shape: {train_dataset.tensors[1].shape}")
    print(f"Test Data Shape: {test_dataset.tensors[0].shape}")
    print(f"Test Labels Shape: {test_dataset.tensors[1].shape}")

    train_loader = DataLoader(train_dataset, **dataloader_params)
    test_loader = DataLoader(test_dataset, **{**dataloader_params, 'shuffle': False})

    return train_loader, test_loader