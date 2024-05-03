import torch
import os
import numpy as np

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def training(model, batch_size, num_epoch, train_loader, device, optimizer, loss_fn):
    model.to(device)
    loss_prev = 100
    count = 0

    for epoch in range(num_epoch):
        timestep_losses = torch.zeros(72).to(device)
        model.train()

        for i, (data, labels) in enumerate(train_loader):
            data, labels = data.to(device), labels.to(device)
            labels_expanded = labels.unsqueeze(1).expand(-1, 72, -1)

            # Reset gradients
            optimizer.zero_grad()

            # Forward pass
            # print(f"batch size: {data.size(0)}")
            states = model.init_state(data.size(0))
            states = tuple(state.to(device) for state in states)
            output, states = model(data, states)
            states = model.detach_states(states)
            # print(f"output shape: {output.shape}")
            # print(f"labels shape: {labels_expanded.shape}")

            # loss (losses -> only for reducti on='none')
            losses = loss_fn(output, labels_expanded)
            loss = losses.mean()
            loss.backward()
            # print(f"timestep_losses shape: {timestep_losses.shape}")
            # print(f"losses sum shape: {losses.sum(dim=0).detach().shape}")

            timestep_losses += losses.sum(dim=0).detach().squeeze()

            # Update the parameters
            optimizer.step()

        timestep_losses /= len(train_loader)
        print(f'Epoch [{epoch+1}/{num_epoch}], Timestep Losses: {timestep_losses.cpu().numpy()}')

        loss_avg = timestep_losses.mean()
        print(f"Average loss: {loss_avg}")

        if (loss_avg > loss_prev):
            count += 1
        else:
            count = 0

        if (count >= 5):
            print(f"No improvement in loss for {count} consecutive times. Terminate training at epoch {epoch}.")
            torch.save(model.state_dict(), f'../data/models/output_lab_age.pth')
            break

        loss_prev = loss_avg
        print(f"count: {count}")
    
    torch.save(model.state_dict(), f'../data/models/output_lab_avg.pth')

def training_bin(model, batch_size, num_epoch, train_loader, device, optimizer, loss_fn, num_bins, keyword):
    print("training_bin")
    model.to(device)
    loss_prev = 100
    count = 0

    for epoch in range(num_epoch):
        timestep_losses = torch.zeros(72).to(device)
        model.train()

        for i, (data, labels) in enumerate(train_loader):
            data, labels = data.to(device), labels.to(device)
            labels_expanded = labels.unsqueeze(1).expand(-1, 72, -1)
            
            # if i == 0:  # Optionally print only for the first batch of each epoch to reduce clutter
            #     print(f"Input data sample (Batch 1): {data[:1].detach().cpu().numpy()}")

            # Check for NaN values
            nan_mask = torch.isnan(data)
            # Replace NaN values with 0
            data[nan_mask] = 0

            # data_cpu = data.cpu().numpy()
            # # Check for NaN values
            # nan_indices = np.isnan(data_cpu)
            # # Count the number of NaN values
            # num_nans = np.sum(nan_indices)
            # print(f"num_nans in input data: {num_nans}")
            # print(f"original shape: {data_cpu.shape}")

            # Reset gradients
            optimizer.zero_grad()

            # Forward pass
            # print(f"batch size: {data.size(0)}")
            states = model.init_state(data.size(0))
            states = tuple(state.to(device) for state in states)
            output, states = model(data, states)
            states = model.detach_states(states)
            # print(f"output shape: {output.shape}")
            # print(f"labels shape: {labels_expanded.shape}")

            # print("Output max:", output.max().item(), "Output min:", output.min().item())

            # if i == 0:  # Optionally, limit printing to the first batch of each epoch to reduce clutter
            #     print(f"Output before loss calculation (Sample from Batch): {output[:1].detach().cpu().numpy()}")

            

            # loss (losses -> only for reduction='none')
            losses = loss_fn(output, labels_expanded)
            loss = losses.mean()

            loss.backward()
            # print(f"timestep_losses shape: {timestep_losses.shape}")
            # print(f"losses sum shape: {losses.sum(dim=0).detach().shape}")

             # Clip gradients to prevent explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            timestep_losses += losses.sum(dim=0).detach().squeeze()

            # Update the parameters
            optimizer.step()

        timestep_losses /= len(train_loader)
        print(f'Epoch [{epoch+1}/{num_epoch}], Timestep Losses: {timestep_losses.cpu().numpy()}')

        loss_avg = timestep_losses.mean()
        print(f"Average loss: {loss_avg}")

        if (loss_avg > loss_prev):
            count += 1
        else:
            count = 0

        if (count >= 5):
            print(f"No improvement in loss for {count} consecutive times. Terminate training at epoch {epoch}.")
            torch.save(model.state_dict(), f'../data/models/output_lab_{keyword}_{num_bins}bin.pth')
            break

        loss_prev = loss_avg
        print(f"count: {count}")
    
    torch.save(model.state_dict(), f'../data/models/output_lab_{keyword}_{num_bins}bin.pth')