import torch

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