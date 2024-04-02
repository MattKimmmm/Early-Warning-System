import torch

import numpy as np

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def evaluate_model(model, test_loader, device, threshold=0.5):
    model.to(device)
    model.eval()

    num_timesteps = 72
    metrics = {
        'accuracy': np.zeros(num_timesteps),
        'precision': np.zeros(num_timesteps),
        'recall': np.zeros(num_timesteps),
        'f1': np.zeros(num_timesteps),
        'auc': np.zeros(num_timesteps),
        'counts': np.zeros(num_timesteps)  # To keep track of non-zero instances per timestep for averaging
    }

    # Initialize lists to store metrics for each timestep
    accuracy_list, precision_list, recall_list, f1_list, auc_list = [], [], [], [], []

    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data.to(device), labels.to(device)
            labels_expanded = labels.unsqueeze(1).expand(-1, 72, -1)
            
            # Forward pass
            states = model.init_state(data.size(0))
            states = tuple(state.to(device) for state in states)
            output, _ = model(data, states)

            predictions = output > threshold  # Apply threshold to get binary predictions

            for timestep in range(output.shape[1]):  # Iterate through each timestep
                timestep_pred = predictions[:, timestep, :].cpu().numpy().flatten()
                timestep_labels = labels_expanded[:, timestep, :].cpu().numpy().flatten()

                # Calculate metrics for this timestep
                accuracy = accuracy_score(timestep_labels, timestep_pred)
                precision = precision_score(timestep_labels, timestep_pred, zero_division=0)
                recall = recall_score(timestep_labels, timestep_pred, zero_division=0)
                f1 = f1_score(timestep_labels, timestep_pred, zero_division=0)

                # ROC AUC requires probability scores, not binary predictions
                # Ensure labels and outputs are flattened to match expected input shape
                auc = roc_auc_score(timestep_labels, output[:, timestep, :].cpu().numpy().flatten())

                accuracy_list.append(accuracy)
                precision_list.append(precision)
                recall_list.append(recall)
                f1_list.append(f1)
                auc_list.append(auc)

                # Calculate metrics for this timestep
                metrics['accuracy'][timestep] += accuracy
                metrics['precision'][timestep] += precision
                metrics['recall'][timestep] += recall
                metrics['f1'][timestep] += f1
                metrics['auc'][timestep] += auc

                # Increment count of samples for this timestep
                metrics['counts'][timestep] += 1

    # Calculate average metrics across all timesteps
    avg_accuracy = np.mean(accuracy_list)
    avg_precision = np.mean(precision_list)
    avg_recall = np.mean(recall_list)
    avg_f1 = np.mean(f1_list)
    avg_auc = np.mean(auc_list)

    print(f"Test Metrics: Accuracy: {avg_accuracy}, Precision: {avg_precision}, Recall: {avg_recall}, F1 Score: {avg_f1}, AUC: {avg_auc}")

    # Correctly average the metrics for each timestep
    for metric in metrics.keys():
        if metric != 'counts':  # Skip 'counts' since it's not a metric
            metrics[metric] /= metrics['counts']

    # Report metrics for each timestep
    for timestep in range(num_timesteps):
        print(f"Timestep {timestep + 1}: "
              f"Accuracy: {metrics['accuracy'][timestep]:.4f}, "
              f"Precision: {metrics['precision'][timestep]:.4f}, "
              f"Recall: {metrics['recall'][timestep]:.4f}, "
              f"F1: {metrics['f1'][timestep]:.4f}, "
              f"AUC: {metrics['auc'][timestep]:.4f}")