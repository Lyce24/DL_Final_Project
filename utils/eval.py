# Evaluate the model
from sklearn.metrics import mean_absolute_error
import numpy as np
import torch

def test_model(model, dataloader, labels, device):
    """
    Tests the model and calculates MAE for each nutritional fact.
    
    Args:
        model (torch.nn.Module): Trained model to evaluate.
        dataloader (DataLoader): DataLoader for the test dataset.
    
    Returns:
        dict: MAE for each nutritional fact.
    """
    
    model.eval()
    all_targets = []
    all_predictions = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            # Move data to device
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Make predictions
            outputs = model(inputs)

            # Store predictions and targets
            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    # Concatenate all predictions and targets
    all_predictions = np.concatenate(all_predictions, axis=0)
    # print a few predictions
    print(all_predictions[:5])
    all_targets = np.concatenate(all_targets, axis=0)

    # Calculate MAE and MAE percentage for each nutritional fact
    results = {}
    for i, fact in enumerate(labels):
        mae = mean_absolute_error(all_targets[:, i], all_predictions[:, i])
        mean_value = np.mean(all_targets[:, i])
        mae_percent = (mae / mean_value) * 100 if mean_value != 0 else 0
        results[fact] = {
            "MAE": mae,
            "MAE (%)": mae_percent
        }

    return results