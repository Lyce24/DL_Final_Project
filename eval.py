# Evaluate the model
from sklearn.metrics import mean_absolute_error
import numpy as np
import torch
import pandas as pd
from models.models import ConvLSTM, InceptionV3Model, ViTModel

min_max_file = './utils/min_max_values.csv'

min_max_df = pd.read_csv(min_max_file)


def test_model(model, dataloader, labels, device, image_path, img_dim, dict_output = True, log_min_max = True):
    """
    Test the model on the test dataset.
    1. Calculate the MSE for each nutritional fact.
    2. Convert back the normalized values to original scale and calculate MAE.
    
    Args:
        model (torch.nn.Module): Trained model to evaluate.
        dataloader (DataLoader): DataLoader for the test dataset.
        labels (List[str]): List of nutritional facts.
        device (torch.device): Device to run the model on.
        dict_output (bool): If True, the model outputs a dictionary of individual tasks. If False, the model outputs a single tensor.
        log_min_max (bool): If True, convert the normalized values back to original scale using log transformation and min-max scaling.
    """
    
    model.eval()
    
    all_predictions = []
    all_ground_truths = []

    with torch.no_grad():
        for batch in dataloader:
            # Move batch data to the appropriate device
            images = batch['image'].to(device)
            targets = batch['label'].to(device)

            # Get model predictions
            outputs = model(images)

            # Handle model output based on dict_output
            if dict_output:
                predictions = {label: outputs[label].cpu().numpy() for label in labels}
            else:
                predictions = outputs.cpu().numpy()

            # Store predictions and ground truth
            all_predictions.append(predictions)
            all_ground_truths.append(targets.cpu().numpy())

    # Combine predictions and ground truths across all batches
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_ground_truths = np.concatenate(all_ground_truths, axis=0)

    # Reverse normalization if log_min_max is True
    if log_min_max:
        min_vals = min_max_df['min'].values
        max_vals = min_max_df['max'].values
        all_predictions = np.exp(all_predictions * (max_vals - min_vals) + min_vals)
        all_ground_truths = np.exp(all_ground_truths * (max_vals - min_vals) + min_vals)

    # Calculate Mean Absolute Error for each nutritional fact
    maes = {}
    for i, label in enumerate(labels):
        maes[label] = mean_absolute_error(all_ground_truths[:, i], all_predictions[:, i])

    # Print the MAE for each label
    for label, mae in maes.items():
        print(f"MAE for {label}: {mae:.4f}")

    return maes
