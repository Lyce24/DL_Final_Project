# Evaluate the model
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import torch
import pandas as pd
import os
from models.models import ConvLSTM, InceptionV3Model, ViTModel

min_max_file = './utils/min_max_values.csv'

min_max_df = pd.read_csv(min_max_file)

dataset_path = '../data/nutrition5k_reconstructed/'

image_path = os.path.join(dataset_path, 'images')

def test_model_mse(model, dataloader, labels, device):
    """
    Test the model on the test dataset.
    1. Calculate the MSE for each nutritional fact.
    2. Convert back the normalized values to original scale and calculate MAE.
    
    Args:
        model (torch.nn.Module): Trained model to evaluate.
        dataloader (DataLoader): DataLoader for the test dataset.
        labels (List[str]): List of nutritional facts.
        device (torch.device): Device to run the model on.
    """
    
    model.eval()
    
    val_loss = 0.0
    # individual losses for each task
    losses = {key: 0.0 for key in model.task_heads.keys()}
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            
            for i, key in enumerate(outputs.keys()):
                loss = mean_absolute_error(outputs[key].squeeze(), targets[:, i])
                losses[key] += loss.item()
            
            val_loss += sum(losses.values())
            
    for key in losses.keys():
        losses[key] /= len(dataloader)
        
    return val_loss / len(dataloader), losses


if __name__ == "__main__":
    from utils.preprocess import prepare_test_loader
    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = InceptionV3Model(["calories", "mass", "fat", "carb", "protein"]).to(device)
    
    # Load the test dataset
    dataset_path = './utils/data/test_labels.csv'
    test_df = pd.read_csv(dataset_path)
    
    # Prepare the test dataloader
    test_loader = prepare_test_loader(test_df, image_path, ["calories", "mass", "fat", "carb", "protein"] , img_dim=299, batch_size=16)
    
    # load model
    model_path = './models/checkpoints/inceptionv3_best.pth'
    model.load_state_dict(torch.load(model_path, map_location=device))
    
        
    # Test the model
    overall_mse, mse_results = test_model_mse(model, test_loader, ["calories", "mass", "fat", "carb", "protein"], device)
    print(f"Overall MSE: {overall_mse}")
    for key, value in mse_results.items():
        print(f"{key}: {value}")