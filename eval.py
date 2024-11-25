# Evaluate the model
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import torch
import pandas as pd
import os
from models.models import ConvLSTM, InceptionV3Model, ViTModel
# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the test dataset
dataset_path = './utils/data/test_labels.csv'
test_df = pd.read_csv(dataset_path)


min_max_file = './utils/min_max_values.csv'

min_max_df = pd.read_csv(min_max_file)

dataset_path = '../data/nutrition5k_reconstructed/'

image_path = os.path.join(dataset_path, 'images')

def test_model_mse(model, dataloader, device):
    """
    Test the model on the test dataset.
    1. Calculate the Overall MSE.
    1. Calculate the MSE for each nutritional fact.
    
    Args:
        model (torch.nn.Module): Trained model to evaluate.
        dataloader (DataLoader): DataLoader for the test dataset.
        device (torch.device): Device to run the model on.
    """
    
    model.eval()
    
    total_mse = 0.0  # Accumulate overall MSE
    task_mse = {task: 0.0 for task in model.task_heads.keys()}  # Initialize per-task MSE
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            
            # Calculate task-specific MSE and accumulate
            for task_idx, task in enumerate(outputs.keys()):
                predictions = outputs[task].squeeze()
                task_targets = targets[:, task_idx]
                
                # Update task-specific MSE
                task_mse[task] += mean_squared_error(
                    predictions.cpu().numpy(),
                    task_targets.cpu().numpy()
                )
                
             # Calculate overall MSE for this batch and accumulate
            batch_mse = sum(
                mean_squared_error(
                    outputs[task].squeeze().cpu().numpy(),
                    targets[:, task_idx].cpu().numpy()
                )
                for task_idx, task in enumerate(outputs.keys())
            )
            total_mse += batch_mse
            
    # Average the accumulated MSE over the number of batches
    num_batches = len(dataloader)
    overall_mse = total_mse / num_batches
    for task in task_mse:
        task_mse[task] /= num_batches
    
    return overall_mse, task_mse

def batch_scaler(y_bar, task):
    """
    Scale the input y_bar based on the task.
    First do inverse min-max, then to inverse log transformation.
    
    Args:
        y_bar (float): A batch of values to scale.
        task (str): Nutritional fact to scale.
    """
    min_val = min_max_df.loc[min_max_df['category'] == task, 'min'].values[0]
    max_val = min_max_df.loc[min_max_df['category'] == task, 'max'].values[0]
    
    y_bar = (y_bar * (max_val - min_val)) + min_val
    
    # then do log transformation
    y_bar = np.exp(y_bar) - 1
    return y_bar

def test_model_mae(model, dataloader, log_min_max, device):
    model.eval()  # Set the model to evaluation mode
    
    task_mae = {task: 0.0 for task in model.task_heads.keys()}  # Initialize MAE for each task
    task_percentage_mae = {task: 0.0 for task in model.task_heads.keys()}  # Initialize percentage MAE for each task
    task_counts = {task: 0 for task in model.task_heads.keys()}  # Track number of samples per task
    
    total_absolute_error = 0.0  # Accumulate total absolute error
    total_percentage_error = 0.0  # Accumulate total percentage error
    total_count = 0  # Track total number of samples across all tasks
    
    with torch.no_grad():  # Disable gradient computation for evaluation
        for inputs, targets in dataloader:
            # Move inputs and targets to the specified device
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Generate predictions
            outputs = model(inputs)
            
            # Process predictions and targets for each task
            for task_idx, task in enumerate(outputs.keys()):
                predictions = outputs[task].squeeze()
                task_targets = targets[:, task_idx]
                
                # Scale predictions and targets
                if log_min_max:
                    scaled_predictions = batch_scaler(predictions.cpu().numpy(), task)
                    scaled_targets = batch_scaler(task_targets.cpu().numpy(), task)
                else:
                    scaled_predictions = np.exp(predictions.cpu().numpy()) - 1
                    scaled_targets = np.exp(task_targets.cpu().numpy()) - 1
                    
                # Compute MAE and percentage error for this batch
                batch_absolute_error = np.abs(scaled_predictions - scaled_targets)
                batch_percentage_error = 100 * batch_absolute_error / np.abs(scaled_targets + 1e-8)  # Avoid division by zero
                
                # Update task-specific MAE and counts
                task_mae[task] += batch_absolute_error.sum()
                task_percentage_mae[task] += batch_percentage_error.sum()
                task_counts[task] += len(scaled_targets)
                
                # Update overall MAE
                total_absolute_error += batch_absolute_error.sum()
                total_percentage_error += batch_percentage_error.sum()
                total_count += len(scaled_targets)
                
                
    # Compute task-specific MAE
    for task in task_mae.keys():
        task_mae[task] /= task_counts[task]
        task_percentage_mae[task] /= task_counts[task]
    
    # Compute overall MAE and percentage MAE
    overall_mae = total_absolute_error / total_count
    overall_percentage_mae = total_percentage_error / total_count
    return {
        "overall_mae": overall_mae,
        "overall_percentage_mae": overall_percentage_mae,
        "task_mae": task_mae,
        "task_percentage_mae": task_percentage_mae
    }
    
def eval(model_name, save_name, log_min_max, s, device):
    if model_name == 'convlstm':
        model = ConvLSTM(["calories", "mass", "fat", "carb", "protein"]).to(device)
    elif model_name == 'vit':
        model = ViTModel(["calories", "mass", "fat", "carb", "protein"]).to(device)
    elif model_name == 'inceptionv3':
        model = InceptionV3Model(["calories", "mass", "fat", "carb", "protein"]).to(device)
    else:
        raise ValueError(f"Invalid model name: {model_name}")
    
    # load model
    model_path = f'./models/checkpoints/{save_name}.pth'
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    
    print(f"Evaluating model: {model_name}")
    
    # Prepare the test dataloader
    IMG_DIM = 299 if model_name == 'inceptionv3' else 224
    test_loader = prepare_test_loader(test_df, image_path, ["calories", "mass", "fat", "carb", "protein"] , img_dim=IMG_DIM, batch_size=16)

    # Evaluate the model
    overall_mse, mse_results = test_model_mse(model, test_loader, device)
    print(f"Overall MSE: {overall_mse}")
    for key, value in mse_results.items():
        print(f"{key}: {value}")
        
    print()
    results = test_model_mae(model, test_loader, log_min_max, device)
    print(f"Overall MAE: {results['overall_mae']}")
    for key, value in results['task_mae'].items():
        print(f"{key}: {value}")
        
    if s:
        save_dir = './results/'
        save_path = os.path.join(save_dir, s)
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        # write at the end of the file
        with open(save_path, 'a') as f:
            f.write(f"Evaluating model: {model_name} saved as {save_name}\n")
            f.write(f"Overall MSE: {overall_mse}\n")
            for key, value in mse_results.items():
                f.write(f"{key}: {value}\n")
            f.write("\n")
            f.write(f"Overall MAE: {results['overall_mae']}\n")
            for key, value in results['task_mae'].items():
                f.write(f"{key}: {value}\n")
            f.write("\n")
    
    return overall_mse, mse_results, results
    

if __name__ == "__main__":
    import argparse
    from utils.preprocess import prepare_test_loader

    # parse arguments
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('--model', type=str, default='inceptionv3', help='Model to eval (inceptionv3, convlstm, vit)')
    parser.add_argument('--model_name', type=str, required=True, help='Name of the model checkpoint to save')
    parser.add_argument('--log_min_max', type=bool, default=True, help='Used log min-max values')
    parser.add_argument('--s', type=str, required=False, help='Name of the file to save the results')
    
    args = parser.parse_args()

    # Evaluate the model
    eval(args.model, args.model_name, args.log_min_max, args.s, device)
            
        