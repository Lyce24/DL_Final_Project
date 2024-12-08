import os
import sys

# change the directory to the root of the project
os.chdir('../../')
sys.path.append('./')

# Evaluate the model
import numpy as np
import torch
import pandas as pd
from models.models import PaperModel
from utils.preprocess import prepare_test_loader
from sklearn.metrics import mean_squared_error

############################################################################################################
'''
Prepare the test dataloader and utility functions
'''
# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

### Data Paths ###
dataset_path = '../data/nutrition5k_reconstructed/'

image_path = os.path.join(dataset_path, 'images')

### Load Nutrition Dataset ###
nutritional_facts_file = './utils/data/test_labels_log.csv'
nutrition_df = pd.read_csv(nutritional_facts_file)

# only select the nutritional facts with original values
nutrition_loader = prepare_test_loader(nutrition_df, image_path, ['original_calories','original_mass','original_fat','original_carb','original_protein'], 224, 16)


############################################################################################################
def perpare_data():
    # Prepare the test dataloader
    IMG_DIM = 299
    
    dataset_path = './utils/data/test_labels_log.csv'

    test_df = pd.read_csv(dataset_path)
    print(test_df.head(3))
            
    return prepare_test_loader(test_df, image_path, ["calories", "mass", "fat", "carb", "protein"], img_dim=IMG_DIM, batch_size=16)

############################################################################################################
'''
Test Model's MSE performance
'''
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

############################################################################################################
'''
Test MAE performance and Specific MAE for each task
For Ingredient dataset, we also calculate classification accuracy
'''

def test_model_mae(model, dataloader, device):
    model.eval()  # Set the model to evaluation mode
    
    task_mae = {task: 0.0 for task in model.task_heads.keys()}  # Initialize MAE for each task
    task_counts = {task: 0 for task in model.task_heads.keys()}  # Track number of samples per task
    
    total_absolute_error = 0.0  # Accumulate total absolute error
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
                scaled_predictions = np.exp(predictions.cpu().numpy()) - 1
                
                task_targets = task_targets.cpu().numpy()
                
                # Compute MAE and percentage error for this batch
                batch_absolute_error = np.abs(scaled_predictions - task_targets)
                
                # Update task-specific MAE and counts
                task_mae[task] += batch_absolute_error.sum()
                task_counts[task] += len(task_targets)
                
                # Update overall MAE
                total_absolute_error += batch_absolute_error.sum()
                total_count += len(task_targets)
                
                
    # Compute task-specific MAE
    for task in task_mae.keys():
        task_mae[task] /= task_counts[task]
    
    # Compute overall MAE and percentage MAE
    overall_mae = total_absolute_error / total_count
    return {
        "overall_mae": overall_mae,
        "task_mae": task_mae,
    }
    
############################################################################################################

def eval(save_name, test_loader, device):
    tasks = ["calories", "mass", "fat", "carb", "protein"]
    
    model = PaperModel(tasks).to(device)
    # load model
    model_path = f'./models/checkpoints/{save_name}.pth'
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    
    # Evaluate the model
    overall_mse, mse_results = test_model_mse(model, test_loader, device)
    print(f"Overall MSE: {overall_mse}")
    for key, value in mse_results.items():
        print(f"{key}: {value}")
    
    # baseline mae
    baseline_mae_dict = {
        'overall_mae': 65.1307957381579,
        'calories': 168.5898685361842,
        'mass': 118.26738935526316,
        'fat': 10.349520950657896,
        'carb': 13.908921615131577,
        'protein': 14.53827823355263
    }   
    
    print()
    results = test_model_mae(model, nutrition_loader, device)
    overall_improvement_percentage = 0.0
    overall_mae_improvement = (baseline_mae_dict['overall_mae'] - results['overall_mae']) / baseline_mae_dict['overall_mae'] * 100
    overall_improvement_percentage += overall_mae_improvement
    print(f"Overall MAE: {results['overall_mae']} (Improvement: {overall_mae_improvement:.2f}%)")
    for task in results['task_mae'].keys():
        improvement_percentage = (baseline_mae_dict[task] - results['task_mae'][task]) / baseline_mae_dict[task] * 100
        overall_improvement_percentage += improvement_percentage
        print(f"{task}: {results['task_mae'][task]} (Improvement: {improvement_percentage:.2f}%)")
        
    print(f"Overall Average Improvement: {overall_improvement_percentage / 6:.2f}%")
    
    save_dir = './results/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_path = os.path.join(save_dir, f'paper_method_eval')
                
    with open(save_path, 'w') as f:
        f.write(f'Evaluation results for model: Paper_Method\n\n')
        f.write(f"Overall MSE: {overall_mse}\n")
        for key, value in mse_results.items():
            f.write(f"{key}: {value}\n")
        f.write('\n')
        f.write(f"Overall MAE: {results['overall_mae']} (Improvement: {overall_mae_improvement:.2f}%)\n")
        for task in results['task_mae'].keys():
            f.write(f"{task}: {results['task_mae'][task]} (Improvement: {(baseline_mae_dict[task] - results['task_mae'][task]) / baseline_mae_dict[task] * 100:.2f}%)\n")
        f.write(f"Overall Average Improvement: {overall_improvement_percentage / 6:.2f}%")
        
            
if __name__ == "__main__":   
    test_set = perpare_data()

    # Evaluate the model
    model_path = 'paper_method'
    eval(model_path, test_set, device)