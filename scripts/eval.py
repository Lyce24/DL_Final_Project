import os
import sys

# change the directory to the root of the project
os.chdir('../')
sys.path.append('./')

# Evaluate the model
import numpy as np
import torch
import pandas as pd
from models.models import ConvLSTM, InceptionV3Model, ViTModel
from utils.preprocess import prepare_test_loader
from torch import nn
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

# Load the min-max values
min_max_file = './utils/min_max_values.csv'
min_max_df = pd.read_csv(min_max_file)

### Load Nutrition Dataset ###
nutritional_facts_file = './utils/data/test_labels_lmm.csv'
nutrition_df = pd.read_csv(nutritional_facts_file)

# only select the nutritional facts with original values
nutrition_loader = prepare_test_loader(nutrition_df, image_path, ['original_calories','original_mass','original_fat','original_carb','original_protein'], 224, 16)

def baseline_mae():
    total_absolute_error = 0.0
    total_count = 0
    task_mae = {task: 0.0 for task in ["calories", "mass", "fat", "carb", "protein"]}
    task_counts = {task: 0 for task in ["calories", "mass", "fat", "carb", "protein"]}
    
    '''
            calories	mass	    fat	        carb	    protein
    mean	255.012738	214.980074	12.727633	19.386111	18.004492
    '''
    nutrition_dict = {
        'calories': 255.012738,
        'mass': 214.980074,
        'fat': 12.727633,
        'carb': 19.386111,
        'protein': 18.004492
    }
    
    for task in task_mae.keys():
        task_targets = nutrition_df[f'original_{task}'].values
        task_mae[task] = np.abs(task_targets - nutrition_dict[task]).sum()
        task_counts[task] = len(task_targets)
        total_absolute_error += task_mae[task]
        total_count += task_counts[task]
        
    for task in task_mae.keys():
        task_mae[task] /= task_counts[task]
    with open ('./results/baseline_mae.txt', 'w') as f:
        f.write(f"Overall MAE: {total_absolute_error / total_count}\n")
        for task in task_mae.keys():
            f.write(f"{task}: {task_mae[task]}\n")


############################################################################################################
def perpare_data(model_backbone, batch_size, log_min_max):
    print(f'Log Min Max: {log_min_max}')
    
    # Prepare the test dataloader
    IMG_DIM = 299 if model_backbone == 'inceptionv3' else 224
    

    if log_min_max:
        dataset_path = './utils/data/test_labels_lmm.csv'
    else:
        dataset_path = './utils/data/test_labels_log.csv'

    test_df = pd.read_csv(dataset_path)
    print(test_df.head(3))
            
    return prepare_test_loader(test_df, image_path, ["calories", "mass", "fat", "carb", "protein"], img_dim=IMG_DIM, batch_size=batch_size)

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

def test_model_mae(model, dataloader, log_min_max, device):
    model.eval()  # Set the model to evaluation mode
    
    task_mae = {task: 0.0 for task in model.task_heads.keys()}  # Initialize MAE for each task
    task_counts = {task: 0 for task in model.task_heads.keys()}  # Track number of samples per task
    
    total_absolute_error = 0.0  # Accumulate total absolute error
    total_count = 0  # Track total number of samples across all tasks
    
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
                else:
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

def eval(model_backbone, save_name, test_loader, log_min_max, s, device):
    if model_backbone == 'convlstm':
        model = ConvLSTM(["calories", "mass", "fat", "carb", "protein"]).to(device)
    elif model_backbone == 'vit':
        model = ViTModel(["calories", "mass", "fat", "carb", "protein"]).to(device)
    elif model_backbone == 'inceptionv3':
        model = InceptionV3Model(["calories", "mass", "fat", "carb", "protein"]).to(device)
    else:
        raise ValueError(f"Invalid model name: {model_backbone}")
    
    # load model
    model_path = f'./models/checkpoints/{save_name}.pth'
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    
    print(f"Evaluating model: {model_backbone} saved as {save_name}")

    # Evaluate the model
    overall_mse, mse_results = test_model_mse(model, test_loader, device)
    print(f"Overall MSE: {overall_mse}")
    for key, value in mse_results.items():
        print(f"{key}: {value}")
    print()
    results = test_model_mae(model, nutrition_loader, log_min_max, device)
    print(f"Overall MAE: {results['overall_mae']}")
    for key, value in results['task_mae'].items():
        print(f"{key}: {value}")
    if s:
        save_dir = './results/'
        save_path = os.path.join(save_dir, s)
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        # write at the end of the file
        with open(save_path, 'w') as f:
            f.write(f"Evaluating model: {model_backbone} saved as {save_name}\n")
            f.write(f"Overall MSE: {overall_mse}\n")
            for key, value in mse_results.items():
                f.write(f"{key}: {value}\n")
            f.write("\n")
            f.write(f"Overall MAE: {results['overall_mae']}\n")
            for key, value in results['task_mae'].items():
                f.write(f"{key}: {value}\n")
            
if __name__ == "__main__":
    import argparse

    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    # parse arguments
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('--model_backbone', type=str, required= True, help='Model to eval')
    parser.add_argument('--model_name', type=str, required=True, help='Name of the model checkpoint to save')
    parser.add_argument('--log_min_max', type=str2bool, default=True, help='Used log min-max values')
    parser.add_argument('--s', type=str, required=False, help='Name of the file to save the results')
    
    args = parser.parse_args()
    
    # baseline_mae()
    
    test_set = perpare_data(args.model_backbone, 16, args.log_min_max)

    # Evaluate the model
    eval(args.model_backbone, args.model_name, test_set, args.log_min_max, args.s, device)