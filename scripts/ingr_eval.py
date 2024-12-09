import os
import sys

# change the directory to the root of the project
os.chdir('../')
sys.path.append('./')

# Evaluate the model
import numpy as np
import torch
import pandas as pd
from models.models import BaselineModel, IngrPredModel, NutriFusionNet
from utils.preprocess import prepare_test_loader, prepare_test_loader_ingr
from torch import nn

############################################################################################################
'''
Prepare the test dataloader and utility functions
'''
# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

### Data Paths ###
dataset_path = '../data/nutrition5k_reconstructed/'

image_path = os.path.join(dataset_path, 'images')
ingr_mata = os.path.join(dataset_path, 'metadata/ingredients_metadata.csv')

# Load the ingredient metadata
ingr_dataset_path = './utils/data/test_labels_ingr_log.csv'
ingr_df = pd.read_csv(ingr_mata)
ingr_index = {}
ingr_indx_df = pd.read_csv(ingr_dataset_path)
colnames = ingr_indx_df.columns[1:-1]
for i in range(len(colnames)):
    ingr_index[i] = colnames[i]

# ingr,id,cal/g,fat(g),carb(g),protein(g)
ingr_dict = {}
for i in range(len(ingr_df)):
    ingr = ingr_df.iloc[i]['ingr']
    cal = ingr_df.iloc[i]['cal/g']
    fat = ingr_df.iloc[i]['fat(g)']
    carb = ingr_df.iloc[i]['carb(g)']
    protein = ingr_df.iloc[i]['protein(g)']
    ingr_dict[ingr] = (cal, fat, carb, protein)

# Load the min-max values of the nutritional facts
min_max_file = './utils/data/min_max_values_ingr.csv'
min_max_df = pd.read_csv(min_max_file)

### Load Nutrition Dataset ###
nutritional_facts_file = './utils/data/test_labels_lmm.csv'
nutrition_df = pd.read_csv(nutritional_facts_file)

# only select the nutritional facts with original values
nutrition_loader = prepare_test_loader(nutrition_df, image_path, ['original_calories','original_mass','original_fat','original_carb','original_protein'], 224, 16)


############################################################################################################
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
    
    dataset_id_path = './utils/data/test_labels_ingr_id.csv'
    if log_min_max:
        dataset_path = './utils/data/test_labels_ingr_lmm.csv'
    else:
        dataset_path = './utils/data/test_labels_ingr_log.csv'
    test_df = pd.read_csv(dataset_path)
    test_id_df = pd.read_csv(dataset_id_path)
    print(test_df.head(3))
    
    return prepare_test_loader_ingr(test_id_df, test_df, image_path, img_dim=IMG_DIM, batch_size=batch_size)
############################################################################################################
'''
Test Model's MSE performance
'''
def ingredient_loss(outputs, targets, mask, alpha=0.9):
    """
    Parameters:
    - outputs: Predicted masses (batch_size, num_ingr)
    - targets: Ground truth masses (batch_size, num_ingr)
    - mask: Binary mask (batch_size, num_ingr), 1 for non-zero, 0 for zero
    Returns:
    - Weighted Mean Squared Error with L2 regularization
    """
    # Compute weighted MSE
    if outputs.shape != targets.shape:
        raise ValueError(f"Output shape {outputs.shape} and target shape {targets.shape} do not match")
    
    mse = (outputs - targets) ** 2
    weights = alpha * mask + (1 - alpha) * (1 - mask)
    weighted_mse = (weights * mse).mean()

    return weighted_mse

def test_model_ingr(model, dataloader, device, threshold):
    """    
    Args:
        model (torch.nn.Module): Trained model to evaluate.
        dataloader (DataLoader): DataLoader for the test dataset.
        device (torch.device): Device to run the model on.
    """
    
    model.eval()
    
    total_regression_loss = 0.0  # Accumulate overall
    
    total_correct = 0
    total_samples = 0
    
    global_true_positives = 0
    global_predicted_positives = 0
    global_actual_positives = 0
    
   
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), [t.to(device) for t in targets]
            outputs = model(inputs)
            # outputs.shape = [(batch_size, 199)]
            
            ### BCE and MSE Loss ###
            regression_loss = ingredient_loss(outputs, targets[1], targets[0])
            
            # Calculate overall MSE for this batch and accumulate            
            total_regression_loss += regression_loss.item()
            
            ### Measure Classification Accuracy ###
            ### Precision, Recall, F1 Score ###
            predicted = (outputs > threshold).float().cpu().numpy()
            targets_classification = targets[0].cpu().numpy()
                    
            total_correct += np.sum(predicted == targets_classification)
            total_samples += targets_classification.size
                        
            global_true_positives += np.sum((predicted == 1) & (targets_classification == 1))
            global_predicted_positives += np.sum(predicted == 1)
            global_actual_positives += np.sum(targets_classification == 1)

    # Average the accumulated MSE over the number of batches
    num_batch = len(dataloader)
    overall_regression_loss = total_regression_loss / num_batch
    classification_accuracy = total_correct / total_samples

    # Compute global precision, recall, F1-score
    precision = global_true_positives / global_predicted_positives if global_predicted_positives > 0 else 0.0
    recall = global_true_positives / global_actual_positives if global_actual_positives > 0 else 0.0
    f1_score = (
        2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    )

    return overall_regression_loss, classification_accuracy, precision, recall, f1_score

############################################################################################################
'''
Test MAE performance and Specific MAE for each task
For Ingredient dataset, we also calculate classification accuracy
''' 
def test_model_ingr_mae(model, dataloader, log_min_max, device, threshold):
    model.eval()  # Set the model to evaluation mode
    
    tasks = ["calories", "mass", "fat", "carbs", "protein"]
    task_mae = {task: 0.0 for task in tasks}
    task_counts = {task: 0 for task in tasks}
    total_absolute_error = 0.0
    total_count = 0
    
    def lmm_scaling(x, mass):
        # x is the ingr name, mass is the predicted mass of the ingredient
        min_val = min_max_df[min_max_df['ingr'] == x]['min'].values[0]
        max_val = min_max_df[min_max_df['ingr'] == x]['max'].values[0]
        
        reverse_min_max_scaling = mass * (max_val - min_val) + min_val
        reverse_log_scaling = np.exp(reverse_min_max_scaling) - 1
        
        return reverse_log_scaling
        
    def calculate_nutritional_facts(outputs, log_min_max, ingr_dict, ingr_index):
        outputs = outputs.cpu().numpy() # outputs.shape = (batch_size, 199)
        total_calories = []
        total_mass = []
        total_fat = []
        total_carbs = []
        total_protein = []
        
        # outputs = (batch_size, 199)
        for batch_idx in range(outputs.shape[0]):
            # Find the indices of all non-zero ingredients
            idx = np.where(outputs[batch_idx] > threshold)[0]
            
            if len(idx) == 0:
                print("No ingredients found for this sample")
                total_calories.append(0)
                total_mass.append(0)
                total_fat.append(0)
                total_carbs.append(0)
                total_protein.append(0)
                continue # Skip this sample
            
            sample_calories = 0
            sample_mass = 0
            sample_fat = 0
            sample_carbs = 0
            sample_protein = 0
            
            for ingr_idx in idx:
                # Find the ingredient name and mass
                ingr_name = ingr_index[ingr_idx]
                mass = outputs[batch_idx][ingr_idx]
                if log_min_max:
                    mass = lmm_scaling(ingr_name, mass)
                else:
                    mass = np.exp(mass) - 1
                    
                # Find the nutritional facts for this ingredient
                cal, fat, carb, protein = ingr_dict[ingr_name]
                
                # Update the sample nutritional facts
                sample_calories += mass * cal
                sample_fat += mass * fat
                sample_carbs += mass * carb
                sample_protein += mass * protein
                sample_mass += mass
            
            # Append the sample nutritional facts
            total_calories.append(sample_calories)
            total_mass.append(sample_mass)
            total_fat.append(sample_fat)
            total_carbs.append(sample_carbs)
            total_protein.append(sample_protein)
                
        return {
            "calories": total_calories,
            "mass": total_mass,
            "fat": total_fat,
            "carbs": total_carbs,
            "protein": total_protein
        }
        
    with torch.no_grad():  # Disable gradient computation for evaluation
        for inputs, targets in dataloader:
            # Move inputs and targets to the specified device
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Generate predictions
            outputs = model(inputs)
            outputs = calculate_nutritional_facts(outputs, log_min_max, ingr_dict, ingr_index)
        
             # Process predictions and targets for each task
            for task_idx, task in enumerate(outputs.keys()):
                predictions = outputs[task]
                task_targets = targets[:, task_idx]
                               
                task_targets = task_targets.cpu().numpy()
                
                # Compute MAE and percentage error for this batch
                batch_absolute_error = np.abs(predictions - task_targets)
                
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
def eval(model_type, model_backbone, save_name, embed_path, test_loader, log_min_max, s, device, lstm_layers, attn_layers, threshold):
    ######## LOAD MODEL ########
    num_ingr = 199
    pretrained = False

    if model_type == "multimodal" or "customized" or "NutriFusionNet":
        embeddings = torch.load(f'./utils/embeddings/ingredient_embeddings_{embed_path}.pt', map_location=device, weights_only=True)
        print(embeddings.shape)

    if model_type == "NutriFusionNet":
        if model_backbone == 'vit' or model_backbone == 'convnx' or model_backbone == 'resnet' or model_backbone == 'incept' or model_backbone == 'effnet' or model_backbone == 'convlstm':
            model = NutriFusionNet(num_ingr, model_backbone, embeddings, pretrained, lstm_layers=lstm_layers, num_layers= attn_layers).to(device)
    elif model_type == "bb_lstm":
        if model_backbone == 'vit' or model_backbone == 'convnx' or model_backbone == 'resnet' or model_backbone == 'incept' or model_backbone == 'effnet' or model_backbone == 'convlstm':
            model = IngrPredModel(num_ingr, model_backbone, pretrained).to(device)
    elif model_type == 'baseline':
        model = BaselineModel(num_ingr).to(device)
    else:
        raise ValueError(f"Invalid model backbone: {model_backbone} or model type: {model_type}")
    
    model_path = f'./models/checkpoints/{save_name}.pth'
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    
    print(f"Evaluating model: {model_type} with {model_backbone} backbone saved as {save_name}")
    print(f"Threshold: {threshold}\n")

    ######## EVALUATION ########
    overall_regression_loss, classification_accuracy, precision, recall, f1_score = test_model_ingr(model, test_loader, device, threshold)
    print(f"Regression Loss (WMSE): {overall_regression_loss}\nClassification Accuracy: {classification_accuracy}\nPrecision: {precision}\nRecall: {recall}\nF1-Score: {f1_score}\n")
    
    # baseline mae
    baseline_mae_dict = {
        'overall_mae': 65.1307957381579,
        'calories': 168.5898685361842,
        'mass': 118.26738935526316,
        'fat': 10.349520950657896,
        'carbs': 13.908921615131577,
        'protein': 14.53827823355263
    }

    results = test_model_ingr_mae(model, nutrition_loader, log_min_max, device, threshold)
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

    if s:
        save_path = os.path.join(save_dir, f'{s}')
    else:
        save_path = os.path.join(save_dir, f'{save_name}_eval')
                
    with open(save_path, 'w') as f:
        f.write(f'Evaluation results for model: {model_backbone} saved as {save_name}\n\n')
        f.write(f"Threshold: {threshold}\n")
        f.write(f"Regression Loss (WMSE): {overall_regression_loss}\nClassification Accuracy: {classification_accuracy}\nPrecision: {precision}\nRecall: {recall}\nF1-Score: {f1_score}\n")
        f.write('\n')
        f.write(f"Overall MAE: {results['overall_mae']} (Improvement: {overall_mae_improvement:.2f}%)\n")
        for task in results['task_mae'].keys():
            f.write(f"{task}: {results['task_mae'][task]} (Improvement: {(baseline_mae_dict[task] - results['task_mae'][task]) / baseline_mae_dict[task] * 100:.2f}%)\n")
        f.write(f"Overall Average Improvement: {overall_improvement_percentage / 6:.2f}%")
    
    return overall_regression_loss, classification_accuracy, precision, recall, f1_score, results, overall_improvement_percentage / 6

def eval_all_models(model_type, model_backbones, embed_path, device, lstm_layers, attn_layers, thresholds):
    test_set = perpare_data('resnet', 16, False)
    
    results = {}
    for model_backbone in model_backbones:
        for lstm_layer in lstm_layers:
            for attn_layer in attn_layers:
                for threshold in thresholds:
                    # NutriFusionNet_vit_3lstm_2attn_gat_v2_pretrained_da_16_75_25_eval
                    save_name = f"{model_type}_{model_backbone}_{lstm_layer}lstm_{attn_layer}attn_{embed_path}_pretrained_da_16_75_25"
                    dict_name = save_name + + "_" + str(threshold)
                    regression_loss, classification_accuracy, precision, recall, f1_score, ingr_results, improvement = eval(model_type, model_backbone, save_name, embed_path, test_set, False, "all", device, lstm_layer, attn_layer, threshold)
                    results[dict_name] = {
                        'regression_loss': regression_loss,
                        'classification_accuracy': classification_accuracy,
                        'precision': precision,
                        'recall': recall,
                        'f1_score': f1_score,
                        'ingredient_results': ingr_results,
                        'overall_improvement': improvement
                    }
    return results

if __name__ == "__main__":
    # import argparse

    # def str2bool(v):
    #     if isinstance(v, bool):
    #         return v
    #     if v.lower() in ('yes', 'true', 't', 'y', '1'):
    #         return True
    #     elif v.lower() in ('no', 'false', 'f', 'n', '0'):
    #         return False
    #     else:
    #         raise argparse.ArgumentTypeError('Boolean value expected.')

    # # parse arguments
    # parser = argparse.ArgumentParser(description='Train a model')
    # parser.add_argument('--model_type', type=str, required=True, help='Type of the model')
    # parser.add_argument('--model_backbone', type=str, required= True, help='Model to eval')
    # parser.add_argument('--model_name', type=str, required=True, help='Name of the model checkpoint to save')
    # parser.add_argument('--log_min_max', type=str2bool, required= False, default=False, help='Used log min-max values')
    # parser.add_argument('--batch_size', type=int, required=False, default=16, help='Batch size for evaluation')
    # parser.add_argument('--embed_path', type=str, required=False, default='bert', help='Path to the ingredient embeddings')
    # parser.add_argument('--s', type=str, required=False, help='Name of the file to save the results')
    # parser.add_argument('--lstm_layers', type=int, required=False, default=1, help='Number of LSTM layers')
    # parser.add_argument('--attn_layers', type=int, required=False, default=1, help='Number of Attention layers')
    # parser.add_argument('--threshold', type=float, required=False, default=0.0, help='Threshold for classification')
    
    # args = parser.parse_args()
    
    # test_set = perpare_data(args.model_backbone, args.batch_size, args.log_min_max)

    # # Evaluate the model
    # eval(args.model_type, args.model_backbone, args.model_name, args.embed_path, test_set, args.log_min_max, args.s, device, args.lstm_layers, args.attn_layers, args.threshold)
        
    model_backbones = ['resnet', 'vit']
    lstm_layers = [2]
    attn_layers = [2]
    thresholds = [0.13, 0.14, 0.15]
    
    results = eval_all_models('NutriFusionNet', model_backbones, 'gat_v2', device, lstm_layers, attn_layers, thresholds)
    print(results)
    
    with open('./results/NutriFusionNet_eval_results.csv', 'w') as f:
        f.write("model_parameters,regression_loss,classification_accuracy,precision,recall,f1_score,calories_mae,mass_mae,fat_mae,carbs_mae,protein_mae,overall_improvement\n")
        for key, value in results.items():
            f.write(f"{key},{value['regression_loss']:.2f},{value['classification_accuracy']:.2f},{value['precision']:.2f},{value['recall']:.2f},{value['f1_score']:.2f},{value['ingredient_results']['task_mae']['calories']:.2f},{value['ingredient_results']['task_mae']['mass']:.2f},{value['ingredient_results']['task_mae']['fat']:.2f},{value['ingredient_results']['task_mae']['carbs']:.2f},{value['ingredient_results']['task_mae']['protein']:.2f},{value['overall_improvement']:.2f}\n")