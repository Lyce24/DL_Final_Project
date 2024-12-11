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
import time

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
ingr_dataset_path = './utils/data/train_labels_ingr_log.csv'
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
nutritional_facts_file = './utils/data/train_labels_lmm.csv'
nutrition_df = pd.read_csv(nutritional_facts_file)

# only select the nutritional facts with original values
nutrition_loader = prepare_test_loader(nutrition_df, image_path, ['original_calories','original_mass','original_fat','original_carb','original_protein'], 224, 16)


############################################################################################################
def perpare_data(model_backbone, batch_size, log_min_max):
    print(f'Log Min Max: {log_min_max}')
    
    # Prepare the test dataloader
    IMG_DIM = 299 if model_backbone == 'inceptionv3' else 224
    
    dataset_id_path = './utils/data/train_labels_ingr_id.csv'
    val_id_path = './utils/data/val_labels_ingr_id.csv'
    test_id_path = './utils/data/test_labels_ingr_id.csv'
    if log_min_max:
        dataset_path = './utils/data/train_labels_ingr_lmm.csv'
    else:
        dataset_path = './utils/data/train_labels_ingr_log.csv'
        val_dataset_path = './utils/data/val_labels_ingr_log.csv'
        test_dataset_path = './utils/data/test_labels_ingr_log.csv'
        
    train_df = pd.read_csv(dataset_path)
    val_df = pd.read_csv(val_dataset_path)
    test_df = pd.read_csv(test_dataset_path)
    
    test_df = pd.concat([val_df, test_df, train_df], ignore_index=True)
    
    train_id_df = pd.read_csv(dataset_id_path)
    val_id_df = pd.read_csv(val_id_path)
    test_id_df = pd.read_csv(test_id_path)
    
    test_id_df = pd.concat([val_id_df, test_id_df, train_id_df], ignore_index=True)
    
    print(test_df.head(3))
    print(len(test_df))
    
    return prepare_test_loader_ingr(test_id_df, test_df, image_path, img_dim=IMG_DIM, batch_size=batch_size)
############################################################################################################

def test_speed(model, dataloader, device):
    """    
    Args:
        model (torch.nn.Module): Trained model to evaluate.
        dataloader (DataLoader): DataLoader for the test dataset.
        device (torch.device): Device to run the model on.
    """
    
    model.eval()
    
    start_time = time.time()
   
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), [t.to(device) for t in targets]
            _ = model(inputs)

    end_time = time.time()
    
    return end_time - start_time            

def eval(model_type, model_backbone, save_name, embed_path, test_loader, device, lstm_layers, attn_layers, extraction_mode):
    ######## LOAD MODEL ########
    num_ingr = 199
    pretrained = False

    if model_type == "multimodal" or "customized" or "NutriFusionNet":
        embeddings = torch.load(f'./embeddings/ingredient_embeddings_{embed_path}.pt', map_location=device, weights_only=True)
        print(embeddings.shape)

    if model_type == "NutriFusionNet":
        if model_backbone == 'vit' or model_backbone == 'convnx' or model_backbone == 'resnet' or model_backbone == 'incept' or model_backbone == 'effnet' or model_backbone == 'convlstm':
            model = NutriFusionNet(num_ingr, model_backbone, embeddings, pretrained, lstm_layers=lstm_layers, num_layers= attn_layers, extraction_mode=extraction_mode).to(device)
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
    
    # Test the speed of the model
    speed = test_speed(model, test_loader, device)
    
    return speed

def eval_all_models(model_type, model_backbones, embed_path, device, lstm_layers, attn_layers, extraction_modes):
    test_set = perpare_data('resnet', 3000, False) # batch size = 16, log_min_max = False
    
    results = {}
    for extraction_mode in extraction_modes:
        for model_backbone in model_backbones:
            for lstm_layer in lstm_layers:
                for attn_layer in attn_layers:
                        if extraction_mode == 'lstm':
                            save_name = f"{model_type}_{model_backbone}_{lstm_layer}lstm_{attn_layer}attn_{embed_path}_pretrained_da_16_75_25"
                        else:
                            save_name = f"{model_type}_{model_backbone}_{extraction_mode}_{lstm_layer}lstm_{attn_layer}attn_{embed_path}_pretrained_da_16_75_25"
                        dict_name = save_name
                        speed = eval(model_type, model_backbone, save_name, embed_path, test_set, device, lstm_layer, attn_layer, extraction_mode)
                        results[dict_name] = {
                            'speed': speed
                        }
    return results

if __name__ == "__main__":
    model_backbones = ['resnet', 'vit'] 
    lstm_layers = [2]
    attn_layers = [2]
    extraction_modes = ['global_pooling', 'attn_pooling','lstm'] # lstm_dropout 'attn_pooling', 'global_pooling', 'lstm_dropout'
    
    results = eval_all_models('NutriFusionNet', model_backbones, 'gat_512', device, lstm_layers, attn_layers, extraction_modes)
    print(results)
    
    with open('./results/NutriFusionNet_speed_results.csv', 'w') as f:
        f.write("model_parameters,speed\n")
        for key, value in results.items():
            f.write(f"{key},{value['speed']}\n")