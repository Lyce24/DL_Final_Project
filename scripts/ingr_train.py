# %%
import os
import sys

# change the directory to the root of the project
os.chdir('../')
sys.path.append('./')

import torch
import pandas as pd
from utils.preprocess import load_ingr_data
from models.models import BaselineModel, IngrPredModel, MultimodalPredictionNetwork
import matplotlib.pyplot as plt
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def perpare_data(model_type, batch_size, log_min_max, da):
    IMG_DIM = 299 if model_type == 'incept' else 224
    dataset_path = '../data/nutrition5k_reconstructed/'
    prepared_path = './utils/data'
    image_path = os.path.join(dataset_path, 'images')
    
    print(f'Log Min Max: {log_min_max}, DA: {da}')
    
    train_id_labels = os.path.join(prepared_path, 'train_labels_ingr_id.csv')
    val_id_labels = os.path.join(prepared_path, 'val_labels_ingr_id.csv')
    df_train_id = pd.read_csv(train_id_labels)
    df_val_id = pd.read_csv(val_id_labels)
    
    if log_min_max:
        train_labels = os.path.join(prepared_path, 'train_labels_ingr_lmm.csv')
        val_labels = os.path.join(prepared_path, 'val_labels_ingr_lmm.csv')
    else:
        train_labels = os.path.join(prepared_path, 'train_labels_ingr_log.csv')
        val_labels = os.path.join(prepared_path, 'val_labels_ingr_log.csv')

    df_train = pd.read_csv(train_labels)
    df_val = pd.read_csv(val_labels)

    print(df_train.head(3))
    
    return load_ingr_data(df_train_ingr_id=df_train_id, df_val_ingr_id=df_val_id, df_train_ingr=df_train, df_val_ingr=df_val, image_path=image_path, img_dim=IMG_DIM, batch_size=batch_size)

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

def validate_ingr_model(model, val_loader):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for inputs, targets in val_loader:
            # Move data to the appropriate device
            inputs = inputs.to(device)
            targets = [t.to(device) for t in targets]

            # Get model outputs
            outputs = model(inputs)

            # Compute loss
            loss = ingredient_loss(outputs, targets[1], targets[0])
            
            # Accumulate losses
            total_loss += loss.item()

    # Compute average losses
    num_batches = len(val_loader)
    avg_total_loss = total_loss / num_batches
    return avg_total_loss

def train(model_backbone, model_type, embed_path, train_loader, val_loader, batch_size, pretrained, epochs, checkpoint_name, learning_rate, patience):
    num_ingr = 199
    
    if model_type == "multimodal" or "customized":
        embeddings = torch.load(f'./utils/data/ingredient_embeddings_{embed_path}.pt', map_location=device, weights_only=True)

        print(embeddings.shape)

    if model_type == "multimodal":
        if model_backbone == 'vit' or model_backbone == 'convnx' or model_backbone == 'resnet' or model_backbone == 'incept' or model_backbone == 'effnet' or model_backbone == 'convlstm':
            model = MultimodalPredictionNetwork(num_ingr, model_backbone, embeddings, pretrained, hidden_dim = 512).to(device)
    elif model_type == "bb_lstm":
        if model_backbone == 'vit' or model_backbone == 'convnx' or model_backbone == 'resnet' or model_backbone == 'incept' or model_backbone == 'effnet' or model_backbone == 'convlstm':
            model = IngrPredModel(num_ingr, model_backbone, pretrained).to(device)
    elif model_type == 'baseline':
        model = BaselineModel(num_ingr).to(device)
    else:
        raise ValueError(f"Invalid model backbone: {model_backbone} or model type: {model_type}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Print the number of trainable parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model Type: {model_type} (Backbone: {model_backbone}, Learning rate: {learning_rate}, Batch Size: {batch_size}, Epochs: {epochs}, Pretrained: {pretrained},  Patience: {patience}, Saved as: {checkpoint_name})")
    print(f"Number of trainable parameters: {num_params}")

    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    
    training_start_time = time.time()
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        start_time = time.time()
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            targets = [targets[0].to(device), targets[1].to(device)]
            inputs = inputs.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            
            loss = ingredient_loss(outputs, targets[1], targets[0])
            
            print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item()}")
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
    
        val_loss = validate_ingr_model(model, val_loader)
        end_time = time.time()
        epoch_time = (end_time - start_time) / 60
        
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss/len(train_loader)}, Val Loss: {val_loss}, Time Used: {epoch_time:.2f}")
            
        train_losses.append(train_loss / len(train_loader))
        val_losses.append(val_loss)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0  # Reset patience counter

            if checkpoint_name is not None:
                torch.save(model.state_dict(), os.path.join(model_checkpoints, f"{checkpoint_name}.pth"))
                print("Model saved")
            else:
                torch.save(model.state_dict(), os.path.join(model_checkpoints, f"{model_backbone}_best.pth"))
                print("Model saved")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
            
    training_end_time = time.time()
    # Plot the training and validation losses
    plt.plot(train_losses, label='Training loss')
    plt.plot(val_losses, label='Validation loss')
    plt.legend()
    # save the plot
    plt.savefig(f'./plots/{checkpoint_name}_loss.png')
    
    print(f"Total training time: {training_end_time - training_start_time:.2f} seconds, Best Validation Loss: {best_val_loss}")
    
if __name__ == '__main__':
    import argparse

    model_checkpoints = './models/checkpoints/'
    
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
    parser.add_argument('--model_type', type=str, required= True, help='Model Type (multimodal, bb_lstm, baseline)')
    parser.add_argument('--model_backbone', type=str, required= True, help='Model Backbone (convlstm, vit, clv2, convnx)')
    parser.add_argument('--embed_path', type=str, required=False, default='bert', help='Path to the ingredient embeddings')
    parser.add_argument('--pretrained', type=str2bool, default=False, help='Use pre-trained weights')
    parser.add_argument('--log_min_max', type=str2bool, default=False, help='Use log min max normalization')
    parser.add_argument('--da', type=str2bool, default=True, help='Use data augmentation')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--save_name', type=str, required=False, help='Name of the model checkpoint to save')
    parser.add_argument('--patience', type=int, default=10, help='Patience for early stopping')  
    
    args = parser.parse_args()
    
    train_loader, val_loader = perpare_data(model_type=args.model_backbone,
                                            batch_size=args.batch_size,
                                            log_min_max=args.log_min_max,
                                            da=args.da)
    
    print('Data Preprocessing Done')
    train(model_backbone=args.model_backbone, 
            model_type=args.model_type,
            embed_path=args.embed_path,
            train_loader=train_loader,
            val_loader=val_loader,
            batch_size=args.batch_size,
            pretrained=args.pretrained,
            epochs=args.epochs,
            checkpoint_name=args.save_name,
            learning_rate=args.lr,
            patience=args.patience)