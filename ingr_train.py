# %%
import os
import torch
import pandas as pd
from utils.preprocess import load_ingr_data
from models.models import ViTIngrModel, CLIngrModel, CLIngrV2, ResNetIngrModel
import torch.nn as nn
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def perpare_data(model_type, batch_size, log_min_max, da):
    IMG_DIM = 299 if model_type == 'inceptionv3' else 224
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

def ingredient_loss(outputs, targets, mask, alpha=0.9, l2_lambda=0.001, model_params=None):
    """
    Parameters:
    - outputs: Predicted masses (batch_size, num_ingr)
    - targets: Ground truth masses (batch_size, num_ingr)
    - mask: Binary mask (batch_size, num_ingr), 1 for non-zero, 0 for zero
    - alpha: Weight for non-zero ingredients
    - l2_lambda: Weight for L2 regularization

    Returns:
    - Weighted Mean Squared Error with L2 regularization
    """
    # Compute weighted MSE
    mse = (outputs - targets) ** 2
    weights = alpha * mask + (1 - alpha) * (1 - mask)
    weighted_mse = (weights * mse).mean()

    # Compute L2 regularization
    l2_reg = 0
    if model_params is not None:
        l2_reg = sum(torch.norm(p) ** 2 for p in model_params)
    
    # Combine losses
    total_loss = weighted_mse + l2_lambda * l2_reg
    return total_loss


def validate_ingr_model(model, val_loader, l2):
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
            regression_loss = ingredient_loss(outputs, targets[1], targets[0], model_params=model.parameters(), l2_lambda=l2)

            # Accumulate losses
            total_loss += regression_loss.item()

    # Compute average losses
    num_batches = len(val_loader)
    avg_total_loss = total_loss / num_batches
    return avg_total_loss

def train(model_backbone, train_loader, val_loader, batch_size, pretrained, epochs, checkpoint_name, learning_rate, patience, l2):
    # print the model
    num_ingr = 199
    
    if model_backbone == 'vit':
        model = ViTIngrModel(num_ingr, pretrained).to(device)
    elif model_backbone == 'convlstm':
        model = CLIngrModel(num_ingr).to(device)
    elif model_backbone == 'clv2':
        model = CLIngrV2(num_ingr).to(device)
    elif model_backbone == 'resnet':
        model = ResNetIngrModel(num_ingr, pretrained).to(device)
    else:
        raise ValueError("Invalid model backbone")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Print the number of trainable parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model Backbone: {model_backbone} (Learning rate: {learning_rate}, Batch Size: {batch_size}, Epochs: {epochs}, Pretrained: {pretrained}, l2: {l2}, Patience: {patience}, Saved as: {checkpoint_name})")
    print(f"Number of trainable parameters: {num_params}")

    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            targets = [targets[0].to(device), targets[1].to(device)]
            inputs = inputs.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            
            # Calculate loss for all tasks
            loss = ingredient_loss(outputs, targets[1], targets[0], model_params=model.parameters(), l2_lambda=l2)
            
            print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item()}")
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
    
        val_loss = validate_ingr_model(model, val_loader, l2)
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss/len(train_loader)}, Val Loss: {val_loss}")
            
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
            
    # Plot the training and validation losses
    plt.plot(train_losses, label='Training loss')
    plt.plot(val_losses, label='Validation loss')
    plt.legend()
    # save the plot
    plt.savefig(f'./plots/{checkpoint_name}_loss.png')

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
    parser.add_argument('--model_backbone', type=str, required= True, help='Model Backbone (inceptionv3, convlstm, vit, effnet)')
    parser.add_argument('--pretrained', type=str2bool, default=False, help='Use pre-trained weights')
    parser.add_argument('--log_min_max', type=str2bool, default=False, help='Use log min max normalization')
    parser.add_argument('--da', type=str2bool, default=True, help='Use data augmentation')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--save_name', type=str, required=False, help='Name of the model checkpoint to save')
    parser.add_argument('--patience', type=int, default=10, help='Patience for early stopping')  
    parser.add_argument('--l2', type=float, default=0.001, help='L2 regularization weight')
    
    args = parser.parse_args()
    
    train_loader, val_loader = perpare_data(args.model_backbone, args.batch_size, args.log_min_max, args.da)
    print('Data Preprocessing Done')
    train(args.model_backbone, train_loader, val_loader, args.batch_size, args.pretrained, args.epochs, args.save_name, args.lr, args.patience, args.l2)