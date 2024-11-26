# %%
import os
import torch
import pandas as pd
from utils.preprocess import load_data
from models.models import ConvLSTM, InceptionV3Model, ViTModel
import argparse
import torch.nn as nn
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def perpare_data(model_type, batch_size, log_min_max):
    if model_type == 'inceptionv3':
        IMG_DIM = 299
    else:
        IMG_DIM = 224
        
    dataset_path = '../data/nutrition5k_reconstructed/'
    prepared_path = './utils/data'

    image_path = os.path.join(dataset_path, 'images')
    if log_min_max:
        train_labels = os.path.join(prepared_path, 'train_labels.csv')
        val_labels = os.path.join(prepared_path, 'val_labels.csv')
    else:
        train_labels = os.path.join(prepared_path, 'train_labels_log.csv')
        val_labels = os.path.join(prepared_path, 'val_labels_log.csv')

    df_train = pd.read_csv(train_labels)
    df_val = pd.read_csv(val_labels)

    labels = ["calories", "mass", "fat", "carb", "protein"]
    return load_data(df_train=df_train, df_val=df_val, image_path=image_path, labels = labels, img_dim = IMG_DIM, batch_size=batch_size) 

def validate_model(model, val_loader, loss_fn=nn.MSELoss()):
    model.eval()
    val_loss = 0.0
    # individual losses for each task
    losses = {key: 0.0 for key in model.task_heads.keys()}
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            
            for i, key in enumerate(outputs.keys()):
                loss = loss_fn(outputs[key].squeeze(), targets[:, i])
                losses[key] += loss.item()
            
            val_loss += sum(losses.values())
            
    for key in losses.keys():
        losses[key] /= len(val_loader)
        
    return val_loss / len(val_loader), losses

def train(model_name, train_loader, val_loader, pretrained, epochs, checkpoint_name, learning_rate=1e-5):
    # print the model
    tasks = ["calories", "mass", "fat", "carb", "protein"]
    
    if model_name == 'inceptionv3':
        model = InceptionV3Model(tasks).to(device)
    elif model_name == 'convlstm':
        model = ConvLSTM(tasks, pretrained).to(device)
    elif model_name == 'vit':
        model = ViTModel(tasks, pretrained).to(device)
    
    mse_loss = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Print the number of trainable parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {model_name} (Learning rate: {learning_rate}, Epochs: {epochs})")
    print(f"Number of trainable parameters: {num_params}")

    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            # Calculate loss for all tasks
            loss = sum(mse_loss(outputs[key].squeeze(), targets[:, i])
                       for i, key in enumerate(outputs.keys()))
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
    
        val_loss, ind_loss = validate_model(model, val_loader, mse_loss)
        ind_loss_str = ", ".join([f"{key}: {val:.4f}" for key, val in ind_loss.items()])
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss/len(train_loader)}, Val Loss: {val_loss}, Ind Loss: {ind_loss_str}")
             
        train_losses.append(train_loss / len(train_loader))
        val_losses.append(val_loss)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if checkpoint_name is not None:
                torch.save(model.state_dict(), os.path.join(model_checkpoints, f"{checkpoint_name}.pth"))
                print("Model saved")
            else:
                torch.save(model.state_dict(), os.path.join(model_checkpoints, f"{model_name}_best.pth"))
                print("Model saved")
            
    # Plot the training and validation losses
    plt.plot(train_losses, label='Training loss')
    plt.plot(val_losses, label='Validation loss')
    plt.legend()
    # save the plot
    plt.savefig(f'./plots/{checkpoint_name}_loss.png')

if __name__ == '__main__':
    model_checkpoints = './models/checkpoints/'
    
    # parse arguments
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('--model', type=str, default='inceptionv3', help='Model to train (inceptionv3, convlstm, vit)')
    parser.add_argument('--pretrained', type=bool, default=True, help='Use pre-trained weights')
    parser.add_argument('--log_min_max', type=bool, default=True, help='Use log min max normalization')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--save_name', type=str, required=False, help='Name of the model checkpoint to save')
    
    args = parser.parse_args()
    
    train_loader, val_loader = perpare_data(args.model, args.batch_size, args.log_min_max)
    train(args.model, train_loader, val_loader, args.pretrained, args.epochs, args.save_name, args.lr)