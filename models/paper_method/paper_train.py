# %%
import os
import sys
import time

# change the directory to the root of the project
os.chdir('../../')
sys.path.append('./')

import torch
import pandas as pd
from utils.preprocess import load_data
from models.models import PaperModel
import torch.nn as nn
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def perpare_data(batch_size, da):
    IMG_DIM = 299 # inceptionv3 requires 299x299
    dataset_path = '../data/nutrition5k_reconstructed/'
    prepared_path = './utils/data'
    image_path = os.path.join(dataset_path, 'images')
    
    print(f'DA: {da}')
    
    train_labels = os.path.join(prepared_path, 'train_labels_log.csv')
    val_labels = os.path.join(prepared_path, 'val_labels_log.csv')
    
    df_train = pd.read_csv(train_labels)
    df_val = pd.read_csv(val_labels)
    
    print(df_train.head(3))
    
    labels = ["calories", "mass", "fat", "carb", "protein"]
    return load_data(df_train=df_train, df_val=df_val, image_path=image_path, labels = labels, img_dim = IMG_DIM, batch_size=batch_size, da=da)

def validate_model(model, val_loader, loss_fn):
    """
    Validate the model on a validation dataset.

    Args:
        model: The model to be evaluated, which should have task-specific heads.
        val_loader: A DataLoader for the validation dataset.
        device: The device to perform computation on (e.g., 'cuda' or 'cpu').
        loss_fn: Loss function to compute the error (default: nn.MSELoss).

    Returns:
        tuple: (average validation loss, dictionary of average losses for each task)
    """
    model.eval()
    total_val_loss = 0.0
    # Dictionary to accumulate task-specific losses
    task_losses = {key: 0.0 for key in model.task_heads.keys()}
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            
            batch_loss = 0.0
            for i, key in enumerate(outputs.keys()):
                loss = loss_fn(outputs[key].squeeze(), targets[:, i])
                    
                task_losses[key] += loss.item()
                batch_loss += loss.item()
            
            total_val_loss += batch_loss
    
    # Compute average losses
    avg_task_losses = {key: task_losses[key] / len(val_loader) for key in task_losses}
    avg_val_loss = total_val_loss / len(val_loader)
    
    return avg_val_loss, avg_task_losses


def train(train_loader, val_loader, epochs, checkpoint_name, learning_rate, patience):
    # print the model
    tasks = ["calories", "mass", "fat", "carb", "protein"]
    
    model = PaperModel(tasks).to(device)
    
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Print the number of trainable parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
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
            targets = targets.to(device) 
            inputs = inputs.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            
            # Calculate loss for all tasks
            loss = sum(loss_fn(outputs[key].squeeze(), targets[:, i]) for i, key in enumerate(outputs.keys()))
            
            print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item()}")
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            
        end_time = time.time()
        epoch_time = (end_time - start_time) / 60

        val_loss, ind_loss = validate_model(model, val_loader, loss_fn=loss_fn)
        ind_loss_str = ", ".join([f"{key}: {val:.4f}" for key, val in ind_loss.items()])
        
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss/len(train_loader)}, Val Loss: {val_loss}, Ind Loss: {ind_loss_str}, Time Used: {epoch_time:.2f}")
             
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
                torch.save(model.state_dict(), os.path.join(model_checkpoints, f"paper_method.pth"))
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
    model_checkpoints = './models/checkpoints/'
    train_loader, val_loader = perpare_data(16, True)
    print('Data Preprocessing Done')
    train(train_loader, val_loader, 75, 'paper_method', 1e-4, 25)