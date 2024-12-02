# %%
import os
import torch
import pandas as pd
from utils.preprocess import load_data
from models.models import ConvLSTM, InceptionV3Model, ViTModel, ENetModel
import torch.nn as nn
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def perpare_data(model_type, batch_size, log_min_max, da):
    IMG_DIM = 299 if model_type == 'inceptionv3' else 224
    dataset_path = '../data/nutrition5k_reconstructed/'
    prepared_path = './utils/data'
    image_path = os.path.join(dataset_path, 'images')
    
    print(f'Log Min Max: {log_min_max}, DA: {da}')
    
    if log_min_max:
        train_labels = os.path.join(prepared_path, 'train_labels_lmm.csv')
        val_labels = os.path.join(prepared_path, 'val_labels_lmm.csv')
    else:
        train_labels = os.path.join(prepared_path, 'train_labels_log.csv')
        val_labels = os.path.join(prepared_path, 'val_labels_log.csv')
    
    df_train = pd.read_csv(train_labels)
    df_val = pd.read_csv(val_labels)
    
    print(df_train.head(3))
    
    labels = ["calories", "mass", "fat", "carb", "protein"]
    return load_data(df_train=df_train, df_val=df_val, image_path=image_path, labels = labels, img_dim = IMG_DIM, batch_size=batch_size, da=da)

def validate_model(model, val_loader, loss_fn=torch.nn.MSELoss()):
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


def train(model_backbone, train_loader, val_loader, batch_size, pretrained, epochs, checkpoint_name, learning_rate, patience):
    # print the model
    tasks = ["calories", "mass", "fat", "carb", "protein"]
    
    if model_backbone == 'inceptionv3':
        model = InceptionV3Model(tasks, pretrained).to(device)
    elif model_backbone == 'convlstm':
        model = ConvLSTM(tasks).to(device)
    elif model_backbone == 'vit':
        model = ViTModel(tasks, pretrained).to(device)
    elif model_backbone == 'enet':
        model = ENetModel(tasks, pretrained).to(device)
    else:
        raise ValueError("Invalid model backbone")
    
    mse_loss = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Print the number of trainable parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model Backbone: {model_backbone} (Learning rate: {learning_rate}, Batch Size: {batch_size}, Epochs: {epochs}, Pretrained: {pretrained}, Patience: {patience}, Saved as: {checkpoint_name})")
    print(f"Number of trainable parameters: {num_params}")

    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            targets = targets.to(device) 
            inputs = inputs.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            
            # Calculate loss for all tasks
            loss = sum(mse_loss(outputs[key].squeeze(), targets[:, i])
                    for i, key in enumerate(outputs.keys()))
            
            print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item()}")
            
            # Backward pass and optimization
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
            patience_counter = 0  # Reset patience counter

            if checkpoint_name is not None:
                torch.save(model.state_dict(), os.path.join(model_checkpoints, f"{checkpoint_name}.pth"))
                print("Model saved")
            else:
                torch.save(model.state_dict(), os.path.join(model_checkpoints, f"{model_backbone}_test.pth"))
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
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--save_name', type=str, required=False, help='Name of the model checkpoint to save')
    parser.add_argument('--patience', type=int, default=10, help='Patience for early stopping')  
    
    args = parser.parse_args()
    
    print(args)
    train_loader, val_loader = perpare_data(args.model_backbone, args.batch_size, args.log_min_max, args.da)
    print('Data Preprocessing Done')
    train(args.model_backbone, train_loader, val_loader, args.batch_size, args.pretrained, args.epochs, args.save_name, args.lr, args.patience)