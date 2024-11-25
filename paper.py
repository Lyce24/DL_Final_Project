# %%
import sys
import os
import torch
import pandas as pd
from utils.preprocess import load_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMG_DIMN = 299
BATCH_SIZE = 16

dataset_path = '../data/nutrition5k_reconstructed/'
prepared_path = './utils/data'

image_path = os.path.join(dataset_path, 'images')
train_labels = os.path.join(prepared_path, 'train_labels.csv')
val_labels = os.path.join(prepared_path, 'val_labels.csv')

df_train = pd.read_csv(train_labels)
df_val = pd.read_csv(val_labels)

labels = ["calories", "mass", "fat", "carb", "protein"]
train_loader, val_loader = load_data(df_train=df_train, df_val=df_val, image_path=image_path, labels = labels, img_dim = IMG_DIMN, batch_size=BATCH_SIZE)

for images, labels in train_loader:
    print(images.shape)
    print(labels.shape)
    break

# %%
import torch
import torch.nn as nn
from torchvision.models import inception_v3, Inception_V3_Weights

# Define the InceptionV3 backbone
class InceptionV3(nn.Module):
    def __init__(self, weights=Inception_V3_Weights.DEFAULT):
        """
        Args:
            weights: Pre-trained weights to use for the InceptionV3 model. Use `None` for no pre-training.
        """
        super().__init__()
        # Load the InceptionV3 model with specified weights
        self.backbone = inception_v3(weights=weights, aux_logits=True)
        self.backbone.fc = nn.Identity()  # Remove the classification head
        
    def forward(self, x):
        # When aux_logits=True, the output is a tuple: (main_output, aux_output)
        x = self.backbone(x)
        return x[0] if isinstance(x, tuple) else x

# test the forward pass
model = InceptionV3()
x = torch.randn(16, 3, 299, 299)
print(model(x).shape)

# %%
from typing import List
class NutritionModel(nn.Module):
    def __init__(self, tasks : List[str]):
        """
        Args:
            num_tasks: Number of tasks (calories, macronutrients, and mass).
        """
        super(NutritionModel, self).__init__()
        self.backbone = InceptionV3()  # Use the corrected backbone
        
        # Shared image feature layers
        self.shared_fc1 = nn.Linear(2048, 4096) # Use 2048 as input size as InceptionV3 has 2048 output features
        self.shared_fc2 = nn.Linear(4096, 4096)
        
        
        # Task-specific heads
        self.task_heads = nn.ModuleDict({
            task: nn.Sequential(
                nn.Linear(4096, 4096),
                nn.ReLU(),
                nn.Linear(4096, 1)
            ) for task in tasks
        })

    def forward(self, image):
        # Process the image through the backbone
        image_features = self.backbone(image)
        image_features = nn.functional.relu(self.shared_fc1(image_features))
        image_features = nn.functional.relu(self.shared_fc2(image_features))
        
        # Pass through task-specific heads
        outputs = {task: head(image_features) for task, head in self.task_heads.items()}        
        return outputs
            
# print the model
tasks = ["calories", "mass", "fat", "carb", "protein"]

model = NutritionModel(tasks)
# print how many parameters the model has
x = torch.randn(16, 3, 299, 299)
y_bar = model(x)
print(y_bar["calories"].shape)

# Print the number of trainable parameters
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Number of trainable parameters: {num_params}")

# %%
from torch.nn.functional import mse_loss

# Training function
def train_model(model, train_loader, val_loader, epochs, checkpoint_path):
    best_val_loss = float('inf')

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

        val_loss, ind_loss = validate_model(model, val_loader)
        ind_loss_str = ", ".join([f"{key}: {val:.4f}" for key, val in ind_loss.items()])
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss/len(train_loader)}, Val Loss: {val_loss}, Ind Loss: {ind_loss_str}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoint_path)
            print("Model saved")
            
def validate_model(model, val_loader):
    model.eval()
    val_loss = 0.0
    # individual losses for each task
    losses = {key: 0.0 for key in model.task_heads.keys()}
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            
            for i, key in enumerate(outputs.keys()):
                loss = mse_loss(outputs[key].squeeze(), targets[:, i])
                losses[key] += loss.item()
            
            val_loss += sum(losses.values())
            
    for key in losses.keys():
        losses[key] /= len(val_loader)
        
    return val_loss / len(val_loader), losses

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define the model
tasks = ["calories", "mass", "fat", "carb", "protein"]

model = NutritionModel(tasks).to(device)

loss_fn = nn.MSELoss()  # Mean Squared Error (MSE) loss
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

# Train the model (testing with 1 epoch)
train_model(model, train_loader, val_loader, epochs=5, checkpoint_path="./models/checkpoints/paper.pth")
