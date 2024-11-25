import os
import pandas as pd
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torch
import torch.optim as optim
from torch.nn.functional import mse_loss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

IMG_DIMN = 224
BATCH_SIZE = 16

dataset_path = '../data/nutrition5k_reconstructed/'
prepared_path = './utils/data'

image_path = os.path.join(dataset_path, 'images')
train_labels = os.path.join(prepared_path, 'train_labels.csv')
val_labels = os.path.join(prepared_path, 'val_labels.csv')

df_train = pd.read_csv(train_labels)
df_val = pd.read_csv(val_labels)


class NutritionDataset(Dataset):
    def __init__(self, dataframe, image_dir, labels, transform=None):
        self.dataframe = dataframe
        self.image_dir = image_dir
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.dataframe.iloc[idx]["img_indx"])
        image = Image.open(img_name).convert("RGB")
        label = torch.tensor(self.dataframe.iloc[idx][self.labels].values.astype(float), dtype=torch.float32)
        if self.transform:
            image = self.transform(image)
        return image, label

# Transforms
train_transforms = transforms.Compose([
    transforms.RandomRotation(30),  # Randomly rotate the image by up to 30 degrees
    transforms.RandomAffine(0, shear=0.15),  # Shear the image by up to 0.15
    transforms.RandomResizedCrop(size=(IMG_DIMN, IMG_DIMN), scale=(0.6, 1.0)),  # Zoom range equivalent
    transforms.ColorJitter(brightness=(0.7, 1.0)),  # Brightness range [0.7, 1.0]
    transforms.RandomHorizontalFlip(),  # Random horizontal flip
    transforms.ToTensor(),  # Convert image to PyTorch tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Rescale and normalize
])

val_transforms = transforms.Compose([
    transforms.Resize((IMG_DIMN, IMG_DIMN)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Datasets and DataLoaders
labels = ["calories", "mass", "fat", "carb", "protein"]
train_dataset = NutritionDataset(df_train, image_path, labels, transform=train_transforms)
val_dataset = NutritionDataset(df_val, image_path, labels, transform=val_transforms)


train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

for images, labels in train_loader:
    print(images.shape)
    print(labels.shape)
    break

# %%
import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights  # Vision Transformer

class ViTBackbone(nn.Module):
    def __init__(self, weights=ViT_B_16_Weights.DEFAULT):
        """
        Args:
            weights: Pre-trained weights to use for the ViT model. Use `None` for no pre-training.
        """
        super().__init__()
        # Load the Vision Transformer with pretrained weights
        self.backbone = vit_b_16(weights=weights)
        self.backbone.heads = nn.Identity()  # Remove the classification head

    def forward(self, x):
        # Forward pass through the ViT backbone
        x = self.backbone(x)
        return x
    
# Test the backbone
backbone = ViTBackbone()
x = torch.randn(16, 3, IMG_DIMN, IMG_DIMN)
print(backbone(x).shape)

class NutritionModel(nn.Module):
    def __init__(self, weights=ViT_B_16_Weights.DEFAULT):
        super(NutritionModel, self).__init__()
        # Use the ViT backbone
        self.backbone = ViTBackbone(weights=weights)

        # Custom fully connected layers for regression
        self.fc1 = nn.Sequential(
            nn.Linear(768, 512),  # 768 is the output size of ViT-B-16
            nn.BatchNorm1d(512),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU()
        )
        self.fc3 = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU()
        )

        # Output layers for each prediction task
        self.out_calories = nn.Linear(512, 1)
        self.out_mass = nn.Linear(512, 1)
        self.out_fat = nn.Linear(512, 1)
        self.out_carb = nn.Linear(512, 1)
        self.out_protein = nn.Linear(512, 1)

    def forward(self, x):
        # Forward pass through the ViT backbone
        x = self.backbone(x)  # Output size: [batch_size, 768]

        # Pass through the fully connected layers
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        # Return predictions for each task
        return {
            "calories": self.out_calories(x),
            "mass": self.out_mass(x),
            "fat": self.out_fat(x),
            "carb": self.out_carb(x),
            "protein": self.out_protein(x)
        }

# Test the model
model = NutritionModel()
x = torch.randn(16, 3, IMG_DIMN, IMG_DIMN)
outputs = model(x)
for task, output in outputs.items():
    print(f"{task}: {output.shape}")
    
# Print the model
print(model)

# print the total trainable parameters and non-trainable parameters
total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"{total_trainable_params:,} trainable parameters.")

model = NutritionModel().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-5)

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

        val_loss = validate_model(model, val_loader)
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss/len(train_loader)}, Val Loss: {val_loss}")


        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoint_path)
            print("Model saved")
            
def validate_model(model, val_loader):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = sum(mse_loss(outputs[key].squeeze(), targets[:, i])
                       for i, key in enumerate(outputs.keys()))
            val_loss += loss.item()
    return val_loss / len(val_loader)

# Train the model
checkpoint_path = './models/checkpoints/vit.pth'
train_model(model, train_loader, val_loader, epochs=50, checkpoint_path=checkpoint_path)