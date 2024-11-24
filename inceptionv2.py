# %%
import torch
import torch.nn as nn
import torchvision.models as models

# Define the InceptionV2 backbone
class InceptionV2Backbone(nn.Module):
    def __init__(self, pretrained=True):
        super(InceptionV2Backbone, self).__init__()
        # Use InceptionV3 as a proxy for InceptionV2
        self.backbone = models.inception_v3(pretrained=pretrained, aux_logits=True)
        self.backbone.fc = nn.Identity()  # Remove the classification head

    def forward(self, x):
        # When aux_logits=True, the output is a tuple: (main_output, aux_output)
        x = self.backbone(x)
        if isinstance(x, tuple):  # Extract the main output
            x = x[0]
        return x


# %%
class NutritionModel(nn.Module):
    def __init__(self, num_tasks=3):
        """
        Args:
            num_tasks: Number of tasks (calories, macronutrients, and mass).
        """
        super(NutritionModel, self).__init__()
        self.backbone = InceptionV2Backbone()  # Use the corrected backbone
        self.shared_fc1 = nn.Linear(2048, 4096)  # Shared fully connected layer
        self.shared_fc2 = nn.Linear(4096, 4096)
        self.task_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(4096, 4096),
                nn.ReLU(),
                nn.Linear(4096, 1)
            ) for _ in range(num_tasks)
        ])

    def forward(self, x):
        x = self.backbone(x)
        x = nn.functional.relu(self.shared_fc1(x))
        x = nn.functional.relu(self.shared_fc2(x))
        outputs = [task_head(x) for task_head in self.task_heads]
        return torch.cat(outputs, dim=1)

# %%
# Define the loss function for multi-task learning
def multi_task_loss(predictions, targets):
    """
    Args:
        predictions: Tensor of shape (batch_size, num_tasks).
        targets: Tensor of shape (batch_size, num_tasks).
    Returns:
        Combined loss for all tasks.
    """
    losses = torch.abs(predictions - targets)  # Mean Absolute Error (MAE)
    return losses.mean()

# %%
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from PIL import Image
import os
import pandas as pd
from torch.utils.data import DataLoader, Dataset, TensorDataset

transform = transforms.Compose([
    transforms.Resize((299, 299)),  # Resize to Inception-compatible dimensions
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize if needed
])

# Step 3: Preprocess the Data
print('Preprocess the Data...')

image_path = '../data/nutrition5k_revised/images/'
meta_path = '../data/nutrition5k_revised/metadata.csv'
nutrition_label_path = '../data/nutrition5k_revised/labels/nutrition.csv'

images, labels = [], []

# Load metadata and labels
data = []
with open(meta_path, 'r') as f:
    lines = f.readlines()
    for line in lines:
        data.append(line.strip())

nutrition_labels = pd.read_csv(nutrition_label_path)

for idx in data:
    img = Image.open(os.path.join(image_path, idx, 'rgb.png')).convert('RGB')
    img = transform(img)
    images.append(img)

labels = nutrition_labels.iloc[:, 1:].values.astype(float)

# Convert to tensor
images = torch.stack(images)
labels = torch.tensor(labels, dtype=torch.float32)

# Split data into training and testing
train_images, test_images = images[:2800], images[2800:]
train_labels, test_labels = labels[:2800], labels[2800:]

# Data Loader
train_dataset = TensorDataset(train_images, train_labels)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
print('Data Preprocessing Done')

# %%
# Example training loop
def train_model(model, dataloader, optimizer, num_epochs=10):
    model = model.to(device)
    criterion = multi_task_loss
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(dataloader)}")
    
    # save model
    torch.save(model.state_dict(), 'model.pth')
    
def test_model(model, dataloader):
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            loss = multi_task_loss(outputs, targets)
            print(f"Test Loss: {loss.item()}")
        

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define the model
model = NutritionModel(num_tasks=5)  # Predicting 5 outputs: calories, mass, fat, carbs, protein

# Example optimizer
optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-4, momentum=0.9)

# Train the model
train_model(model, train_loader, optimizer, num_epochs=10)