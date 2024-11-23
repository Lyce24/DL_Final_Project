# %%
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from PIL import Image
import os
import pandas as pd

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# %%
class CRNNNutritionNet(nn.Module):
    def __init__(self):
        super(CRNNNutritionNet, self).__init__()
        # CNN Feature Extractor
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        # LSTM for temporal dependency
        self.rnn = nn.LSTM(input_size=256*28*28, hidden_size=512, num_layers=2, batch_first=True, dropout=0.5)
        
        # Fully Connected Layers
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 5)  # Predicting 5 outputs: calories, mass, fat, carbs, protein
        )

    def forward(self, x):
        batch_size = x.size(0)
        x = self.cnn(x)
        x = x.view(batch_size, -1)  # Flatten
        x = x.unsqueeze(1)  # Add sequence dimension for LSTM
        x, _ = self.rnn(x)
        x = x[:, -1, :]  # Take the last output of LSTM
        x = self.fc(x)
        return x

# %%
from torch.utils.data import DataLoader, Dataset, TensorDataset

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
train_images, test_images = images[:2500], images[2500:]
train_labels, test_labels = labels[:2500], labels[2500:]

# Data Loader
train_dataset = TensorDataset(train_images, train_labels)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
print('Data Preprocessing Done')
# %%
# Step 5: Train the model
model = CRNNNutritionNet()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

model.to(device)

num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item()}')
