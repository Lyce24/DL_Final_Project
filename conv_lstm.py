# %%
import os
import torch
import pandas as pd
from utils.preprocess import load_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMG_DIMN = 224
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
import torch.nn.functional as F

class ConvLSTM(nn.Module):
    def __init__(self, hidden_dim=512, num_classes=5):
        super(ConvLSTM, self).__init__()
        # Define convolutional layers to extract spatial features
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        
        # Batch normalization layers
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(512)
        self.bn5 = nn.BatchNorm2d(512)
        
        # Define LSTM layer
        self.lstm = nn.LSTM(input_size=512 * 7 * 7, hidden_size=hidden_dim, num_layers=2, batch_first=True, dropout=0.5)
        
        # Define fully connected output layer
        self.fc1 = nn.Linear(hidden_dim, 256)
        self.fc2 = nn.Linear(256, num_classes)
        
    def forward(self, x):
        batch_size = x.size(0)

        # Pass input through convolutional layers with batch normalization and pooling
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2, 2)  # Output: (batch, 64, 112, 112)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2, 2)  # Output: (batch, 128, 56, 56)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, 2, 2)  # Output: (batch, 256, 28, 28)
        
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.max_pool2d(x, 2, 2)  # Output: (batch, 512, 14, 14)
        
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.max_pool2d(x, 2, 2)  # Output: (batch, 512, 7, 7)
        
        # Flatten the output from conv layers for LSTM
        x = x.view(batch_size, 1, -1)  # Output: (batch, 1, 512 * 7 * 7)
        
        # Pass through LSTM
        lstm_out, _ = self.lstm(x)  # Output: (batch, 1, hidden_dim)
        
        # Extract the output from the LSTM
        lstm_out = lstm_out[:, -1, :]  # Output: (batch, hidden_dim)
        
        # Pass through the fully connected layers
        x = F.relu(self.fc1(lstm_out))  # Output: (batch, 256)
        out = self.fc2(x)  # Output: (batch, num_classes)
        
        return out
    
x = torch.randn(16, 3, 224, 224)
model = ConvLSTM()
output = model(x)
print(output.shape)

# print out the total trainable parameters
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Number of trainable parameters: {num_params}")

# %%
import torch.optim as optim
import torch.nn as nn

model = ConvLSTM(len(labels)).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(train_loader)

def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for i, (images, labels) in enumerate(val_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
    return running_loss / len(val_loader)

n_epochs = 30
train_losses = []
val_losses = []
min_loss = float('inf')
for epoch in range(n_epochs):
    train_loss = train(model, train_loader, criterion, optimizer, device)
    val_loss = validate(model, val_loader, criterion, device)
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    
    print(f"Epoch {epoch+1}/{n_epochs} Train Loss: {train_loss:.4f} Validation Loss: {val_loss:.4f}")

    if val_loss < min_loss:
        min_loss = val_loss
        torch.save(model.state_dict(), './models/checkpoints/convlstm_best.pth')
        print("Model saved with loss: ", min_loss)
        
import matplotlib.pyplot as plt

plt.plot(train_losses, label='Training loss')
plt.plot(val_losses, label='Validation loss')
plt.legend()
# save the plot
plt.savefig('./plots/convlstm_loss.png')

