import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from torchvision.models import inception_v3, Inception_V3_Weights, vit_b_16, ViT_B_16_Weights

############################################################################################################
class ConvLSTM(nn.Module):
    def __init__(self, tasks : List[str], hidden_dim=512):
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
        
        self.task_heads = nn.ModuleDict({
            task: nn.Sequential(
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, 1)
            ) for task in tasks
        })
        
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
        
        # Pass through task-specific heads
        outputs = {task: head(x) for task, head in self.task_heads.items()}
        
        return outputs
    
############################################################################################################
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

class InceptionV3Model(nn.Module):
    def __init__(self, tasks : List[str]):
        """
        Args:
            num_tasks: Number of tasks (calories, macronutrients, and mass).
        """
        super(InceptionV3Model, self).__init__()
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
    
############################################################################################################
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

class ViTModel(nn.Module):
    def __init__(self, tasks : List[str]):
        super(ViTModel, self).__init__()
        
        # Use the ViT backbone
        self.backbone = ViTBackbone()

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
        
        # Task-specific heads
        self.task_heads = nn.ModuleDict({
            task: nn.Sequential(
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 1)
            ) for task in tasks
        })

    def forward(self, x):
        # Forward pass through the ViT backbone
        x = self.backbone(x)  # Output size: [batch_size, 768]

        # Pass through the fully connected layers
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

    def forward(self, x):
        # Process the image through the backbone
        x = self.backbone(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
                
        # Pass through task-specific heads
        outputs = {task: head(x) for task, head in self.task_heads.items()}  
        return outputs
   
############################################################################################################
if __name__ == "__main__":
    x = torch.randn(16, 3, 224, 224) # For ConvLSTM model and ViT model
    tasks = ["calories", "mass", "fat", "carb", "protein"]
    
    # Test the ConvLSTM model
    print("Testing the ConvLSTM model")
    model = ConvLSTM(tasks)
    output = model(x)
    for task, output in output.items():
        print(f"{task}: {output.shape}") # Should be [16, 1]
    
    # Test the ViT model
    print("\nTesting the ViT model")
    model = ViTModel(tasks)
    output = model(x)
    for task, output in output.items():
        print(f"{task}: {output.shape}") # Should be [16, 1]

    print("\nTesting the InceptionV3 model")
    # Test the InceptionV3 model
    x = torch.randn(16, 3, 299, 299)
    model = InceptionV3Model(tasks)
    output = model(x)
    for task, output in output.items():
        print(f"{task}: {output.shape}") # Should be [16, 1]