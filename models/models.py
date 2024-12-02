import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from torchvision.models import inception_v3, Inception_V3_Weights, vit_b_16, ViT_B_16_Weights, efficientnet_v2_m, EfficientNet_V2_M_Weights, resnet50, ResNet50_Weights

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
    
    
class CLIngrModel(nn.Module):
    def __init__(self, num_ingr = 199, hidden_dim=512):
        super(CLIngrModel, self).__init__()
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
        self.fc1 = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # 2 task heads: ingredient prediction and mass prediction
        # self.task_head = nn.ModuleList([
        #     # for ingredient prediction
        #     nn.Sequential(
        #         nn.Linear(256, 256),
        #         nn.ReLU(),
        #         nn.Linear(256, num_ingr) # Logits for each ingredient (Loss Function: BCEWithLogitsLoss)
        #     ),
        #     # for mass prediction
        #     nn.Sequential(
        #         nn.Linear(256, 256),
        #         nn.ReLU(),
        #         nn.Linear(256, num_ingr),
        #         nn.ReLU() # ReLU activation for mass prediction (non-negative)
        #     )
        # ])
        
        self.mass_prediction = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_ingr),
            nn.ReLU() # ReLU activation for mass prediction (non-negative)
        )
        
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
        x = self.fc1(lstm_out)  # Output: (batch, 256)
        
        # # Pass through task-specific heads
        # outputs = self.task_head[0](x), self.task_head[1](x)
        outputs = self.mass_prediction(x)
        
        return outputs
    
class CLIngrV2(nn.Module):
    def __init__(self, num_ingr = 199, hidden_dim=512):
        super(CLIngrV2, self).__init__()
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
        
        # Task-specific heads
        self.ingredient_head = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, num_ingr) # Logits for each ingredient (Loss Function: BCEWithLogitsLoss)
        )
        
        # Refinement module for ingredient embeddings
        self.refinement = nn.Sequential(
            nn.Linear(num_ingr, 128),
            nn.ReLU(),
            nn.Linear(128, num_ingr)
        )
        
        # Mass prediction head
        self.mass_head = nn.Sequential(
            nn.Linear(256 + num_ingr, 256),  # Input size adjusted for refined ingredient embeddings
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_ingr),
            nn.ReLU()  # ReLU activation for non-negative predictions
        )
        
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

        # Shared features
        x = F.relu(self.fc1(lstm_out)) # Output: (batch, 256)

        # Ingredient prediction
        ingr_logits = self.ingredient_head(x)  # Output: (batch, num_ingr)
        
        # Refinement module
        refined_ingr = self.refinement(ingr_logits)  # Output: (batch, num_ingr)
        
        # Concatenate refined ingredient embeddings with shared features
        x = torch.cat([x, refined_ingr], dim=1)  # Output: (batch, 256 + num_ingr)
        
        # Mass prediction
        mass_pred = self.mass_head(x)  # Output: (batch, num_ingr)
        
        return ingr_logits, mass_pred 

############################################################################################################
'''
BackBones
- InceptionV3
- Vision Transformer (ViT)
- EfficientNet
'''
class InceptionV3(nn.Module):
    def __init__(self, pretrained = True):
        """
        Args:
            weights: Pre-trained weights to use for the InceptionV3 model. Use `None` for no pre-training.
        """
        super().__init__()
        # Load the InceptionV3 model with specified weights
        if pretrained:
            self.backbone = inception_v3(weights=Inception_V3_Weights.DEFAULT, aux_logits=True)
        else:
            self.backbone = inception_v3(weights=None, aux_logits=True)            
        self.backbone.fc = nn.Identity()  # Remove the classification head
        
    def forward(self, x):
        # When aux_logits=True, the output is a tuple: (main_output, aux_output)
        x = self.backbone(x)
        return x[0] if isinstance(x, tuple) else x

class ViTBackbone(nn.Module):
    def __init__(self, pretrained = True):
        """
        Args:
            weights: Pre-trained weights to use for the ViT model. Use `None` for no pre-training.
        """
        super().__init__()
        # Load the Vision Transformer with pretrained weights
        if pretrained:
            self.backbone = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
        else:
            self.backbone = vit_b_16(weights=None)
        self.backbone.heads = nn.Identity()  # Remove the classification head

    def forward(self, x):
        # Forward pass through the ViT backbone
        x = self.backbone(x)
        return x
    
class EfficientNetBackbone(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        # Load EfficientNet with pre-trained weights
        if pretrained:
            self.backbone = efficientnet_v2_m(weights=EfficientNet_V2_M_Weights.DEFAULT)
        else:
            self.backbone = efficientnet_v2_m(weights=None)
        self.backbone.classifier = nn.Identity()  # Remove the classification head

    def forward(self, x):
        # Forward pass through EfficientNet backbone
        return self.backbone(x)
    
class ResNet50Backbone(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        # Load ResNet50 with pre-trained weights
        if pretrained:
            self.backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        else:
            self.backbone = resnet50(weights=None) # Input size: 224x224
        self.backbone.fc = nn.Identity()  # Remove the classification head

    def forward(self, x):
        # Forward pass through ResNet50 backbone
        return self.backbone(x)
    
############################################################################################################
'''
For Nutritional Facts Prediction (MTL)
'''
class InceptionV3Model(nn.Module):
    def __init__(self, tasks : List[str], pretrained = True):
        """
        Args:
            num_tasks: Number of tasks (calories, macronutrients, and mass).
        """
        super(InceptionV3Model, self).__init__()
        self.backbone = InceptionV3(pretrained)  # Use the corrected backbone
        
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
    
class ViTModel(nn.Module):
    def __init__(self, tasks : List[str], pretrained = True):
        super(ViTModel, self).__init__()
        
        # Use the ViT backbone
        self.backbone = ViTBackbone(pretrained)

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

class ENetModel(nn.Module):
    def __init__(self, tasks: List[str], pretrained=True):
        super(ENetModel, self).__init__()
        
        # Use the EfficientNet backbone
        self.backbone = EfficientNetBackbone(pretrained)
        
        # Custom fully connected layers for regression
        self.fc1 = nn.Sequential(
            nn.Linear(1280, 512),  # 1280 is the output size of EfficientNet
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
        # Forward pass through the EfficientNet backbone
        x = self.backbone(x)  # Output size: [batch_size, 1280]

        # Pass through the fully connected layers
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        # Pass through task-specific heads
        outputs = {task: head(x) for task, head in self.task_heads.items()}
        return outputs

############################################################################################################
'''
For Ingredient + Mass Prediction (Regression)
'''
class ViTIngrModel(nn.Module):
    def __init__(self, num_ingr = 199, pretrained = True, hidden_dim=512, dropout_rate=0.3):
        super(ViTIngrModel, self).__init__()
        
        # Use the ViT backbone
        self.backbone = ViTBackbone(pretrained)
        
        # Custom fully connected layers for regression
        self.fc_layers = nn.Sequential(
            nn.Linear(768, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Task-specific regression head
        # self.task_heads = nn.ModuleList([
        #     nn.Sequential(
        #         nn.Linear(hidden_dim, hidden_dim),
        #         nn.ReLU(),
        #         nn.Linear(hidden_dim, num_ingr) # Logits for each ingredient (Loss Function: BCEWithLogitsLoss)
        #     ),
        #     nn.Sequential(
        #         nn.Linear(hidden_dim, hidden_dim),
        #         nn.ReLU(),
        #         nn.Linear(hidden_dim, num_ingr),
        #         nn.ReLU() # ReLU activation for mass prediction (non-negative)
        #     )
        # ])
        self.mass_prediction = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_ingr),
            nn.ReLU() # ReLU activation for mass prediction (non-negative)
        )
        

    def forward(self, x):
        # Forward pass through ViT backbone
        x = self.backbone(x)  # Output: [batch_size, 768]
        
        # Pass through fully connected layers
        x = self.fc_layers(x)
        
        # Task-specific head
        # outputs = [head(x) for head in self.task_heads]
        outputs = self.mass_prediction(x)
        return outputs
    
class ResNetIngrModel(nn.Module):
    def __init__(self, num_ingr = 199, pretrained = False, hidden_dim = 512, dropout_rate = 0.3):
        super(ResNetIngrModel, self).__init__()
        
        # Use the ResNet50 backbone
        self.backbone = ResNet50Backbone(pretrained)
        
        # Custom fully connected layers for regression
        self.fc_layers = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(1024, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Task-specific regression head
        self.mass_prediction = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_ingr),
            nn.ReLU() # ReLU activation for mass prediction (non-negative)
        )
        
    def forward(self, x):
        # Forward pass through ViT backbone
        x = self.backbone(x)  # Output: [batch_size, 768]
        
        # Pass through fully connected layers
        x = self.fc_layers(x)
        
        # Task-specific head
        # outputs = [head(x) for head in self.task_heads]
        outputs = self.mass_prediction(x)
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
        
    print("\nTesting the EfficientNet model")
    # Test the EfficientNet model
    x = torch.randn(16, 3, 224, 224)
    model = ENetModel(tasks)
    output = model(x)
    for task, output in output.items():
        print(f"{task}: {output.shape}") # Should be [16, 1]
        
    x = torch.randn(16, 3, 224, 224) # For ConvLSTM model and ViT model
    num_ingr = 199
    
    model = CLIngrModel(num_ingr)
    output = model(x)
    print(f"ConvLSTM Ingr model output shape: {output[0].shape}") # Expected output shape: [16, 199]
    
    model = CLIngrV2(num_ingr)
    output = model(x)
    print(f"ConvLSTM Ingr v2 model output shape: {output[0].shape}") # Expected output shape: [16, 199]
    
    # Test the ViT model
    print("\nTesting the ViT model")
    model = ViTIngrModel(num_ingr)
    output = model(x)
    print(f"ViT Ingr model output shape: {output[0].shape}") # Expected output shape: [16, 199]
    
    print("\nTesting the ResNet model")
    model = ResNetIngrModel(num_ingr)
    output = model(x)
    print(f"ResNet Ingr model output shape: {output[0].shape}") # Expected output shape: [16, 199]
        
    print("All tests passed!")