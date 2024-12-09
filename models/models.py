import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from torchvision.models import (
    inception_v3, Inception_V3_Weights, 
    vit_b_16, ViT_B_16_Weights, 
    efficientnet_v2_m, EfficientNet_V2_M_Weights, 
    convnext_small, ConvNeXt_Small_Weights, 
    resnet50, ResNet50_Weights
    )

############################################################################################################
'''
BackBones
- InceptionV3
- Vision Transformer (ViT)
- EfficientNet
- ConvNeXt
- ResNet50
'''
class ResNetBackbone(nn.Module):
    def __init__(self, pretrained = True):
        super().__init__()
        # Load ResNet50 with pre-trained weights
        if pretrained:
            self.backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        else:
            self.backbone = resnet50(weights=None)
        self.backbone.fc = nn.Identity()  # Remove the classification head

    def forward(self, x):
        # Forward pass through ResNet50 backbone
        return self.backbone(x)
    
class ConvNextBackbone(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        # Load ConvNeXt with pre-trained weights
        if pretrained:
            self.backbone = convnext_small(weights=ConvNeXt_Small_Weights.IMAGENET1K_V1)
        else:
            self.backbone = convnext_small(weights=None) # input size: 3x224x224
        self.backbone.classifier = nn.Identity()  # Remove the classification head

    def forward(self, x):
        # Forward pass through ConvNeXt backbone
        features = self.backbone(x)
        output_features = features.view(features.size(0), -1)
        return output_features
    
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
    
############################################################################################################
'''
For Ingredient + Mass Prediction (Regression)
- Our Implementation of Mass Prediction Model
    - BaselineModel: CNN-based model for mass prediction
    - IngrPredModel: Pretrained backbone-based + LSTM-based model for ingredient prediction
    - NutriFusionNet: Stacking Attention Encoder-Decoder Network for ingredient prediction
'''

class BaselineModel(nn.Module):
    def __init__(self, num_ingr = 199, hidden_dim=512):
        super(BaselineModel, self).__init__()
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
        
        # flatten the output from conv layers for FC
        self.fc1 = nn.Sequential(
            nn.Linear(512 * 7 * 7, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
 
        self.mass_prediction = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_ingr),
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
        
        # Flatten the output from conv layers for FC
        x = x.view(batch_size, -1)  # Output: (batch, 512 * 7 * 7)
        
        # Pass through the fully connected layers
        x = self.fc1(x)  # Output: (batch, hidden_dim)
        
        # Pass through task-specific heads
        outputs = self.mass_prediction(x)
        
        return outputs
        
class IngrPredModel(nn.Module):
    def __init__(self, num_ingr = 199, backbone = 'resnet', pretrained = True, hidden_dim = 512, dropout_rate=0.3):
        super(IngrPredModel, self).__init__()
        
        # Use the specified backbone
        self.backbone_type = backbone
        if backbone == "vit":
            self.backbone = ViTBackbone(pretrained)
            input_size = 768
        elif backbone == "convnx":
            self.backbone = ConvNextBackbone(pretrained)
            input_size = 768
        elif backbone == "resnet":
            self.backbone = ResNetBackbone(pretrained)
            input_size = 2048
        elif backbone == "effnet":
            self.backbone = EfficientNetBackbone(pretrained)
            input_size = 1280
        elif backbone == "incept":
            self.backbone = InceptionV3(pretrained)
            input_size = 2048
        else:
            raise ValueError(f"Invalid backbone {backbone}")
        
        # Define LSTM layer
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_dim, num_layers=2, batch_first=True, dropout=0.5)
        
        # Custom fully connected layers for regression
        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )

        self.mass_prediction = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_ingr),
        )
        
    def forward(self, x):
        batch_size = x.size(0)
        # Forward pass through ViT backbone
        x = self.backbone(x)  # Output: [batch_size, input_size]
        
        x = x.view(batch_size, 1, -1)  # Output: (batch, 1, input_size)
        # Pass through LSTM layer
        lstm_out, _ = self.lstm(x)
        
        # Extract the output from the LSTM
        lstm_out = lstm_out[:, -1, :]  # Output: (batch, hidden_dim)
        
        x = self.fc_layers(lstm_out)
        
        # Task-specific head
        outputs = self.mass_prediction(x)
        
        return outputs
 
############## Multimodal Encoder-Decoder Network ##########################
class IngredientFeatureExtractor(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, lstm_layers):
        super(IngredientFeatureExtractor, self).__init__()
        # 300-dimensional GloVe embeddings for ingredients
        # 768-dimensional BERT embeddings for ingredients
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=lstm_layers, batch_first=True)

    def forward(self, ingredient_embeddings):
        # ingredient_embeddings: (num_ingr, embedding_dim) as each image will have the same candidate ingredients
        # Pass through LSTM
        lstm_out, _ = self.lstm(ingredient_embeddings)  # Output: (batch, 1, hidden_dim)
        
        # Extract the output from the LSTM
        ingr_feature = lstm_out[:, -1, :]  # Output: (batch, hidden_dim)
        
        return ingr_feature
    
class ImageFeatureExtractor(nn.Module):
    def __init__(self, backbone='resnet', pretrained=True, hidden_dim=256):
        super(ImageFeatureExtractor, self).__init__()
        
        # Use the specified backbone
        self.backbone_type = backbone
        if backbone == "vit":
            self.backbone = ViTBackbone(pretrained)
            input_size = 768
        elif backbone == "convnx":
            self.backbone = ConvNextBackbone(pretrained)
            input_size = 768
        elif backbone == "resnet":
            self.backbone = ResNetBackbone(pretrained)
            input_size = 2048
        elif backbone == "effnet":
            self.backbone = EfficientNetBackbone(pretrained)
            input_size = 1280
        elif backbone == "incept":
            self.backbone = InceptionV3(pretrained)
            input_size = 2048
        else:
            raise ValueError(f"Invalid backbone {backbone}")

        self.image_fc = nn.Linear(input_size, hidden_dim)

    def forward(self, images):
        # images: (batch_size, 3, 224, 224)
        image_features = self.backbone(images) # (batch_size, 768)
        image_features = self.image_fc(image_features) # (batch_size, hidden_dim)
        return image_features

class AttentionEncoderDecoder(nn.Module):
    def __init__(self, num_heads, d_model, dropout=0.3, epsilon=1e-6):
        super(AttentionEncoderDecoder, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.dropout = dropout
        self.epsilon = epsilon

        # Encoder layers (for ingredient features)
        # self-attention
        self.ingredients_sa = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, dropout=dropout)
        self.ingredients_sa_normalized = nn.LayerNorm(d_model, eps=epsilon)
        self.ingredients_aligned = nn.Linear(d_model, d_model)
        self.ingredients_encoded = nn.LayerNorm(d_model, eps=epsilon)

        # Decoder layers (for dish image features)
        # guided attention + self-attention
        self.dish_image_ga = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, dropout=dropout)
        self.dish_image_ga_normalized = nn.LayerNorm(d_model, eps=epsilon)
        
        # self-attention to model spatial relationships among dish image regions
        self.dish_image_sa = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, dropout=dropout)
        self.dish_image_sa_normalized = nn.LayerNorm(d_model, eps=epsilon)
        self.dish_image_aligned = nn.Linear(d_model, d_model)
        self.dish_image_decoded = nn.LayerNorm(d_model, eps=epsilon)

    def forward(self, ingredients, dish_image):
        # Encoder: Ingredients Self-Attention
        ingredients_sa, _ = self.ingredients_sa(ingredients, ingredients, ingredients) # (batch_size, d_model)
        ingredients_sa = self.ingredients_sa_normalized(ingredients_sa + ingredients) # (batch_size, d_model)
        ingredients_aligned = F.relu(self.ingredients_aligned(ingredients_sa)) # (batch_size, dff)
        ingredients_encoded = self.ingredients_encoded(ingredients_sa + ingredients_aligned) # (batch_size, d_model)

        # Decoder: Dish Image Guided Attention + Self-Attention
        dish_image_ga, _ = self.dish_image_ga(dish_image, ingredients_encoded, ingredients_encoded)
        dish_image_ga = self.dish_image_ga_normalized(dish_image_ga + dish_image)
        dish_image_sa, _ = self.dish_image_sa(dish_image_ga, dish_image_ga, dish_image_ga)
        dish_image_sa = self.dish_image_sa_normalized(dish_image_sa + dish_image_ga)
        dish_image_aligned = F.relu(self.dish_image_aligned(dish_image_sa))
        dish_image_decoded = self.dish_image_decoded(dish_image_sa + dish_image_aligned)

        return ingredients_encoded, dish_image_decoded

############################################################################################################
'''
Stacking Attention Encoder-Decoder Network
'''

class SMEDA(nn.Module):
    def __init__(self, num_heads, d_model, dropout=0.3, num_layers=3, epsilon=1e-6):
        super(SMEDA, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.dropout = dropout
        self.epsilon = epsilon
        self.num_layers = num_layers
        
        # Define stacks of Ingredient Encoders and Dish Decoders
        self.ingredient_encoders = nn.ModuleList([
            nn.ModuleDict({
                'self_attention': nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, dropout=dropout),
                'self_attention_norm': nn.LayerNorm(d_model, eps=epsilon),
                'alignment': nn.Linear(d_model, d_model),
                'encoding': nn.LayerNorm(d_model, eps=epsilon),
            })
            for _ in range(num_layers)
        ])
        
        self.dish_image_decoders = nn.ModuleList([
            nn.ModuleDict({
                'guided_attention': nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, dropout=dropout),
                'guided_attention_norm': nn.LayerNorm(d_model, eps=epsilon),
                'self_attention': nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, dropout=dropout),
                'self_attention_norm': nn.LayerNorm(d_model, eps=epsilon),
                'alignment': nn.Linear(d_model, d_model),
                'decoding': nn.LayerNorm(d_model, eps=epsilon),
            })
            for _ in range(num_layers)
        ])

    def forward(self, ingredients, dish_image):
        # Encoder: Ingredients Self-Attention
        
        for i in range(self.num_layers):
            # Ingredient Encoding
            ingr_self_attn, _ = self.ingredient_encoders[i]['self_attention'](ingredients, ingredients, ingredients)
            ingr_self_attn = self.ingredient_encoders[i]['self_attention_norm'](ingr_self_attn + ingredients)
            ingr_aligned = F.relu(self.ingredient_encoders[i]['alignment'](ingr_self_attn))
            ingr_encoded = self.ingredient_encoders[i]['encoding'](ingr_self_attn + ingr_aligned) # (num_ingr, d_model)

            # Dish Image Decoding with Cross-Attention
            dish_guided_attn, _ = self.dish_image_decoders[i]['guided_attention'](dish_image, ingr_encoded, ingr_encoded)
            dish_guided_attn = self.dish_image_decoders[i]['guided_attention_norm'](dish_guided_attn + dish_image)
            
            # Dish Image self-attention
            dish_self_attn, _ = self.dish_image_decoders[i]['self_attention'](dish_guided_attn, dish_guided_attn, dish_guided_attn)
            dish_self_attn = self.dish_image_decoders[i]['self_attention_norm'](dish_self_attn + dish_guided_attn)
            dish_aligned = F.relu(self.dish_image_decoders[i]['alignment'](dish_self_attn))
            dish_decoded = self.dish_image_decoders[i]['decoding'](dish_self_attn + dish_aligned) # (batch_size, d_model)
            
            # Update the dish image features for the next layer
            dish_image = dish_decoded
            
            # Update the ingredient features for the next layer
            ingredients = ingr_encoded
            

        return ingr_encoded, dish_decoded
        
class NutriFusionNet(nn.Module):
    def __init__(self, num_ingr = 199, backbone = "resnet", ingredient_embedding = None, pretrained = True, hidden_dim = 512, lstm_layers = 1, num_heads = 8, dropout = 0.3, epsilon = 1e-6, num_layers = 2):
        super(NutriFusionNet, self).__init__()
        
        self.ingredient_embedding = ingredient_embedding
        
        self.ingredient_extractor = IngredientFeatureExtractor(embedding_dim=ingredient_embedding.size(1), hidden_dim=hidden_dim, lstm_layers=lstm_layers)
        self.image_extractor = ImageFeatureExtractor(backbone=backbone, pretrained = pretrained, hidden_dim=hidden_dim)
        
        if num_layers == 1:
            self.attn_enc_dec = AttentionEncoderDecoder(num_heads, hidden_dim, dropout, epsilon)
        elif num_layers > 1:
            self.attn_enc_dec = SMEDA(num_heads, hidden_dim, dropout, num_layers, epsilon)
        else:
            raise ValueError(f"Invalid number of layers {num_layers}")

        self.fusion_layer = nn.Linear(hidden_dim * 2, hidden_dim)
        
        # Mass prediction
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_ingr),
        )

    def forward(self, images):
        # Extract features
        ingredient_embeddings = self.ingredient_embedding.unsqueeze(0).repeat(images.size(0), 1, 1)
        ingredients_features = self.ingredient_extractor(ingredient_embeddings)
        dish_image_features = self.image_extractor(images)
                
        # Attention Encoder-Decoder
        ingredients_encoded, dish_image_decoded = self.attn_enc_dec(ingredients_features, dish_image_features)

        # Global pooling
        # Multimodal fusion
        fused_features = torch.cat([ingredients_encoded, dish_image_decoded], dim=1) # (batch_size, d_model * 2)
        fused_features = self.fusion_layer(fused_features) # (batch_size, d_model)

        # Mass prediction
        prediction = self.mlp(fused_features) # (batch_size, num_ingr)
        return prediction
    
############################################################################################################
if __name__ == "__main__":
    x = torch.randn(16, 3, 224, 224) # For ConvLSTM model and ViT model
    num_ingr = 199

    embed = torch.randn(num_ingr, 768) # Assume 768-dimensional embeddings
    tasks = ["calories", "mass", "fat", "carb", "protein"]
        
    print("\nTesting the BaselineModel")
    test_model = BaselineModel(num_ingr)
    output = test_model(x)
    print(f"Output shape: {output.shape}")
    
    print("\nTesting the IngrModel")
    model = IngrPredModel(num_ingr, "vit", pretrained=False)
    output = model(x)
    print(f"Output shape: {output.shape}")       

    print("\nTesting the NutriFusionNet")
    model = NutriFusionNet(num_ingr, "vit", embed, hidden_dim=512)
    output = model(x)
    print(f"Output shape: {output.shape}")
    
    print("\nAll tests passed!")