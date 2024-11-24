from torchvision import transforms
import os
from utils.preprocess import prepare_dataloaders
import torch
import torch.nn as nn
import torchvision.models as models
# Evaluate the model
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
from torchvision.models import inception_v3, Inception_V3_Weights



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

transform = transforms.Compose([
    transforms.Resize((299, 299)),  # Resize to Inception-compatible dimensions
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize if needed
])


dataset_path = '../data/nutrition5k_reconstructed/'
prepared_path = './utils/data'

image_path = os.path.join(dataset_path, 'images')
train_labels = os.path.join(prepared_path, 'train_labels.csv')
val_labels= os.path.join(prepared_path, 'val_labels.csv')
test_labels = os.path.join(prepared_path, 'test_labels.csv')
mass_inputs = os.path.join(prepared_path, 'mass_inputs.csv')

# If mass is true, then mass is passed as an input to the model
train_loader, val_loader, test_loader = prepare_dataloaders(image_path, mass_inputs, train_labels, val_labels, test_labels, transform, batch_size = 32, shuffle = True, mass = True)


# Define the InceptionV2 backbone
class InceptionV2Backbone(nn.Module):
    def __init__(self, weights=Inception_V3_Weights.DEFAULT):
        """
        Args:
            weights: Pre-trained weights to use for the InceptionV3 model. Use `None` for no pre-training.
        """
        super().__init__()
        # Load the InceptionV3 model with specified weights
        self.backbone = inception_v3(weights=weights, aux_logits=True)
        self.backbone.fc = nn.Identity()  # Remove the classification head
    
class NutritionModel(nn.Module):
    def __init__(self, num_tasks=3):
        """
        Args:
            num_tasks: Number of tasks (calories, macronutrients, and mass).
        """
        super(NutritionModel, self).__init__()
        self.backbone = InceptionV2Backbone()  # Use the corrected backbone
        
        # Shared image feature layers
        self.shared_fc1 = nn.Linear(2048, 4096)
        self.shared_fc2 = nn.Linear(4096, 4096)
        
        # Mass input processing
        self.mass_fc1 = nn.Linear(1, 128)
        self.mass_fc2 = nn.Linear(128, 256)
        
        # Task-specific heads
        self.task_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(4096 + 256, 4096),  # Adjust input size to account for concatenation
                nn.ReLU(),
                nn.Linear(4096, 1)
            ) for _ in range(num_tasks)
        ])

    def forward(self, image, mass):
        # Process the image through the backbone
        image_features = self.backbone(image)
        image_features = nn.functional.relu(self.shared_fc1(image_features))
        image_features = nn.functional.relu(self.shared_fc2(image_features))
        
        # Process the mass input
        mass_features = nn.functional.relu(self.mass_fc1(mass))
        mass_features = nn.functional.relu(self.mass_fc2(mass_features))
        
        # Concatenate image and mass features
        combined_features = torch.cat([image_features, mass_features], dim=1)
        
        # Pass through task-specific heads
        outputs = [task_head(combined_features) for task_head in self.task_heads]
        return torch.cat(outputs, dim=1)
    
    
def train_model(
    model, 
    dataloader, 
    val_dataloader, 
    criterion, 
    optimizer, 
    num_epochs=50, 
    save_path="best_model.pth"
):
    model = model.to(device)  # Move model to device
    current_best_loss = float('inf')

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0

        for batch in dataloader:
            # Unpack and move data to device
            images, masses, targets = (data.to(device) for data in batch)

            # Forward pass and optimization
            optimizer.zero_grad()
            outputs = model(images, masses)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(dataloader)
        print(f"Epoch [{epoch + 1}/{num_epochs}] - Training Loss: {avg_train_loss:.4f}")

        # Validation phase
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch in val_dataloader:
                images, masses, targets = (data.to(device) for data in batch)
                outputs = model(images, masses)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_dataloader)
        print(f"Epoch [{epoch + 1}/{num_epochs}] - Validation Loss: {avg_val_loss:.4f}")

        # Save the best model
        if avg_val_loss < current_best_loss:
            current_best_loss = avg_val_loss
            torch.save(model.state_dict(), save_path)
            print(f"Model improved. Saved at epoch {epoch + 1} with validation loss: {avg_val_loss:.4f}")

    print("Training complete!")

        
# Define the model
model = NutritionModel(num_tasks=4)  # Predicting 4 outputs: calories, fat, carbs, protein
model.to(device)

loss_fn = nn.MSELoss()  # Mean Squared Error (MSE) loss
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

# Train the model
train_model(model, train_loader, val_loader, loss_fn, optimizer, num_epochs=50, save_path="./baseline_inv2.pth")


def test_model(model, dataloader):
    """
    Tests the model and calculates MAE for each nutritional fact.
    
    Args:
        model (torch.nn.Module): Trained model to evaluate.
        dataloader (DataLoader): DataLoader for the test dataset.
    
    Returns:
        dict: MAE for each nutritional fact.
    """
    model.eval()
    all_targets = []
    all_predictions = []

    with torch.no_grad():
        for images, masses, targets in dataloader:
            # Move data to device
            images = images.to(device)
            masses = masses.to(device)
            targets = targets.to(device)

            # Make predictions
            outputs = model(images, masses)

            # Store predictions and targets
            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    # Concatenate all predictions and targets
    all_predictions = np.concatenate(all_predictions, axis=0)
    # print a few predictions
    print(all_predictions[:10])
    all_targets = np.concatenate(all_targets, axis=0)

    # Calculate MAE and MAE percentage for each nutritional fact
    results = {}
    nutritional_facts = ["calories", "fat", "carb", "protein"]
    for i, fact in enumerate(nutritional_facts):
        mae = mean_absolute_error(all_targets[:, i], all_predictions[:, i])
        mse = mean_squared_error(all_targets[:, i], all_predictions[:, i])
        mean_value = np.mean(all_targets[:, i])
        mae_percent = (mae / mean_value) * 100 if mean_value != 0 else 0
        mse_percent = (mse / mean_value) * 100 if mean_value != 0 else 0
        results[fact] = {
            "MAE": mae,
            "MSE": mse,
            "MAE (%)": mae_percent,
            "MSE (%)": mse_percent
        }

    return results

# Load the best model
model.load_state_dict(torch.load("./baseline_inv2_updated.pth"))
model.to(device)

print("Evaluating the model...")
results = test_model(model, test_loader)
print("MAE Results:")
for fact, metrics in results.items():
    print(f"{fact.capitalize()}: {metrics['MAE']:.2f} ({metrics['MAE (%)']:.2f}%), {metrics['MSE']:.2f} ({metrics['MSE (%)']:.2f}%)")