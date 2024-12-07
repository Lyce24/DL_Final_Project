import torch
import torch.nn as nn
from typing import List
from torchvision.models import inception_v3, Inception_V3_Weights

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

class PaperModel(nn.Module):
    def __init__(self, tasks : List[str]):
        """
        Args:
            num_tasks: Number of tasks (calories, macronutrients, and mass).
        """
        super(PaperModel, self).__init__()
        self.backbone = InceptionV3(pretrained=True)
        
        # Shared image feature layers
        self.shared_fc1 = nn.Linear(2048, 4096) # Use 2048 as input size as InceptionV3 has 2048 output features
        self.shared_fc2 = nn.Linear(4096, 4096)
        
        # Task-specific heads
        self.task_heads = nn.ModuleDict({
            task: nn.Sequential(
                nn.Linear(4096, 4096), # work as FC 3
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
    
if __name__ == '__main__':
    # Test the InceptionV3 model
    tasks = ["calories", "mass", "fat", "carb", "protein"]
    model = PaperModel(tasks=tasks)
    image = torch.randn(32, 3, 299, 299)  # Batch size 1, 3 channels, 299x299 image
    outputs = model(image)
    for task, output in outputs.items():
        print(f"{task}: {output.size()}")