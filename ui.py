import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import torch
from models.models import BaselineModel, IngrPredModel, MultimodalPredictionNetwork
from torchvision import transforms
import os
import pandas as pd
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#============== Load the model ==============#
embeddings = torch.load(f'./utils/data/ingredient_embeddings_gnn_gat.pt', map_location=device, weights_only=True)

model = MultimodalPredictionNetwork(num_ingr=199, backbone='resnet', ingredient_embedding=embeddings).to(device)
model.load_state_dict(torch.load("models/checkpoints/multimodal_resnet_gat_pretrained_da_16_75_25.pth", map_location=device, weights_only=True))

# Preprocess the image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def preprocess_image(image_path):
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)
    return image.to(device)

#============== Load the ingredient metadata ==============#
dataset_path = '../data/nutrition5k_reconstructed/'

image_path = os.path.join(dataset_path, 'images')
ingr_mata = os.path.join(dataset_path, 'metadata/ingredients_metadata.csv')

# Load the ingredient metadata
ingr_dataset_path = './utils/data/test_labels_ingr_log.csv'
ingr_df = pd.read_csv(ingr_mata)
ingr_index = {}
ingr_indx_df = pd.read_csv(ingr_dataset_path)
colnames = ingr_indx_df.columns[1:-1]
for i in range(len(colnames)):
    ingr_index[i] = colnames[i]

# ingr,id,cal/g,fat(g),carb(g),protein(g)
ingr_dict = {}
for i in range(len(ingr_df)):
    ingr = ingr_df.iloc[i]['ingr']
    cal = ingr_df.iloc[i]['cal/g']
    fat = ingr_df.iloc[i]['fat(g)']
    carb = ingr_df.iloc[i]['carb(g)']
    protein = ingr_df.iloc[i]['protein(g)']
    ingr_dict[ingr] = (cal, fat, carb, protein)

#============== Inference function ==============#
def calculate_ingrdients_and_nutritional_facts(outputs, ingr_dict, ingr_index, k = 3):
    # outputs will always be (1, 199) since we are processing one image at a time
    outputs = outputs.squeeze(0).cpu().detach().numpy()
    sample_calories = 0
    sample_mass = 0
    sample_fat = 0
    sample_carbs = 0
    sample_protein = 0
    top_k_ingr = []
    
    idx = np.where(outputs > 0.0)[0]
    if len(idx) != 0:
        # Dictionary to store ingredient mass for top k selection
        ingredient_masses = {}
        for ingr_idx in idx:
            mass = outputs[ingr_idx]
            # Convert back from log scale
            mass = np.exp(mass) - 1
            ingr_name = ingr_index[ingr_idx]
            cal, fat, carb, protein = ingr_dict[ingr_name]
            
            sample_calories += mass * cal
            sample_fat += mass * fat
            sample_carbs += mass * carb
            sample_protein += mass * protein
            sample_mass += mass
            
            # Store ingredient name and mass
            ingredient_masses[ingr_name] = mass
            
        # Sort ingredients by mass and select the top 3
        sorted_ingredients = sorted(ingredient_masses.items(), key=lambda x: x[1], reverse=True)
        top_k_ingr = [ingr for ingr, _ in sorted_ingredients[:k]]
    else:
        top_k_ingr = ["No ingredients detected"]
        
    return {
        "calories": sample_calories,
        "mass": sample_mass,
        "fat": sample_fat,
        "carbs": sample_carbs,
        "protein": sample_protein,
        "top_k_ingr": top_k_ingr
    }
    
#============== Process the image ==============#
# Function to process the image and predict
def process_image():
    # Get the file path
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")])
    if not file_path:
        # No file selected, output a message
        ingredients_text.set("No image selected. Please try again.")
        stats_text.set("")
        return

    # Load and display the image
    image = Image.open(file_path)
    image.thumbnail((300, 300))  # Resize for display
    img_display = ImageTk.PhotoImage(image)
    img_label.config(image=img_display)
    img_label.image = img_display
    
    # Preprocess the image
    image_tensor = preprocess_image(file_path)
    
    outputs = model(image_tensor)
    results = calculate_ingrdients_and_nutritional_facts(outputs, ingr_dict, ingr_index)
    ingredients = results["top_k_ingr"]
    nutritional_facts = {
        "Calories": results["calories"],
        "Total Mass (g)": results["mass"],
        "Fat (g)": results["fat"],
        "Carbs (g)": results["carbs"],
        "Protein (g)": results["protein"],
    }

    # Update the UI with the predictions
    ingredients_text.set("\n".join(ingredients))
    stats_text.set("\n".join([f"{k}: {v:.2f}" for k, v in nutritional_facts.items()]))
    
    return ingredients, nutritional_facts

# Create the UI
root = tk.Tk()
root.title("Image to Ingredients & Nutrition Facts")

# File selection and display frame
frame_top = tk.Frame(root)
frame_top.pack(pady=10)

select_button = tk.Button(frame_top, text="Select Image", command=process_image)
select_button.pack(side=tk.LEFT, padx=10)

img_label = tk.Label(frame_top)
img_label.pack(side=tk.RIGHT, padx=10)

# Results frame
frame_results = tk.Frame(root)
frame_results.pack(pady=10)

# Ingredients
ingredients_label = tk.Label(frame_results, text="Ingredients:", font=("Arial", 14))
ingredients_label.grid(row=0, column=0, sticky=tk.W, padx=10, pady=5)
ingredients_text = tk.StringVar()
ingredients_display = tk.Label(frame_results, textvariable=ingredients_text, justify=tk.LEFT, font=("Arial", 12))
ingredients_display.grid(row=1, column=0, sticky=tk.W, padx=10)

# Nutritional Facts
stats_label = tk.Label(frame_results, text="Nutritional Facts:", font=("Arial", 14))
stats_label.grid(row=0, column=1, sticky=tk.W, padx=10, pady=5)
stats_text = tk.StringVar()
stats_display = tk.Label(frame_results, textvariable=stats_text, justify=tk.LEFT, font=("Arial", 12))
stats_display.grid(row=1, column=1, sticky=tk.W, padx=10)

if __name__ == '__main__':
    root.mainloop()
    # # test the model
    # image_file = '../data/nutrition5k_reconstructed/images/dish_1568048053.jpeg'
    # process_image(image_file)
