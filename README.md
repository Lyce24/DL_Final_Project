# What has been done
## 1. Data Reconstruction
Nutrition5k dataset contains both overhead images and side-view videos (NOT IMAGE). In this project, we only focus on the overhead images. The dataset is reconstructed to a new dataset with the following structure:
```
nutrition5k_reconstructed
/images
    - dish_{}.jpeg
    - dish_{}.jpeg
    ...
/labels
    /labels
/metadata
    /train_ids.csv
    /test_ids.csv
```
The `images` folder contains all the overhead images of the dishes. The `labels` folder contains the labels of the dishes including the nutrition information (calories, mass, fat, carbs, and protein). The `metadata` folder contains the train and test ids. Note that the dataset contains incremental nature of dishes, which means that the same dish can appear multiple times in the dataset. The train and test ids are from the original dataset where we don't have any overlapping images of the same plate.

## 2. Data Preprocessing and Splitting
In `utils/preprocess.py` and `utils/data_preparation.ipynb`, we preprocess the images by resizing them to certain dimensions and normalizing the pixel values. The labels are also preprocessed by normalizing the values. The images are then split into train and test sets. Furthermore, the test set is split into validation and test sets. The data is saved in the `data` folder.

## 3. Model Training
### 3.1. Baseline Model
In `baseline_model.py`, we train a simple CRNN model.

### 3.2. InceptionV3 Model 
InceptionV3 Backbone + Fully Connected Heads for each output. The model is trained in `inceptionv3_model.py`. Note that we pass the masses of the dishes as the input to the model. 

Input: (224, 224, 3) image, (1,) mass
Output: (1,) calories, (1,) fat, (1,) carbs, (1,) protein

## 4. LLM Inference
In `llm_inference.py`, we use Llama 3.2 from Facebook AI to infer the nutrition information of the dishes. 


# Future Work
## 1. Data Augmentation
We can apply data augmentation techniques to the images to increase the size of the dataset.

## 2. Model Improvement and Training
We can improve the model by using more complex architectures and training for more epochs.

## 3. LLM Improvement
Prompt Engineering to ensure that it outputs the correct nutrition information in the right format.

## 4. Distillation Process
When LLM is improved, we can use the LLM model to distill the knowledge to the CRNN model and InceptionV3 model. Or, from InceptionV3 to CRNN model.

## 5. Model Explanation using SHAP
We can use SHAP to explain the model's predictions.