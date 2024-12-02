import os
import pandas as pd
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torch

##################################################
class IngredientDataset(Dataset):
    def __init__(self, dataframe_id, dataframe, image_dir, transform=None):
        self.dataframe_id = dataframe_id
        self.dataframe = dataframe
        self.image_dir = image_dir
        # labels or from the second column to the first to the last column
        self.labels = self.dataframe.columns[1 : -1]
        self.transform = transform
        
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.dataframe.iloc[idx]["img_indx"])
        image = Image.open(img_name).convert("RGB")
        label_classification = torch.tensor(self.dataframe_id.iloc[idx][self.labels].values.astype(float), dtype=torch.float32)
        label_regression = torch.tensor(self.dataframe.iloc[idx][self.labels].values.astype(float), dtype=torch.float32)
        label = (label_classification, label_regression)
        if self.transform:
            image = self.transform(image)
        return image, label

class NutritionDataset(Dataset):
    def __init__(self, dataframe, image_dir, labels, transform=None):
        self.dataframe = dataframe
        self.image_dir = image_dir
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.dataframe.iloc[idx]["img_indx"])
        image = Image.open(img_name).convert("RGB")
        label = torch.tensor(self.dataframe.iloc[idx][self.labels].values.astype(float), dtype=torch.float32)
        if self.transform:
            image = self.transform(image)
        return image, label
    
##################################################
def load_data(df_train, df_val, image_path, labels, img_dim, batch_size = 16, da = True):
    IMG_DIMN = img_dim
    BATCH_SIZE = batch_size

    if da == True:
        # Transforms
        train_transforms = transforms.Compose([
            transforms.RandomRotation(30),  # Randomly rotate the image by up to 30 degrees
            transforms.RandomAffine(0, shear=0.15),  # Shear the image by up to 0.15
            transforms.RandomResizedCrop(size=(IMG_DIMN, IMG_DIMN), scale=(0.6, 1.0)),  # Zoom range equivalent
            transforms.ColorJitter(brightness=(0.7, 1.0)),  # Brightness range [0.7, 1.0]
            transforms.RandomHorizontalFlip(),  # Random horizontal flip
            transforms.ToTensor(),  # Convert image to PyTorch tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Rescale and normalize
        ])
    else:
        train_transforms = transforms.Compose([
            transforms.Resize((IMG_DIMN, IMG_DIMN)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    val_transforms = transforms.Compose([
        transforms.Resize((IMG_DIMN, IMG_DIMN)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Datasets and DataLoaders
    train_dataset = NutritionDataset(df_train, image_path, labels, transform=train_transforms)
    val_dataset = NutritionDataset(df_val, image_path, labels, transform=val_transforms)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, val_loader

def load_ingr_data(df_train_ingr_id, df_val_ingr_id, df_train_ingr, df_val_ingr, image_path, img_dim, batch_size = 16, da = True):
    IMG_DIMN = img_dim
    BATCH_SIZE = batch_size

    # Transforms
    if da == True:
        train_transforms = transforms.Compose([
            transforms.RandomRotation(30),  # Randomly rotate the image by up to 30 degrees
            transforms.RandomAffine(0, shear=0.15),  # Shear the image by up to 0.15
            transforms.RandomResizedCrop(size=(IMG_DIMN, IMG_DIMN), scale=(0.6, 1.0)),  # Zoom range equivalent
            transforms.ColorJitter(brightness=(0.7, 1.0)),  # Brightness range [0.7, 1.0]
            transforms.RandomHorizontalFlip(),  # Random horizontal flip
            transforms.ToTensor(),  # Convert image to PyTorch tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Rescale and normalize
        ])
    else:
        train_transforms = transforms.Compose([
            transforms.Resize((IMG_DIMN, IMG_DIMN)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    val_transforms = transforms.Compose([
        transforms.Resize((IMG_DIMN, IMG_DIMN)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Datasets and DataLoaders
    train_dataset = IngredientDataset(df_train_ingr_id, df_train_ingr, image_path, transform=train_transforms)
    val_dataset = IngredientDataset(df_val_ingr_id, df_val_ingr, image_path, transform=val_transforms)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, val_loader

def prepare_test_loader(df_test, image_path, labels, img_dim, batch_size = 16):
    IMG_DIMN = img_dim
    BATCH_SIZE = batch_size

    # Transforms
    test_transforms = transforms.Compose([
        transforms.Resize((IMG_DIMN, IMG_DIMN)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Datasets and DataLoaders
    test_dataset = NutritionDataset(df_test, image_path, labels, transform=test_transforms)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return test_loader

def prepare_test_loader_ingr(df_test_id, df_test, image_path, img_dim, batch_size = 16):
    IMG_DIMN = img_dim
    BATCH_SIZE = batch_size
    
    # Transforms
    test_transforms = transforms.Compose([
        transforms.Resize((IMG_DIMN, IMG_DIMN)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Datasets and DataLoaders    
    test_dataset = IngredientDataset(df_test_id, df_test, image_path, transform=test_transforms)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    return test_loader


if __name__ == '__main__':
    dataset_path = '../../data/nutrition5k_reconstructed/'
    prepared_path = './data'

    image_path = os.path.join(dataset_path, 'images')
    train_labels = os.path.join(prepared_path, 'train_labels_log.csv')
    val_labels = os.path.join(prepared_path, 'val_labels_log.csv')
    test_labels = os.path.join(prepared_path, 'test_labels_log.csv')

    df_train = pd.read_csv(train_labels)
    df_val = pd.read_csv(val_labels)
    df_test = pd.read_csv(test_labels)

    train_loader, val_loader = load_data(df_train, df_val, image_path, labels=["calories", "mass", "fat", "carb", "protein"], img_dim=224,batch_size=16)
    test_loader = prepare_test_loader(df_test, image_path, labels=["calories", "mass", "fat", "carb", "protein"], img_dim=224, batch_size=16)
    print('Data Preprocessing Done')
    for i, (images, labels) in enumerate(train_loader):
        print(images.shape, labels.shape)
        break
    for i, (images, labels) in enumerate(val_loader):
        print(images.shape, labels.shape)
        break
    for i, (images, labels) in enumerate(test_loader):
        print(images.shape, labels.shape)
        break
    
    dataset_path = '../../data/nutrition5k_reconstructed/'
    prepared_path = './data'

    image_path = os.path.join(dataset_path, 'images')
    train_labels = os.path.join(prepared_path, 'train_labels_ingr_log.csv')
    val_labels = os.path.join(prepared_path, 'val_labels_ingr_log.csv')
    test_labels = os.path.join(prepared_path, 'test_labels_ingr_log.csv')
    
    train_id = os.path.join(prepared_path, 'train_labels_ingr_id.csv')
    val_id = os.path.join(prepared_path, 'val_labels_ingr_id.csv')
    test_id = os.path.join(prepared_path, 'test_labels_ingr_id.csv')

    df_train = pd.read_csv(train_labels)
    df_val = pd.read_csv(val_labels)
    df_test = pd.read_csv(test_labels)
    
    df_train_id = pd.read_csv(train_id)
    df_val_id = pd.read_csv(val_id)
    df_test_id = pd.read_csv(test_id)
    
    train_loader, val_loader = load_ingr_data(df_train_id, df_val_id, df_train, df_val, image_path, img_dim=224, batch_size=16)
    test_loader = prepare_test_loader_ingr(df_test_id, df_test, image_path, img_dim=224, batch_size=16)
    print('Data Preprocessing Done')
    for i, (images, labels) in enumerate(train_loader):
        print(images.shape)
        print(labels[0].shape, labels[1].shape)
        print(labels[0][0], labels[1][0])
        break
    for i, (images, labels) in enumerate(val_loader):
        print(images.shape)
        print(labels[0].shape, labels[1].shape)
        break
    
    for i, (images, labels) in enumerate(test_loader):
        print(images.shape)
        print(labels[0].shape, labels[1].shape)
        break