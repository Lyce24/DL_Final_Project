import os
import pandas as pd
import torch
from torchvision import transforms
from PIL import Image
import os
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset


def load_images(image_path, transform):
    image_dict = {}
    for img in os.listdir(image_path):
        idx = img.split('.')[0]
        abs_path = os.path.join(image_path, img)
        img = Image.open(abs_path).convert('RGB')
        img = transform(img)
        image_dict[idx] = img
    return image_dict

def load_masses(file_path):
    mass_dict = {}
    df = pd.read_csv(file_path)
    for idx, row in df.iterrows():
        mass_dict[row['id']] = torch.tensor(row['mass'])
        
    return mass_dict

# If mass is true, then mass is passed as an input to the model
def prepare_dataloader(labels_path, image_dict, mass_dict=None, batch=32, shuffle=True, mass=True):
    # Load the DataFrame once
    df = pd.read_csv(labels_path)
    
    # Extract IDs
    ids = df['id'].values
    
    # Prepare inputs and labels
    inputs = torch.stack([image_dict[idx] for idx in ids])
    labels = torch.tensor(df.drop(columns=['id', 'mass'] if mass else ['id']).values, dtype=torch.float32)

    if mass:
        # Prepare mass inputs only if mass is True
        masses = torch.tensor([mass_dict[idx] for idx in ids], dtype=torch.float32).view(-1, 1)
        dataset = TensorDataset(inputs, masses, labels)
    else:
        dataset = TensorDataset(inputs, labels)

    # Create the DataLoader
    dataloader = DataLoader(dataset, batch_size=batch, shuffle=shuffle)
    
    return dataloader

    
def prepare_dataloaders(image_path, mass_path, train_labels, val_labels, test_labels, img_tranform, batch_size = 32, shuffle = True, mass = True):
    image_dict = load_images(image_path, img_tranform)
    mass_dict = load_masses(mass_path)
    
    train_loader = prepare_dataloader(train_labels, image_dict, mass_dict, batch = batch_size, shuffle = shuffle, mass = mass)
    val_loader = prepare_dataloader(val_labels, image_dict, mass_dict, batch = batch_size, shuffle = shuffle, mass = mass)
    test_loader = prepare_dataloader(test_labels, image_dict, mass_dict, batch = batch_size, shuffle = shuffle, mass = mass)
    
    return train_loader, val_loader, test_loader
    
        
        
if __name__ == '__main__':
    dataset_path = '../../data/nutrition5k_reconstructed/'
    prepared_path = './data'

    image_path = os.path.join(dataset_path, 'images')
    train_labels = os.path.join(prepared_path, 'train_labels.csv')
    val_labels= os.path.join(prepared_path, 'val_labels.csv')
    test_labels = os.path.join(prepared_path, 'test_labels.csv')
    mass_inputs = os.path.join(prepared_path, 'mass_inputs.csv')

    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor()
    ])
    
    train_loader, val_loader, test_loader = prepare_dataloaders(image_path, mass_inputs, train_labels, val_labels, test_labels, transform, batch_size = 32, shuffle = True, mass = True)
    for (train_batch, val_batch, test_batch) in zip(train_loader, val_loader, test_loader):
        print(train_batch[0].shape, train_batch[1].shape, train_batch[2].shape)
        print(val_batch[0].shape, val_batch[1].shape, val_batch[2].shape)
        print(test_batch[0].shape, test_batch[1].shape, test_batch[2].shape)
        break