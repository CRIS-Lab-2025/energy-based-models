import numpy as np
import h5py
import torch
import tqdm 
def extract_hidden_representations(model, data_loader, device):
    """
    Extract hidden layer representations from the model for all samples in the data loader.
    
    Args:
        model (EnergyBasedModel): Trained model
        data_loader (DataLoader): DataLoader containing the dataset
        device (torch.device): Device to run on
        
    Returns:
        dict: Dictionary containing images, labels, and hidden representations
    """
    model.eval()
    hidden_reps = {f'layer_{i}': [] for i in range(len(model.layer_sizes))}
    images = []
    labels = []
    
    with torch.no_grad():
        for data, target in tqdm(data_loader, desc='Extracting representations'):
            data, target = data.to(device), target.to(device)
            
            # Get hidden representations
            states = model.negative(data)
            
            # Store representations
            for i, state in enumerate(states):
                hidden_reps[f'layer_{i}'].append(state.cpu().numpy())
            
            # Store images and labels
            images.append(data.cpu().numpy())
            labels.append(target.cpu().numpy())
    
    # Concatenate all batches
    for layer in hidden_reps:
        hidden_reps[layer] = np.concatenate(hidden_reps[layer], axis=0)
    images = np.concatenate(images, axis=0)
    labels = np.concatenate(labels, axis=0)
    
    return {
        'images': images,
        'labels': labels,
        'hidden_representations': hidden_reps
    }

def save_to_hdf5(data_dict, filename):
    """
    Save the extracted representations to an HDF5 file.
    
    Args:
        data_dict (dict): Dictionary containing the data to save
        filename (str): Path to save the HDF5 file
    """
    with h5py.File(filename, 'w') as f:
        # Save images
        f.create_dataset('images', data=data_dict['images'])
        
        # Save labels
        f.create_dataset('labels', data=data_dict['labels'])
        
        # Save hidden representations
        hidden_group = f.create_group('hidden_representations')
        for layer_name, data in data_dict['hidden_representations'].items():
            hidden_group.create_dataset(layer_name, data=data)