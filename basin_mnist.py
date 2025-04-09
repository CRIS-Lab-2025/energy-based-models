import os
import gzip
import pickle
import numpy as np
import torch
from urllib.request import urlretrieve
import matplotlib.pyplot as plt
from tqdm import tqdm
from ebm.model import Network
import networkx as nx
from ebm.external_world import ExternalWorld, MNISTExternalWorld
import h5py  # Add h5py import for HDF5 file support

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

def train_net(net: Network, plot_graph = False):
    history = {"Energy": [], "Cost": [], "Error": []}
    epochs, batch_size = net.hyperparameters["n_epochs"], net.hyperparameters["batch_size"]
    n_batches = net.dataset_size // batch_size
    n_it_neg, n_it_pos, alphas = net.hyperparameters["n_it_neg"], net.hyperparameters["n_it_pos"], net.hyperparameters["alphas"]


    snapshot_epochs = np.linspace(0, epochs - 1, 5, dtype=int)
    with tqdm(total=epochs, desc="Training Progress", unit="epoch") as epoch_bar:
        for epoch in range(epochs):
            for i in range(n_batches):
                net.update_mini_batch_index(i)
                net.negative_phase(n_it_neg)
                net.positive_phase(n_it_pos, *alphas)

            # Measure and log
            E, C, error = net.measure()
            history["Energy"].append(E)
            history["Cost"].append(C)
            history["Error"].append(error * 100)

            # Update progress bar description instead of using set_postfix()
            epoch_bar.set_description(f"Epoch {epoch+1}/{epochs} | E={E:.2f} C={C:.5f} Error={error*100:.2f}%")
            epoch_bar.update(1)

    # Plot results
    fig, axes = plt.subplots(1, 3, figsize=(18, 4))
    for ax, (key, color) in zip(axes, [("Energy", "blue"), ("Cost", "orange"), ("Error", "red")]):
        ax.plot(history[key], label=key, color=color)
        ax.set_title(f"{key} over Epochs")
    plt.tight_layout()
    plt.show()
net=Network(
        name="mnist", 
        external_world=MNISTExternalWorld(), 
        hyperparameters={
            "hidden_sizes": [1000],
            "n_epochs": 10,
            "batch_size": 20,
            "n_it_neg": 1,
            "n_it_pos": 1,
            "alphas": [np.float32(0.4), np.float32(0.1), np.float32(0.008)],
            "output_size": 10,
            "activation": "pi"
                                                                    }
    )
train_net(net)


# MNIST Dataloader 
# Create a DataLoader for MNIST dataset
mnist_world = MNISTExternalWorld()
batch_size = 20
mnist_data = torch.utils.data.TensorDataset(mnist_world.x, mnist_world.y)
mnist_loader = torch.utils.data.DataLoader(mnist_data, batch_size=batch_size, shuffle=True)

# Create HDF5 file and initialize datasets
# Construct path relative to the script directory
forward_h5_path = os.path.join(script_dir, 'forward_data.h5')
with h5py.File(forward_h5_path, 'w') as f:
    # Get total size to pre-allocate datasets
    total_samples = len(mnist_data)
    
    # Create datasets with initial shape but expandable
    hidden_shape = (0, net.hyperparameters["hidden_sizes"][0])
    images_shape = (0,) + tuple(mnist_world.x.shape[1:])
    labels_shape = (0, 10)  # One-hot encoded labels
    
    # Create resizable datasets
    hidden_dset = f.create_dataset('hidden_states', shape=hidden_shape, 
                                  maxshape=(total_samples,) + hidden_shape[1:],
                                  chunks=True, dtype='float32')
    images_dset = f.create_dataset('images', shape=images_shape, 
                                  maxshape=(total_samples,) + images_shape[1:],
                                  chunks=True, dtype='float32')
    labels_dset = f.create_dataset('labels', shape=labels_shape, 
                                  maxshape=(total_samples,) + labels_shape[1:],
                                  chunks=True, dtype='float32')
    
    # Process data in batches and update HDF5 file
    current_idx = 0
    with torch.no_grad():
        for data, label in tqdm(mnist_loader, desc="Processing MNIST data"):
            # Forward pass through the network
            En, Cn, error = net.forward((data, label), 10)
            
            # Get batch size for this iteration
            batch_current_size = data.shape[0]
            
            # Convert to numpy and one-hot encode labels
            hidden_batch = net.layers[1].detach().numpy()
            images_batch = data.detach().numpy()
            labels_batch = label.detach().numpy()
            labels_one_hot = np.eye(10)[labels_batch]
            
            # Resize datasets to accommodate new data
            new_size = current_idx + batch_current_size
            hidden_dset.resize(new_size, axis=0)
            images_dset.resize(new_size, axis=0)
            labels_dset.resize(new_size, axis=0)
            
            # Store data in HDF5 file
            hidden_dset[current_idx:new_size] = hidden_batch
            images_dset[current_idx:new_size] = images_batch
            labels_dset[current_idx:new_size] = labels_one_hot
            
            # Update index
            current_idx = new_size

# Create HDF5 file for streaming data
# Construct path relative to the script directory
backward_h5_path = os.path.join(script_dir, 'backward_data.h5')
with h5py.File(backward_h5_path, 'w') as f:
    # Create resizable datasets that we'll append to
    images_dset = f.create_dataset('images', shape=(0, 28, 28), maxshape=(None, 28, 28), 
                                  chunks=True, dtype='float32')
    targets_dset = f.create_dataset('labels', shape=(0, 10), maxshape=(None, 10), 
                                   chunks=True, dtype='float32')
    hidden_dset = f.create_dataset('hidden_states', shape=(0, 1000), maxshape=(None, 1000), 
                                  chunks=True, dtype='float32')
    
    # Set metadata
    f.attrs['description'] = 'Reverse MNIST with random probability distributions'
    
    # Track total samples for dataset resizing
    total_samples = 0
    
    for prob in torch.arange(1.0, 0.0, -0.1):
        # For each digit (0-9), create 10 different random distributions
        batch_targets = []
        
        for digit in range(10):
            for variant in range(10):  # 10 random distributions per digit and probability
                # Create target with the main digit having probability 'prob'
                current_target = torch.zeros(1, 10)
                current_target[0, digit] = prob
                
                # Distribute remaining probability randomly among other classes
                remaining_prob = 1.0 - prob
                
                # Generate random values for other classes
                other_classes = [j for j in range(10) if j != digit]
                random_values = torch.rand(9)
                
                # Normalize to sum to remaining_prob
                random_values = remaining_prob * random_values / random_values.sum()
                
                # Assign random probabilities to other classes
                for idx, j in enumerate(other_classes):
                    current_target[0, j] = random_values[idx]
                
                # Add to our batch targets
                batch_targets.append(current_target)
        
        # Convert batch targets to tensor
        target = torch.cat(batch_targets, dim=0)
        
        # Generate images using reverse inference (in batches)
        batch_size = 20  # Process in smaller batches to avoid memory issues
        num_samples = target.shape[0]
        
        for i in range(0, num_samples, batch_size):
            end_idx = min(i + batch_size, num_samples)
            batch_target = target[i:end_idx]
            
            # Generate images
            batch_images = net.backward(batch_target, 10)
            
            
            # Get current batch size
            current_batch_size = batch_target.shape[0]
            
            # Resize datasets to accommodate new data
            images_dset.resize(total_samples + current_batch_size, axis=0)
            targets_dset.resize(total_samples + current_batch_size, axis=0)
            hidden_dset.resize(total_samples + current_batch_size, axis=0)
            
            # Store data directly to HDF5
            images_dset[total_samples:total_samples + current_batch_size] = batch_images.detach().cpu().numpy().reshape(current_batch_size, 28, 28)
            targets_dset[total_samples:total_samples + current_batch_size] = batch_target.detach().cpu().numpy()
            hidden_dset[total_samples:total_samples + current_batch_size] = net.layers[1].detach().cpu().numpy()
            
            # Update total samples count
            total_samples += current_batch_size
            
            # Flush to ensure data is written
            f.flush()


# Now for both hidden_states and hidden_states_backward, do uMAP and reduce to 2D. Then plot the 2D scatter plot of the two with the labels
# Removed the direct call to umap_basin to prevent running it automatically
# import umap_basin # This line is removed/commented out

# The following UMAP processing block is removed as it's now handled by umap_basin.py
# # Load data from HDF5 files
# with h5py.File(forward_h5_path, 'r') as f_forward, h5py.File(backward_h5_path, 'r') as f_backward:
#     # ... (rest of the UMAP code previously here is removed) ...

