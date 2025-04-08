import umap
import h5py
import numpy as np
import matplotlib.pyplot as plt

# Load data from HDF5 files
with h5py.File('forward_data.h5', 'r') as f_forward, h5py.File('backward_data.h5', 'r') as f_backward:
    # Load data in chunks to save memory
    chunk_size = 1000
    
    # Get total sizes
    forward_size = f_forward['hidden_states'].shape[0]
    backward_size = f_backward['hidden_states'].shape[0]
    
    # Initialize UMAP model
    umap_model = umap.UMAP(n_components=2, random_state=42)
    
    # Process forward data in chunks for fitting UMAP
    all_forward_data = []
    all_forward_labels = []
    
    for i in range(0, forward_size, chunk_size):
        end_idx = min(i + chunk_size, forward_size)
        # Load and flatten chunk
        chunk_data = f_forward['hidden_states'][i:end_idx]
        chunk_data_flat = chunk_data.reshape(chunk_data.shape[0], -1)
        all_forward_data.append(chunk_data_flat)
        
        # Load labels
        chunk_labels = f_forward['labels'][i:end_idx].argmax(axis=1)
        all_forward_labels.append(chunk_labels)
    
    # Concatenate all chunks
    forward_data_flat = np.vstack(all_forward_data)
    forward_labels = np.concatenate(all_forward_labels)
    
    # Fit UMAP on forward data
    umap_result = umap_model.fit_transform(forward_data_flat)
    
    # Process backward data in chunks for transforming with UMAP
    all_backward_data = []
    all_backward_labels = []
    
    for i in range(0, backward_size, chunk_size):
        end_idx = min(i + chunk_size, backward_size)
        # Load and flatten chunk
        chunk_data = f_backward['hidden_states'][i:end_idx]
        chunk_data_flat = chunk_data.reshape(chunk_data.shape[0], -1)
        all_backward_data.append(chunk_data_flat)
        
        # Load labels
        chunk_labels = f_backward['labels'][i:end_idx].argmax(axis=1)
        all_backward_labels.append(chunk_labels)
    
    # Concatenate all chunks
    backward_data_flat = np.vstack(all_backward_data)
    backward_labels = np.concatenate(all_backward_labels)
    
    # Transform backward data using the fitted UMAP model
    umap_result_backward = umap_model.transform(backward_data_flat)
    
    # Plot the 2D scatter plot
    plt.figure(figsize=(12, 6))
    
    # Create scatter plots with different markers for forward and backward
    forward_scatter = plt.scatter(umap_result[:, 0], umap_result[:, 1], c=forward_labels, cmap='viridis')
    backward_scatter = plt.scatter(umap_result_backward[:, 0], umap_result_backward[:, 1], c=backward_labels, cmap='viridis')
    
    # # Add number labels to each point
    # for i, (x, y, label) in enumerate(zip(umap_result[:, 0], umap_result[:, 1], forward_labels)):
    #     plt.text(x, y, str(label), fontsize=8)
    
    # for i, (x, y, label) in enumerate(zip(umap_result_backward[:, 0], umap_result_backward[:, 1], backward_labels)):
    #     plt.text(x, y, str(label), fontsize=8)
    
    plt.colorbar(label='Class')
    plt.legend()
    plt.title('UMAP Projection with Class Labels')
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.savefig('umap_basin.png')
    plt.show()
